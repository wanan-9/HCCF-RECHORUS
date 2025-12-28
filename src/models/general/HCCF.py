# -*- coding: UTF-8 -*-
# @Author  : Machine Learning Final Project
# @Reference: "Hypergraph Contrastive Collaborative Filtering" Xia et al., SIGIR'2022

""" HCCF
Reference:
    "Hypergraph Contrastive Collaborative Filtering"
    Xia et al., SIGIR'2022.
    arXiv: 2204.12200
CMD example:
    python main.py --model_name HCCF --emb_size 64 --n_layers 2 --lr 1e-3 --l2 1e-6 --dataset 'Grocery_and_Gourmet_Food'
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp

from models.BaseModel import GeneralModel


class HCCF(GeneralModel):
    reader = 'BaseReader'
    runner = 'BaseRunner'
    extra_log_args = ['emb_size', 'n_layers', 'temp', 'ssl_reg', 'keep_rate', 'batch_size']

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of embedding vectors.')
        parser.add_argument('--n_layers', type=int, default=2,
                            help='Number of GNN layers.')
        parser.add_argument('--temp', type=float, default=0.2,
                            help='Temperature for contrastive loss.')
        parser.add_argument('--ssl_reg', type=float, default=0.1,
                            help='Weight for contrastive loss.')
        parser.add_argument('--keep_rate', type=float, default=0.5,
                            help='Keep rate for edge dropout.')
        parser.add_argument('--leaky', type=float, default=0.5,
                            help='Slope for LeakyReLU.')
        return GeneralModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self.emb_size = args.emb_size
        self.n_layers = args.n_layers
        self.temp = args.temp
        self.ssl_reg = args.ssl_reg
        self.keep_rate = args.keep_rate
        self.leaky = args.leaky
        
        # Build adjacency matrix
        self.norm_adj = self._build_adj_matrix(corpus)
        # Build hypergraph incidence matrix
        self.hyper_adj = self._build_hypergraph(corpus)
        
        self._define_params()
        self.apply(self.init_weights)

    def _build_adj_matrix(self, corpus):
        """Build normalized adjacency matrix for graph convolution (LightGCN style)"""
        user_count = corpus.n_users
        item_count = corpus.n_items
        train_mat = corpus.train_clicked_set
        
        R = sp.dok_matrix((user_count, item_count), dtype=np.float32)
        for user in train_mat:
            for item in train_mat[user]:
                R[user, item] = 1
        R = R.tolil()

        adj_mat = sp.dok_matrix((user_count + item_count, user_count + item_count), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        adj_mat[:user_count, user_count:] = R
        adj_mat[user_count:, :user_count] = R.T
        adj_mat = adj_mat.todok()

        # D^-1/2 * A * D^-1/2
        rowsum = np.array(adj_mat.sum(1)) + 1e-10
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        norm_adj = d_mat_inv_sqrt.dot(adj_mat).dot(d_mat_inv_sqrt)
        
        return norm_adj.tocsr()

    def _build_hypergraph(self, corpus):
        """Build hypergraph incidence matrix
        In HCCF, hyperedges are defined by user neighborhoods
        H: (n_users + n_items) x n_hyperedges
        """
        user_count = corpus.n_users
        item_count = corpus.n_items
        train_mat = corpus.train_clicked_set
        
        # Each user's clicked items form a hyperedge
        # H[i, j] = 1 if node i belongs to hyperedge j
        n_hyperedges = user_count  # Each user defines a hyperedge
        
        H = sp.dok_matrix((user_count + item_count, n_hyperedges), dtype=np.float32)
        
        for user in train_mat:
            # User node belongs to its own hyperedge
            H[user, user] = 1
            # All items clicked by this user belong to this hyperedge
            for item in train_mat[user]:
                H[user_count + item, user] = 1
        
        H = H.tocsr()
        
        # Compute hypergraph Laplacian: D_v^-1/2 H D_e^-1 H^T D_v^-1/2
        # D_v: node degree matrix
        # D_e: hyperedge degree matrix
        
        D_v = np.array(H.sum(1)).flatten() + 1e-10
        D_e = np.array(H.sum(0)).flatten() + 1e-10
        
        D_v_inv_sqrt = sp.diags(np.power(D_v, -0.5))
        D_e_inv = sp.diags(np.power(D_e, -1))
        
        # Hypergraph Laplacian
        hyper_adj = D_v_inv_sqrt.dot(H).dot(D_e_inv).dot(H.T).dot(D_v_inv_sqrt)
        
        return hyper_adj.tocsr()

    def _convert_sp_mat_to_sp_tensor(self, X):
        """Convert scipy sparse matrix to torch sparse tensor"""
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def _define_params(self):
        self.user_embedding = nn.Embedding(self.user_num, self.emb_size)
        self.item_embedding = nn.Embedding(self.item_num, self.emb_size)
        
        # Convert sparse matrices to tensors
        self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.norm_adj).to(self.device)
        self.sparse_hyper_adj = self._convert_sp_mat_to_sp_tensor(self.hyper_adj).to(self.device)
        
        # Activation
        self.leaky_relu = nn.LeakyReLU(self.leaky)

    def _edge_dropout(self, adj, keep_rate):
        """Random edge dropout for graph augmentation"""
        if keep_rate >= 1.0:
            return adj
        
        vals = adj._values()
        idxs = adj._indices()
        
        # Random mask
        mask = torch.rand(vals.size()) < keep_rate
        mask = mask.to(vals.device)
        
        new_vals = vals * mask / keep_rate
        
        return torch.sparse.FloatTensor(idxs, new_vals, adj.shape)

    def graph_conv(self, ego_embeddings, adj):
        """Graph convolution layer (LightGCN style)"""
        all_embeddings = [ego_embeddings]
        
        for _ in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(adj, ego_embeddings)
            all_embeddings.append(ego_embeddings)
        
        all_embeddings = torch.stack(all_embeddings, dim=1)
        final_embeddings = torch.mean(all_embeddings, dim=1)
        
        return final_embeddings

    def hypergraph_conv(self, ego_embeddings, hyper_adj):
        """Hypergraph convolution layer"""
        all_embeddings = [ego_embeddings]
        
        for _ in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(hyper_adj, ego_embeddings)
            ego_embeddings = self.leaky_relu(ego_embeddings)
            all_embeddings.append(ego_embeddings)
        
        all_embeddings = torch.stack(all_embeddings, dim=1)
        final_embeddings = torch.mean(all_embeddings, dim=1)
        
        return final_embeddings

    def contrastive_loss(self, z1, z2, batch_idx):
        """InfoNCE contrastive loss between two views"""
        # Normalize embeddings
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        # Get batch embeddings
        z1_batch = z1[batch_idx]
        z2_batch = z2[batch_idx]
        
        # Positive pairs
        pos_score = torch.sum(z1_batch * z2_batch, dim=1) / self.temp
        
        # Negative pairs (all nodes in batch)
        neg_score = torch.mm(z1_batch, z2.T) / self.temp
        
        # InfoNCE loss
        loss = -torch.log(torch.exp(pos_score) / (torch.exp(neg_score).sum(dim=1) + 1e-10))
        
        return loss.mean()

    def forward(self, feed_dict):
        self.check_list = []
        user_ids = feed_dict['user_id']  # [batch_size]
        item_ids = feed_dict['item_id']  # [batch_size, n_candidates]
        
        # Initial embeddings
        ego_embeddings = torch.cat([
            self.user_embedding.weight,
            self.item_embedding.weight
        ], dim=0)
        
        # Graph view with edge dropout
        if self.training:
            adj_dropped = self._edge_dropout(self.sparse_norm_adj, self.keep_rate)
        else:
            adj_dropped = self.sparse_norm_adj
        
        # Graph convolution
        graph_embeddings = self.graph_conv(ego_embeddings, adj_dropped)
        
        # Hypergraph convolution
        hyper_embeddings = self.hypergraph_conv(ego_embeddings, self.sparse_hyper_adj)
        
        # Combine embeddings (average)
        final_embeddings = (graph_embeddings + hyper_embeddings) / 2
        
        # Split user and item embeddings
        user_all_emb = final_embeddings[:self.user_num]
        item_all_emb = final_embeddings[self.user_num:]
        
        # Get batch embeddings
        user_emb = user_all_emb[user_ids]  # [batch_size, emb_size]
        item_emb = item_all_emb[item_ids]  # [batch_size, n_candidates, emb_size]
        
        # Prediction
        prediction = (user_emb[:, None, :] * item_emb).sum(dim=-1)  # [batch_size, n_candidates]
        
        out_dict = {
            'prediction': prediction.view(feed_dict['batch_size'], -1),
            'graph_embeddings': graph_embeddings,
            'hyper_embeddings': hyper_embeddings,
            'user_ids': user_ids
        }
        
        return out_dict

    def loss(self, out_dict):
        """Combined loss: BPR loss + Contrastive loss"""
        # BPR loss (from parent class)
        prediction = out_dict['prediction']
        pos_pred = prediction[:, 0]
        neg_pred = prediction[:, 1:]
        neg_softmax = F.softmax(neg_pred, dim=-1)
        bpr_loss = -torch.log(torch.sigmoid(pos_pred[:, None] - neg_pred) + 1e-10)
        bpr_loss = (bpr_loss * neg_softmax).sum(dim=-1).mean()
        
        # Contrastive loss (only during training)
        if self.training:
            graph_emb = out_dict['graph_embeddings']
            hyper_emb = out_dict['hyper_embeddings']
            user_ids = out_dict['user_ids']
            
            # User contrastive loss
            user_graph_emb = graph_emb[:self.user_num]
            user_hyper_emb = hyper_emb[:self.user_num]
            user_cl_loss = self.contrastive_loss(user_graph_emb, user_hyper_emb, user_ids)
            
            # Item contrastive loss (use items from batch)
            item_graph_emb = graph_emb[self.user_num:]
            item_hyper_emb = hyper_emb[self.user_num:]
            # Sample some items for contrastive loss
            item_idx = torch.randint(0, self.item_num, (len(user_ids),), device=self.device)
            item_cl_loss = self.contrastive_loss(item_graph_emb, item_hyper_emb, item_idx)
            
            cl_loss = (user_cl_loss + item_cl_loss) / 2
            
            total_loss = bpr_loss + self.ssl_reg * cl_loss
        else:
            total_loss = bpr_loss
        
        return total_loss
