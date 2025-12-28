@echo off
cd /d "c:\Users\王唱晓\Downloads\ReChorus-master\ReChorus-master\src"
call c:\Users\王唱晓\Desktop\机器学习\hccf_env\Scripts\activate.bat
python main.py --model_name HCCF --emb_size 64 --dataset Grocery_and_Gourmet_Food --path ../data/ --epoch 5
pause
