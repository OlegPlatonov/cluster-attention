python main.py --name GCN --dataset amazon-ratings --model GCN --lr 3e-05 1e-04 3e-04 1e-03 3e-03 --dropout 0 0.1 0.2 --max_steps 3000 --amp True --compile False --device cuda:0
python main.py --name GCN_CLATT --dataset amazon-ratings --model GCN-CLATT --clusterings la --lr 3e-05 1e-04 3e-04 1e-03 3e-03 --dropout 0 0.1 0.2 --max_steps 3000 --amp True --compile True --device cuda:0

python main.py --name GraphSAGE --dataset amazon-ratings --model GraphSAGE --lr 3e-05 1e-04 3e-04 1e-03 3e-03 --dropout 0 0.1 0.2 --max_steps 3000 --amp True --compile False --device cuda:0
python main.py --name GraphSAGE_CLATT --dataset amazon-ratings --model GraphSAGE-CLATT --clusterings la --lr 3e-05 1e-04 3e-04 1e-03 3e-03 --dropout 0 0.1 0.2 --max_steps 3000 --amp True --compile True --device cuda:0

python main.py --name LGT --dataset amazon-ratings --model GT --lr 3e-05 1e-04 3e-04 1e-03 3e-03 --dropout 0 0.1 0.2 --max_steps 3000 --amp True --compile False --device cuda:0
python main.py --name LGT_CLATT --dataset amazon-ratings --model GT-CLATT --clusterings la --lr 3e-05 1e-04 3e-04 1e-03 3e-03 --dropout 0 0.1 0.2 --max_steps 3000 --amp True --compile True --device cuda:0
