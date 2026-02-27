python main.py --name GCN --dataset amazon-ratings --model GCN --lr 3e-05 1e-04 3e-04 1e-03 3e-03 --dropout 0 0.1 0.2 --max_steps 3000 --amp True --compile False --device cuda:0
python main.py --name GCN_CLATT --dataset amazon-ratings --model GCN-CLATT --clusterings la bpp --lr 3e-05 1e-04 3e-04 1e-03 3e-03 --dropout 0 0.1 0.2 --max_steps 3000 --amp True --compile True --device cuda:0

python main.py --name GraphSAGE --dataset amazon-ratings --model GraphSAGE --lr 3e-05 1e-04 3e-04 1e-03 3e-03 --dropout 0 0.1 0.2 --max_steps 3000 --amp True --compile False --device cuda:0
python main.py --name GraphSAGE_CLATT --dataset amazon-ratings --model GraphSAGE-CLATT --clusterings la bpp --lr 3e-05 1e-04 3e-04 1e-03 3e-03 --dropout 0 0.1 0.2 --max_steps 3000 --amp True --compile True --device cuda:0

python main.py --name LGT --dataset amazon-ratings --model GT --lr 3e-05 1e-04 3e-04 1e-03 3e-03 --dropout 0 0.1 0.2 --max_steps 3000 --amp True --compile False --device cuda:0
python main.py --name LGT_CLATT --dataset amazon-ratings --model GT-CLATT --clusterings la bpp --lr 3e-05 1e-04 3e-04 1e-03 3e-03 --dropout 0 0.1 0.2 --max_steps 3000 --amp True --compile True --device cuda:0

python main.py --name GGT_DW --dataset amazon-ratings --model GGT --node_embeddings embs_dw --lr 3e-05 1e-04 3e-04 1e-03 3e-03 --dropout 0 0.1 0.2 --max_steps 3000 --amp True --compile True --device cuda:0
python main.py --name GGT_DW_CLATT --dataset amazon-ratings --model GGT-CLATT --node_embeddings embs_dw  --clusterings la bpp --lr 3e-05 1e-04 3e-04 1e-03 3e-03 --dropout 0 0.1 0.2 --max_steps 3000 --amp True --compile True --device cuda:0

python main.py --name GGT_Lap --dataset amazon-ratings --model GGT --node_embeddings embs_lap --lr 3e-05 1e-04 3e-04 1e-03 3e-03 --dropout 0 0.1 0.2 --max_steps 3000 --amp True --compile True --device cuda:0
python main.py --name GGT_Lap_CLATT --dataset amazon-ratings --model GGT-CLATT --node_embeddings embs_lap  --clusterings la bpp --lr 3e-05 1e-04 3e-04 1e-03 3e-03 --dropout 0 0.1 0.2 --max_steps 3000 --amp True --compile True --device cuda:0
