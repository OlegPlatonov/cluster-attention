python main.py --name GCN --dataset questions --model GCN --lr 3e-05 1e-04 3e-04 1e-03 3e-03 --dropout 0 0.1 0.2 --amp True --compile False --device cuda:0
python main.py --name GCN_CLATT --dataset questions --model GCN-CLATT --clusterings la km --max_cluster_size 2048 --lr 3e-05 1e-04 3e-04 1e-03 3e-03 --dropout 0 0.1 0.2 --amp True --compile True --device cuda:0

python main.py --name GraphSAGE --dataset questions --model GraphSAGE --lr 3e-05 1e-04 3e-04 1e-03 3e-03 --dropout 0 0.1 0.2 --amp True --compile False --device cuda:0
python main.py --name GraphSAGE_CLATT --dataset questions --model GraphSAGE-CLATT --clusterings la km --max_cluster_size 2048 --lr 3e-05 1e-04 3e-04 1e-03 3e-03 --dropout 0 0.1 0.2 --amp True --compile True --device cuda:0

python main.py --name LGT --dataset questions --model GT --lr 3e-05 1e-04 3e-04 1e-03 3e-03 --dropout 0 0.1 0.2 --amp True --compile False --device cuda:0
python main.py --name LGT_CLATT --dataset questions --model GT-CLATT --clusterings la km --max_cluster_size 2048 --lr 3e-05 1e-04 3e-04 1e-03 3e-03 --dropout 0 0.1 0.2 --amp True --compile True --device cuda:0

python main.py --name GGT_DW --dataset questions --model GGT --node_embeddings embs_dw --lr 3e-05 1e-04 3e-04 1e-03 3e-03 --dropout 0 0.1 0.2 --amp True --compile True --device cuda:0
python main.py --name GGT_DW_CLATT --dataset questions --model GGT-CLATT --node_embeddings embs_dw  --clusterings la km --max_cluster_size 2048 --lr 3e-05 1e-04 3e-04 1e-03 3e-03 --dropout 0 0.1 0.2 --amp True --compile True --device cuda:0

python main.py --name GGT_Lap --dataset questions --model GGT --node_embeddings embs_lap --lr 3e-05 1e-04 3e-04 1e-03 3e-03 --dropout 0 0.1 0.2 --amp True --compile True --device cuda:0
python main.py --name GGT_Lap_CLATT --dataset questions --model GGT-CLATT --node_embeddings embs_lap  --clusterings la km --max_cluster_size 2048 --lr 3e-05 1e-04 3e-04 1e-03 3e-03 --dropout 0 0.1 0.2 --amp True --compile True --device cuda:0
