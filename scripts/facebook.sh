python main.py --name GCN --dataset facebook --model GCN --lr 3e-05 1e-04 3e-04 1e-03 3e-03 --dropout 0 0.1 0.2 --amp True --compile False --device cuda:0
python main.py --name GCN_CLATT --dataset facebook --model GCN-CLATT --clusterings h1 la --lr 3e-05 1e-04 3e-04 1e-03 3e-03 --dropout 0 0.1 0.2 --amp True --compile True --device cuda:0

python main.py --name GraphSAGE --dataset facebook --model GraphSAGE --lr 3e-05 1e-04 3e-04 1e-03 3e-03 --dropout 0 0.1 0.2 --amp True --compile False --device cuda:0
python main.py --name GraphSAGE_CLATT --dataset facebook --model GraphSAGE-CLATT --clusterings h1 la --lr 3e-05 1e-04 3e-04 1e-03 3e-03 --dropout 0 0.1 0.2 --amp True --compile True --device cuda:0

python main.py --name LGT --dataset facebook --model GT --lr 3e-05 1e-04 3e-04 1e-03 3e-03 --dropout 0 0.1 0.2 --amp True --compile False --device cuda:0
python main.py --name LGT_CLATT --dataset facebook --model GT-CLATT --clusterings h1 la --lr 3e-05 1e-04 3e-04 1e-03 3e-03 --dropout 0 0.1 0.2 --amp True --compile True --device cuda:0

python main.py --name GGT_DW --dataset facebook --model GGT --node_embeddings embs_dw --lr 3e-05 1e-04 3e-04 1e-03 3e-03 --dropout 0 0.1 0.2 --amp True --compile True --device cuda:0
python main.py --name GGT_DW_CLATT --dataset facebook --model GGT-CLATT --node_embeddings embs_dw  --clusterings h1 la --lr 3e-05 1e-04 3e-04 1e-03 3e-03 --dropout 0 0.1 0.2 --amp True --compile True --device cuda:0

python main.py --name GGT_Lap --dataset facebook --model GGT --node_embeddings embs_lap --lr 3e-05 1e-04 3e-04 1e-03 3e-03 --dropout 0 0.1 0.2 --amp True --compile True --device cuda:0
python main.py --name GGT_Lap_CLATT --dataset facebook --model GGT-CLATT --node_embeddings embs_lap  --clusterings h1 la --lr 3e-05 1e-04 3e-04 1e-03 3e-03 --dropout 0 0.1 0.2 --amp True --compile True --device cuda:0
