python main.py --name GCN --dataset lastfm-asia --model GCN --lr 3e-05 1e-04 3e-04 1e-03 3e-03 --dropout 0 0.1 0.2 --amp True --compile False --device cuda:0
python main.py --name GCN_CLATT --dataset lastfm-asia --model GCN-CLATT --clusterings la h1 bpp --lr 3e-05 1e-04 3e-04 1e-03 3e-03 --dropout 0 0.1 0.2 --amp True --compile True --device cuda:0

python main.py --name GraphSAGE --dataset lastfm-asia --model GraphSAGE --lr 3e-05 1e-04 3e-04 1e-03 3e-03 --dropout 0 0.1 0.2 --amp True --compile False --device cuda:0
python main.py --name GraphSAGE_CLATT --dataset lastfm-asia --model GraphSAGE-CLATT --clusterings la h1 bpp --lr 3e-05 1e-04 3e-04 1e-03 3e-03 --dropout 0 0.1 0.2 --amp True --compile True --device cuda:0

python main.py --name LGT --dataset lastfm-asia --model GT --lr 3e-05 1e-04 3e-04 1e-03 3e-03 --dropout 0 0.1 0.2 --amp True --compile False --device cuda:0
python main.py --name LGT_CLATT --dataset lastfm-asia --model GT-CLATT --clusterings la h1 bpp --lr 3e-05 1e-04 3e-04 1e-03 3e-03 --dropout 0 0.1 0.2 --amp True --compile True --device cuda:0

python main.py --name GGT_DW --dataset lastfm-asia --model GGT --node_embeddings embs_dw --lr 3e-05 1e-04 3e-04 1e-03 3e-03 --dropout 0 0.1 0.2 --amp True --compile True --device cuda:0
python main.py --name GGT_DW_CLATT --dataset lastfm-asia --model GGT-CLATT --node_embeddings embs_dw  --clusterings la h1 bpp --lr 3e-05 1e-04 3e-04 1e-03 3e-03 --dropout 0 0.1 0.2 --amp True --compile True --device cuda:0

python main.py --name GGT_Lap --dataset lastfm-asia --model GGT --node_embeddings embs_lap --lr 3e-05 1e-04 3e-04 1e-03 3e-03 --dropout 0 0.1 0.2 --amp True --compile True --device cuda:0
python main.py --name GGT_Lap_CLATT --dataset lastfm-asia --model GGT-CLATT --node_embeddings embs_lap  --clusterings la h1 bpp --lr 3e-05 1e-04 3e-04 1e-03 3e-03 --dropout 0 0.1 0.2 --amp True --compile True --device cuda:0
