# Unveiling Strong Ties in Graph Adaptive Neural Network Against Poisoning Attack: Trust Zones with Provable Confidence


## Requirements

To install the required dependencies, use the following command:

```bash
pip install -r requirements.txt
```

## Usage Instructions

### **1. Adaptive Adversarial Attacks**

For adaptive attacks, use the following commands to evaluate performance on the given datasets.

#### **GANN**
```bash
python3 gann.py --dataset cora --alpha 0.3 --beta 0.7
python3 gann.py --dataset citeseer --alpha 0.5 --beta 0.5
python3 gann.py --dataset cora_ml --alpha 0.5 --beta 0.5 --k 2
python3 gann.py --dataset polblogs --alpha 0.4 --beta 0.6
```

#### **Dir-Gnn**
```bash
python3 directedgnn.py --dataset cora
python3 directedgnn.py --dataset citeseer
python3 directedgnn.py --dataset cora_ml
python3 directedgnn.py --dataset polblogs
```

#### **Magnet**
```bash
python3 Magnet.py --dataset cora
python3 Magnet.py --dataset citeseer 
python3 Magnet.py --dataset cora_ml
python3 Magnet.py --dataset polblogs
```


### **2. Non-adaptive adversarial attack**

#### GANN
```bash
python3 gann.py --dataset cora --alpha 0.3 --beta 0.7 --decay 5e-4 --ptb_rate 0.25
python3 gann.py --dataset cora --alpha 0.3 --beta 0.7 --decay 5e-4 --ptb_rate 0.5
python3 gann.py --dataset cora --alpha 0.3 --beta 0.7 --decay 5e-4 --ptb_rate 0.75

python3 gann.py --dataset citeseer --alpha 0.5 --beta 0.5 --decay 5e-2 --ptb_rate 0.25
python3 gann.py --dataset citeseer --alpha 0.5 --beta 0.5 --decay 5e-2 --ptb_rate 0.5
python3 gann.py --dataset citeseer --alpha 0.5 --beta 0.5 --decay 5e-2 --ptb_rate 0.75

python3 gann.py --dataset pubmed --alpha 0.5 --beta 0.5 --decay 5e-4 --ptb_rate 0.25
python3 gann.py --dataset pubmed --alpha 0.5 --beta 0.5 --decay 5e-4 --ptb_rate 0.5
python3 gann.py --dataset pubmed --alpha 0.5 --beta 0.5 --decay 5e-4 --ptb_rate 0.75
```


#### GCN
```bash
python gcn.py --dataset cora --ptb_rate 0.25 
# We can replace dataset and ptb_rate  
```

#### GAT
```bash
python gat.py --dataset cora --ptb_rate 0.25  
# We can replace dataset and ptb_rate  
```

#### APPNP
```bash
python appnp.py --dataset cora --ptb_rate 0.25  
# We can replace dataset and ptb_rate  
```

#### SSGC
```bash
python ssgc.py --epochs 100 --lr 0.2 --weight_decay 1e-5 --alpha 0.05 --degree 16 --dataset cora --ptb_rate 0.25  
# We can replace ptb_rate  

python ssgc.py --epochs 150 --lr 0.2 --weight_decay 1e-4 --alpha 0.05 --degree 16 --dataset citeseer --ptb_rate 0.25  
# We can replace ptb_rate  

python ssgc.py --epochs 100 --lr 0.2 --weight_decay 2e-5 --alpha 0.05 --dataset pubmed --ptb_rate 0.25  
# We can replace ptb_rate  
```

#### NAGphormer
```bash
python main.py --batch_size 2000 --dropout 0.1 --hidden_dim 512 --hops 3  --n_heads 8 --n_layers 1 --pe_dim 3 --peak_lr 0.01  --weight_decay=1e-05 --dataset cora --ptb_rate 0.25  
# We can replace ptb_rate  

python main.py --batch_size 2000 --dropout 0.3 --hidden_dim 512  --hops 7  --n_heads 8 --n_layers 1 --pe_dim 3 --peak_lr 0.001  --weight_decay=1e-05 --dataset citeseer --ptb_rate 0.25  
# We can replace ptb_rate  

python main.py --batch_size 2000 --dropout 0.1 --hidden_dim 512 --hops 7  --n_heads 8 --n_layers 1 --pe_dim 15 --peak_lr 0.001  --weight_decay=1e-05 --dataset pubmed --ptb_rate 0.25  
# We can replace ptb_rate  
```

#### Robust GCN
```bash
python rgcn.py --dataset cora --ptb_rate 0.25  
# We can replace dataset and ptb_rate  
```

#### GCN Jaccard
```bash
python gcn_jaccard.py --dataset cora --ptb_rate 0.25  
# We can replace dataset and ptb_rate  
```

#### GCN SVD
```bash
python gcn_svd.py --dataset cora --ptb_rate 0.25  
# We can replace dataset and ptb_rate  
```

#### Pro-GNN
```bash
python prognn.py --alpha 5e-4 --beta 1.5 --gamma 1 --lambda_ 0.001 --lr 5e-4 --epoch 1000 --dataset cora --ptb_rate 0.25  
# We can replace ptb_rate  

python prognn.py --alpha 5e-4 --beta 1.5 --gamma 1 --lambda_ 0.0001 --lr 5e-4 --epoch 1000 --dataset citeseer --ptb_rate 0.25
# We can replace ptb_rate  

python prognn.py --alpha 0.3 --beta 2.5 --gamma 1 --lambda_ 0.001 --lr 1e-2 --epoch 100 --inner_steps 30 --dataset pubmed --ptb_rate 0.25  
# We can replace ptb_rate  
```

#### GNNGuard
```bash
python main.py --modelname GCN --GNNGuard True --dataset cora --ptb_rate 0.25  
# We can replace dataset and ptb_rate  
```

#### Elastic GNN
```bash
cd code
python main.py --random_splits 1 --runs 10 --lr 0.01 --K 10 --lambda1 9 --lambda2 3 --weight_decay 0.0005 --hidden 16 --normalize_features False --dataset Cora-adv --ptb_rate 0.25  
# We can replace ptb_rate 

python main.py --random_splits 1 --runs 10 --lr 0.01 --K 10 --lambda1 9 --lambda2 3 --weight_decay 0.0005 --hidden 16 --normalize_features False --dataset CiteSeer-adv --ptb_rate 0.25 
# We can replace ptb_rate 

python main.py --random_splits 1 --runs 10 --lr 0.01 --K 10 --lambda1 9 --lambda2 3 --weight_decay 0.0005 --hidden 16 --normalize_features False --dataset PubMed-adv --ptb_rate 0.25  
# We can replace ptb_rate 
```

#### HANG-quad
```bash
python main.py --function hangquad --block constant --lr 0.005 --dropout 0.4 --input_dropout 0.4 --batch_norm --time 8 --hidden_dim 64 --step_size 1 --runtime 10 --add_source --batch_norm --gpu 4 --epochs 800 --patience 150 --dataset cora --ptb_rate 0.25  
# We can replace ptb_rate 

python main.py --function hangquad --block constant --lr 0.005 --dropout 0.4 --input_dropout 0.4 --batch_norm --time 12 --hidden_dim 64 --step_size 1 --runtime 10 --add_source --batch_norm --gpu 4 --epochs 800 --patience 150 --dataset citeseer --ptb_rate 0.25  
# We can replace ptb_rate 

python main.py --function hangquad --block constant --lr 0.005 --dropout 0.4 --input_dropout 0.4 --batch_norm --time 6 --hidden_dim 64 --step_size 1 --runtime 10 --add_source --batch_norm --gpu 4 --epochs 800 --patience 150 --dataset pubmed --ptb_rate 0.25   
# We can replace ptb_rate 
```

#### STABLE
```bash
python main.py --alpha 0.6 --beta 2 --k 7  --jt 0.03 --cos 0.25 --dataset cora --ptb_rate 0.25  
# We can replace ptb_rate 

python main.py --alpha 0.1 --beta 2 --k 5  --jt 0.03 --cos 0.1 --dataset citeseer --ptb_rate 0.25  
# We can replace ptb_rate 

python main.py --alpha 0.1 --beta 2 --k 5  --jt 0.03 --cos 0.1 --dataset pubmed --ptb_rate 0.25  
# We can replace ptb_rate 
```

#### GCN-GARNET
```bash
python main.py --device 0 --backbone gcn --dataset cora --attack meta --ptb_rate 0.25 --perturbed
# We can replace dataset and ptb_rate  
```

#### EvenNet
```bash
python main.py --runs 100 --dataset cora --ptb_rate 0.25 --alpha 0.9  
# We can replace ptb_rate 

python main.py --runs 100 --dataset citeseer --ptb_rate 0.25 --alpha 0.9    
# We can replace ptb_rate 

python main.py --runs 100 --dataset pubmed --ptb_rate 0.25 --alpha 0.5 
python main.py --runs 100 --dataset pubmed --ptb_rate 0.5 --alpha 0.9
python main.py --runs 100 --dataset pubmed --ptb_rate 0.75 --alpha 0.9
```


#### GADC
```bash
python gadc.py --degree 6 --lam 1 --lr 0.02 --epochs 100 --weight_decay 1e-5 --hidden 32 --dataset cora --ptb_rate 0.25  
python gadc.py --degree 3 --lam 1 --lr 0.02 --epochs 100 --weight_decay 1e-5 --hidden 32 --dataset cora --ptb_rate 0.5  
python gadc.py --degree 1 --lam 1 --lr 0.02 --epochs 100 --weight_decay 1e-5 --hidden 32 --dataset cora --ptb_rate 0.75  
python gadc.py --degree 6 --lam 1 --lr 0.02 --epochs 100 --weight_decay 1e-5 --hidden 32 --dataset citeseer --ptb_rate 0.25  
python gadc.py --degree 3 --lam 1 --lr 0.02 --epochs 100 --weight_decay 1e-5 --hidden 32 --dataset citeseer --ptb_rate 0.5  
python gadc.py --degree 1 --lam 1 --lr 0.02 --epochs 100 --weight_decay 1e-5 --hidden 32 --dataset citeseer --ptb_rate 0.75  
python gadc.py --degree 2 --lam 1 --lr 0.02 --epochs 200 --weight_decay 1e-5 --hidden 32 --dataset pubmed --ptb_rate 0.25  
python gadc.py --degree 1 --lam 1 --lr 0.02 --epochs 200 --weight_decay 1e-5 --hidden 32 --dataset pubmed --ptb_rate 0.5 
python gadc.py --degree 1 --lam 1 --lr 0.02 --epochs 200 --weight_decay 1e-4 --hidden 32 --dataset pubmed --ptb_rate 0.75 
```
