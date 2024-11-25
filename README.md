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
## Reference
https://github.com/SongtaoLiu0823/GADC/edit/main/README.md
