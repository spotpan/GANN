from tqdm import trange
from deeprobust.graph.data import Dataset
import torch
from gann_utils import *
import argparse
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
import networkx as nx
from scipy.sparse import csr_matrix
import scipy.sparse as sp
from deeprobust.graph.data import Dataset
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from tqdm import tqdm
#import utils
import math
import json

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--dataset', type=str, default='pubmed', choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed'], help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.5,  help='pertubation rate')
parser.add_argument('--runs', type=int, default=10)

parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')

parser.add_argument('--k', type=int, default=6, help='iteration number.')
parser.add_argument('--alpha', type=float, default=0.5, help='message passing parameter.')
parser.add_argument('--beta', type=float, default=0.5, help='message passing parameter.')
parser.add_argument('--decay', type=float, default=5e-4, help='weight decay.')

args = parser.parse_args()
args.cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

class Encoder(Module):

    def __init__(self, in_features, out_features, with_bias=True):
        super(Encoder, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if with_bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GANN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.5, with_relu=True, with_bias=True):
        super(GANN, self).__init__()

        self.layer1 = Encoder(nfeat, nhid, with_bias=with_bias)
        self.layer2 = Encoder(nhid, nclass, with_bias=with_bias)
        self.dropout = dropout
        self.with_relu = with_relu


    def forward(self, x, adj):

        authority=x
        hub=x
        intervalue=x
        for i in range(args.k):
            authority =  torch.clamp(adj @ intervalue,min=0,max=1)
            intervalue=torch.clamp(args.alpha*authority+args.beta*hub,min=0,max=1)
            hub = torch.clamp(adj.T @ intervalue,min=0,max=1)

        x=args.alpha*authority+args.beta*hub

        if self.with_relu:
            x = F.relu(self.layer1(x, adj))
        else:
            x = self.layer1(x, adj)

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.layer2(x, adj)


        return F.log_softmax(x, dim=1)

    def initialize(self):
        self.layer1.reset_parameters()
        self.layer2.reset_parameters()


#get splitted data
data = Dataset(root='attack_data/', name=args.dataset, seed=args.seed, setting='prognn')
_, features, labels = data.adj, data.features, data.labels
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test



#adj, features, labels = load_data(dataset=args.dataset)
nclass = max(labels) + 1
adj = sp.load_npz("meta_adj_directed/%s_meta_adj_%s.npz" % (args.dataset, args.ptb_rate))
adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False)


if device != 'cpu':
    adj = adj.to(device)
    features = features.to(device)
    labels = labels.to(device)

def test(adj):
    ''' test on GANN '''

    adj = normalize_adj_tensor(adj)
    gann = GANN(nfeat=features.shape[1],
              nhid=args.hidden,
              nclass=labels.max().item() + 1,
              dropout=0.5)

    if device != 'cpu':
        gann = gann.to(device)

    optimizer = optim.Adam(gann.parameters(),
                           lr=args.lr, weight_decay=args.decay)

    gann.train()

    for epoch in range(args.epochs):
        optimizer.zero_grad()
        output = gann(features, adj)
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()

    gann.eval()
    output = gann(features, adj)

    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])


    return acc_test.item()


def main():
 
    
    modified_adj = adj
    runs = 10
    clean_acc = []
    attacked_acc = []

    print('=== testing GANN on attacked graph ===')
    for i in range(runs):
        attacked_acc.append(test(modified_adj))

    print(np.mean(attacked_acc), np.std(attacked_acc))


if __name__ == '__main__':
    main()

