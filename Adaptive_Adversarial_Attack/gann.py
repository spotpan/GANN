import torch
from utils import *
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
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from tqdm import tqdm
import utils
import math
import json

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dataset', type=str, default='citeseer',
                    choices=['cora', 'cora_ml', 'citeseer', 'polblogs'], help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.05,  help='pertubation rate')
parser.add_argument('--model', type=str, default='Meta-Self', choices=['A-Meta-Self', 'Meta-Self'], help='model variant')
parser.add_argument('--k', type=int, default=6, help='iteration number of HITS score.')
parser.add_argument('--alpha', type=float, default=0.5, help='message passing parameter.')
parser.add_argument('--beta', type=float, default=0.5, help='message passing parameter.')



args = parser.parse_args()
cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if device != 'cpu':
    torch.cuda.manual_seed(args.seed)

# === Inductive demo of GANN ===

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


class BaseMeta(Module):

    def __init__(self, nfeat, hidden_sizes, nclass, nnodes, dropout, train_iters, attack_features, lambda_, device, with_bias=False, lr=0.01, with_relu=False):
        super(BaseMeta, self).__init__()

        self.hidden_sizes = hidden_sizes
        self.nfeat = nfeat
        self.nclass = nclass
        self.with_bias = with_bias
        self.with_relu = with_relu

        self.gann = GANN(nfeat=nfeat,
                       nhid=hidden_sizes[0],
                       nclass=nclass,
                       dropout=0.5,
                       with_relu=False)

        self.train_iters = train_iters
        self.surrogate_optimizer = optim.Adam(self.gann.parameters(), lr=lr, weight_decay=5e-4)

        self.attack_features = attack_features
        self.lambda_ = lambda_
        self.device = device
        self.nnodes = nnodes

        self.adj_changes = Parameter(torch.FloatTensor(nnodes, nnodes))
        self.adj_changes.data.fill_(0)

    def filter_potential_singletons(self, modified_adj):
        """
        Computes a mask for entries potentially leading to singleton nodes, i.e. one of the two nodes corresponding to
        the entry have degree 1 and there is an edge between the two nodes.

        Returns
        -------
        torch.Tensor shape [N, N], float with ones everywhere except the entries of potential singleton nodes,
        where the returned tensor has value 0.

        """

        degrees = modified_adj.sum(0)
        degree_one = (degrees == 1)
        resh = degree_one.repeat(modified_adj.shape[0], 1).float()

        l_and = resh * modified_adj
        logical_and_symmetric = l_and + l_and.t()
        flat_mask = 1 - logical_and_symmetric
        return flat_mask

    def train_surrogate(self, features, adj, labels, idx_train, train_iters=200):
        print('=== training surrogate model to predict unlabled data for self-training')
        surrogate = self.gann
        surrogate.initialize()

        adj_norm = utils.normalize_adj_tensor(adj)
        surrogate.train()
        for i in range(train_iters):
            self.surrogate_optimizer.zero_grad()
            output = surrogate(features, adj_norm)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            self.surrogate_optimizer.step()

        # Predict the labels of the unlabeled nodes to use them for self-training.
        surrogate.eval()
        output = surrogate(features, adj_norm)
        labels_self_training = output.argmax(1)
        labels_self_training[idx_train] = labels[idx_train]
        # reset parameters for later updating
        surrogate.initialize()
        return labels_self_training


    def log_likelihood_constraint(self, modified_adj, ori_adj, ll_cutoff):
        """
        Computes a mask for entries that, if the edge corresponding to the entry is added/removed, would lead to the
        log likelihood constraint to be violated.
        """

        t_d_min = torch.tensor(2.0).to(self.device)
        t_possible_edges = np.array(np.triu(np.ones((self.nnodes, self.nnodes)), k=1).nonzero()).T
        allowed_mask, current_ratio = utils.likelihood_ratio_filter(t_possible_edges,
                                                                    modified_adj,
                                                                    ori_adj, t_d_min,
                                                                    ll_cutoff)

        return allowed_mask, current_ratio


# === Poisoning Attack: Metattack ===

class Metattack(BaseMeta):

    def __init__(self, nfeat, hidden_sizes, nclass, nnodes, dropout, train_iters,
                 attack_features, device, lambda_=0.5, with_relu=False, with_bias=False, lr=0.1, momentum=0.9):

        super(Metattack, self).__init__(nfeat, hidden_sizes, nclass, nnodes, dropout, train_iters, attack_features, lambda_, device, with_bias=with_bias, with_relu=with_relu)

        self.momentum = momentum
        self.lr = lr

        self.weights = []
        self.biases = []
        self.w_velocities = []
        self.b_velocities = []

        previous_size = nfeat
        for ix, nhid in enumerate(self.hidden_sizes):
            weight = Parameter(torch.FloatTensor(previous_size, nhid).to(device))
            bias = Parameter(torch.FloatTensor(nhid).to(device))
            w_velocity = torch.zeros(weight.shape).to(device)
            b_velocity = torch.zeros(bias.shape).to(device)
            previous_size = nhid

            self.weights.append(weight)
            self.biases.append(bias)
            self.w_velocities.append(w_velocity)
            self.b_velocities.append(b_velocity)

        output_weight = Parameter(torch.FloatTensor(previous_size, nclass).to(device))
        output_bias = Parameter(torch.FloatTensor(nclass).to(device))
        output_w_velocity = torch.zeros(output_weight.shape).to(device)
        output_b_velocity = torch.zeros(output_bias.shape).to(device)

        self.weights.append(output_weight)
        self.biases.append(output_bias)
        self.w_velocities.append(output_w_velocity)
        self.b_velocities.append(output_b_velocity)

        self._initialize()

    def _initialize(self):

        for w, b in zip(self.weights, self.biases):
            stdv = 1. / math.sqrt(w.size(1))
            w.data.uniform_(-stdv, stdv)
            b.data.uniform_(-stdv, stdv)


    def inner_train(self, features, adj_norm, idx_train, idx_unlabeled, labels):
        self._initialize()

        for ix in range(len(self.hidden_sizes) + 1):
            self.weights[ix] = self.weights[ix].detach()
            self.weights[ix].requires_grad = True
            self.w_velocities[ix] = self.w_velocities[ix].detach()
            self.w_velocities[ix].requires_grad = True

            if self.with_bias:
                self.biases[ix] = self.biases[ix].detach()
                self.biases[ix].requires_grad = True
                self.b_velocities[ix] = self.b_velocities[ix].detach()
                self.b_velocities[ix].requires_grad = True

        for j in range(self.train_iters):
            hidden = features
            for ix, w in enumerate(self.weights):
                b = self.biases[ix] if self.with_bias else 0
                if self.sparse_features:
                    hidden = adj_norm @ torch.spmm(hidden, w) + b
                else:
                    hidden = adj_norm @ hidden @ w + b
                if self.with_relu:
                    hidden = F.relu(hidden)

            output = F.log_softmax(hidden, dim=1)
            loss_labeled = F.nll_loss(output[idx_train], labels[idx_train])

            weight_grads = torch.autograd.grad(loss_labeled, self.weights, create_graph=True)
            self.w_velocities = [self.momentum * v + g for v, g in zip(self.w_velocities, weight_grads)]
            if self.with_bias:
                bias_grads = torch.autograd.grad(loss_labeled, self.biases, create_graph=True)
                self.b_velocities = [self.momentum * v + g for v, g in zip(self.b_velocities, bias_grads)]

            self.weights = [w - self.lr * v for w, v in zip(self.weights, self.w_velocities)]
            if self.with_bias:
                self.biases = [b - self.lr * v for b, v in zip(self.biases, self.b_velocities)]

    def get_meta_grad(self, features, adj_norm, idx_train, idx_unlabeled, labels, labels_self_training):

        hidden = features
        for ix, w in enumerate(self.weights):
            b = self.biases[ix] if self.with_bias else 0
            if self.sparse_features:
                hidden = adj_norm @ torch.spmm(hidden, w) + b
            else:
                hidden = adj_norm @ hidden @ w + b
            if self.with_relu:
                hidden = F.relu(hidden)
        output = F.log_softmax(hidden, dim=1)

        loss_labeled = F.nll_loss(output[idx_train], labels[idx_train])


        loss_unlabeled = F.nll_loss(output[idx_unlabeled], labels_self_training[idx_unlabeled])

        loss_test_val = F.nll_loss(output[idx_unlabeled], labels[idx_unlabeled])
        
        if self.lambda_ == 1:
            attack_loss = loss_labeled
        elif self.lambda_ == 0:
            attack_loss = loss_unlabeled
        else:
            attack_loss = self.lambda_ * loss_labeled + (1 - self.lambda_) * loss_unlabeled

        adj_grad = torch.autograd.grad(attack_loss, self.adj_changes, retain_graph=True)[0]
        print(f'GANN loss on unlabled data: {loss_test_val.item()}')
        print(f'GANN acc on unlabled data: {utils.accuracy(output[idx_unlabeled], labels[idx_unlabeled]).item()}')
        print(f'attack loss: {attack_loss.item()}')

        return adj_grad


    def forward(self, features, ori_adj, labels, idx_train, idx_unlabeled, perturbations, ll_constraint=True, ll_cutoff=0.004):
        self.sparse_features = sp.issparse(features)

        labels_self_training = self.train_surrogate(features, ori_adj, labels, idx_train)


        for i in tqdm(range(perturbations), desc="Perturbing graph"):
            adj_changes_square = self.adj_changes - torch.diag(torch.diag(self.adj_changes, 0))
            ind = np.diag_indices(self.adj_changes.shape[0])

            
            adj_changes_symm = torch.clamp(adj_changes_square + torch.transpose(adj_changes_square, 1, 0), -1, 1)

            modified_adj = adj_changes_symm + ori_adj

            adj_norm = utils.normalize_adj_tensor(modified_adj)

            self.inner_train(features, adj_norm, idx_train, idx_unlabeled, labels)

            #
            adj_grad = self.get_meta_grad(features, adj_norm, idx_train, idx_unlabeled, labels, labels_self_training)


            adj_meta_grad = adj_grad * (-2 * modified_adj + 1)
            adj_meta_grad -= adj_meta_grad.min()
            adj_meta_grad -= torch.diag(torch.diag(adj_meta_grad, 0))
            singleton_mask = self.filter_potential_singletons(modified_adj)
            adj_meta_grad = adj_meta_grad *  singleton_mask

            if ll_constraint:
                allowed_mask, self.ll_ratio = self.log_likelihood_constraint(modified_adj, ori_adj, ll_cutoff)
                allowed_mask = allowed_mask.to(self.device)
                adj_meta_grad = adj_meta_grad * allowed_mask

            # Get argmax of the meta gradients.
            adj_meta_argmax = torch.argmax(adj_meta_grad)
            row_idx, col_idx = utils.unravel_index(adj_meta_argmax, ori_adj.shape)
            self.adj_changes.data[row_idx][col_idx] += (-2 * modified_adj[row_idx][col_idx] + 1)
            self.adj_changes.data[col_idx][row_idx] += (-2 * modified_adj[row_idx][col_idx] + 1)

            if self.attack_features:
                pass

        return self.adj_changes + ori_adj


class MetaApprox(BaseMeta):

    def __init__(self, nfeat, hidden_sizes, nclass, nnodes, dropout, train_iters, attack_features, lambda_, device, with_relu=False, with_bias=False, lr=0.01):
        super(MetaApprox, self).__init__(nfeat, hidden_sizes, nclass, nnodes, dropout, train_iters, attack_features, lambda_, device, with_bias=with_bias, with_relu=with_relu)

        self.lr = lr
        self.adj_meta_grad = None
        self.features_meta_grad = None

        self.grad_sum = torch.zeros(nnodes, nnodes).to(device)

        self.weights = []
        self.biases = []
        previous_size = nfeat
        for ix, nhid in enumerate(self.hidden_sizes):
            weight = Parameter(torch.FloatTensor(previous_size, nhid).to(device))
            bias = Parameter(torch.FloatTensor(nhid).to(device))
            previous_size = nhid

            self.weights.append(weight)
            self.biases.append(bias)

        output_weight = Parameter(torch.FloatTensor(previous_size, nclass).to(device))
        output_bias = Parameter(torch.FloatTensor(nclass).to(device))
        self.weights.append(output_weight)
        self.biases.append(output_bias)
        self._initialize()

    def _initialize(self):
        for w, b in zip(self.weights, self.biases):
            stdv = 1. / math.sqrt(w.size(1))
            w.data.uniform_(-stdv, stdv)
            b.data.uniform_(-stdv, stdv)

        self.optimizer = optim.Adam(self.weights + self.biases, lr=self.lr) # , weight_decay=5e-4)

    def inner_train(self, features, modified_adj, idx_train, idx_unlabeled, labels, labels_self_training):
        adj_norm = utils.normalize_adj_tensor(modified_adj)
        for j in range(self.train_iters):
            hidden = features
            for w, b in zip(self.weights, self.biases):
                b = b if self.with_bias else 0
                if self.sparse_features:
                    hidden = adj_norm @ torch.spmm(hidden, w) + b
                else:
                    hidden = adj_norm @ hidden @ w + b
                if self.with_relu:
                    hidden = F.relu(hidden)

            output = F.log_softmax(hidden, dim=1)
            loss_labeled = F.nll_loss(output[idx_train], labels[idx_train])
            loss_unlabeled = F.nll_loss(output[idx_unlabeled], labels_self_training[idx_unlabeled])

            if self.lambda_ == 1:
                attack_loss = loss_labeled
            elif self.lambda_ == 0:
                attack_loss = loss_unlabeled
            else:
                attack_loss = self.lambda_ * loss_labeled + (1 - self.lambda_) * loss_unlabeled

            self.optimizer.zero_grad()
            loss_labeled.backward(retain_graph=True)

            self.adj_changes.grad.zero_()
            self.grad_sum += torch.autograd.grad(attack_loss, self.adj_changes, retain_graph=True)[0]
            self.optimizer.step()

        loss_test_val = F.nll_loss(output[idx_unlabeled], labels[idx_unlabeled])
        print(f'GANN loss on unlabled data: {loss_test_val.item()}')
        print(f'GANN acc on unlabled data: {utils.accuracy(output[idx_unlabeled], labels[idx_unlabeled]).item()}')


    def forward(self, features, ori_adj, labels, idx_train, idx_unlabeled, perturbations, ll_constraint=True, ll_cutoff=0.004):
        labels_self_training = self.train_surrogate(features, ori_adj, labels, idx_train)
        self.sparse_features = sp.issparse(features)

       

        for i in tqdm(range(perturbations), desc="Perturbing graph"):

            

            adj_changes_square = self.adj_changes - torch.diag(torch.diag(self.adj_changes, 0))
            ind = np.diag_indices(self.adj_changes.shape[0])
            adj_changes_symm = torch.clamp(adj_changes_square + torch.transpose(adj_changes_square, 1, 0), -1, 1)
            modified_adj = adj_changes_symm + ori_adj

            self._initialize()
            self.grad_sum.data.fill_(0)
            self.inner_train(features, modified_adj, idx_train, idx_unlabeled, labels, labels_self_training)

            adj_meta_grad = self.grad_sum * (-2 * modified_adj + 1)
            adj_meta_grad -= adj_meta_grad.min()
            singleton_mask = self.filter_potential_singletons(modified_adj)
            adj_meta_grad = adj_meta_grad *  singleton_mask

            if ll_constraint:
                allowed_mask, self.ll_ratio = self.log_likelihood_constraint(modified_adj, ori_adj, ll_cutoff)
                allowed_mask = allowed_mask.to(self.device)
                adj_meta_grad = adj_meta_grad * allowed_mask

            # Get argmax of the approximate meta gradients.
            adj_meta_approx_argmax = torch.argmax(adj_meta_grad)
            row_idx, col_idx = utils.unravel_index(adj_meta_approx_argmax, ori_adj.shape)

            self.adj_changes.data[row_idx][col_idx] += (-2 * modified_adj[row_idx][col_idx] + 1)
            self.adj_changes.data[col_idx][row_idx] += (-2 * modified_adj[row_idx][col_idx] + 1)

            if self.attack_features:
                pass

        return self.adj_changes + ori_adj


def load_indices_from_json(dataset, directory='splits'):
    """
    Load the train, val, and test indices from a JSON file based on the dataset name.
    """
    filename = f'{directory}/{dataset}_splits.json'
    with open(filename, 'r') as f:
        data = json.load(f)
    print(f"Indices loaded from {filename}")
    return np.array(data['idx_train']), np.array(data['idx_val']), np.array(data['idx_test'])


# === loading dataset ===
adj, features, labels = load_data(dataset=args.dataset)
nclass = max(labels) + 1
idx_train, idx_val, idx_test = load_indices_from_json(args.dataset)
idx_unlabeled = np.union1d(idx_val, idx_test)
perturbations = int(args.ptb_rate * (adj.sum()//2))

adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False)

# set up attack model
if 'Self' in args.model:
    lambda_ = 0
if 'Train' in args.model:
    lambda_ = 1
if 'Both' in args.model:
    lambda_ = 0.5

if 'A' in args.model:
    model = MetaApprox(nfeat=features.shape[1], hidden_sizes=[args.hidden],
                       nnodes=adj.shape[0], nclass=nclass, dropout=0.5,
                       train_iters=100, attack_features=False, lambda_=lambda_, device=device)

else:
    model = Metattack(nfeat=features.shape[1], hidden_sizes=[args.hidden],
                       nnodes=adj.shape[0], nclass=nclass, dropout=0.5,
                       train_iters=100, attack_features=False, lambda_=lambda_, device=device)

if device != 'cpu':
    adj = adj.to(device)
    features = features.to(device)
    labels = labels.to(device)
    model = model.to(device)


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
                           lr=args.lr, weight_decay=5e-4)

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
    # print("Test set results:",
    #       "loss= {:.4f}".format(loss_test.item()),
    #       "accuracy= {:.4f}".format(acc_test.item()))

    return acc_test.item()


def main():
 
    modified_adj = model(features, adj, labels, idx_train,
                         idx_unlabeled, perturbations, ll_constraint=False)
    modified_adj = modified_adj.detach()

    runs = 10
    attacked_acc = []


    print('=== testing GANN on attacked graph ===')
    for i in range(runs):
        attacked_acc.append(test(modified_adj))
    #print("attacked_acc:",attacked_acc)

    print(np.mean(attacked_acc), np.std(attacked_acc))

if __name__ == '__main__':
    main()

