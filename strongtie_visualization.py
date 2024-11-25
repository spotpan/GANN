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


# Argument parser for user inputs
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=16, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--hidden', type=int, default=16, help='Number of hidden units.')
parser.add_argument('--dataset', type=str, default='citeseer', choices=['cora', 'cora_ml', 'citeseer', 'polblogs'], help='Dataset to use.')
parser.add_argument('--ptb_rate', type=float, default=0.05, help='Perturbation rate.')
parser.add_argument('--model', type=str, default='Meta-Self', choices=['A-Meta-Self', 'Meta-Self'], help='Model variant.')
parser.add_argument('--k', type=int, default=6, help='iteration number of HITS score.')
parser.add_argument('--alpha', type=float, default=0.5, help='message passing parameter.')
parser.add_argument('--beta', type=float, default=0.5, help='message passing parameter.')


args = parser.parse_args()
cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")

# Set random seeds for reproducibility
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if cuda:
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



# === Load Dataset ===
adj, features, labels = load_data(dataset=args.dataset)
nclass = max(labels) + 1

val_size = 0.164
test_size = 0.8
train_size = 1 - test_size - val_size

idx = np.arange(adj.shape[0])
idx_train, idx_val, idx_test = get_train_val_test(idx, train_size, val_size, test_size, stratify=labels)
idx_unlabeled = np.union1d(idx_val, idx_test)
perturbations = int(args.ptb_rate * (adj.sum() // 2))

adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False)

# Initialize the attack model
if 'A' in args.model:
    model = MetaApprox(nfeat=features.shape[1], hidden_sizes=[args.hidden], nnodes=adj.shape[0],
                       nclass=nclass, dropout=0.5, train_iters=100, attack_features=False,
                       lambda_=0, device=device)
else:
    model = Metattack(nfeat=features.shape[1], hidden_sizes=[args.hidden], nnodes=adj.shape[0],
                      nclass=nclass, dropout=0.5, train_iters=100, attack_features=False,
                      lambda_=0, device=device)

adj, features, labels = adj.to(device), features.to(device), labels.to(device)
model = model.to(device)

def get_neighbors(graph, nodes, direction, depth=1):
    """Get neighbors in a specific direction up to a given depth."""
    neighbors = set(nodes)
    for _ in range(depth):
        new_neighbors = {n for node in neighbors for n in 
                         (graph.successors(node) if direction == 'successor' else graph.predecessors(node))}
        neighbors.update(new_neighbors)
    return neighbors

def extract_strong_ties(added_edges, removed_edges, idx_train):
    """Extract strong tie nodes from manipulated edges, excluding training nodes."""
    manipulated_nodes = {n for edge in added_edges | removed_edges for n in edge}
    return manipulated_nodes - set(idx_train)

def assign_primary_role(roles):
    """Assign each node a primary role based on a predefined priority."""
    node_to_role = {}
    priority_order = ['successor', 'predecessor', 'predecessor_successor']

    for role in priority_order:
        for node in roles[role]:
            if node not in node_to_role:
                node_to_role[node] = role

    primary_roles = {role: [] for role in roles}
    for node, role in node_to_role.items():
        primary_roles[role].append(node)

    return primary_roles
def filter_2hop_overlap(roles):
    """Remove successor and predecessor nodes from 2-hop neighbor roles."""
    filtered_roles = roles.copy()

    # Remove successors from successor_successor and successor_predecessor roles
    filtered_roles['succ_succ'] = [
        node for node in filtered_roles['succ_succ'] 
        if node not in filtered_roles['succ']
    ]
    filtered_roles['succ_pred'] = [
        node for node in filtered_roles['succ_pred'] 
        if node not in filtered_roles['succ']
    ]

    filtered_roles['pred_pred'] = [
        node for node in filtered_roles['pred_pred'] 
        if node not in filtered_roles['succ']
    ]
    filtered_roles['pred_succ'] = [
        node for node in filtered_roles['pred_succ'] 
        if node not in filtered_roles['succ']
    ]

    # Remove predecessors from predecessor_predecessor and predecessor_successor roles
    filtered_roles['pred_pred'] = [
        node for node in filtered_roles['pred_pred'] 
        if node not in filtered_roles['pred']
    ]
    filtered_roles['pred_succ'] = [
        node for node in filtered_roles['pred_succ'] 
        if node not in filtered_roles['pred']
    ]

    filtered_roles['succ_succ'] = [
        node for node in filtered_roles['succ_succ'] 
        if node not in filtered_roles['pred']
    ]
    filtered_roles['succ_pred'] = [
        node for node in filtered_roles['succ_pred'] 
        if node not in filtered_roles['succ']
    ]

    return filtered_roles

def get_strong_tie_distribution(roles, strong_tie_nodes):
    """Calculate total nodes and strong ties for each role."""
    total_distribution = {}
    strong_tie_distribution = {}

    for role, nodes in roles.items():
        total_count = len(nodes)
        strong_tie_count = len(set(nodes) & strong_tie_nodes)  # Strong ties in this role

        total_distribution[role] = total_count
        strong_tie_distribution[role] = strong_tie_count

    return total_distribution, strong_tie_distribution

def visualize_strong_ties(clean_adj, attacked_adj, idx_train):
    """Visualize graph with stacked bar chart aligned along the Y-axis."""
    clean_adj = csr_matrix(clean_adj.cpu().numpy())
    attacked_adj = csr_matrix(attacked_adj.cpu().numpy())

    G_clean = nx.from_scipy_sparse_array(clean_adj, create_using=nx.DiGraph)
    G_attacked = nx.from_scipy_sparse_array(attacked_adj, create_using=nx.DiGraph)

    # Identify added and removed edges
    added_edges = set(G_attacked.edges()) - set(G_clean.edges())
    removed_edges = set(G_clean.edges()) - set(G_attacked.edges())

    # Extract strong tie nodes excluding training nodes
    strong_tie_nodes = extract_strong_ties(added_edges, removed_edges, idx_train)

    # Define roles with renamed labels (priority for visualization only)
    successors = get_neighbors(G_clean, idx_train, 'successor', depth=1)
    predecessors = get_neighbors(G_clean, idx_train, 'predecessor', depth=1)

    roles = {
        'succ': list(successors),
        'pred': list(predecessors),
        'succ_succ': list(get_neighbors(G_clean, successors, 'successor', depth=1)),
        'pred_pred': list(get_neighbors(G_clean, predecessors, 'predecessor', depth=1)),
        'succ_pred': list(get_neighbors(G_clean, successors, 'predecessor', depth=1)),
        'pred_succ': list(get_neighbors(G_clean, predecessors, 'successor', depth=1)),
    }

    # Apply the filter function for graph visualization to remove overlapping nodes
    roles_for_viz = filter_2hop_overlap(roles)

    # Determine residual nodes and add to roles_for_viz after filtering
    all_neighbors = set().union(*roles_for_viz.values())
    residual_nodes = set(G_clean.nodes()) - all_neighbors - set(idx_train)
    roles_for_viz['rsl'] = list(residual_nodes)

    # Calculate the strong tie distribution based on filtered roles for visualization
    _, strong_tie_distribution = get_strong_tie_distribution(roles_for_viz, strong_tie_nodes)

    # Calculate the total raw counts for each role without priority or overlap filtering
    raw_counts = {role: len(nodes) for role, nodes in roles.items()}
    raw_counts['rsl'] = len(residual_nodes)  # Add raw count for residual nodes

    # === Visualization ===
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={'width_ratios': [3, 1]})

    # ======= Graph Visualization (Left Side) ========
    pos = generate_layout(G_clean, roles_for_viz, idx_train, residual_nodes)
    role_colors = role_colors_map()

    # Set training nodes' positions to the center
    for node in idx_train:
        pos[node] = (0, 0)

    # Draw nodes with appropriate colors and shapes based on priority
    for role, (color, shape) in role_colors.items():
        nx.draw_networkx_nodes(G_clean, pos, nodelist=roles_for_viz[role], node_color=color, node_shape=shape, node_size=120, ax=ax1)

    # Draw gray training nodes at the center with a larger size and border
    nx.draw_networkx_nodes(G_clean, pos, nodelist=list(idx_train), node_color='gray', node_size=200, edgecolors='black', ax=ax1)

    # Overlay strong ties in red
    strong_tie_overlay = [node for node in strong_tie_nodes if node in pos]
    nx.draw_networkx_nodes(G_clean, pos, nodelist=strong_tie_overlay, node_color='red', node_size=150, edgecolors='black', ax=ax1)

    # Draw edges safely
    draw_edges_safely(G_clean, pos, added_edges, removed_edges, ax=ax1)

    # Add dataset name and perturbation percentage to the title
    perturbation_percentage = args.ptb_rate * 100
    ax1.set_title(f"Graph Visualization - {args.dataset}\nPerturbation: {perturbation_percentage:.2f}%", fontsize=16)

    # ======= Vertical Stacked Bar Chart (Right Side) ========
    roles_with_residual = list(roles.keys()) + ['rsl']
    total_counts = np.array([raw_counts[role] for role in roles_with_residual])
    strong_tie_counts = np.array([strong_tie_distribution.get(role, 0) for role in roles_with_residual])
    non_strong_tie_counts = total_counts - strong_tie_counts

    # Colors matching the graph visualization
    bar_colors = [role_colors.get(role, ('brown', 'o'))[0] for role in roles_with_residual]

    # Plot bar chart with raw counts for each role without filtering for overlaps
    bars_non_strong = ax2.barh(roles_with_residual, non_strong_tie_counts, height=0.3, color=bar_colors, alpha=0.5)
    bars_strong = ax2.barh(roles_with_residual, strong_tie_counts, height=0.3, left=non_strong_tie_counts, color='red')

    # Add type labels inside the bars
    for bar, role in zip(bars_non_strong, roles_with_residual):
        ax2.text(bar.get_width() / 2, bar.get_y() + bar.get_height() / 2, role, 
                 ha='center', va='center', fontsize=10, color='black')

    # Formatting the bar chart
    ax2.set_xlabel('Number of Nodes', fontsize=12)
    ax2.set_yticks([])  # Remove Y-axis ticks
    ax2.set_title('Strong Tie Distribution', fontsize=14)
    ax2.grid(axis='x', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()



def generate_layout(graph, roles, idx_train, residual_nodes):
    """Generate layout with separation and assign nodes to proper roles."""
    pos = {}
    base_radius = 2.0
    increment = 8.0

    # Assign radii for each role, including the 'residual' role
    role_radii = {
        'succ': base_radius + 4.0,
        'pred': base_radius + increment,
        'succ_succ': base_radius + increment * 2,
        'pred_pred': base_radius + increment * 3,
        'succ_pred': base_radius + increment * 5,
        'pred_succ': base_radius + increment * 4,
        'rsl': base_radius + increment * 6,  # Radius for residual nodes
    }

    # Generate positions for each role
    for role, nodes in roles.items():
        radius = role_radii[role]  # Use the radius value from the dictionary
        pos.update(generate_positions_in_circle(nodes, center=(0, 0), radius=radius))

    return pos

def generate_positions_in_circle(nodes, center, radius, random_perturbation=0.3):
    """Generate positions in a circle with slight perturbation."""
    positions = {}
    for i, node in enumerate(nodes):
        angle = 2 * np.pi * i / len(nodes)
        perturb = np.random.uniform(-random_perturbation, random_perturbation, size=2)
        positions[node] = center + radius * np.array([np.cos(angle), np.sin(angle)]) + perturb
    return positions

# Draw edges safely if both nodes are present in the layout
def draw_edges_safely(graph, pos, added_edges, removed_edges, ax=None):
    """Draw edges safely only if both nodes are in the layout."""
    valid_added_edges = [(u, v) for u, v in added_edges if u in pos and v in pos]
    valid_removed_edges = [(u, v) for u, v in removed_edges if u in pos and v in pos]

    # Use the appropriate axes if provided
    if ax is not None:
        nx.draw_networkx_edges(graph, pos, edgelist=valid_added_edges, edge_color='red', style='dashed', ax=ax)
        nx.draw_networkx_edges(graph, pos, edgelist=valid_removed_edges, edge_color='blue', style='dotted', ax=ax)
    else:
        nx.draw_networkx_edges(graph, pos, edgelist=valid_added_edges, edge_color='red', style='dashed')
        nx.draw_networkx_edges(graph, pos, edgelist=valid_removed_edges, edge_color='blue', style='dotted')

def role_colors_map():
    """Return colors and markers for roles."""
    return {
        'succ': ('green', '8'),
        'pred': ('blue', 's'),
        'succ_succ': ('purple', 'o'),
        'pred_pred': ('pink', '^'),
        'succ_pred': ('orange', 'h'),
        'pred_succ': ('cyan', 'p'),
        'rsl': ('brown', 'd'),  # Brown for residual nodes
    }

def main():
    modified_adj = model(features, adj, labels, idx_train, idx_unlabeled, perturbations, ll_constraint=False).detach()
    visualize_strong_ties(adj, modified_adj, idx_train)

if __name__ == '__main__':
    main()
