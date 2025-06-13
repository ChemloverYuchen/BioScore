from collections import namedtuple
from copy import deepcopy
import esm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from torch_scatter import scatter_mean, scatter_sum, scatter_add
from torch_geometric.nn import global_add_pool

from data.pdb_utils import VOCAB
from utils.nn_utils import BatchEdgeConstructor, _knn_edges, _radial_edges, _radial_edges_intra
from .GET.modules.tools import BlockEmbedding, KNNBatchEdgeConstructor, _block_edge_dist

from torch.distributions import Normal
import math
from utils.logger import print_log
from .GET.modules.radial_basis import RadialBasis

import pandas as pd

#intramolecular edge: protein: 10Å; small molecule: 2Å; nucleic acid: 10Å
#intermolecular edge: 8Å
class ModifiedKNNBatchEdgeConstructor(BatchEdgeConstructor):
    def __init__(self, k_neighbors, global_message_passing=True, global_node_id_vocab=[], delete_self_loop=True,std=1.0):
        super().__init__(global_node_id_vocab, delete_self_loop)
        self.k_neighbors = k_neighbors
        self.global_message_passing = global_message_passing
        self.std = std

    def _construct_intra_edges(self, S, batch_id, segment_ids, symbol, **kwargs):
        all_intra_edges = super()._construct_intra_edges(S, batch_id, segment_ids)
        X, block_id = kwargs['X'], kwargs['block_id']
        
        # Compute distances 
        src_dst = all_intra_edges.T
        dist = _block_edge_dist(X, block_id, src_dst)
        
        # Distance-threshold-aware edge construction
        intra_edges, dist_neighbors = _radial_edges_intra(dist, src_dst, segment_ids, symbol, dist_cut_offs=torch.tensor([10.0/self.std, 2.0/self.std, 10.0/self.std], device=src_dst.device))
        
        return intra_edges, dist_neighbors
    
    def _construct_inter_edges(self, S, batch_id, segment_ids, symbol, **kwargs):
        all_inter_edges = super()._construct_inter_edges(S, batch_id, segment_ids)
        X, block_id = kwargs['X'], kwargs['block_id']
        
        src_dst = all_inter_edges.T
        dist = _block_edge_dist(X, block_id, src_dst)

        # Distance-threshold-aware edge construction
        inter_edges, dist_neighbors = _radial_edges(dist, src_dst, dist_cut_off=8/self.std)
        
        return inter_edges, dist_neighbors
    
    def _construct_global_edges(self, S, batch_id, segment_ids, **kwargs):
        if self.global_message_passing:
            return super()._construct_global_edges(S, batch_id, segment_ids, **kwargs)
        else:
            return None, None

    def _construct_seq_edges(self, S, batch_id, segment_ids, **kwargs):
        return None
    
    @torch.no_grad()
    def __call__(self, B, batch_id, segment_ids, symbol, **kwargs):
        self._prepare(B, batch_id, segment_ids)
        intra_edges, intra_dist = self._construct_intra_edges(B, batch_id, segment_ids, symbol, **kwargs)
        inter_edges, inter_dist = self._construct_inter_edges(B, batch_id, segment_ids, symbol, **kwargs)
        
        global_global_edges, global_normal_edges = self._construct_global_edges(B, batch_id, segment_ids, **kwargs)

        self._reset_buffer()
        
        return intra_edges, intra_dist, inter_edges, inter_dist, global_global_edges, global_normal_edges, None

ReturnValue = namedtuple(
    'ReturnValue',
    ['energy', 'total_potential',
     'unit_repr', 'block_repr', 'graph_repr',
     'batch_id', 'block_id',
     'mdn_loss',
     ],
    )

def construct_edges(edge_constructor, B, batch_id, segment_ids, X, block_id, symbol, complexity=-1):
    if complexity == -1:  # don't do splicing
        intra_edges, intra_dist, inter_edges, global_global_edges, global_normal_edges, _ = edge_constructor(
            B, batch_id, segment_ids, symbol, X=X, block_id=block_id)
        return intra_edges, intra_dist, inter_edges, global_global_edges, global_normal_edges

    # do splicing
    offset, bs_id_start, bs_id_end = 0, 0, 0
    mini_intra_edges, mini_intra_dists, mini_inter_edges, mini_global_global_edges, mini_global_normal_edges = [], [], [], [], []
    mini_inter_dists = []
    with torch.no_grad():
        batch_size = batch_id.max() + 1
        unit_batch_id = batch_id[block_id]
        lengths = scatter_sum(torch.ones_like(batch_id), batch_id, dim=0)

        while bs_id_end < batch_size:
            bs_id_start = bs_id_end
            bs_id_end += 1
            while bs_id_end + 1 <= batch_size and \
                  (lengths[bs_id_start:bs_id_end + 1] * lengths[bs_id_start:bs_id_end + 1].max()).sum() < complexity:
                bs_id_end += 1

            block_is_in = (batch_id >= bs_id_start) & (batch_id < bs_id_end)
            unit_is_in = (unit_batch_id >= bs_id_start) & (unit_batch_id < bs_id_end)
            B_mini, batch_id_mini, segment_ids_mini = B[block_is_in], batch_id[block_is_in], segment_ids[block_is_in]
            symbol_mini = symbol[block_is_in]
            X_mini, block_id_mini = X[unit_is_in], block_id[unit_is_in]
            
            intra_edges, intra_dist, inter_edges, inter_dist, global_global_edges, global_normal_edges, _ = edge_constructor(
                B_mini, batch_id_mini - bs_id_start, segment_ids_mini, symbol_mini, X=X_mini, block_id=block_id_mini - offset)
            
            if not hasattr(edge_constructor, 'given_intra_edges'):
                mini_intra_edges.append(intra_edges + offset)
                mini_intra_dists.append(intra_dist)
            if not hasattr(edge_constructor, 'given_inter_edges'):
                mini_inter_edges.append(inter_edges + offset)
                mini_inter_dists.append(inter_dist)
            if global_global_edges is not None:
                mini_global_global_edges.append(global_global_edges + offset)
            if global_normal_edges is not None:
                mini_global_normal_edges.append(global_normal_edges + offset)
            offset += B_mini.shape[0]

        if hasattr(edge_constructor, 'given_intra_edges'):
            intra_edges = edge_constructor.given_intra_edges
            intra_dist = None
        else:
            intra_edges = torch.cat(mini_intra_edges, dim=1)
            intra_dist = torch.cat(mini_intra_dists, dim=0)
        
        if hasattr(edge_constructor, 'given_inter_edges'):
            inter_edges = edge_constructor.given_inter_edges
        else:
            inter_edges = torch.cat(mini_inter_edges, dim=1)
            inter_dist = torch.cat(mini_inter_dists, dim=0)
        
        if global_global_edges is not None:
            global_global_edges = torch.cat(mini_global_global_edges, dim=1)
        if global_normal_edges is not None:
            global_normal_edges = torch.cat(mini_global_normal_edges, dim=1)

    return intra_edges, intra_dist, inter_edges, inter_dist, global_global_edges, global_normal_edges

def mdn_loss_fn(pi, sigma, mu, y, eps=1e-10):
    while len(y.shape) < len(mu.shape):
        y = y.unsqueeze(-1)
    normal = Normal(mu, sigma)
    loglik = normal.log_prob(y.expand_as(normal.loc))
    loss = -torch.logsumexp(torch.log(pi + eps) + loglik, dim=1)

    return loss

def calculate_probablity(pi, sigma, mu, y, eps=0):
    while len(y.shape) < len(mu.shape):
        y = y.unsqueeze(-1)
    normal = Normal(mu, sigma)
    logprob = normal.log_prob(y.expand_as(normal.loc))
    logprob += torch.log(pi + eps)
    prob = logprob.exp().sum(1)
    prob += eps
    return prob

class BioScoreModel(nn.Module):
    def __init__(self, model_type, hidden_size, n_channel,
                 n_rbf=1, cutoff=7.0, n_head=1,
                 radial_size=16, edge_size=64, k_neighbors=9, n_layers=3,
                 dropout=0.1, std=1.0, global_message_passing=True,
                 atom_level=False, hierarchical=False, no_block_embedding=False) -> None:
        super().__init__()
        # for GET block #
        self.model_type = model_type
        self.hidden_size = hidden_size
        self.n_channel = n_channel
        self.n_rbf = n_rbf
        self.n_head = n_head
        self.radial_size = radial_size
        self.edge_size = edge_size
        self.k_neighbors = k_neighbors
        self.n_layers = n_layers
        self.dropout = dropout
        self.std = std
        self.cutoff = cutoff / self.std
        self.global_message_passing = global_message_passing
        self.atom_level = atom_level
        self.hierarchical = hierarchical
        self.no_block_embedding = no_block_embedding

        self.alpha = nn.Parameter(torch.tensor(1.0))

        assert not (self.hierarchical and self.atom_level), 'Hierarchical model is incompatible with atom-level model'
        self.global_block_id = VOCAB.symbol_to_idx(VOCAB.GLB)

        self.block_embedding = BlockEmbedding(
            num_block_type=len(VOCAB),
            num_atom_type=VOCAB.get_num_atom_type(),
            num_atom_position=VOCAB.get_num_atom_pos(),
            embed_size=hidden_size,
            no_block_embedding=no_block_embedding
        )
        
        self.edge_constructor = ModifiedKNNBatchEdgeConstructor(
            k_neighbors=k_neighbors,
            global_message_passing=global_message_passing,
            global_node_id_vocab=[self.global_block_id],
            delete_self_loop=True,
            std=self.std)
        
        # [intra / inter / global_global / global_normal]
        self.edge_embedding = nn.Embedding(4, self.edge_size)

        z_requires_grad = False
        if model_type == 'GET':
            from .GET.encoder import GETEncoder
            self.encoder = GETEncoder(
                hidden_size, radial_size, n_channel,
                n_rbf, cutoff, edge_size, n_layers,
                n_head, dropout=dropout,
                z_requires_grad=z_requires_grad
            )
        else:
            raise NotImplementedError(f'Model type {model_type} not implemented!')
        
        if self.hierarchical:
            self.top_encoder = deepcopy(self.encoder)
            
        self.energy_ffn = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, 1, bias=False)
        )
    
        #For MDN
        in_channels = hidden_size * 2
        hidden_dim = 128
        dropout_rate = 0.1
        n_gaussians = 10
        
        self.MLP = nn.Sequential(nn.Linear(in_channels, hidden_dim),
                         nn.BatchNorm1d(hidden_dim), nn.ELU(),
                         nn.Dropout(p=dropout_rate))

        self.z_pi = nn.Linear(hidden_dim, n_gaussians)
        self.z_sigma = nn.Linear(hidden_dim, n_gaussians)
        self.z_mu = nn.Linear(hidden_dim, n_gaussians)
 
    @torch.no_grad()
    def choose_receptor(self, batch_size, device):
#         segment_retain = (torch.randn((batch_size, ), device=device) > 0).long()  # [bs], 0 or 1
        segment_retain = torch.zeros((batch_size,), device=device).long()  # [bs], all zeros
        return segment_retain

    @torch.no_grad()
    def normalize(self, Z, B, block_id, batch_id, segment_ids, receptor_segment):
        # centering
        center = Z[(B[block_id] == self.global_block_id) & (segment_ids[block_id] == receptor_segment[batch_id][block_id])]  # [bs]       
        Z = Z - center[batch_id][block_id]
        # normalize
        Z = Z / self.std
        return Z
 
    @torch.no_grad()
    def update_global_block(self, Z, B, block_id):
        is_global = B[block_id] == self.global_block_id  # [Nu]
        scatter_ids = torch.cumsum(is_global.long(), dim=0) - 1  # [Nu]        
        not_global = ~is_global
        centers = scatter_mean(Z[not_global], scatter_ids[not_global], dim=0)  # [Nglobal, n_channel, 3], Nglobal = batch_size * 2
        Z = Z.clone()
        Z[is_global] = centers
        return Z, not_global

    def get_edges(self, B, batch_id, segment_ids, Z, block_id, symbol):
        intra_edges, intra_dist, inter_edges, inter_dist, global_global_edges, global_normal_edges = construct_edges(
    self.edge_constructor, B, batch_id, segment_ids, Z, block_id, symbol, complexity=2000**2)
        
        if self.global_message_passing:
            edges = torch.cat([intra_edges, inter_edges, global_global_edges, global_normal_edges], dim=1)
            edge_attr = torch.cat([
                torch.zeros_like(intra_edges[0]),
                torch.ones_like(inter_edges[0]),
                torch.ones_like(global_global_edges[0]) * 2,
                torch.ones_like(global_normal_edges[0]) * 3])
        else:
            edges = torch.cat([intra_edges, inter_edges], dim=1)
            edge_attr = torch.cat([torch.zeros_like(intra_edges[0]), torch.ones_like(inter_edges[0])])
        edge_attr = self.edge_embedding(torch.zeros_like(intra_edges[0]))
        return intra_edges, intra_dist, inter_edges, inter_dist, edges, edge_attr
    
    def forward(self, Z, B, A, atom_positions, block_lengths, lengths, segment_ids, label, symbol, mdn_th=7.0, score_th=5.0) -> ReturnValue:
        with torch.no_grad():
            
            batch_id = torch.zeros_like(segment_ids)  # [Nb]

            batch_id[torch.cumsum(lengths, dim=0)[:-1]] = 1
            batch_id.cumsum_(dim=0)  # [Nb], item idx in the batch
            
            block_id = torch.zeros_like(A) # [Nu]
            
            block_id[torch.cumsum(block_lengths, dim=0)[:-1]] = 1
            block_id.cumsum_(dim=0)  # [Nu], block (residue) id of each unit (atom)

            batch_size = lengths.shape[0]

            # select receptor
            receptor_segment = self.choose_receptor(batch_size, batch_id.device)
            
            # normalize
            Z = self.normalize(Z, B, block_id, batch_id, segment_ids, receptor_segment)
            Z, _ = self.update_global_block(Z, B, block_id)
        

        Z.requires_grad_(True)
        
        # Initialize embedding
        H_0 = self.block_embedding(B, A, atom_positions, block_id) #Nu,128

        # edges
        intra_edges, intra_dist, inter_edges, inter_dist, edges, edge_attr = self.get_edges(B, batch_id, segment_ids, Z, block_id, symbol)
        
        unit_repr, block_repr, graph_repr, pred_Z = self.encoder(H_0, Z, block_id, batch_id, intra_edges, edge_attr)
        
        distances = inter_dist
        pair_vectors_struct = torch.cat([block_repr[inter_edges[0]], block_repr[inter_edges[1]]], dim=-1)
        edge_batch_id_1 = batch_id[inter_edges[0]]
        edge_batch_id_2 = batch_id[inter_edges[1]]
        
        assert torch.equal(edge_batch_id_1, edge_batch_id_2), "The two endpoints of the edge do not belong to the same complex"
        edge_batch_id = edge_batch_id_1
        
        C = self.MLP(pair_vectors_struct)
        
        #############################  For Docking/Screening Pipeline  ##########################
        mu = F.elu(self.z_mu(C)) + 1
        sigma = F.elu(self.z_sigma(C)) + 1.1
        pi = F.softmax(self.z_pi(C), -1)

        dist_threshold = mdn_th / self.std
        
        mdn_loss = mdn_loss_fn(pi, sigma, mu, distances)
        mdn_loss = mdn_loss[torch.where(distances <= dist_threshold)[0]]
        mdn_loss = mdn_loss.mean()
        
        prob = calculate_probablity(pi, sigma, mu, distances)
        valid_mask = distances <= dist_threshold
        filtered_prob = prob[valid_mask]
        filtered_edge_batch_id = edge_batch_id[valid_mask]
        
        if not torch.any(valid_mask):
            print(f"Warning: valid_mask is all zeros: no edges in cutoff: {dist_threshold}")
            return ReturnValue(
            energy=0,
            total_potential=torch.full((batch_size,), 0, dtype=mdn_loss.dtype, device=mdn_loss.device),

            # representations
            unit_repr=None,
            block_repr=None,
            graph_repr=None,

            # batch information
            batch_id=batch_id,
            block_id=block_id,

            # loss
            mdn_loss=mdn_loss,
        )
    
        #Confidence Score
        penal = torch.log(1.0 * scatter_add(torch.ones_like(filtered_prob),filtered_edge_batch_id, dim=0))
        penal[torch.isinf(penal)] = -1
        num_small_atom = scatter_add(segment_ids, batch_id, dim=0)-1

        potential = -torch.log(filtered_prob)
        total_potential = - scatter_mean(potential, filtered_edge_batch_id, dim=0)

        total_potential = total_potential + 1 * penal
        
        while total_potential.shape[0]< num_small_atom.shape[0]:
            total_potential = torch.cat([total_potential, torch.full((1,), -1).to(penal.device)])

        #############################  For Scoring/Ranking Pipeline  ##########################
        pred_energy = 0

        dist_thr = score_th / self.std
        valid_mask_score = distances <= dist_thr
        filtered_C = pair_vectors_struct[valid_mask_score]
        filtered_edge_batch_id_score = edge_batch_id[valid_mask_score]
        pred_energy_unit = self.energy_ffn(filtered_C).squeeze(-1)
        
        pred_energy = scatter_mean(pred_energy_unit, filtered_edge_batch_id_score, dim=0)
        
        penal = torch.log(1.0 * scatter_add(torch.ones_like(filtered_edge_batch_id_score),filtered_edge_batch_id_score, dim=0))
        penal[torch.isinf(penal)] = -1

        pred_energy -= self.alpha * penal

        while pred_energy.shape[0]< num_small_atom.shape[0]:
            pred_energy = torch.cat([pred_energy, torch.full((1,), -1).to(pred_energy.device)])
        
        return ReturnValue(

            energy=pred_energy,
            total_potential=total_potential,

            # representations
            unit_repr=None,
            block_repr=None,
            graph_repr=None,

            # batch information
            batch_id=batch_id,
            block_id=block_id,

            # loss
            mdn_loss=mdn_loss,
        )
