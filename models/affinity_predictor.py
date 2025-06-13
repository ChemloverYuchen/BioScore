#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum

from .score_model import BioScoreModel, ReturnValue

class AffinityPredictor(BioScoreModel):

    def __init__(self, model_type, hidden_size, n_channel, loss_type, **kwargs) -> None:
        super().__init__(model_type, hidden_size, n_channel, **kwargs)
        self.loss_type = loss_type
        print(f"----------- current loss type: {loss_type}-------------")

    def forward(self, Z, B, A, atom_positions, block_lengths, lengths, segment_ids, label, symbol) -> ReturnValue:
        return_value = super().forward(Z, B, A, atom_positions, block_lengths, lengths, segment_ids, label, symbol, mdn_th=7.0, score_th=5.0)
        loss_mdn = return_value.mdn_loss
        
        if self.loss_type == "mdn":
            loss = loss_mdn
        
        elif self.loss_type == "merge":
            score = -return_value.energy
            if score.shape[0] < label.shape[0]:
                label = label[:score.shape[0]]
            
            if torch.isnan(score).any() or torch.isinf(score).any():
                print(score)
                raise ValueError("there is nan or inf in pred socres")
                
            if torch.isnan(label).any() or torch.isinf(label).any():
                print(label)
                raise ValueError("there is nan or inf in labels")
            
            if label.shape[0]<2:
                loss_corr = - 0.5
            else:
                loss_corr = - torch.corrcoef(torch.stack([score, label]))[1, 0]
                
            loss_mse = F.mse_loss(score, label)
            
            loss = 0.5 * loss_mse + 5 * loss_corr + 1 * loss_mdn
            
            print(f"loss_corr: {loss_corr:.2f}, loss_mse: {loss_mse:.2f}, loss_mdn: {loss_mdn:.2f}")
            
        return loss
            
    def infer(self, batch, output_type):
        return_value = super().forward(
            Z=batch['X'], B=batch['B'], A=batch['A'],
            atom_positions=batch['atom_positions'],
            block_lengths=batch['block_lengths'],
            lengths=batch['lengths'],
            segment_ids=batch['segment_ids'],
            label=None, 
            symbol=batch['symbol'],
            mdn_th=5.0,
            score_th=5.0 
        )
        # Docking/Screening pipeline
        if output_type == "docking/screening":
            print("Current score is used for Docking/Screening pipeline")
            score = return_value.total_potential
       
        # Scoring/Ranking pipeline
        elif output_type == "scoring/ranking":
            print("Current score is used for Scoring/Ranking pipeline")
            score = -return_value.energy
        else:
            raise NotImplementedError(f'Output type {output_type} not implemented!')
        return score
    
    @classmethod
    def load_from_pretrained(cls, pretrain_ckpt, loss_type, **kwargs):
        pretrained_model: BioScoreModel = torch.load(pretrain_ckpt, map_location='cpu')
        model = cls(
            model_type=pretrained_model.model_type,
            hidden_size=pretrained_model.hidden_size,
            n_channel=pretrained_model.n_channel,

            n_rbf=pretrained_model.n_rbf,
            cutoff=pretrained_model.cutoff,
            n_head=pretrained_model.n_head,

            radial_size=pretrained_model.radial_size,
            edge_size=pretrained_model.edge_size,
            k_neighbors=pretrained_model.k_neighbors,
            n_layers=pretrained_model.n_layers,
            
            dropout=pretrained_model.dropout,
            std=pretrained_model.std,

            atom_level=pretrained_model.atom_level,
            hierarchical=pretrained_model.hierarchical,
            no_block_embedding=pretrained_model.no_block_embedding,

            loss_type=loss_type,

            **kwargs
        )
        model.load_state_dict(pretrained_model.state_dict(), strict=False)

        return model