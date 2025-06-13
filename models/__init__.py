#!/usr/bin/python
# -*- coding:utf-8 -*-
from .score_model import BioScoreModel
from .affinity_predictor import AffinityPredictor

def create_model(args):
    model_type = args.model_type
    add_params = {}
    if args.task == 'PPI' or args.task == 'PLI' or args.task == 'PLI+PPI' or args.task == 'PLI+PPI+PNI' or args.task == 'PNI' or args.task == 'PLI+PNI' or args.task == 'PLI+PPI+PNI+NLI' or args.task == 'NLI-TEST':
        Model = AffinityPredictor
    else:
        raise NotImplementedError(f'Model for task {args.task} not implemented')

    if args.pretrain_ckpt:
        model = Model.load_from_pretrained(args.pretrain_ckpt, loss_type=args.loss_type, **add_params)
        assert model.model_type == model_type
        return model
    else:
        return Model(
            model_type=model_type,
            hidden_size=args.hidden_size,
            n_channel=args.n_channel,

            n_rbf=args.n_rbf,
            cutoff=args.cutoff,
            n_head=args.n_head,

            radial_size=args.radial_size,
            edge_size=args.edge_size,
            k_neighbors=args.k_neighbors,
            n_layers=args.n_layers,

            atom_level=args.atom_level,
            hierarchical=args.hierarchical,
            no_block_embedding=args.no_block_embedding,

            loss_type=args.loss_type,

            **add_params
        )
