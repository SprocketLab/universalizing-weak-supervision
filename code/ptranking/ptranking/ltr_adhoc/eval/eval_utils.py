#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Description
Given the neural ranker, compute nDCG values.
"""

import torch
import numpy as np

from ptranking.data.data_utils import LABEL_TYPE
from ptranking.metric.adhoc_metric import torch_nDCG_at_k, torch_nDCG_at_ks, torch_kendall_tau, torch_precision_at_k

def ndcg_at_k(pred=None, ranker=None, test_data=None, k=10, label_type=LABEL_TYPE.MultiLabel, gpu=False, device=None):
    '''
    There is no check based on the assumption (say light_filtering() is called) that each test instance Q includes at least k documents,
    and at least one relevant document. Or there will be errors.
    '''
    sum_ndcg_at_k = torch.zeros(1)
    cnt = torch.zeros(1)
    already_sorted = True if test_data.presort else False

    for i, (qid, batch_ranking, batch_labels) in enumerate(test_data): # _, [batch, ranking_size, num_features], [batch, ranking_size]
        if batch_labels.size(1) < k: continue # skip the query if the number of associated documents is smaller than k

        if gpu: batch_ranking = batch_ranking.to(device)

        if (ranker is None) and (pred is not None):
            batch_rele_preds = pred[i].unsqueeze(dim=0)
        else:
            batch_rele_preds = ranker.predict(batch_ranking)
        if gpu: batch_rele_preds = batch_rele_preds.cpu()

        _, batch_sorted_inds = torch.sort(batch_rele_preds, dim=1, descending=True)

        batch_sys_sorted_labels = torch.gather(batch_labels, dim=1, index=batch_sorted_inds)
        if already_sorted:
            batch_ideal_sorted_labels = batch_labels
        else:
            batch_ideal_sorted_labels, _ = torch.sort(batch_labels, dim=1, descending=True)

        batch_ndcg_at_k = torch_nDCG_at_k(batch_sys_sorted_labels=batch_sys_sorted_labels,
                                          batch_ideal_sorted_labels=batch_ideal_sorted_labels,
                                          k = k, label_type=label_type)

        sum_ndcg_at_k += torch.sum(batch_ndcg_at_k)
        cnt += batch_ndcg_at_k.shape[0]

    avg_ndcg_at_k = sum_ndcg_at_k/cnt
    return  avg_ndcg_at_k

def ndcg_at_ks(pred=None, ranker=None, test_data=None, ks=[1, 5, 10], label_type=LABEL_TYPE.MultiLabel, gpu=False, device=None):
    '''
    There is no check based on the assumption (say light_filtering() is called)
    that each test instance Q includes at least k(k=max(ks)) documents, and at least one relevant document.
    Or there will be errors.
    '''
    sum_ndcg_at_ks = torch.zeros(len(ks))
    cnt = torch.zeros(1)
    already_sorted = True if test_data.presort else False
    for i, (qid, batch_ranking, batch_labels) in enumerate(test_data): # _, [batch, ranking_size, num_features], [batch, ranking_size]
        if gpu: batch_ranking = batch_ranking.to(device)
        if (ranker is None) and (pred is not None):
            batch_rele_preds = pred[i].unsqueeze(dim=0)
        else:
            batch_rele_preds = ranker.predict(batch_ranking)
        if gpu: batch_rele_preds = batch_rele_preds.cpu()

        _, batch_sorted_inds = torch.sort(batch_rele_preds, dim=1, descending=True)

        batch_sys_sorted_labels = torch.gather(batch_labels, dim=1, index=batch_sorted_inds)
        if already_sorted:
            batch_ideal_sorted_labels = batch_labels
        else:
            batch_ideal_sorted_labels, _ = torch.sort(batch_labels, dim=1, descending=True)

        # print('batch_sys_sorted_labels.shape', batch_sys_sorted_labels.shape)
        # print('batch_ideal_sorted_labels.shape', batch_ideal_sorted_labels.shape)

        batch_ndcg_at_ks = torch_nDCG_at_ks(batch_sys_sorted_labels=batch_sys_sorted_labels,
                                            batch_ideal_sorted_labels=batch_ideal_sorted_labels,
                                            ks=ks, label_type=label_type)

        # print('batch_ndcg_at_ks', batch_ndcg_at_ks)
        # default batch_size=1 due to testing data
        sum_ndcg_at_ks = torch.add(sum_ndcg_at_ks, torch.squeeze(batch_ndcg_at_ks, dim=0))
        cnt += 1

    avg_ndcg_at_ks = sum_ndcg_at_ks/cnt
    # print('sum_ndcg_at_ks', sum_ndcg_at_ks, 'cnt', cnt, 'avg_ndcg_at_ks', avg_ndcg_at_ks)
    return avg_ndcg_at_ks


def kendall_tau(pred=None, ranker=None, test_data=None, label_type=LABEL_TYPE.MultiLabel, gpu=False, device=None):
    '''
    Calculate kendall_tau
    '''

    tau_list = []
    cnt = torch.zeros(1)
    already_sorted = True if test_data.presort else False
    for i, (qid, batch_ranking, batch_labels) in enumerate(test_data): # _, [batch, ranking_size, num_features], [batch, ranking_size]
        # print('batch_ranking.shape in kendall_tau', batch_ranking.shape)
        # print('batch_labels.shape in kendall_tau', batch_labels.shape)
        if gpu: batch_ranking = batch_ranking.to(device)
        if (ranker is None) and (pred is not None):
            batch_rele_preds = pred[i].unsqueeze(dim=0)
        else:
            batch_rele_preds = ranker.predict(batch_ranking)

        # print('batch_rele_preds.shape', batch_rele_preds.shape)
        if gpu: batch_rele_preds = batch_rele_preds.cpu()

        # get sorted index of batch_rele_preds so that lowest score comes first
        # print('batch_rele_preds.shape', batch_rele_preds.shape)
        _, batch_sorted_inds = torch.sort(batch_rele_preds, dim=-1, descending=False)
        # print('batch_sorted_inds.shape', batch_sorted_inds.shape)

        # sort based on batch_sorted_inds
        # if two rankings are the same with each other, batch_labels should be ascending
        # it's because torch_kendall_tau's natural_ascending is True as default
        batch_sys_sorted_labels = torch.gather(batch_labels, dim=-1, index=batch_sorted_inds)
        if batch_sys_sorted_labels.shape[0] == 1:
            kendall_tau = torch_kendall_tau(batch_sys_sorted_labels.view(-1))
            tau_list.append(kendall_tau)
        else:
            kendall_tau_list = [torch_kendall_tau(x) for x in batch_sys_sorted_labels]
            tau_list.extend(kendall_tau_list)
    avg_kendall_tau = torch.mean(torch.tensor(tau_list))
    return avg_kendall_tau

def precision_at_k(pred=None, ranker=None, test_data=None, k=20, label_type=LABEL_TYPE.MultiLabel, gpu=False, device=None):
    '''
    There is no check based on the assumption (say light_filtering() is called) that each test instance Q includes at least k documents,
    and at least one relevant document. Or there will be errors.
    '''
    sum_precision_at_k = torch.zeros(1)
    cnt = torch.zeros(1)
    already_sorted = True if test_data.presort else False
    for i, (qid, batch_ranking, batch_labels) in enumerate(test_data): # _, [batch, ranking_size, num_features], [batch, ranking_size]
        if batch_labels.size(1) < k: continue # skip the query if the number of associated documents is smaller than k

        if gpu: batch_ranking = batch_ranking.to(device)
        if (ranker is None) and (pred is not None):
            batch_rele_preds = pred[i].unsqueeze(dim=0)
        else:
            batch_rele_preds = ranker.predict(batch_ranking)
        if gpu: batch_rele_preds = batch_rele_preds.cpu()

        _, batch_sorted_inds = torch.sort(batch_rele_preds, dim=1, descending=False)

        batch_precision_at_k = torch_precision_at_k(batch_sys_sorted_labels=batch_sorted_inds, k = k)

        sum_precision_at_k += torch.sum(batch_precision_at_k)
        cnt += batch_precision_at_k.shape[0]

    avg_precision_at_k = sum_precision_at_k/cnt
    return  avg_precision_at_k