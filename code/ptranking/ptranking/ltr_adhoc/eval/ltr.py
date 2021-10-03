#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Description
A general framework for evaluating traditional learning-to-rank methods.
"""

import os
import sys
import datetime
import numpy as np

import torch

#from tensorboardX import SummaryWriter
from ptranking.base.ranker import LTRFRAME_TYPE
from ptranking.utils.bigdata.BigPickle import pickle_save
from ptranking.metric.metric_utils import metric_results_to_string
from ptranking.data.data_utils import get_data_meta, SPLIT_TYPE, LABEL_TYPE
from ptranking.ltr_adhoc.eval.eval_utils import ndcg_at_ks, ndcg_at_k, kendall_tau, precision_at_k
from ptranking.data.data_utils import LTRDataset, YAHOO_LTR, ISTELLA_LTR, MSLETOR_SEMI, MSLETOR_LIST, MOVIES, BOARDGAMES, SYNTHETIC
from ptranking.ltr_adhoc.eval.parameter import ModelParameter, DataSetting, EvalSetting, ScoringFunctionParameter

from ptranking.ltr_adhoc.pointwise.rank_mse   import RankMSE
from ptranking.ltr_adhoc.pairwise.ranknet     import RankNet, RankNetParameter
from ptranking.ltr_adhoc.listwise.rank_cosine import RankCosine
from ptranking.ltr_adhoc.listwise.listnet     import ListNet
from ptranking.ltr_adhoc.listwise.listmle     import ListMLE, ListMLEParameter
from ptranking.ltr_adhoc.listwise.st_listnet  import STListNet, STListNetParameter

from ptranking.ltr_adhoc.listwise.approxNDCG        import ApproxNDCG, ApproxNDCGParameter
from ptranking.ltr_adhoc.listwise.wassrank.wassRank import WassRank, WassRankParameter
from ptranking.ltr_adhoc.listwise.lambdarank        import LambdaRank, LambdaRankParameter
from ptranking.ltr_adhoc.listwise.lambdaloss import LambdaLoss, LambdaLossParameter

LTR_ADHOC_MODEL = ['RankMSE',
                   'RankNet',
                   'RankCosine', 'ListNet', 'ListMLE', 'STListNet', 'ApproxNDCG', 'WassRank', 'LambdaRank', 'LambdaLoss']

class LTREvaluator():
    """
    The class for evaluating different ltr_adhoc methods.
    """
    def __init__(self, frame_id=LTRFRAME_TYPE.Adhoc):
        self.frame_id = frame_id

        if torch.cuda.is_available():
            self.gpu, self.device = True, 'cuda:'+str(torch.cuda.current_device())
        else:
            self.gpu, self.device = False, 'cpu'

    def display_information(self, data_dict, model_para_dict):
        """
        Display some information.
        :param data_dict:
        :param model_para_dict:
        :return:
        """
        if self.gpu: print('-- GPU({}) is launched --'.format(self.device))
        print(' '.join(['\nStart {} on {} >>>'.format(model_para_dict['model_id'], data_dict['data_id'])]))

    def check_consistency(self, data_dict, eval_dict, sf_para_dict):
        """
        Check whether the settings are reasonable in the context of adhoc learning-to-rank
        """
        ''' Part-1: data loading '''

        if data_dict['data_id'] == 'Istella':
            assert eval_dict['do_validation'] is not True # since there is no validation data

        if data_dict['data_id'] in MSLETOR_SEMI:
            assert data_dict['train_presort'] is not True # due to the non-labeled documents
            if data_dict['binary_rele']: # for unsupervised dataset, it is required for binarization due to '-1' labels
                assert data_dict['unknown_as_zero']
        else:
            assert data_dict['unknown_as_zero'] is not True  # since there is no non-labeled documents

        if data_dict['data_id'] in MSLETOR_LIST: # for which the standard ltr_adhoc of each query is unique
            assert 1 == data_dict['train_batch_size']

        if data_dict['scale_data']:
            scaler_level = data_dict['scaler_level'] if 'scaler_level' in data_dict else None
            assert not scaler_level== 'DATASET' # not supported setting

        assert data_dict['validation_presort'] # Rule of thumb setting for adhoc learning-to-rank
        assert data_dict['test_presort'] # Rule of thumb setting for adhoc learning-to-rank
        assert 1 == data_dict['validation_batch_size'] # Rule of thumb setting for adhoc learning-to-rank
        assert 1 == data_dict['test_batch_size'] # Rule of thumb setting for adhoc learning-to-rank

        ''' Part-2: evaluation setting '''

        if eval_dict['mask_label']: # True is aimed to use supervised data to mimic semi-supervised data by masking
            assert not data_dict['data_id'] in MSLETOR_SEMI

        ''' Part-1: network setting '''

        if data_dict['train_batch_size']>1:
            assert sf_para_dict['one_fits_all']['BN'] == False # a kind of feature normalization


    def determine_files(self, data_dict, fold_k=None):
        """
        Determine the file path correspondingly.
        :param data_dict:
        :param fold_k:
        :return:
        """
        if data_dict['data_id'] in YAHOO_LTR:
            file_train, file_vali, file_test = os.path.join(data_dict['dir_data'], data_dict['data_id'].lower() + '.train.txt'),\
                                               os.path.join(data_dict['dir_data'], data_dict['data_id'].lower() + '.valid.txt'),\
                                               os.path.join(data_dict['dir_data'], data_dict['data_id'].lower() + '.test.txt')

        elif data_dict['data_id'] in ISTELLA_LTR:
            if data_dict['data_id'] == 'Istella_X' or data_dict['data_id']=='Istella_S':
                file_train, file_vali, file_test = data_dict['dir_data'] + 'train.txt', data_dict['dir_data'] + 'vali.txt', data_dict['dir_data'] + 'test.txt'
            else:
                file_vali = None
                file_train, file_test = data_dict['dir_data'] + 'train.txt', data_dict['dir_data'] + 'test.txt'

        elif (data_dict['data_id'].upper() in MOVIES) or (data_dict['data_id'].upper() in BOARDGAMES):
            file_train, file_test = os.path.join(data_dict['dir_data'], 'train.npz'),\
                                        os.path.join(data_dict['dir_data'], 'test.npz')

        else:
            print('Fold-', fold_k)
            fold_k_dir = data_dict['dir_data'] + 'Fold' + str(fold_k) + '/'
            file_train, file_vali, file_test = fold_k_dir + 'train.txt', fold_k_dir + 'vali.txt', fold_k_dir + 'test.txt'

        return file_train, file_vali, file_test


    def load_data(self, eval_dict, data_dict, fold_k):
        """
        Load the dataset correspondingly.
        :param eval_dict:
        :param data_dict:
        :param fold_k:
        :param model_para_dict:
        :return:
        """
        file_train, file_vali, file_test = self.determine_files(data_dict, fold_k=fold_k)

        train_batch_size, train_presort = data_dict['train_batch_size'], data_dict['train_presort']
        input_eval_dict = eval_dict if eval_dict['mask_label'] else None # required when enabling masking data
        train_data = LTRDataset(file=file_train, split_type=SPLIT_TYPE.Train, batch_size=train_batch_size,
                                shuffle=True, presort=train_presort, data_dict=data_dict, eval_dict=input_eval_dict)

        test_data = LTRDataset(file=file_test, split_type=SPLIT_TYPE.Test, shuffle=False, data_dict=data_dict,
                               batch_size=data_dict['test_batch_size'])

        if eval_dict['do_validation'] or eval_dict['do_summary']: # vali_data is required
            vali_data = LTRDataset(file=file_vali, split_type=SPLIT_TYPE.Validation, shuffle=False,
                                   batch_size=data_dict['validation_batch_size'], data_dict=data_dict)
        else:
            vali_data = None

        return train_data, test_data, vali_data

    def set_and_load_data(self, X_train, X_test, Y_train, Y_test,
                          eval_dict, data_dict,
                          qid_train=None, qid_test=None,
                          root_path = '.',
                          file_train='train.npz',
                          file_test='test.npz',
                          file_vali=None):

        if type(X_train) != np.ndarray:
            X_train, X_test, Y_train, Y_test = X_train.numpy(), X_test.numpy(), Y_train.numpy(), Y_test.numpy()
        qid_train = np.array(list(range(X_train.shape[0])))
        qid_test = np.array(list(range(X_test.shape[0]))) + X_train.shape[0]  # sum to prevent duplication

        # save files
        save_path = os.path.join(root_path, self.data_setting.data_dict['dir_data'])
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            print("processed data path", save_path, "generated")

        print('set_and_load_data in LTREvaluator')
        print(X_train.shape, Y_train.shape, qid_train.shape)
        np.savez(os.path.join(save_path, file_train), X=X_train, Y=Y_train, qid=qid_train)
        np.savez(os.path.join(save_path, file_test), X=X_test, Y=Y_test, qid=qid_test)

        train_batch_size, train_presort = data_dict['train_batch_size'], data_dict['train_presort']
        input_eval_dict = eval_dict if eval_dict['mask_label'] else None  # required when enabling masking data
        num_features = X_train.shape[2]

        # load files
        train_data = LTRDataset(file=os.path.join(save_path, file_train), split_type=SPLIT_TYPE.Train, batch_size=train_batch_size,
                                shuffle=True, presort=train_presort, data_dict=data_dict, eval_dict=input_eval_dict,
                                num_features=num_features)

        test_data = LTRDataset(file=os.path.join(save_path, file_test), split_type=SPLIT_TYPE.Test, shuffle=False, data_dict=data_dict,
                               batch_size=data_dict['test_batch_size'], num_features=num_features)

        if file_vali is not None:  # vali_data is required
            vali_data = LTRDataset(file=file_vali, split_type=SPLIT_TYPE.Validation, shuffle=False,
                                   batch_size=data_dict['validation_batch_size'], data_dict=data_dict,
                                   num_features=num_features)
        else:
            vali_data = None

        return train_data, test_data, vali_data

    def load_ranker(self, sf_para_dict, model_para_dict, opt='Adam', lr = 1e-3, weight_decay=1e-3):
        """
        Load a ranker correspondingly
        :param sf_para_dict:
        :param model_para_dict:
        :param kwargs:
        :return:
        """
        model_id = model_para_dict['model_id']


        if model_id in ['RankMSE', 'ListNet', 'RankCosine']:
            ranker = globals()[model_id](sf_para_dict=sf_para_dict, gpu=self.gpu, device=self.device,
                                         opt=opt, lr=lr, weight_decay=weight_decay)

        elif model_id in ['RankNet', 'ListMLE', 'LambdaRank', 'STListNet', 'ApproxNDCG', 'DirectOpt',
                          'LambdaLoss', 'MarginLambdaLoss']:
            ranker = globals()[model_id](sf_para_dict=sf_para_dict, model_para_dict=model_para_dict, gpu=self.gpu,
                                         device=self.device, opt=opt, lr=lr, weight_decay=weight_decay)

        elif model_id == 'WassRank':
            ranker = WassRank(sf_para_dict=sf_para_dict, wass_para_dict=model_para_dict,
                              dict_cost_mats=self.dict_cost_mats, dict_std_dists=self.dict_std_dists,
                              gpu=self.gpu, device=self.device,
                              opt=opt, lr=lr, weight_decay=weight_decay)
        else:
            raise NotImplementedError

        return ranker


    def setup_output(self, data_dict=None, eval_dict=None):
        """
        Update output directory
        :param data_dict:
        :param eval_dict:
        :param sf_para_dict:
        :param model_para_dict:
        :return:
        """
        model_id = self.model_parameter.model_id
        grid_search, do_vali, dir_output = eval_dict['grid_search'], eval_dict['do_validation'], eval_dict['dir_output']
        mask_label = eval_dict['mask_label']

        if grid_search:
            dir_root = dir_output + '_'.join(['gpu', 'grid', model_id]) + '/' if self.gpu else dir_output + '_'.join(['grid', model_id]) + '/'
        else:
            dir_root = dir_output

        eval_dict['dir_root'] = dir_root
        # if not os.path.exists(dir_root): os.makedirs(dir_root)

        sf_str = self.sf_parameter.to_para_string()
        data_eval_str = '_'.join([self.data_setting.to_data_setting_string(),
                                  self.eval_setting.to_eval_setting_string()])
        if mask_label:
            data_eval_str = '_'.join([data_eval_str, 'MaskLabel', 'Ratio', '{:,g}'.format(eval_dict['mask_ratio'])])

        file_prefix = '_'.join([model_id, 'SF', sf_str, data_eval_str])

        if data_dict['scale_data']:
            if data_dict['scaler_level'] == 'QUERY':
                file_prefix = '_'.join([file_prefix, 'QS', data_dict['scaler_id']])
            else:
                file_prefix = '_'.join([file_prefix, 'DS', data_dict['scaler_id']])

        dir_run = dir_root + file_prefix + '/'  # run-specific outputs

        model_para_string = self.model_parameter.to_para_string()
        if len(model_para_string) > 0:
            dir_run = dir_run + model_para_string + '/'

        eval_dict['dir_run'] = dir_run
        # if not os.path.exists(dir_run):
        #     os.makedirs(dir_run)

        return dir_run


    def setup_eval(self, data_dict, eval_dict, sf_para_dict, model_para_dict):
        """
        Finalize the evaluation setting correspondingly
        :param data_dict:
        :param eval_dict:
        :param sf_para_dict:
        :param model_para_dict:
        :return:
        """
        # update data_meta given the debug mode
        if sf_para_dict['id'] == 'ffnns':
            sf_para_dict['ffnns'].update(dict(num_features=data_dict['num_features']))
        else:
            raise NotImplementedError

        self.dir_run  = self.setup_output(data_dict, eval_dict)
        if eval_dict['do_log']: sys.stdout = open(self.dir_run + 'log.txt', "w")
        #if self.do_summary: self.summary_writer = SummaryWriter(self.dir_run + 'summary')


    def log_max(self, data_dict=None, max_cv_avg_scores=None, sf_para_dict=None,  eval_dict=None, log_para_str=None):
        ''' Log the best performance across grid search and the corresponding setting '''
        dir_root, cutoffs = eval_dict['dir_root'], eval_dict['cutoffs']
        data_id = data_dict['data_id']

        sf_str = self.sf_parameter.to_para_string(log=True)

        data_eval_str = self.data_setting.to_data_setting_string(log=True) +'\n'+ self.eval_setting.to_eval_setting_string(log=True)

        with open(file=dir_root + '/' + data_id + '_max.txt', mode='w') as max_writer:
            max_writer.write('\n\n'.join([data_eval_str, sf_str, log_para_str, metric_results_to_string(max_cv_avg_scores, cutoffs)]))


    def train_ranker(self, ranker, train_data, model_para_dict=None, epoch_k=None, reranking=False):
        '''	One-epoch train of the given ranker '''
        presort, label_type = train_data.presort, train_data.label_type
        epoch_loss = torch.cuda.FloatTensor([0.0]) if self.gpu else torch.FloatTensor([0.0])
        for qid, batch_rankings, batch_stds in train_data: # size: _, [batch, ranking_size, num_features], [batch, ranking_size]
            if self.gpu: batch_rankings, batch_stds = batch_rankings.to(self.device), batch_stds.to(self.device)

            batch_loss, stop_training = ranker.train(batch_rankings, batch_stds, qid=qid, epoch_k=epoch_k, presort=presort, label_type=label_type)

            if stop_training:
                break
            else:
                epoch_loss += batch_loss.item()

        return epoch_loss, stop_training

    def kfold_cv_eval(self, data_dict=None, eval_dict=None, sf_para_dict=None, model_para_dict=None):
        """
        Evaluation learning-to-rank methods via k-fold cross validation if there are k folds, otherwise one fold.
        :param data_dict:       settings w.r.t. data
        :param eval_dict:       settings w.r.t. evaluation
        :param sf_para_dict:    settings w.r.t. scoring function
        :param model_para_dict: settings w.r.t. the ltr_adhoc model
        :return:
        """
        self.display_information(data_dict, model_para_dict)
        self.check_consistency(data_dict, eval_dict, sf_para_dict)
        self.setup_eval(data_dict, eval_dict, sf_para_dict, model_para_dict)

        model_id = model_para_dict['model_id']
        fold_num = data_dict['fold_num']
        # for quick access of common evaluation settings
        epochs, loss_guided = eval_dict['epochs'], eval_dict['loss_guided']
        vali_k, log_step, cutoffs   = eval_dict['vali_k'], eval_dict['log_step'], eval_dict['cutoffs']
        do_vali, do_summary = eval_dict['do_validation'], eval_dict['do_summary']

        ranker = self.load_ranker(model_para_dict=model_para_dict, sf_para_dict=sf_para_dict)

        time_begin = datetime.datetime.now()       # timing
        l2r_cv_avg_scores = np.zeros(len(cutoffs)) # fold average

        for fold_k in range(1, fold_num + 1):   # evaluation over k-fold data
            ranker.reset_parameters()           # reset with the same random initialization

            train_data, test_data, vali_data = self.load_data(eval_dict, data_dict, fold_k)

            if do_vali: fold_optimal_ndcgk = 0.0
            if do_summary: list_epoch_loss, list_fold_k_train_eval_track, list_fold_k_test_eval_track, list_fold_k_vali_eval_track = [], [], [], []
            if not do_vali and loss_guided:
                first_round = True
                threshold_epoch_loss = torch.cuda.FloatTensor([10000000.0]) if self.gpu else torch.FloatTensor([10000000.0])

            for epoch_k in range(1, epochs + 1):
                torch_fold_k_epoch_k_loss, stop_training = self.train_ranker(ranker=ranker, train_data=train_data, model_para_dict=model_para_dict, epoch_k=epoch_k)

                ranker.scheduler.step()  # adaptive learning rate with step_size=40, gamma=0.5

                if stop_training:
                    print('training is failed !')
                    break

                if (do_summary or do_vali) and (epoch_k % log_step == 0 or epoch_k == 1):  # stepwise check
                    if do_vali:     # per-step validation score
                        vali_eval_tmp = ndcg_at_k(ranker=ranker, test_data=vali_data, k=vali_k, gpu=self.gpu, device=self.device,
                                                  label_type=self.data_setting.data_dict['label_type'])
                        vali_eval_v = vali_eval_tmp.data.numpy()
                        if epoch_k > 1:  # further validation comparison
                            curr_vali_ndcg = vali_eval_v
                            if (curr_vali_ndcg > fold_optimal_ndcgk) or (epoch_k == epochs and curr_vali_ndcg == fold_optimal_ndcgk):  # we need at least a reference, in case all zero
                                print('\t', epoch_k, '- nDCG@{} - '.format(vali_k), curr_vali_ndcg)
                                fold_optimal_ndcgk = curr_vali_ndcg
                                fold_optimal_checkpoint = '-'.join(['Fold', str(fold_k)])
                                fold_optimal_epoch_val = epoch_k
                                ranker.save(dir=self.dir_run + fold_optimal_checkpoint + '/', name='_'.join(['net_params_epoch', str(epoch_k)]) + '.pkl')  # buffer currently optimal model
                            else:
                                print('\t\t', epoch_k, '- nDCG@{} - '.format(vali_k), curr_vali_ndcg)

                    if do_summary:  # summarize per-step performance w.r.t. train, test
                        fold_k_epoch_k_train_ndcg_ks = ndcg_at_ks(ranker=ranker, test_data=train_data, ks=cutoffs, gpu=self.gpu, device=self.device,
                                                                  label_type=self.data_setting.data_dict['label_type'])
                        np_fold_k_epoch_k_train_ndcg_ks = fold_k_epoch_k_train_ndcg_ks.cpu().numpy() if self.gpu else fold_k_epoch_k_train_ndcg_ks.data.numpy()
                        list_fold_k_train_eval_track.append(np_fold_k_epoch_k_train_ndcg_ks)

                        fold_k_epoch_k_test_ndcg_ks  = ndcg_at_ks(ranker=ranker, test_data=test_data, ks=cutoffs, gpu=self.gpu, device=self.device,
                                                                  label_type=self.data_setting.data_dict['label_type'])
                        np_fold_k_epoch_k_test_ndcg_ks  = fold_k_epoch_k_test_ndcg_ks.cpu().numpy() if self.gpu else fold_k_epoch_k_test_ndcg_ks.data.numpy()
                        list_fold_k_test_eval_track.append(np_fold_k_epoch_k_test_ndcg_ks)

                        fold_k_epoch_k_loss = torch_fold_k_epoch_k_loss.cpu().numpy() if self.gpu else torch_fold_k_epoch_k_loss.data.numpy()
                        list_epoch_loss.append(fold_k_epoch_k_loss)

                        if do_vali: list_fold_k_vali_eval_track.append(vali_eval_v)

                elif loss_guided:  # stopping check via epoch-loss
                    if first_round and torch_fold_k_epoch_k_loss >= threshold_epoch_loss:
                        print('Bad threshold: ', torch_fold_k_epoch_k_loss, threshold_epoch_loss)

                    if torch_fold_k_epoch_k_loss < threshold_epoch_loss:
                        first_round = False
                        print('\tFold-', str(fold_k), ' Epoch-', str(epoch_k), 'Loss: ', torch_fold_k_epoch_k_loss)
                        threshold_epoch_loss = torch_fold_k_epoch_k_loss
                    else:
                        print('\tStopped according epoch-loss!', torch_fold_k_epoch_k_loss, threshold_epoch_loss)
                        break

            if do_summary:  # track
                sy_prefix = '_'.join(['Fold', str(fold_k)])
                fold_k_train_eval = np.vstack(list_fold_k_train_eval_track)
                fold_k_test_eval  = np.vstack(list_fold_k_test_eval_track)
                pickle_save(fold_k_train_eval, file=self.dir_run + '_'.join([sy_prefix, 'train_eval.np']))
                pickle_save(fold_k_test_eval, file=self.dir_run + '_'.join([sy_prefix, 'test_eval.np']))

                fold_k_epoch_loss = np.hstack(list_epoch_loss)
                pickle_save((fold_k_epoch_loss, train_data.__len__()), file=self.dir_run + '_'.join([sy_prefix, 'epoch_loss.np']))
                if do_vali:
                    fold_k_vali_eval = np.hstack(list_fold_k_vali_eval_track)
                    pickle_save(fold_k_vali_eval, file=self.dir_run + '_'.join([sy_prefix, 'vali_eval.np']))

            if do_vali: # using the fold-wise optimal model for later testing based on validation data
                buffered_model = '_'.join(['net_params_epoch', str(fold_optimal_epoch_val)]) + '.pkl'
                ranker.load(self.dir_run + fold_optimal_checkpoint + '/' + buffered_model)
                fold_optimal_ranker = ranker
            else:            # buffer the model after a fixed number of training-epoches if no validation is deployed
                fold_optimal_checkpoint = '-'.join(['Fold', str(fold_k)])
                ranker.save(dir=self.dir_run + fold_optimal_checkpoint + '/', name='_'.join(['net_params_epoch', str(epoch_k)]) + '.pkl')
                fold_optimal_ranker = ranker

            torch_fold_ndcg_ks = ndcg_at_ks(ranker=fold_optimal_ranker, test_data=test_data, ks=cutoffs, gpu=self.gpu, device=self.device,
                                            label_type=self.data_setting.data_dict['label_type'])
            fold_ndcg_ks = torch_fold_ndcg_ks.data.numpy()

            performance_list = [model_id + ' Fold-' + str(fold_k)]      # fold-wise performance
            for i, co in enumerate(cutoffs):
                performance_list.append('nDCG@{}:{:.4f}'.format(co, fold_ndcg_ks[i]))
            performance_str = '\t'.join(performance_list)
            print('\t', performance_str)

            l2r_cv_avg_scores = np.add(l2r_cv_avg_scores, fold_ndcg_ks) # sum for later cv-performance

        time_end = datetime.datetime.now()  # overall timing
        elapsed_time_str = str(time_end - time_begin)
        print('Elapsed time:\t', elapsed_time_str + "\n\n")

        l2r_cv_avg_scores = np.divide(l2r_cv_avg_scores, fold_num)
        eval_prefix = str(fold_num) + '-fold cross validation scores:' if do_vali else str(fold_num) + '-fold average scores:'
        print(model_id, eval_prefix, metric_results_to_string(list_scores=l2r_cv_avg_scores, list_cutoffs=cutoffs))  # print either cv or average performance

        return l2r_cv_avg_scores

    def naive_train(self, ranker, eval_dict, train_data=None, test_data=None, vali_data=None):
        """
        A simple train and test, namely train based on training data & test based on testing data
        :param ranker:
        :param eval_dict:
        :param train_data:
        :param test_data:
        :param vali_data:
        :return:
        """
        ranker.reset_parameters()  # reset with the same random initialization

        assert train_data is not None
        assert test_data  is not None

        list_losses = []
        list_train_ndcgs = []
        list_test_ndcgs = []

        epochs, cutoffs = eval_dict['epochs'], eval_dict['cutoffs']

        for i in range(epochs):
            epoch_loss = torch.zeros(1).to(self.device) if self.gpu else torch.zeros(1)
            for qid, batch_rankings, batch_stds in train_data:
                # print(qid, ba)
                if self.gpu: batch_rankings, batch_stds = batch_rankings.to(self.device), batch_stds.to(self.device)
                batch_loss, stop_training = ranker.train(batch_rankings, batch_stds, qid=qid)
                epoch_loss += batch_loss.item()

            np_epoch_loss = epoch_loss.cpu().numpy() if self.gpu else epoch_loss.data.numpy()
            list_losses.append(np_epoch_loss)

            test_ndcg_ks = ndcg_at_ks(ranker=ranker, test_data=test_data, ks=cutoffs, label_type=LABEL_TYPE.MultiLabel, gpu=self.gpu, device=self.device,)
            np_test_ndcg_ks = test_ndcg_ks.data.numpy()
            list_test_ndcgs.append(np_test_ndcg_ks)

            train_ndcg_ks = ndcg_at_ks(ranker=ranker, test_data=train_data, ks=cutoffs, label_type=LABEL_TYPE.MultiLabel, gpu=self.gpu, device=self.device,)
            np_train_ndcg_ks = train_ndcg_ks.data.numpy()
            list_train_ndcgs.append(np_train_ndcg_ks)

        test_ndcgs = np.vstack(list_test_ndcgs)
        train_ndcgs = np.vstack(list_train_ndcgs)

        return list_losses, train_ndcgs, test_ndcgs

    def custom_train(self, ranker, eval_dict, verbose, train_data=None, test_data=None):
        """
        A simple train and test, namely train based on training data & test based on testing data
        :param ranker:
        :param eval_dict:
        :param train_data:
        :param test_data:
        :param vali_data:
        :return:
        """

        assert train_data is not None
        assert test_data  is not None

        list_losses = []
        list_train_tau = []
        list_test_tau = []
        # # TODO: metric selection based on configuration
        list_train_ndcg1 = []
        list_train_ndcg3 = []
        list_train_ndcg5 = []
        list_test_ndcg1 = []
        list_test_ndcg3 = []
        list_test_ndcg5 = []

        epochs = eval_dict['epochs']

        label_type = LABEL_TYPE.Permutation

        for i in range(epochs):
            epoch_loss = torch.zeros(1).to(self.device) if self.gpu else torch.zeros(1)
            for qid, batch_rankings, batch_stds in train_data:
                # print(qid)
                # print('batch_rankings, batch_stds', batch_rankings.shape, batch_stds.shape)
                # print(batch_rankings)
                if self.gpu: batch_rankings, batch_stds = batch_rankings.to(self.device), batch_stds.to(self.device)
                batch_loss, stop_training = ranker.train(batch_rankings, batch_stds, qid=qid)
                epoch_loss += batch_loss.item()

            np_epoch_loss = epoch_loss.cpu().numpy() if self.gpu else epoch_loss.data.numpy()
            list_losses.append(np_epoch_loss)

            test_tau = (1 - kendall_tau(ranker=ranker, test_data=test_data, label_type=label_type,
                                   gpu=self.gpu, device=self.device,)) / 2
            list_test_tau.append(test_tau)

            train_tau = (1 - kendall_tau(ranker=ranker, test_data=train_data, label_type=label_type,
                                   gpu=self.gpu, device=self.device, )) / 2
            list_train_tau.append(train_tau)

            # ndcg metrics
            train_ndcg1 = ndcg_at_k(ranker=ranker, test_data=train_data, k=1,
                                    label_type=LABEL_TYPE.Permutation, gpu=self.gpu, device=self.device)
            train_ndcg3 = ndcg_at_k(ranker=ranker, test_data=train_data, k=3,
                                    label_type=LABEL_TYPE.Permutation, gpu=self.gpu, device=self.device)
            train_ndcg5 = ndcg_at_k(ranker=ranker, test_data=train_data, k=5,
                                    label_type=LABEL_TYPE.Permutation, gpu=self.gpu, device=self.device)

            test_ndcg1 = ndcg_at_k(ranker=ranker, test_data=test_data, k=1,
                                    label_type=LABEL_TYPE.Permutation, gpu=self.gpu, device=self.device)
            test_ndcg3 = ndcg_at_k(ranker=ranker, test_data=test_data, k=3,
                                    label_type=LABEL_TYPE.Permutation, gpu=self.gpu, device=self.device)
            test_ndcg5 = ndcg_at_k(ranker=ranker, test_data=test_data, k=5,
                                    label_type=LABEL_TYPE.Permutation, gpu=self.gpu, device=self.device)

            list_train_ndcg1.append(train_ndcg1)
            list_train_ndcg3.append(train_ndcg3)
            list_train_ndcg5.append(train_ndcg5)
            list_test_ndcg1.append(test_ndcg1)
            list_test_ndcg3.append(test_ndcg3)
            list_test_ndcg5.append(test_ndcg5)

            if (verbose == 1):
                print(f"epoch {i}, loss {np_epoch_loss}, train tau {train_tau}, test_tau {test_tau},"
                      f"train_ndcg@1 {train_ndcg1}, test_ndcg@1 {test_ndcg1}")

        test_tau = np.vstack(list_test_tau)
        train_tau = np.vstack(list_train_tau)

        result_summary = {}
        result_summary['loss'] = list_losses
        result_summary['train_tau'] = train_tau
        result_summary['test_tau'] = test_tau
        result_summary['train_ndcg1'] = list_train_ndcg1
        result_summary['train_ndcg3'] = list_train_ndcg3
        result_summary['train_ndcg5'] = list_train_ndcg5
        result_summary['test_ndcg1'] = list_test_ndcg1
        result_summary['test_ndcg3'] = list_test_ndcg3
        result_summary['test_ndcg5'] = list_test_ndcg5

        return ranker, result_summary

    def custom_train_ir(self, ranker, eval_dict, verbose, train_data=None, test_data=None):
        """
        A simple train and test, namely train based on training data & test based on testing data
        :param ranker:
        :param eval_dict:
        :param train_data:
        :param test_data:
        :param vali_data:
        :return:
        """

        assert train_data is not None
        assert test_data  is not None

        list_losses = []
        list_train_tau = []
        list_test_tau = []
        # # TODO: metric selection based on configuration
        list_train_ndcg1 = []
        list_test_ndcg1 = []
        list_train_p1 = []
        list_test_p1 = []
        epochs = eval_dict['epochs']

        label_type = LABEL_TYPE.Permutation


        for i in range(epochs):
            epoch_loss = torch.zeros(1).to(self.device) if self.gpu else torch.zeros(1)
            qid_count = 0
            for qid, batch_rankings, batch_stds in train_data:
                # print('qid', qid, 'batch_rankings', batch_rankings.shape, 'batch_stds', batch_stds.shape)
                # print(batch_stds)
                
                if self.gpu: batch_rankings, batch_stds = batch_rankings.to(self.device), batch_stds.to(self.device)
                batch_loss, stop_training = ranker.train(batch_rankings, batch_stds, qid=qid)
                epoch_loss += batch_loss.item()

                qid_count += 1
                # if int(qid_count) > 100: # tmp for testing
                #     break

            np_epoch_loss = epoch_loss.cpu().numpy() if self.gpu else epoch_loss.data.numpy()
            list_losses.append(np_epoch_loss)

            test_tau = (1 - kendall_tau(ranker=ranker, test_data=test_data, label_type=label_type,
                                   gpu=self.gpu, device=self.device,)) / 2

            train_tau = (1 - kendall_tau(ranker=ranker, test_data=train_data, label_type=label_type,
                                   gpu=self.gpu, device=self.device, )) / 2

            list_test_tau.append(test_tau)
            list_train_tau.append(train_tau)

            # ndcg metrics
            train_ndcg1 = ndcg_at_k(ranker=ranker, test_data=train_data, k=1,
                                    label_type=LABEL_TYPE.Permutation, gpu=self.gpu, device=self.device)

            test_ndcg1 = ndcg_at_k(ranker=ranker, test_data=test_data, k=1,
                                    label_type=LABEL_TYPE.Permutation, gpu=self.gpu, device=self.device)

            list_train_ndcg1.append(train_ndcg1)
            list_test_ndcg1.append(test_ndcg1)

            # precion metrics
            train_p1 = precision_at_k(ranker=ranker, test_data=train_data, k=1,
                                    label_type=LABEL_TYPE.Permutation, gpu=self.gpu, device=self.device)
            test_p1 = precision_at_k(ranker=ranker, test_data=test_data, k=1,
                                       label_type=LABEL_TYPE.Permutation, gpu=self.gpu, device=self.device)

            list_train_p1.append(train_p1)
            list_test_p1.append(test_p1)

            if (verbose == 1):
                print(f"epoch {i}, loss {np_epoch_loss}, train tau {train_tau}, test_tau {test_tau},"
                      f"train_ndcg@1 {train_ndcg1}, test_ndcg@1 {test_ndcg1}"
                      f"train_p@1 {train_p1}, test_p@1, {test_p1}")

        test_tau = np.vstack(list_test_tau)
        train_tau = np.vstack(list_train_tau)

        result_summary = {}
        result_summary['loss'] = list_losses
        result_summary['train_tau'] = train_tau
        result_summary['test_tau'] = test_tau
        result_summary['train_ndcg1'] = list_train_ndcg1
        result_summary['test_ndcg1'] = list_test_ndcg1
        result_summary['train_p1'] = list_train_p1
        result_summary['test_p1'] = list_test_p1

        return ranker, result_summary

    def eval_with_pred(self, pred, eval_dict, verbose, train_data=None, test_data=None):
        """
        A simple train and test, namely train based on training data & test based on testing data
        :param ranker:
        :param eval_dict:
        :param train_data:
        :param test_data:
        :param vali_data:
        :return:
        """

        assert train_data is not None
        assert test_data  is not None

        label_type = LABEL_TYPE.Permutation

        test_tau = (1 - kendall_tau(pred=pred, test_data=test_data, label_type=label_type,
                                   gpu=self.gpu, device=self.device,)) / 2

        test_ndcg1 = ndcg_at_k(pred=pred, test_data=test_data, k=1,
                                label_type=LABEL_TYPE.Permutation, gpu=self.gpu, device=self.device)

        test_p1 = precision_at_k(pred=pred, test_data=test_data, k=1,
                                       label_type=LABEL_TYPE.Permutation, gpu=self.gpu, device=self.device)

        if (verbose == 1):
            print(f"test_tau {test_tau},"
                  f"test_ndcg@1 {test_ndcg1}"
                  f"test_p@1, {test_p1}")

        result_summary = {}
        result_summary['test_tau'] = test_tau
        result_summary['test_ndcg1'] = test_ndcg1
        result_summary['test_p1'] = test_p1

        return result_summary

    def set_data_setting(self, data_json=None, debug=False, data_id=None, dir_data=None):
        if data_json is not None:
            self.data_setting = DataSetting(data_json=data_json)
        else:
            self.data_setting = DataSetting(debug=debug, data_id=data_id, dir_data=dir_data)

    def get_default_data_setting(self):
        return self.data_setting.default_setting()

    def iterate_data_setting(self):
        return self.data_setting.grid_search()

    def set_eval_setting(self, eval_json=None, debug=False, dir_output=None):
        if eval_json is not None:
            self.eval_setting = EvalSetting(debug=debug, eval_json=eval_json)
        else:
            self.eval_setting = EvalSetting(debug=debug, dir_output=dir_output)

    def get_default_eval_setting(self):
        if self.data_setting.data_dict['data_id'].upper() in MOVIES:
            return self.eval_setting.tmdb_setting()
        elif self.data_setting.data_dict['data_id'].upper() in MOVIES:
            return self.eval_setting.boardgame_setting()
        else:
            return self.eval_setting.default_setting()

    def get_imdb_eval_setting(self):
        return self.eval_setting.tmdb_setting()

    def iterate_eval_setting(self):
        return self.eval_setting.grid_search()

    def set_scoring_function_setting(self, sf_json=None, debug=None, data_dict=None):
        if sf_json is not None:
            self.sf_parameter = ScoringFunctionParameter(sf_json=sf_json)
        else:
            self.sf_parameter = ScoringFunctionParameter(debug=debug, data_dict=data_dict)

    def get_default_scoring_function_setting(self):
        return self.sf_parameter.default_para_dict()

    def iterate_scoring_function_setting(self, data_dict=None):
        return self.sf_parameter.grid_search(data_dict=data_dict)

    def set_model_setting(self, model_id=None, dir_json=None, debug=False):
        """
        Initialize the parameter class for a specified model
        :param debug:
        :param model_id:
        :return:
        """
        if model_id in ['RankMSE', 'ListNet', 'RankCosine']: # ModelParameter is sufficient
            self.model_parameter = ModelParameter(model_id=model_id)
        else:
            if dir_json is not None:
                para_json = dir_json + model_id + "Parameter.json"
                self.model_parameter = globals()[model_id + "Parameter"](para_json=para_json)
            else: # the 3rd type, where debug-mode enables quick test
                self.model_parameter = globals()[model_id + "Parameter"](debug=debug)

    def get_default_model_setting(self):
        return self.model_parameter.default_para_dict()

    def iterate_model_setting(self):
        return self.model_parameter.grid_search()

    def declare_global(self, model_id=None):
        """
        Declare global variants if required, such as for efficiency
        :param model_id:
        :return:
        """
        if model_id == 'WassRank': # global buffering across a number of runs with different model parameters
            self.dict_cost_mats, self.dict_std_dists = dict(), dict()


    def point_run(self, debug=False, model_id=None, data_id=None, dir_data=None, dir_output=None):
        """
        Perform one-time run based on given setting.
        :param debug:
        :param model_id:
        :param data_id:
        :param dir_data:
        :param dir_output:
        :return:
        """
        self.set_eval_setting(debug=debug, dir_output=dir_output)
        self.set_data_setting(debug=debug, data_id=data_id, dir_data=dir_data)
        data_dict = self.get_default_data_setting()
        eval_dict = self.get_default_eval_setting()

        self.set_scoring_function_setting(debug=debug, data_dict=data_dict)
        sf_para_dict = self.get_default_scoring_function_setting()

        self.set_model_setting(debug=debug, model_id=model_id)
        model_para_dict = self.get_default_model_setting()

        self.declare_global(model_id=model_id)

        self.kfold_cv_eval(data_dict=data_dict, eval_dict=eval_dict, model_para_dict=model_para_dict, sf_para_dict=sf_para_dict)


    def grid_run(self, model_id=None, dir_json=None, debug=False, data_id=None, dir_data=None, dir_output=None):
        """
        Explore the effects of different hyper-parameters of a model based on grid-search
        :param debug:
        :param model_id:
        :param data_id:
        :param dir_data:
        :param dir_output:
        :return:
        """
        if dir_json is not None:
            data_eval_sf_json = dir_json + 'Data_Eval_ScoringFunction.json'
            self.set_eval_setting(debug=debug, eval_json=data_eval_sf_json)
            self.set_data_setting(data_json=data_eval_sf_json)
            self.set_scoring_function_setting(sf_json=data_eval_sf_json)
            self.set_model_setting(model_id=model_id, dir_json=dir_json)
        else:
            self.set_eval_setting(debug=debug, dir_output=dir_output)
            self.set_data_setting(debug=debug, data_id=data_id, dir_data=dir_data)
            self.set_scoring_function_setting(debug=debug)
            self.set_model_setting(debug=debug, model_id=model_id)

        self.declare_global(model_id=model_id)

        ''' select the best setting through grid search '''
        vali_k, cutoffs = 5, [1, 3, 5, 10, 20, 50]
        max_cv_avg_scores = np.zeros(len(cutoffs))  # fold average
        k_index = cutoffs.index(vali_k)
        max_common_para_dict, max_sf_para_dict, max_model_para_dict = None, None, None

        for data_dict in self.iterate_data_setting():
            for eval_dict in self.iterate_eval_setting():
                assert self.eval_setting.check_consistence(vali_k=vali_k, cutoffs=cutoffs) # a necessary consistence

                for sf_para_dict in self.iterate_scoring_function_setting(data_dict=data_dict):
                    for model_para_dict in self.iterate_model_setting():
                        curr_cv_avg_scores = self.kfold_cv_eval(data_dict=data_dict, eval_dict=eval_dict,
                                                            sf_para_dict=sf_para_dict, model_para_dict=model_para_dict)
                        if curr_cv_avg_scores[k_index] > max_cv_avg_scores[k_index]:
                            max_cv_avg_scores, max_sf_para_dict, max_eval_dict, max_model_para_dict = \
                                                           curr_cv_avg_scores, sf_para_dict, eval_dict, model_para_dict

        # log max setting
        self.log_max(data_dict=data_dict, eval_dict=max_eval_dict,
                     max_cv_avg_scores=max_cv_avg_scores, sf_para_dict=max_sf_para_dict,
                     log_para_str=self.model_parameter.to_para_string(log=True, given_para_dict=max_model_para_dict))


    def run(self, debug=False, model_id=None, config_with_json=None, dir_json=None,
            data_id=None, dir_data=None, dir_output=None, grid_search=False):
        if config_with_json:
            assert dir_json is not None
            self.grid_run(debug=debug, model_id=model_id, dir_json=dir_json)
        else:
            if grid_search:
                self.grid_run(debug=debug, model_id=model_id, data_id=data_id, dir_data=dir_data, dir_output=dir_output)
            else:
                self.point_run(debug=debug, model_id=model_id, data_id=data_id, dir_data=dir_data, dir_output=dir_output)
