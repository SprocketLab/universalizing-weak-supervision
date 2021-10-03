import os
import pickle
import numpy as np
from ptranking.ltr_adhoc.eval.ltr import LTREvaluator

RESULT_FILE_NAME = 'result_summary.pkl'

class PtrankingWrapper:
    def __init__(self, data_conf, weak_sup_conf, l2r_training_conf, result_path,
                 wl_kt_distance=None, individual_kt=None):
        self.data_conf = data_conf
        self.weak_sup_conf = weak_sup_conf
        self.l2r_training_conf =l2r_training_conf
        self.result_path = os.path.join(self.data_conf['project_root'], result_path)
        self.debug = l2r_training_conf['debug']
        self.ltr_evaluator = LTREvaluator()
        self.wl_kt_distance = wl_kt_distance  # save the kt distance between weak labels & true labels
        self.individual_kt = individual_kt

        ''' using the default setting for loading dataset & using the default setting for evaluation '''
        ''' mainly parameters for ptranking package'''
        self.ltr_evaluator.set_eval_setting(debug=self.debug, dir_output=result_path)
        self.ltr_evaluator.set_data_setting(debug=self.debug, data_id=data_conf['dataset_name'],
                                            dir_data=data_conf['processed_data_path'])
        self.data_dict = self.ltr_evaluator.get_default_data_setting()
        self.eval_dict = self.ltr_evaluator.get_default_eval_setting()
        self.ltr_evaluator.set_scoring_function_setting(debug=self.debug, data_dict=self.data_dict)
        self.ltr_evaluator.set_model_setting(debug=self.debug, model_id=l2r_training_conf['model'])  # data_dict argument is required
        self.model_para_dict = self.ltr_evaluator.get_default_model_setting()
        self.sf_para_dict = self.ltr_evaluator.get_default_scoring_function_setting()
        # model parameters setup in sf_para_dict
        """
        Default
        {'id': 'ffnns',
         'ffnns': {'num_layers': 5, 'HD_AF': 'R', 'HN_AF': 'R', 'TL_AF': 'S', 'apply_tl_af': True, 'BN': True,
                   'RD': False, 'FBN': True, 'num_features': 10}}
        ```
        """
        self.sf_para_dict['ffnns']['num_layers'] = 3
        self.sf_para_dict['ffnns']['h_dim'] = 30
        self.sf_para_dict['ffnns']['BN'] = True
        self.sf_para_dict['ffnns']['FBN'] = True
        self.sf_para_dict['ffnns']['apply_tl_af'] = False

        self.data_dict['num_features'] = len(data_conf['features']) - 1 # -1: the label feature
        self.data_dict['train_batch_size'] = l2r_training_conf['train_batch_size']
        self.data_dict['test_batch_size'] = l2r_training_conf['test_batch_size']
        self.eval_dict['epochs'] = l2r_training_conf['epochs']
        self.ltr_evaluator.setup_eval(data_dict=self.data_dict, eval_dict=self.eval_dict,
                                      sf_para_dict=self.sf_para_dict, model_para_dict=self.model_para_dict)

    def set_data(self, X_train, X_test, Y_train, Y_test, qid_train=None, qid_test=None):
        """

        Parameters
        ----------
        X_train
        X_test
        Y_train
        Y_test

        Returns
        -------
        """
        print('Training data shape, X_train.shape', X_train.shape, 'Y_train.shape', Y_train.shape)
        train_data, test_data, _ = self.ltr_evaluator.set_and_load_data(X_train=X_train, X_test=X_test,
                                                                        Y_train=Y_train, Y_test=Y_test,
                                                                        qid_train=qid_train,
                                                                        qid_test=qid_test,
                                                                        eval_dict=self.eval_dict,
                                                                        data_dict=self.data_dict,
                                                                        root_path=self.data_conf['project_root'])
        self.train_data = train_data
        self.test_data = test_data


    def get_model(self):
        """
        Get model based on parameter setup
        Returns
        -------

        """
        model = self.ltr_evaluator.load_ranker(sf_para_dict=self.sf_para_dict, model_para_dict=self.model_para_dict,
                                               opt=self.l2r_training_conf['optimizer'],
                                               lr=self.l2r_training_conf['learning_rate'],
                                               weight_decay=self.l2r_training_conf['weight_decay'])
        return model

    def train_model(self, model, IR=False, verbose=0, model_save=False):
        """
        train model based on train_data, test_data
        Parameters
        ----------
        model

        Returns
        -------

        """
        if IR:
            ranker, result_summary = self.ltr_evaluator.custom_train_ir(ranker=model, eval_dict=self.eval_dict,
                                                                     train_data=self.train_data,
                                                                     test_data=self.test_data, verbose=verbose)
        else:
            ranker, result_summary = self.ltr_evaluator.custom_train(ranker=model, eval_dict=self.eval_dict,
        train_data=self.train_data, test_data=self.test_data, verbose=verbose)
        if model_save:
            model_save_path = os.path.join(self.data_conf['project_root'], self.l2r_training_conf['model_checkpoint'], 'model.pkl')
            ranker.save(dir=os.path.join(self.data_conf['project_root'], self.l2r_training_conf['model_checkpoint']),
                        name='model.pkl')
            print('model saved in', model_save_path)

        # save result
        self.save_result(result_summary)

        return result_summary

    def eval(self, pred, verbose=0):
        """Evaluation with pred - it's for evaluation of label model itself

        Args:
            pred ([type]): Prediction, generated by weak labels
            verbose (int, optional): [description]. Defaults to 0.

        Returns:
            [type]: [description]
        """
        result_summary = self.ltr_evaluator.eval_with_pred(pred=pred, eval_dict=self.eval_dict,
        train_data=self.train_data, test_data=self.test_data, verbose=verbose)

        # save result
        self.save_result(result_summary)

        return result_summary

    def load_model_checkpoint(self, model, save_path):
        """
        load model checkpoint
        Parameters
        ----------
        model
        save_path

        Returns
        -------

        """
        model = model.load(save_path)
        print("Model loaded from", save_path)
        return model



    def save_result(self, result_summary):
        """

        Parameters
        ----------
        result_summary

        Returns
        -------

        """
        # append configurations to result_summary
        result_summary['data_conf'] = self.data_conf
        result_summary['weak_sup_conf'] = self.weak_sup_conf
        result_summary['l2r_training_conf'] = self.l2r_training_conf
        result_summary['wl_kt_distance'] = self.wl_kt_distance
        result_summary['individual_kt'] = self.individual_kt
        save_path = self.result_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            print("result data path", save_path, "generated")

        with open(os.path.join(save_path, RESULT_FILE_NAME), 'wb') as fp:
            pickle.dump(result_summary, fp)
            print('The experiment result is saved in', os.path.join(save_path, 'result_summary.pkl'))
