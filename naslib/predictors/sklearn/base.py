from typing import Dict, List, Union
# import torch
import numpy as np
from naslib.predictors.predictor import Predictor


class ZCOnlyPredictor(Predictor):
    def __init__(self, encoding_type='adjacency_one_hot', ss_type='nasbench201', zc=False, zc_only=False, hpo_wrapper=False):
        super(Predictor, self).__init__()
        self.encoding_type = encoding_type
        self.ss_type = ss_type
        self.zc = zc
        self.zc_names = None
        self.zc_only = zc_only
        self.hyperparams = self.default_hyperparams
        self.hpo_wrapper = hpo_wrapper

    @property
    def default_hyperparams(self):
        return {}

    def get_x(self, encodings):
        return np.array(encodings)
    
    def get_y(self, labels):
        return np.array(labels)

    def train(self, train_data, **kwargs):
        return NotImplementedError('Train method not implemented')

    def predict(self, data, **kwargs):
        return self.model.predict(data, **kwargs)

    def fit(self, xtrain, ytrain, train_info=None, params=None, **kwargs):
        raise NotImplementedError()
    
    def query(self, xtest, info=None):
        zc_scores = [self.create_zc_feature_vector(data['zero_cost_scores']) for data in info]
        xtest = zc_scores
        # xtest = self.get(xtest)

        x = self.get_x(xtest)
        return np.squeeze(self.model.predict(x)) * self.std + self.mean

    # def get_random_hyperparams(self):
    #     pass

    def create_zc_feature_vector(self, zero_cost_scores: Union[List[Dict], Dict]) -> Union[List[List], List]:
        zc_features = []

        def _make_features(zc_scores):
            zc_values = []
            for zc_name in self.zc_names:
                zc_values.append(zc_scores[zc_name])

            zc_features.append(zc_values)

        if isinstance(zero_cost_scores, list):
            for zc_scores in zero_cost_scores:
                _make_features(zc_scores)
        elif isinstance(zero_cost_scores, dict):
            _make_features(zero_cost_scores)
            zc_features = zc_features[0]

        return zc_features

    def set_hyperparams(self, params):
        self.hyperparams = params

    def _get_zc_to_feature_mapping(self, zc_names, xtrain):
        x = xtrain[0] # Consider one datapoint

        n_zc = len(zc_names)
        assert n_zc == len(x)
        mapping = {}

        for zc_name in zc_names:
            mapping[zc_name] = zc_name

        return mapping

    def set_pre_computations(self, unlabeled=None, xtrain_zc_info=None, xtest_zc_info=None, unlabeled_zc_info=None):
        if xtrain_zc_info is not None:
            self.xtrain_zc_info = xtrain_zc_info
            self._verify_zc_info(xtrain_zc_info['zero_cost_scores'])
            self._set_zc_names(xtrain_zc_info['zero_cost_scores'])
            self.zc_features = self.create_zc_feature_vector(xtrain_zc_info['zero_cost_scores'])

    def _verify_zc_info(self, zero_cost_scores):
        zc_names = [set(zc_scores.keys()) for zc_scores in zero_cost_scores]
    
        assert len(zc_names) > 0, 'No ZC values found in zero_cost_scores'
        assert zc_names.count(zc_names[0]) == len(zc_names), 'All models do not have the same number of ZC values'

    def _set_zc_names(self, zero_cost_scores):
        zc_names = sorted(zero_cost_scores[0].keys())
        self.zc_names = zc_names
