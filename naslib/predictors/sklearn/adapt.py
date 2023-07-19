# from adapt.utils import make_regression_da
from adapt.instance_based import TrAdaBoostR2
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor

import numpy as np
from naslib.predictors.sklearn import ZCOnlyPredictor, register_model
# from sklearn.linear_model import BayesianRidge
# model = TrAdaBoostR2(Ridge(), n_estimators=10, Xt=Xt[:10], yt=yt[:10], random_state=0)

# class AdaptBase(ZCOnlyPredictor)

class AdaptRegression(ZCOnlyPredictor):
    def __init__(self, encoding_type='adjacency_one_hot', ss_type='nasbench201', zc=False, zc_only=False, hpo_wrapper=False, pretrained_model=None):
        super().__init__(encoding_type=encoding_type, ss_type=ss_type, zc=zc, zc_only=zc_only, hpo_wrapper=hpo_wrapper)
        self.pretrained_model = pretrained_model
        # if isinstance(pretrained_model, str):

        # elif pretrained_model:
        #     self.pretrained_model = pretrained_model
        # else:
        #     raise ValueError("no pretrained model")

    @property
    def default_hyperparams(self):
        return {'n_estimators': 100}

    def train(self, x, y, **kwargs):
        # init model
        model = TrAdaBoostR2(self.pretrained_model, x, y, random_state=0)
        # train model
        model.fit(x, y)
        return model

    # def predict(self, data, **kwargs):
    #     return self.model.predict(self.get_x(data), **kwargs)

    def fit(self, xtrain, ytrain, train_info=None, params=None, **kwargs):

        # normalize accuracies
        self.mean = np.mean(ytrain)
        self.std = np.std(ytrain)

        # xtrain = self.zc_features


        if self.zc:
            self.zc_to_features_map = self._get_zc_to_feature_mapping(self.zc_names, xtrain)

        # convert to the right representation
        x, y = self.get_x(xtrain), self.get_y(ytrain)

        # fit to the training data
        self.model = self.train(x, y)

        # predict
        train_pred = np.squeeze(self.predict(x))
        train_error = np.mean(abs(train_pred-y))

        return train_error

    def set_hyperparams(self, params):
        self.hyperparams = params

    def get_fscore(self):
        return dict([(name, weight) for name, weight in zip(self.zc_names, self.model.predict_weights(domain='target'))])


class AdaptRegression2(ZCOnlyPredictor):
    def __init__(self, encoding_type='adjacency_one_hot', ss_type='nasbench201', zc=False, zc_only=False, hpo_wrapper=False, pretrained_model=None):
        super().__init__(encoding_type=encoding_type, ss_type=ss_type, zc=zc, zc_only=zc_only, hpo_wrapper=hpo_wrapper)
        self.pretrained_model = ExtraTreesRegressor()

    @property
    def default_hyperparams(self):
        return {'n_estimators': 10}

    def train(self, x, y, xt, yt, **kwargs):
        # init model
        # model = TrAdaBoostR2(self.pretrained_model, random_state=0)
        # train model
        # model.fit(x, y)
        # print(model.score(xt,yt))
        model = TrAdaBoostR2(self.pretrained_model, random_state=0)
        # train model
        model.fit(x, y, xt, yt)
        return model

    # def predict(self, data, **kwargs):
    #     return self.model.predict(self.get_x(data), **kwargs)

    def fit(self, xtrain, ytrain, xtarg, ytarg, train_info, targ_info, params=None, **kwargs):

        # normalize accuracies
        self.mean = np.mean(ytarg)
        self.std = np.std(ytarg)
        # self.targ_mean = np.mean(ytarg)
        # self.targ_std = np.std(ytarg)

        self.set_pre_computations(xtrain_zc_info=train_info)
        x_zc = self.zc_features
        self.set_pre_computations(xtrain_zc_info=targ_info)
        xtarg_zc = self.zc_features
        # xtarg = self.targ_zc_features

        
        self.zc_to_features_map = self._get_zc_to_feature_mapping(self.zc_names, x_zc)

        # convert to the right representation
        x, y = self.get_x(x_zc), self.get_y(ytrain)
        x2, y2 = self.get_x(xtarg_zc), self.get_y(ytarg)

        # fit to the training data
        self.model = self.train(x, y, x2, y2)

        # predict
        train_pred = np.squeeze(self.predict(x))
        train_error = np.mean(abs(train_pred-y))

        return train_error

    def set_hyperparams(self, params):
        self.hyperparams = params

    def get_fscore(self):
        return dict([(name, weight) for name, weight in zip(self.zc_names, self.model.predict_weights(domain='target'))])
    
class AdaptMLP()