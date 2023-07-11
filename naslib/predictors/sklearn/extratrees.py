import numpy as np
from naslib.predictors.sklearn import ZCOnlyPredictor
from sklearn.ensemble import ExtraTreesRegressor



class TreesRegression(ZCOnlyPredictor):
    @property
    def default_hyperparams(self):
        return {'n_estimators': 100}

    def train(self, x, y, **kwargs):
        # init model
        model = ExtraTreesRegressor(**self.hyperparams)
        # train model
        model.fit(x, y)
        return model

    # def predict(self, data, **kwargs):
    #     return self.model.predict(self.get_x(data), **kwargs)

    def fit(self, xtrain, ytrain, train_info=None, params=None, **kwargs):

        # normalize accuracies
        self.mean = np.mean(ytrain)
        self.std = np.std(ytrain)

        xtrain = self.zc_features


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
        return dict([(name, weight) for name, weight in zip(self.zc_names, self.model.feature_importances_)])


