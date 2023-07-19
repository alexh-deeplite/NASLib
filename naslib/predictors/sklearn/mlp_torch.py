import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms

from typing import Dict, List, Union
import numpy as np
from naslib.predictors.sklearn import ZCOnlyPredictor
from sklearn.neural_network import MLPRegressor



class TorchMLPRegression(ZCOnlyPredictor):
    @property
    def default_hyperparams(self):
        return {'hidden_layer_sizes': (20,)}
    
    # def build_model(self):
    #     modules = []
    #     layer_sizes = self.hyperparams['hidden_layer_sizes']
    #     return MLP(layer_sizes)
    def get_x(self, encodings):
        x = np.array(encodings)
        xmax = np.max(x, 0)
        xmin = np.min(x, 0)
        return torch.tensor((x - xmin) / (xmax - xmin), dtype=torch.float)
    
    def get_y(self, labels):
        y = np.array(labels)
        return torch.tensor((y - self.mean) / self.std, dtype=torch.float)

    def train(self, x, y, **kwargs):
        # breakpoint()
        # init model
        # model = MLPRegressor(**self.hyperparams)
        model = MLP(self.hyperparams['hidden_layer_sizes'])
        loss_func = torch.nn.MSELoss()
        # xt = torch.tensor(x, dtype=torch.float)
        # normalized_x = (normalized_x - normalized_x.min(dim=0)) / 
        # xmax, xmin = torch.max(xt, dim=0)[0], torch.min(xt, dim=0)[0]
        # breakpoint()  # xmin is -1e8 for zc 0, 5, 7, 12. Should that be clipped?
        # xt = (xt - xmin) / (xmax - xmin)

        # yt = torch.tensor(y, dtype=torch.float)
        # ymax, ymin = torch.max(yt, dim=0), torch.min(yt, dim=0)
        # yt = (yt - xmin) / (xmax - xmin)
        # normalized_x = (normalized_x - normalized_x.min(dim=0)) / 
        # tfy = transforms.Normalize(mean=torch.mean(yt, dim=0),
        #                      std=torch.std(yt, dim=0))
        # yt = tfy(yt)

        dataset = TensorDataset(x, y)
        train_loader = DataLoader(dataset, batch_size=32)
        optimizer = torch.optim.SGD([m for m in model.parameters() if m.requires_grad], lr=.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=4)
        epochs = 50
        # train model
        # model.fit(x, y)
        # some of these 
        model.train()
        model.to('cuda')
        for epoch in range(epochs):
            total_loss = 0
            for i, batch in enumerate(train_loader):
                x_batch, y_batch = batch[0].cuda(), batch[1].cuda().unsqueeze(1)/100
                # breakpoint()
                optimizer.zero_grad()
                pred = model(x_batch)
                loss = loss_func(y_batch, pred)
                loss.backward()
                optimizer.step()
                # print(loss)
                total_loss += loss.item()
            scheduler.step(total_loss)
            print(total_loss)
            
        return model

    def predict(self, data, **kwargs):
        self.model.eval()
        self.model.cpu()
        return self.model(self.get_x(data))
        # return self.model.predict(self.get_x(data), **kwargs)

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
        train_pred = np.squeeze(self.predict(x).detach().numpy())
        train_error = np.mean(abs(train_pred-y.numpy()))

        return train_error

    def query(self, xtest, info=None):
        zc_scores = [self.create_zc_feature_vector(data['zero_cost_scores']) for data in info]
        xtest = zc_scores
        # xtest = self.get(xtest)

        x = self.get_x(xtest)
        return np.squeeze(self.model(x).detach().numpy()) * self.std + self.mean

    # def get_random_hyperparams(self):
    #     pass


    def set_hyperparams(self, params):
        self.hyperparams = params

    def get_fscore(self):
        out = dict([(name, weight) for name, weight in zip(self.zc_names, np.abs(self.model.layers[0].weight.detach().cpu().numpy()).sum(axis=1).astype(np.float64))])
        return out


# class ZCDataset(Dataset):
#     #Constructor for initially loading
#     def __init__(self,x,y):
#         # Store the inputs and outputs
#         self.X = x
#         self.y = y #Assuming your outcome variable is in the first column
        
#     # Get the number of rows in the dataset
#     def __len__(self):
#         return len(self.X)
#     # Get a row at an index
#     def __getitem__(self,idx):
#         return [self.X[idx], self.y[idx]]
#     # Create custom class method - instead of dunder methods
#     def split_data(self, split_ratio=0.2):
#         test_size = round(split_ratio * len(self.X))
#         train_size = len(self.X) - test_size
#         return random_split(self, [train_size, test_size])


class MLP(torch.nn.Module):
    def __init__(self, layer_sizes):
        super().__init__()
        modules = []
        modules.append(torch.nn.Linear(13, layer_sizes[0]))
        modules.append(torch.nn.ReLU())
        if len(layer_sizes) > 1:
            for i,layer_size in enumerate(layer_sizes[1:]):
                modules.append(torch.nn.Linear(layer_sizes[i - 1], layer_size))
            modules.append(torch.nn.ReLU())
        modules.append(torch.nn.Linear(layer_sizes[-1], 1))
        self.layers = torch.nn.ModuleList(modules)
    
    def forward(self, x):
        for m in self.layers:
            x = m(x)
        return x