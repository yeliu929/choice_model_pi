
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import datetime
import time
from sklearn.linear_model import LogisticRegression
import pyblp
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
import pickle
import os


class SmallDeepSet(nn.Module):
    def __init__(self, x_d, pool="sum"):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_features=x_d, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=64),
        )
        self.share_enc = nn.Sequential(
            nn.Linear(in_features=x_d, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=64),
        )
        self.dec = nn.Sequential(
            nn.Linear(in_features=64, out_features=300),
            nn.ReLU(),
            nn.Linear(in_features=300, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=64),                                                                                    
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=1),
            #nn.Sigmoid()
        )
        self.pool = pool

    def forward(self, shares, x):
        x = self.enc(x)
        shares = self.share_enc(shares)
        x = x.sum(dim=1) + shares.sum(dim=1)
        x = self.dec(x)
        return x.squeeze()    



class SingleNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size = 128, num_hidden_layers = 3):
        super(SingleNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.output_size = output_size
        
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(num_hidden_layers)
        ])
        self.output_layer = nn.Sequential(nn.Linear(hidden_size, output_size),
                                          nn.Sigmoid())
        
    def forward(self, x):
        x = self.input_layer(x)
        x = nn.functional.relu(x)
        for i in range(self.num_hidden_layers):
            x = self.hidden_layers[i](x)
            x = nn.functional.relu(x)
        x = self.output_layer(x)
        return x



### input: a large vector x_d * J, each market has one data point
class SingleNN_untuned(nn.Module):
    def __init__(self, x_d, J, pool="sum"):
        super().__init__()
        self.nn = nn.Sequential(
            nn.Linear(in_features=x_d, out_features=300),
            nn.ReLU(),
            nn.Linear(in_features=300, out_features=300),
            nn.ReLU(),
            nn.Linear(in_features=300, out_features=300),
            nn.ReLU(),
            nn.Linear(in_features=300, out_features=J),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.nn(x)
        ##x = self.softmax()
        return x
    


