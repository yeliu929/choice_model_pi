import numpy as np
import torch
import torch.nn as nn

def x_transform_mm_varying(data):
    X = data['X']
    M = data['M']
    K = data['K']

    # Find the maximum number of products across all markets
    max_products = data['J']
    J_list = data['J_list']

    # Pad the data and create masks
    x_1 = np.zeros((len(X), 1, K + 1))
    x_2 = np.zeros((len(X), max_products - 1, K + 1))
    mask = np.zeros((len(X), max_products - 1))

    for m in range(M):
        market_rows = X.iloc[J_list[m]: J_list[m+1],]
        num_products = len(market_rows)

        for j, row in enumerate(market_rows.iterrows()):
            i = row[0]
            x_1[i] = row[1].values.astype(np.float32)

            other_products = market_rows.drop(index=i)
            x_2[i, :num_products - 1] = other_products.values.astype(np.float32)
            mask[i, :num_products - 1] = 1

    x_1, x_2, mask = x_1.astype(np.float32), x_2.astype(np.float32), mask.astype(np.float32)

    return x_1, x_2, mask

class SmallDeepSetVarying(nn.Module):
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

    def forward(self, shares, x, mask):
        x = self.enc(x)
        shares = self.share_enc(shares)
        shares = shares * (mask.unsqueeze(-1))  # Apply the mask
        x = x.sum(dim=1) + shares.sum(dim=1)
        x = self.dec(x)
        return x.squeeze()

    
def train_deep_varying(data):
    K = data['K']
    x_1, x_2, mask = x_transform_mm_varying(data)
    y = np.log(data['Y'])
    #y = data['Y']
    model = SmallDeepSetVarying(x_d=K + 1)
    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss().cuda()
    losses = []
    x_1, x_2, mask, y = torch.from_numpy(x_1).float().cuda(), torch.from_numpy(x_2).float().cuda(), torch.from_numpy(mask).float().cuda(), torch.from_numpy(y).float().cuda()
    iteration = 0
    for _ in range(5000):
        loss = criterion(model(x_2, x_1, mask), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return model, losses

def pred_deep_varying(data, model):
    K = data['K']
    x_1, x_2, mask = x_transform_mm_varying(data)
    x_1, x_2, mask = torch.from_numpy(x_1).float().cuda(), torch.from_numpy(x_2).float().cuda(), torch.from_numpy(mask).float().cuda()
    y_pred = model(x_2, x_1, mask)
    
    return y_pred.cpu().detach().numpy() #y_pred.cpu().detach().numpy() #