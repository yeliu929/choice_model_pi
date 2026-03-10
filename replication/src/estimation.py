import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
from data_generation import *
from neural_networks import *


def x_transform_mm(data):
    ### Two inputs for the nn
    X = data['X']
    J = data['J']
    M = data['M']
    K = data['K']
    ## x_1: the focal product's features; shape should be (number of products * number of markets, 1, number of features)
    Total_J = J * M
    X_1 = np.ones((Total_J,1,K+1))
    for i in range(Total_J):
        X_1[i][:] = np.array(X.iloc[i,]).astype(np.float32)
        X_1 = X_1.astype(np.float32)  

    ## x_2: Other products' features (permutation invariant) within the same market; 
    ## shape should be (number of products * number of markets, number of products-1, number of features)
    X_2 = np.ones((Total_J,J-1,K+1))
    for m in range(M):
        for j in range(J):
            i = m * J + j
            ### select other products in the same market 
            X_2[i][:] = np.array(X.loc[(X.index != i) & (X.index >= m * J) & (X.index < (m+1) * J),]).astype(np.float32)

    return X_1, X_2 


def train_deep(data):
    K = data['K']
    x_1, x_2 = x_transform_mm(data)
    # y = np.log(data['Y'])
    y = data['Y']
    model = SmallDeepSet(x_d = K+1)
    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    #criterion = nn.BCELoss().cuda()
    criterion = nn.MSELoss().cuda()
    losses = []
    x_1, x_2, y = torch.from_numpy(x_1).float().cuda(), torch.from_numpy(x_2).float().cuda(), torch.from_numpy(y).float().cuda()
    iteration=0
    for _ in range(5000):
        loss = criterion(model(x_2, x_1), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return model, losses


def x_transform_single(data):
    X = data['X']
    J = data['J']
    M = data['M']
    K = data['K']
    Y = data['Y']
    
    X_1 = np.ones((M, J * (K+1)))
    for i in range(M):
        X_1[i] = np.array(X.iloc[(i * J): ( (i+1) * J),]).flatten().astype(np.float32)
    X_1 = X_1.astype(np.float32)  

    y = np.ones((M, J))
    for i in range(M):
        y[i] = np.array(Y[(i * J): ( (i+1) * J)]).astype(np.float32)
    y = y.astype(np.float32)  
    
    return X_1, y

    
def train_single(data, hyper_params):
    K = data['K']
    J = data['J']
    x_1, y = x_transform_single(data)
    
    model = SingleNN((K+1) * J, J, hyper_params['hidden_size'], hyper_params['num_hidden_layers'])
    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=hyper_params['learning_rate'])
    criterion = nn.BCELoss().cuda()
    losses = []
    x_1, y = torch.from_numpy(x_1).float().cuda(), torch.from_numpy(y).float().cuda()
    iteration=0
    for _ in range(hyper_params['n_epochs']):
        loss = criterion(model(x_1), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return model, losses

def cross_val_accuracy(data, hyper_params):
    x_1, y = x_transform_single(data)
    x_1, y = torch.from_numpy(x_1).float().cuda(), torch.from_numpy(y).float().cuda()
    
    input_size = (data['K']+1) * data['J']
    output_size = data['J']
        
    model = SingleNN(input_size, output_size, hyper_params['hidden_size'], hyper_params['num_hidden_layers'])
    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=hyper_params['learning_rate'])
    criterion = nn.BCELoss().cuda()
    
    kf = KFold(n_splits=5)
    val_loss = []
    for train_indices, val_indices in kf.split(x_1):
        #print(1)
        losses = []
        # get the training and testing data for this fold
        train_x_1 = x_1[train_indices]
        val_x_1 = x_1[val_indices]
        train_y = y[train_indices]
        val_y = y[val_indices]
        
        for epoch in range(hyper_params['n_epochs']):
            loss = criterion(model(train_x_1), train_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
    
        with torch.no_grad():
        # evaluate the model on the validation set
            val_loss.append(criterion(model(val_x_1), val_y))
            # print(val_loss)

    # return the average cross-validation accuracy for these hyperparameters
    return sum(val_loss).cpu().detach().numpy() / len(val_loss)

def get_best_hyper(data, hyper_params_all):
    
    rslt_list = []
    
    for i in range(3*3*3*3):
        print(i, datetime.datetime.now())
        hyper_params_one = {'hidden_size': hyper_params_all['hidden_size'][i//27],
                    'num_hidden_layers': hyper_params_all['num_hidden_layers'][(i//9)%3], 
                    'n_epochs': hyper_params_all['n_epochs'][(i//3)%3],
                    'learning_rate': hyper_params_all['learning_rate'][(i)%3]
        }
        rslt_list.append(cross_val_accuracy(data, hyper_params_one))
    
    best_i = np.argmin(rslt_list)
    hyper_params_best = {'hidden_size': hyper_params_all['hidden_size'][best_i//27],
                    'num_hidden_layers': hyper_params_all['num_hidden_layers'][(best_i//9)%3], 
                    'n_epochs': hyper_params_all['n_epochs'][(best_i//3)%3],
                    'learning_rate': hyper_params_all['learning_rate'][(best_i)%3]
        }
    
    return hyper_params_best, rslt_list




def pred_deep(data, model):
    K = data['K']
    x_1, x_2 = x_transform_mm(data)
    x_1, x_2 = torch.from_numpy(x_1).float().cuda(), torch.from_numpy(x_2).float().cuda()
    y_pred = model(x_2, x_1)
    
    return y_pred.cpu().detach().numpy() #np.exp(y_pred.cpu().detach().numpy())


def pred_single(data, model):
    K = data['K']
    x_1, y = x_transform_single(data)
    x_1 = torch.from_numpy(x_1).float().cuda()
    y_pred = model( x_1)
    
    return y_pred.cpu().detach().numpy().flatten()


def pred_random(data):
    J = data['J']
    M = data['M']
    mae_list = []
    for i in range(1,1000):
        j = i * 0.001
        y_pred = np.ones(J*M) * j
        mae = np.mean(np.abs(y_pred - data['Y']))
        mae_list.append(mae)
    y_pred = (np.argmin(mae_list)+1)/1000 * np.ones(J*M)
    return y_pred

def get_errors(y_pred, y_test):
    mae = np.mean(np.abs(y_pred - y_test))
    rmse = np.std(y_pred - y_test)

    return [mae, rmse] 


def get_errors_2(y_pred, y_test):
    mse = np.mean((y_pred - y_test)**2)
    mae = np.mean(np.abs(y_pred - y_test))
    rmse = np.std(y_pred - y_test)
    mape =  np.median(np.abs((y_pred- y_test) / (y_test  + 1e-10)))
    return [ mae, rmse, mape] 


def cal_true_elasticity(data, from_x_to_y, prod_id, delta, seed):
    
    X = data['X'].copy()
    b = data['params']
    J = data['J']
    K = data['K']
    M = data['M']
    
    record = pd.DataFrame({'old_price' : X['price'].copy()})
    
    for m in range(M):
        new_id = prod_id + m * J
        X['price'].iloc[new_id] = (1 + delta) * X['price'].iloc[new_id]

    
    ## get the true elasticity 
    if from_x_to_y == mnl or from_x_to_y == mnl_choice:
        y_true_new = from_x_to_y(X, b, J, K, M, seed)['Y']
    elif from_x_to_y == mnl_choice_fe:
        y_true_new = from_x_to_y(X, b, J, K-J, M, seed)['Y']
    elif from_x_to_y == rcl or from_x_to_y == rcl_fe or from_x_to_y == rcl_mix:
        y_true_new = rcl_regenerate(X, data)
    elif from_x_to_y == rcl_in3 or from_x_to_y == rcl_in3_fe:
        y_true_new = rcl_regenerate_in3(X, data)
    elif from_x_to_y == rcl_log:
        y_true_new = rcl_regenerate_log(X, data)
    elif from_x_to_y == rcl_log_fe:
        y_true_new = rcl_regenerate_log_fe(X, data)
    elif from_x_to_y == rcl_sin:
        y_true_new = rcl_regenerate_sin(X, data)
    elif from_x_to_y == rcl_sin_fe:
        y_true_new = rcl_regenerate_sin_fe(X, data)
    else:
        print('not defined regeneration')
    
        
    record['new_true_share'] = y_true_new
    record['old_true_share'] = data['Y']
    
    record['true_elasticity'] = (record['new_true_share'] - record['old_true_share']) / (delta * record['old_true_share'])
    
    elasticity_id = 'true_elasticity' + str(prod_id)
    data[elasticity_id] = record['true_elasticity'].copy()
    
    return data



def cal_true_share_change(data, from_x_to_y, prod_id, delta, seed):
    
    X = data['X'].copy()
    b = data['params']
    J = data['J']
    K = data['K']
    M = data['M']
    
    record = pd.DataFrame({'old_price' : X['price'].copy()})
    
    for m in range(M):
        new_id = prod_id + m * J
        X['price'].iloc[new_id] = (1 + delta) * X['price'].iloc[new_id]

    
    ## get the true elasticity 
    if from_x_to_y == mnl or from_x_to_y == mnl_choice:
        y_true_new = from_x_to_y(X, b, J, K, M, seed)['Y']
    elif from_x_to_y == mnl_choice_fe:
        y_true_new = from_x_to_y(X, b, J, K-J, M, seed)['Y']
    elif from_x_to_y == rcl or from_x_to_y == rcl_fe or from_x_to_y == rcl_mix:
        y_true_new = rcl_regenerate(X, data)
    elif from_x_to_y == rcl_in3 or from_x_to_y == rcl_in3_fe:
        y_true_new = rcl_regenerate_in3(X, data)
    elif from_x_to_y == rcl_log:
        y_true_new = rcl_regenerate_log(X, data)
    elif from_x_to_y == rcl_log_fe:
        y_true_new = rcl_regenerate_log_fe(X, data)
    elif from_x_to_y == rcl_sin:
        y_true_new = rcl_regenerate_sin(X, data)
    elif from_x_to_y == rcl_sin_fe:
        y_true_new = rcl_regenerate_sin_fe(X, data)
    else:
        print('not defined regeneration')
        
    
    #print(y_true_new)
    
    record['new_true_share'] = y_true_new
    record['old_true_share'] = data['Y']
    
    record['true_change'] = (record['new_true_share'] - record['old_true_share'])    
    
    return record

def cal_elasticity(data, pred_method, model, prod_id, delta):
    J = data['J']
    K = data['K']
    M = data['M']
    X = data['X'].copy()
    ## save the record in a new dataframe
    record = pd.DataFrame({'old_price' : X['price'].copy()})

    ## get the prediction 
    y_pred_old = pred_method(data, model)

    ## update price of id in all markets
    record['market_id'] = data['market_id']
    record['type'] = 'cross'

    for m in range(M):
        new_id = prod_id + m * J
        X['price'].iloc[new_id] = (1 + delta) * X['price'].iloc[new_id]
        record['type'].iloc[new_id] = "self"

    record['new_price'] = X['price'].copy()
    
    ## predict based on new price
    datax = {'X': X, 'Y': data['Y'], 'J': J, "K": K, "M": M, "params": data['params'], 'market_id': data['market_id']}
    y_new = pred_method(datax, model)
    
    record['old_pred_share'] = y_pred_old
    record['new_pred_share'] = y_new

    record['pred_elasticity'] = (record['new_pred_share'] - record['old_pred_share']) / (delta * record['old_pred_share'])
    
    elasticity_id = 'true_elasticity' + str(prod_id)
    record['true_elasticity'] = data[elasticity_id]
    
    record['Y'] = data['Y'].copy()
    
    record_cross = record.loc[(record.type=='cross') &(np.isfinite(record['true_elasticity']) == True) ].copy()

    
    record_self = record.loc[(record.type=='self') &(np.isfinite(record['true_elasticity']) == True) ].copy()

    mae_c = np.mean(np.abs(record_cross['true_elasticity'] - record_cross['pred_elasticity']))
    max_error_c = np.max(np.abs(record_cross['true_elasticity'] - record_cross['pred_elasticity']))
    mape_c = np.mean(np.abs((record_cross['true_elasticity'] - record_cross['pred_elasticity']) / record_cross['true_elasticity'] ))
    
    mae_s = np.mean(np.abs(record_self['true_elasticity'] - record_self['pred_elasticity']))
    max_error_s = np.max(np.abs(record_self['true_elasticity'] - record_self['pred_elasticity']))
    mape_s = np.mean(np.abs((record_self['true_elasticity'] - record_self['pred_elasticity']) / record_self['true_elasticity']))
    
    return mae_c, max_error_c, mape_c, mae_s, max_error_s, mape_s


def elasticity_mae(model, pred_method, data, delta):
    rslt_list = []
    
    J = data['J']
    for prod_id in range(J):
        mae_c, max_error_c, mape_c, mae_s, max_error_s, mape_s  = cal_elasticity(data, pred_method, model, prod_id, delta)  
        rslt_list.append([mae_c,  mae_s])#max_error_c, mape_c,, max_error_s, mape_s
    
    rslt_df = pd.DataFrame(rslt_list, columns=['mae_c', 'mae_s'])#'max_error_c', 'mape_c', , 'max_error_s', 'mape_s'

    return rslt_df.mean().to_list()


def logit_torch(X, J, K, M, params):
    Total_J = J * M
    b = params[0]
    N = 1

    ## generate random coeffcient for each individual
    b_random = []
    
    ## use the same seed for each iteration
    #torch.manual_seed(999999)
    
    for i in range(len(b)):
        b_random.append((torch.ones((M, N)).to('cuda') *  b[i]).repeat_interleave(J, dim=0))

    u_i = b_random[0] * torch.ones((Total_J, N)).to('cuda')

    ## get the utility of each user, the output is (M * J) * N
    for k in range(1,K+2):
        u_i = u_i + b_random[k] * (X[:,k-1].view(Total_J,1))

    exp_u_i = torch.exp(u_i)

    Y = torch.zeros((M*J,)).to('cuda')
    u_m = exp_u_i.view(M, J, N)

    # Calculate the probability of purchasing for each individual
    sum_u_m = torch.sum(u_m, dim=1, keepdim=True)
    ccp_m = (u_m / (sum_u_m + 1)).transpose(1, 2)

    # Aggregate individual choice by market
    share_m = torch.mean(ccp_m, dim=1)

    # Flatten share_m from (M, J) to (M*J,)
    Y = share_m.flatten()

    return Y

    

def rcl_torch(X,J,K,M, params):
    Total_J = J * M
    b = params[0]
    sigma = params[1]
    N = 10000

    ## generate random coeffcient for each individual
    b_random = []
    
    ## use the same seed for each iteration
    torch.manual_seed(999999)
    
    for i in range(len(b)):
        b_random.append((torch.randn((M, N)).to('cuda') * sigma[i] + b[i]).repeat_interleave(J, dim=0))

    u_i = b_random[0] * torch.ones((Total_J, N)).to('cuda')

    ## get the utility of each user, the output is (M * J) * N
    for k in range(1,K+2):
        u_i = u_i + b_random[k] * (X[:,k-1].view(Total_J,1))

    exp_u_i = torch.exp(u_i)

    Y = torch.zeros((M*J,)).to('cuda')
    
    u_m = exp_u_i.view(M, J, N)

    # Calculate the probability of purchasing for each individual
    sum_u_m = torch.sum(u_m, dim=1, keepdim=True)
    ccp_m = (u_m / (sum_u_m + 1)).transpose(1, 2)

    # Aggregate individual choice by market
    share_m = torch.mean(ccp_m, dim=1)

    # Flatten share_m from (M, J) to (M*J,)
    Y = share_m.flatten()

    return Y


def train_rcl(data):
    J = data['J']
    M = data['M']
    K = data['K']
    y = data['Y']
    X = data['X']
    N = 10000

    params = nn.ParameterList([nn.Parameter(torch.randn(K+2, requires_grad=True)), 
                               nn.Parameter(torch.ones(K+2, requires_grad=True))]).cuda()
    
    optimizer = torch.optim.SGD(list(params), lr=10)
    criterion = nn.MSELoss().cuda()
    losses = []
    X, y = torch.from_numpy(X.to_numpy()).float().cuda(), torch.from_numpy(y).float().cuda()

    for _ in range(2000):
        loss = criterion(rcl_torch(X,J,K,M,params), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
    return list(params), losses


def train_logit(data):
    J = data['J']
    M = data['M']
    K = data['K']
    y = data['Y']
    X = data['X']
    N = 1

    params = nn.ParameterList([nn.Parameter(torch.randn(K+2, requires_grad=True))]).cuda()

    optimizer = torch.optim.SGD(list(params), lr=2)
    criterion = nn.MSELoss().cuda()
    losses = []
    X, y = torch.from_numpy(X.to_numpy()).float().cuda(), torch.from_numpy(y).float().cuda()

    for _ in range(2000):
        loss = criterion(logit_torch(X,J,K,M,params), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        
    return list(params), losses



def pred_rcl(data,est_params):
    J = data['J']
    M = data['M']
    K = data['K']
    X = data['X']
    N = 10000
    Total_J = J * M
    
    X = torch.from_numpy(X.to_numpy()).float().cuda()
    b = est_params[0]
    sigma = est_params[1]
    
    torch.manual_seed(999999)

    ## generate random coeffcient for each individual
    b_random = []

    for i in range(len(b)):
        b_random.append((torch.randn((M, N)).to('cuda') * sigma[i] + b[i]).repeat_interleave(J, dim=0))

    u_i = b_random[0] * torch.ones((Total_J, N)).to('cuda')

    ## get the utility of each user, the output is (M * J) * N
    for k in range(1,K+2):
        u_i = u_i + b_random[k] * (X[:,k-1].view(Total_J,1))

    exp_u_i = torch.exp(u_i)

    Y = torch.zeros((M*J,)).to('cuda')
    for m in range(M):
        ## get the sub-matrix (J * N) of this market
        u_m = exp_u_i[(m*J):((m+1)*J), :]

        ## calculate the probability of purchasing for each individual
        sum_u_m = torch.sum(u_m, dim=0, keepdim=True)
        ccp_m = (u_m / (sum_u_m + 1)).T

        ## aggregate individual choice by market
        share_m = torch.mean(ccp_m, dim=0)
        Y[(m*J):((m+1)*J)] = share_m
    
    return Y.cpu().detach().numpy()


def pred_logit2(data,est_params):
    J = data['J']
    M = data['M']
    K = data['K']
    X = data['X']
    N = 1
    Total_J = J * M
    
    X = torch.from_numpy(X.to_numpy()).float().cuda()
    Y = logit_torch(X,J,K,M,est_params)
    
    return Y.cpu().detach().numpy()
    
    

def new_product_data(data, from_x_to_y, seed):
    ### 
    X = data['X'].copy()
    b = data['params']
    J = data['J']
    K = data['K']
    M = data['M']
    
    ## update the product list, add one new product for each market
    np.random.seed(seed)
    new_product = np.random.normal(loc=0, scale=1, size=(1,K+1))
    ## change the price to be positive
    new_product[0,-1] = np.abs(new_product[0,-1])
    
    ## add this product to all markets
    i = 0 
    for m in range(M): 
        # get the first slice of the DataFrame up to the insert position
        new_id = i + (m+1) * J 
        df1 = X.iloc[:new_id].copy()
        # get the second slice of the DataFrame from the insert position to the end
        df2 = X.iloc[new_id:].copy()
        # create a new DataFrame from the concatenation of the slices and the new row
        X = pd.concat([df1.copy(), pd.DataFrame(new_product, index=[new_id]).rename(columns={K:'price'}).copy(), df2.copy()]).reset_index(drop=True)
        #print(new_id)
        i = i + 1
    
    data2 = data.copy()
    data2['X'] = X 
    data2['J'] = J + 1
    data2['market_id'] = market_id_gen(data2['J'], data2['M'])
    
    ## get the true y 
    if from_x_to_y == mnl  or from_x_to_y == mnl_choice:
        y_true_new = from_x_to_y(X, b, J+1, K, M, seed)['Y']
    elif from_x_to_y == rcl:
        y_true_new = from_x_to_y(X, b, J+1, K, M, seed, N=10000)['Y']
    
    data2['Y'] = y_true_new
    
    return data2




def split_train_test(data, p = 0.8):
    M = data['M']
    J = data['J']
    train_size = int(M * J * p)
    
    
    data_train = {
        'X': data['X'].iloc[0: train_size],
        'Y': data['Y'][0: train_size],
        'M': int(M * p),
        'J': J, 
        'K': data['K'], 
        'params': data['params'], 
        'generation_seed': data['generation_seed'], 
        'market_id' : data['market_id'][0: train_size]}
        
    data_test = {
        'X': data['X'].iloc[train_size: ,].reset_index(drop = True),
        'Y': data['Y'][train_size: ],
        'M': M - int(M*p),
        'J': J, 
        'K': data['K'], 
        'params': data['params'], 
        'generation_seed': data['generation_seed'], 
        'market_id' : data['market_id'][train_size:]}
    
    #print(data_test['M'])
    if 'b_random' in data.keys():
        data_train['b_random'] = [arr[0:train_size,:] for arr in data['b_random']]
        data_test['b_random'] = [arr[train_size:,:] for arr in data['b_random']]
    
    return data_train, data_test




def new_prod_eval4(data, dg, model_list, seed):
    setup = [data['J'], data['K'], data['M'], str(dg).split()[1], seed]
    new_data = new_product_data(data, dg, 9)
    m1_deep = model_list[0]
    m1_rcl = model_list[1]
    m1_mnl = model_list[2]
    
    y_pred_deep1 = pred_deep(data, m1_deep)
    y_pred_deep2 = pred_deep(new_data, m1_deep)
    error_deep1 = get_errors_2(y_pred_deep1, data['Y'])
    error_deep2 = get_errors_2(y_pred_deep2, new_data['Y'])
    
    y_pred_rcl1 = pred_rcl(data, m1_rcl)
    y_pred_rcl2 = pred_rcl(new_data, m1_rcl)
    error_rcl1 = get_errors_2(y_pred_rcl1, data['Y'])
    error_rcl2 = get_errors_2(y_pred_rcl2, new_data['Y'])
    
    y_pred_mnl1 = pred_logit2(data, m1_mnl)
    y_pred_mnl2 = pred_logit2(new_data, m1_mnl)
    error_mnl1 = get_errors_2(y_pred_mnl1, data['Y'])
    error_mnl2 = get_errors_2(y_pred_mnl2, new_data['Y'])
    
    y_pred_random1 = np.ones(y_pred_deep1.shape) * np.mean(data['Y'])
    y_pred_random2 = np.ones(y_pred_deep2.shape) * np.mean(data['Y'])
    error_random1 = get_errors_2(y_pred_random1, data['Y'])
    error_random2 = get_errors_2(y_pred_random2, new_data['Y'])
    
    error_all = setup + error_deep1 + error_deep2 +  error_rcl1 + error_rcl2 + error_mnl1 + error_mnl2 + error_random1 + error_random2 
    
    error_df = pd.DataFrame(columns=[
        'J', 'K', 'M', 'dg', 'seed',
        "mae_deep", "rmse_deep", "mape_deep",
        "mae_deep2", "rmse_deep2", "mape_deep2",
        "mae_rcl", "rmse_rcl", "mape_rcl",
        "mae_rcl2", "rmse_rcl2", "mape_rcl2",
        "mae_mnl", "rmse_mnl", "mape_mnl",
        "mae_mnl2", "rmse_mnl2", "mape_mnl2",
        "mae_random", "rmse_random", "mape_random",
        "mae_random2", "rmse_random2", "mape_random2"])
    
    error_df = pd.concat([error_df, pd.DataFrame([error_all], columns= error_df.columns)],ignore_index=True)
    error_df.to_csv('new_product_prediction_error.csv')
                    
    return error_all



def cal_elasticity_record(data, pred_method, model, prod_id, delta):
    J = data['J']
    K = data['K']
    M = data['M']
    X = data['X'].copy()
    ## save the record in a new dataframe
    record = pd.DataFrame({'old_price' : X['price'].copy()})

    ## get the prediction 
    y_pred_old = pred_method(data, model)

    ## update price of id in all markets
    record['market_id'] = data['market_id']
    record['type'] = 'cross'
    record['price_i'] = 0

    for m in range(M):
        new_id = prod_id + m * J
        price_i = X['price'].iloc[new_id]
        record['price_i'].iloc[m*J:(m+1)*J,] = price_i 
        X['price'].iloc[new_id] = (1 + delta) * price_i
        record['type'].iloc[new_id] = "self"

    record['new_price'] = X['price'].copy()
    
    ## predict based on new price
    datax = {'X': X, 'Y': data['Y'], 'J': J, "K": K, "M": M, "params": data['params'], 'market_id': data['market_id']}
    y_new = pred_method(datax, model)
    
    record['old_pred_share'] = y_pred_old
    record['new_pred_share'] = y_new

    record['pred_elasticity'] = (record['new_pred_share'] - record['old_pred_share']) / (delta * record['old_pred_share'])
    
    elasticity_id = 'true_elasticity' + str(prod_id)
    record['true_elasticity'] = data[elasticity_id]
    
    record['Y'] = data['Y'].copy()

    return record[['market_id','old_price','new_price','type',
                  'true_elasticity','pred_elasticity','price_i']]


def elas_record(data, model_list, dg, seed, delta):
    J = data['J']
    data_train, data_test = split_train_test(data)
    large_record = pd.DataFrame(columns = ['market_id','i','j','old_price','new_price','type',
                                           'true_elasticity','pred_elasticity_deep','pred_elasticity_rcl','pred_elasticity_mnl','price_i'])
    for prod_id in range(J):
        data_train = cal_true_elasticity(data_train, dg, prod_id, delta, seed)
        record = cal_elasticity_record(data_train, pred_deep, model_list[0], prod_id, delta)
        record = record.rename(columns={'pred_elasticity': 'pred_elasticity_deep'})
        record['pred_elasticity_rcl'] = cal_elasticity_record(data_train, pred_rcl, model_list[1], prod_id, delta)['pred_elasticity']
        record['pred_elasticity_mnl'] = cal_elasticity_record(data_train, pred_logit2, model_list[2], prod_id, delta)['pred_elasticity']
        record['i'] = prod_id
        record['j'] = record.index % data_train['J']
        large_record = pd.concat([large_record, record]).reset_index(drop=True)

    large_record['mae_deep'] = np.abs(large_record['pred_elasticity_deep'] - large_record['true_elasticity'])
    large_record['mae_rcl'] = np.abs(large_record['pred_elasticity_rcl'] - large_record['true_elasticity'])
    large_record['mae_mnl'] = np.abs(large_record['pred_elasticity_mnl'] - large_record['true_elasticity'])
    
    return large_record



def full_one_iteration(J, M, K, seed, dg, params, hyper_params, delta):
    setup = [J, K, M, str(dg).split()[1], seed]
    ### step 1: data generation 
    datax = data_generation(params, J, K, M, seed, dg)
    data_file_name = 'data_'+ 'J' + str(J) + 'K' +  str(K) +'M' + str(M) + '_'+str(seed) + str(dg).split()[1] +".pickle"
    
    data_train, data_test = split_train_test(datax, p = 0.8)
    print('Data is generated.')
    ### step 2: train models 
    m1_deep, loss_deep = train_deep(data_train)
    print("Main model is trained.")
    m1_rcl, losses_rcl = train_rcl(data_train)
    print("RCL is estimated.")
    m1_mnl, losses_mnl = train_logit(data_train)
    print("MNL is estimated.")
    m1_single,losses_single = train_single(data_train, hyper_params)
    print("NP is estimated.")
    
    ### step 3: evaluation
    data = data_train.copy()

    y_pred_deep1 = pred_deep(data, m1_deep)
    error_deep1 = get_errors(y_pred_deep1, data['Y'])
    y_pred_rcl1 = pred_rcl(data, m1_rcl)
    error_rcl1 = get_errors(y_pred_rcl1, data['Y'])
    y_pred_logit1 = pred_logit2(data, m1_mnl)
    error_logit1 = get_errors(y_pred_logit1, data['Y'])
    y_pred_single1 = pred_single(data, m1_single)
    error_single1 = get_errors(y_pred_single1, data['Y'])
    
    train_pred_df = pd.DataFrame({'true':data['Y'], 'pred_deep':y_pred_deep1, 'pred_rcl': y_pred_rcl1, 'pred_mnl': y_pred_logit1, 'pred_single': y_pred_single1})
    train_pred_df['pred_random'] = np.mean(data_train['Y'])
    
    train_pred_df.to_csv("train"+ 'J' + str(J) + 'K' +  str(K) +'M' + str(M) + '_'+str(seed) + str(dg).split()[1] +".csv")
    
    data = data_test.copy()

    y_pred_deep1 = pred_deep(data, m1_deep)
    error_deep1 = get_errors(y_pred_deep1, data['Y'])
    y_pred_rcl1 = pred_rcl(data, m1_rcl)
    error_rcl1 = get_errors(y_pred_rcl1, data['Y'])
    y_pred_logit1 = pred_logit2(data, m1_mnl)
    error_logit1 = get_errors(y_pred_logit1, data['Y'])
    y_pred_single1 = pred_single(data, m1_single)
    error_single1 = get_errors(y_pred_single1, data['Y'])
    y_pred_random1 = np.mean(data_train['Y']) * np.ones(y_pred_deep1.shape)
    error_random1 = get_errors(y_pred_random1, data['Y'])

    
    test_pred_df = pd.DataFrame({'true':data['Y'], 'pred_deep':y_pred_deep1, 'pred_rcl': y_pred_rcl1, 'pred_mnl': y_pred_logit1, 'pred_single': y_pred_single1})
    test_pred_df['pred_random'] = np.mean(data_train['Y'])
    
    test_pred_df.to_csv("test"+ 'J' + str(J) + 'K' +  str(K) +'M' + str(M) + '_'+str(seed) + str(dg).split()[1] +".csv")
    
    error_all = setup + error_deep1 + error_rcl1 + error_logit1 + error_single1 + error_random1
    error_file_name = 'error_'+ 'J' + str(J) + 'K' +  str(K) +'M' + str(M) + '_'+str(seed) + str(dg).split()[1] +".pickle"
    
    with open(error_file_name, "wb") as f:
        # dump the dictionary into the file using pickle.dump()
        pickle.dump(error_all, f)
    
    print("Prediction errors are calculated.")
    
    model_list = [m1_deep, m1_rcl, m1_mnl, m1_single]

    ### step 5: elasticity
    J = data_train['J']
    large_record = pd.DataFrame(columns = ['market_id','i','j','old_price','new_price','type',
                                           'true_elasticity','pred_elasticity_deep','pred_elasticity_rcl','pred_elasticity_mnl','pred_elasticity_single','price_i'])
    for prod_id in range(J):
        data_train = cal_true_elasticity(data_train, dg, prod_id, delta, seed)
        record = cal_elasticity_record(data_train, pred_deep, model_list[0], prod_id, delta)
        record = record.rename(columns={'pred_elasticity': 'pred_elasticity_deep'})
        record['pred_elasticity_rcl'] = cal_elasticity_record(data_train, pred_rcl, model_list[1], prod_id, delta)['pred_elasticity']
        record['pred_elasticity_mnl'] = cal_elasticity_record(data_train, pred_logit2, model_list[2], prod_id, delta)['pred_elasticity']
        record['pred_elasticity_single'] = cal_elasticity_record(data_train, pred_single, model_list[3], prod_id, delta)['pred_elasticity']
        record['i'] = prod_id
        record['j'] = record.index % data_train['J']
        large_record = pd.concat([large_record, record]).reset_index(drop=True)

    large_record['error_deep'] = large_record['pred_elasticity_deep'] - large_record['true_elasticity']
    large_record['error_rcl'] = large_record['pred_elasticity_rcl'] - large_record['true_elasticity']
    large_record['error_mnl'] = large_record['pred_elasticity_mnl'] - large_record['true_elasticity']
    large_record['error_single'] = large_record['pred_elasticity_single'] - large_record['true_elasticity']
    
    large_record.to_csv('elas_record_'+ 'J' + str(J) + 'K' +  str(K) +'M' + str(M) + '_'+str(seed) + str(dg).split()[1] +".csv", index = False)
    
    elas_all = large_record.copy()
    elas_all['ae_deep'] = np.abs(elas_all['error_deep'])
    elas_all['ae_mnl'] = np.abs(elas_all['error_mnl'])
    elas_all['ae_rcl'] = np.abs(elas_all['error_rcl'])
    elas_all['ae_single'] = np.abs(elas_all['error_single'])
    
    elas_all.to_csv('elas_all' + 'J' + str(J) + 'K' +  str(K) +'M' + str(M)+str(dg).split()[1] +'.csv', index = False)
    # print('elas_all' + 'J' + str(J) + 'K' +  str(K) +'M' + str(M)+str(dg).split()[1] +'.csv')
    elas_valid = elas_all.loc[(elas_all.true_elasticity.isna()==False) & (np.isfinite(elas_all.true_elasticity) == True)].copy()
    
    elas_rslt_median = elas_valid.groupby(['type'])[['ae_deep','ae_mnl','ae_rcl','ae_single']].median().reset_index()
    elas_rslt_mean = elas_valid.groupby(['type'])[['ae_deep','ae_mnl','ae_rcl','ae_single']].mean().reset_index()
    
    print("Elasticity errors are calculated.")
    
    return 1 




def report(J, K, M, seed_list, dg):
    
    elas_all = pd.DataFrame()
    error_df = pd.DataFrame(columns = ['J', 'K', 'M', 'dg', 'seed',"mse_deep","mae_deep", "mse_rcl","mae_rcl", "mse_mnl", "mae_mnl","mse_single", "mae_single", 'mse_random','mae_random'])
    for seed in seed_list:
        error_file = 'error_'+ 'J' + str(J) + 'K' +  str(K) +'M' + str(M) + '_'+str(seed) + str(dg).split()[1] +".pickle"
        elas_file = 'elas_record_'+ 'J' + str(J) + 'K' +  str(K) +'M' + str(M) + '_'+str(seed) + str(dg).split()[1] +".csv"
        elas_record = pd.read_csv(elas_file)
        elas_record['seed'] = seed
        with open(error_file, 'rb') as f:
            error_all = pickle.load(f) 
            
        error_df = pd.concat([error_df, pd.DataFrame([error_all], columns= error_df.columns)])
        elas_all = pd.concat([elas_all, elas_record])
    
    error_df.to_csv('error_df' + 'J' + str(J) + 'K' +  str(K) +'M' + str(M)+str(dg).split()[1] +'.csv')
    
    elas_all['ae_deep'] = np.abs(elas_all['error_deep'])
    elas_all['ae_mnl'] = np.abs(elas_all['error_mnl'])
    elas_all['ae_rcl'] = np.abs(elas_all['error_rcl'])
    elas_all['ae_single'] = np.abs(elas_all['error_single'])
    
    elas_all['mse_deep'] = elas_all['error_deep']**2
    elas_all['mse_mnl'] = elas_all['error_mnl']**2
    elas_all['mse_rcl'] = elas_all['error_rcl']**2
    elas_all['mse_single'] = elas_all['error_single']**2
    
    elas_all.to_csv('elas_all' + 'J' + str(J) + 'K' +  str(K) +'M' + str(M)+str(dg).split()[1] +'.csv', index = False)
    
    elas_valid = elas_all.loc[(elas_all.true_elasticity.isna()==False) & (np.isfinite(elas_all.true_elasticity) == True)].copy()
    
    elas_rslt_median = elas_valid.groupby(['type'])[['ae_deep','ae_mnl','ae_rcl','ae_single']].median().reset_index()
    elas_rslt_mean = elas_valid.groupby(['type'])[['ae_deep','ae_mnl','ae_rcl','ae_single']].mean().reset_index()
    
    elas_rslt_rmse = elas_valid.groupby(['type'])[['mse_deep', 'mse_mnl', 'mse_rcl', 'mse_single']].mean().reset_index()
    elas_rslt_rmse[['rmse_deep', 'rmse_mnl', 'rmse_rcl', 'rmse_single']] = np.sqrt(elas_rslt_rmse[['mse_deep', 'mse_mnl', 'mse_rcl', 'mse_single']])

    elas_rslt_median.to_csv('elas_median_' + 'J' + str(J) + 'K' +  str(K) +'M' + str(M)+str(dg).split()[1] +'.csv', index = False)
    elas_rslt_mean.to_csv('elas_mae_' + 'J' + str(J) + 'K' +  str(K) +'M' + str(M)+str(dg).split()[1] +'.csv', index = False)
    elas_rslt_rmse.to_csv('elas_rmse_' + 'J' + str(J) + 'K' +  str(K) +'M' + str(M)+str(dg).split()[1] +'.csv', index = False)
    
    return 1
    
def output(JKM_dg_list):
    elas_median_df = pd.DataFrame()
    elas_mae_df = pd.DataFrame()
    elas_rmse_df = pd.DataFrame()
    error_df = pd.DataFrame()

    for [J,K,M,dg] in JKM_dg_list:
        elas_mae_name = 'elas_mae_' + 'J' + str(J) + 'K' +  str(K) +'M' + str(M)+str(dg).split()[1] +'.csv'
        elas_median_name = 'elas_median_' + 'J' + str(J) + 'K' +  str(K) +'M' + str(M)+str(dg).split()[1] +'.csv' 
        elas_rmse_name = 'elas_rmse_' + 'J' + str(J) + 'K' +  str(K) +'M' + str(M)+str(dg).split()[1] +'.csv' 
        error_df_name = 'error_df' + 'J' + str(J) + 'K' +  str(K) +'M' + str(M)+str(dg).split()[1] +'.csv'

        file_names = [elas_mae_name, elas_median_name, elas_rmse_name, error_df_name]
        file_names = [elas_mae_name, elas_median_name,  error_df_name]

        elas_mae = pd.read_csv(elas_mae_name)
        elas_median = pd.read_csv(elas_median_name)
        elas_rmse = pd.read_csv(elas_rmse_name)
        error = pd.read_csv(error_df_name)

        for df in [elas_mae, elas_median, elas_rmse]: 
            df['J'] = J
            df['M'] = M
            df['K'] = K
            df['dg'] = str(dg).split()[1]

        elas_median_df = pd.concat([elas_median_df, elas_median])
        elas_mae_df = pd.concat([elas_mae_df, elas_mae])
        elas_rmse_df = pd.concat([elas_rmse_df, elas_rmse])

        error_df = pd.concat([error_df, error])
        
    return elas_median_df, elas_mae_df, elas_rmse_df, error_df  #


### BLP functions

def train_blp(data):
    X1_formulation = pyblp.Formulation('0 + prices', absorb='C(product_ids)')
    X2_formulation = pyblp.Formulation('1 + prices + sugar + mushy')
    product_formulations = (X1_formulation, X2_formulation)

    mc_integration = pyblp.Integration('monte_carlo', size=50, specification_options={'seed': 0})
    mc_problem = pyblp.Problem(product_formulations, data, integration=mc_integration)
    bfgs = pyblp.Optimization('bfgs', {'gtol': 1e-4})
    results1 = mc_problem.solve(sigma=np.ones((4, 4)), optimization=bfgs)
    
    return results1


def pred_blp(data, model):
    ## only on train
    return model.compute_shares(data['prices']).squeeze()



def full_one_iteration_fe(J, M, K, seed, dg, params, hyper_params, delta):
    setup = [J, K, M, str(dg).split()[1], seed]
    ### step 1: data generation 
    datax = data_generation_fe(params, J, K, M, seed, dg)
    data_file_name = 'data_'+ 'J' + str(J) + 'K' +  str(K) +'M' + str(M) + '_'+str(seed) + str(dg).split()[1] +".pickle"
    # with open(data_file_name, "wb") as f:
    # # dump the dictionary into the file using pickle.dump()
    #     pickle.dump(datax, f)
    
    data_train, data_test = split_train_test(datax, p = 0.8)
    ### step 2: train models 
    m1_deep, loss_deep = train_deep(data_train)
    m1_rcl, losses_rcl = train_rcl(data_train)
    m1_mnl, losses_mnl = train_logit(data_train)
    m1_single,losses_single = train_single(data_train, hyper_params)
    
    ### step 3: evaluation
    data = data_train.copy()

    y_pred_deep1 = pred_deep(data, m1_deep)
    error_deep1 = get_errors(y_pred_deep1, data['Y'])
    y_pred_rcl1 = pred_rcl(data, m1_rcl)
    error_rcl1 = get_errors(y_pred_rcl1, data['Y'])
    y_pred_logit1 = pred_logit2(data, m1_mnl)
    error_logit1 = get_errors(y_pred_logit1, data['Y'])
    y_pred_single1 = pred_single(data, m1_single)
    error_single1 = get_errors(y_pred_single1, data['Y'])
    
    train_pred_df = pd.DataFrame({'true':data['Y'], 'pred_deep':y_pred_deep1, 'pred_rcl': y_pred_rcl1, 'pred_mnl': y_pred_logit1, 'pred_single': y_pred_single1})
    train_pred_df['pred_random'] = np.mean(data_train['Y'])
    
    train_pred_df.to_csv("train_"+ 'J' + str(J) + 'K' +  str(K) +'M' + str(M) + '_'+str(seed) + str(dg).split()[1] +".csv")

    # print("J=", data['J'],"K=", data['K'],"M=", data['M'], "\n", str(dg).split()[1], 'training error',
    #       "rcl error: ", error_rcl1[1], "\n",
    #       "mnl error: ", error_logit1[1], "\n",
    #       "deep error: ", error_deep1[1], "\n",
    #      "single error: ", error_single1[1], "\n") 
    
    data = data_test.copy()

    y_pred_deep1 = pred_deep(data, m1_deep)
    error_deep1 = get_errors(y_pred_deep1, data['Y'])
    y_pred_rcl1 = pred_rcl(data, m1_rcl)
    error_rcl1 = get_errors(y_pred_rcl1, data['Y'])
    y_pred_logit1 = pred_logit2(data, m1_mnl)
    error_logit1 = get_errors(y_pred_logit1, data['Y'])
    y_pred_single1 = pred_single(data, m1_single)
    error_single1 = get_errors(y_pred_single1, data['Y'])
    y_pred_random1 = np.mean(data_train['Y']) * np.ones(y_pred_deep1.shape)
    error_random1 = get_errors(y_pred_random1, data['Y'])
    
    # print("J=", data['J'],"K=", data['K'],"M=", data['M'], "\n",  str(dg).split()[1], 'test error'
    #       "rcl error: ", error_rcl1[1], "\n",
    #       "mnl error: ", error_logit1[1], "\n",
    #       "deep error: ", error_deep1[1], "\n",
    #      "single error: ", error_single1[1], "\n") 
    
    test_pred_df = pd.DataFrame({'true':data['Y'], 'pred_deep':y_pred_deep1, 'pred_rcl': y_pred_rcl1, 'pred_mnl': y_pred_logit1, 'pred_single': y_pred_single1})
    test_pred_df['pred_random'] = np.mean(data_train['Y'])
    
    test_pred_df.to_csv("test"+ 'J' + str(J) + 'K' +  str(K) +'M' + str(M) + '_'+str(seed) + str(dg).split()[1] +".csv")
    
    error_all = setup + error_deep1 + error_rcl1 + error_logit1 + error_single1 + error_random1
    error_file_name = 'error_'+ 'J' + str(J) + 'K' +  str(K) +'M' + str(M) + '_'+str(seed) + str(dg).split()[1] +".pickle"
    
    with open(error_file_name, "wb") as f:
        # dump the dictionary into the file using pickle.dump()
        pickle.dump(error_all, f)
    
    
    model_list = [m1_deep, m1_rcl, m1_mnl, m1_single]
    model_file_name = 'model_'+ 'J' + str(J) + 'K' +  str(K) +'M' + str(M) + '_'+str(seed) + str(dg).split()[1] +".pickle"
    
    # with open(model_file_name, "wb") as f:
    #     # dump the dictionary into the file using pickle.dump()
    #     pickle.dump(model_list, f)
        

    ### step 5: elasticity
    J = data_train['J']
    large_record = pd.DataFrame(columns = ['market_id','i','j','old_price','new_price','type','true_elasticity','pred_elasticity_deep','pred_elasticity_rcl','pred_elasticity_mnl','pred_elasticity_single','price_i'])
    for prod_id in range(J):
        data_train = cal_true_elasticity(data_train, dg, prod_id, delta, seed)
        record = cal_elasticity_record(data_train, pred_deep, model_list[0], prod_id, delta)
        record = record.rename(columns={'pred_elasticity': 'pred_elasticity_deep'})
        record['pred_elasticity_rcl'] = cal_elasticity_record(data_train, pred_rcl, model_list[1], prod_id, delta)['pred_elasticity']
        record['pred_elasticity_mnl'] = cal_elasticity_record(data_train, pred_logit2, model_list[2], prod_id, delta)['pred_elasticity']
        record['pred_elasticity_single'] = cal_elasticity_record(data_train, pred_single, model_list[3], prod_id, delta)['pred_elasticity']
        record['i'] = prod_id
        record['j'] = record.index % data_train['J']
        large_record = pd.concat([large_record, record]).reset_index(drop=True)

    large_record['error_deep'] = large_record['pred_elasticity_deep'] - large_record['true_elasticity']
    large_record['error_rcl'] = large_record['pred_elasticity_rcl'] - large_record['true_elasticity']
    large_record['error_mnl'] = large_record['pred_elasticity_mnl'] - large_record['true_elasticity']
    large_record['error_single'] = large_record['pred_elasticity_single'] - large_record['true_elasticity']
    
    large_record.to_csv('elas_record_'+ 'J' + str(J) + 'K' +  str(K) +'M' + str(M) + '_'+str(seed) + str(dg).split()[1] +".csv", index = False)
    
    elas_all = large_record.copy()
    elas_all['ae_deep'] = np.abs(elas_all['error_deep'])
    elas_all['ae_mnl'] = np.abs(elas_all['error_mnl'])
    elas_all['ae_rcl'] = np.abs(elas_all['error_rcl'])
    elas_all['ae_single'] = np.abs(elas_all['error_single'])
    
    elas_all.to_csv('elas_all' + 'J' + str(J) + 'K' +  str(K) +'M' + str(M)+str(dg).split()[1] +'.csv', index = False)
    
    elas_valid = elas_all.loc[(elas_all.true_elasticity.isna()==False) & (np.isfinite(elas_all.true_elasticity) == True)].copy()
    
    elas_rslt_median = elas_valid.groupby(['type'])[['ae_deep','ae_mnl','ae_rcl','ae_single']].median().reset_index()
    elas_rslt_mean = elas_valid.groupby(['type'])[['ae_deep','ae_mnl','ae_rcl','ae_single']].mean().reset_index()
    
    print('elas_mean', elas_rslt_mean)
    print('elas_median', elas_rslt_median)
    
    return 1 




def full_one_iteration_tri(J, M, K, seed, dg, params, hyper_params, delta):
    setup = [J, K, M, str(dg).split()[1], seed]
    ### step 1: data generation 
    datax = data_generation_tri(params, J, K, M, seed, dg)
    
    data_file_name = 'data_'+ 'J' + str(J) + 'K' +  str(K) +'M' + str(M) + '_'+str(seed) + str(dg).split()[1] +".pickle"
    with open(data_file_name, "wb") as f:
    # dump the dictionary into the file using pickle.dump()
        pickle.dump(datax, f)
    
    data_train, data_test = split_train_test(datax, p = 0.8)
    ### step 2: train models 
    m1_deep, loss_deep = train_deep(data_train)
    m1_rcl, losses_rcl = train_rcl(data_train)
    m1_mnl, losses_mnl = train_logit(data_train)
    m1_single,losses_single = train_single(data_train, hyper_params)
    
    ### step 3: evaluation
    data = data_train.copy()

    y_pred_deep1 = pred_deep(data, m1_deep)
    error_deep1 = get_errors(y_pred_deep1, data['Y'])
    y_pred_rcl1 = pred_rcl(data, m1_rcl)
    error_rcl1 = get_errors(y_pred_rcl1, data['Y'])
    y_pred_logit1 = pred_logit2(data, m1_mnl)
    error_logit1 = get_errors(y_pred_logit1, data['Y'])
    y_pred_single1 = pred_single(data, m1_single)
    error_single1 = get_errors(y_pred_single1, data['Y'])
    
    train_pred_df = pd.DataFrame({'true':data['Y'], 'pred_deep':y_pred_deep1, 'pred_rcl': y_pred_rcl1, 'pred_mnl': y_pred_logit1, 'pred_single': y_pred_single1})
    train_pred_df['pred_random'] = np.mean(data_train['Y'])
    
    train_pred_df.to_csv("train"+ 'J' + str(J) + 'K' +  str(K) +'M' + str(M) + '_'+str(seed) + str(dg).split()[1] +".csv")
    print("train"+ 'J' + str(J) + 'K' +  str(K) +'M' + str(M) + '_'+str(seed) + str(dg).split()[1] +".csv")
    print("J=", data['J'],"K=", data['K'],"M=", data['M'], "\n", str(dg).split()[1], 'training error',
          "rcl error: ", error_rcl1[1], "\n",
          "mnl error: ", error_logit1[1], "\n",
          "deep error: ", error_deep1[1], "\n",
         "single error: ", error_single1[1], "\n") 
    
    data = data_test.copy()

    y_pred_deep1 = pred_deep(data, m1_deep)
    error_deep1 = get_errors(y_pred_deep1, data['Y'])
    y_pred_rcl1 = pred_rcl(data, m1_rcl)
    error_rcl1 = get_errors(y_pred_rcl1, data['Y'])
    y_pred_logit1 = pred_logit2(data, m1_mnl)
    error_logit1 = get_errors(y_pred_logit1, data['Y'])
    y_pred_single1 = pred_single(data, m1_single)
    error_single1 = get_errors(y_pred_single1, data['Y'])
    y_pred_random1 = np.mean(data_train['Y']) * np.ones(y_pred_deep1.shape)
    error_random1 = get_errors(y_pred_random1, data['Y'])
    
    print("J=", data['J'],"K=", data['K'],"M=", data['M'], "\n",  str(dg).split()[1], 'test error'
          "rcl error: ", error_rcl1[1], "\n",
          "mnl error: ", error_logit1[1], "\n",
          "deep error: ", error_deep1[1], "\n",
         "single error: ", error_single1[1], "\n") 
    
    test_pred_df = pd.DataFrame({'true':data['Y'], 'pred_deep':y_pred_deep1, 'pred_rcl': y_pred_rcl1, 'pred_mnl': y_pred_logit1, 'pred_single': y_pred_single1})
    test_pred_df['pred_random'] = np.mean(data_train['Y'])
    
    test_pred_df.to_csv("test"+ 'J' + str(J) + 'K' +  str(K) +'M' + str(M) + '_'+str(seed) + str(dg).split()[1] +".csv")
    
    error_all = setup + error_deep1 + error_rcl1 + error_logit1 + error_single1 + error_random1
    error_file_name = 'error_'+ 'J' + str(J) + 'K' +  str(K) +'M' + str(M) + '_'+str(seed) + str(dg).split()[1] +".pickle"
    
    with open(error_file_name, "wb") as f:
        # dump the dictionary into the file using pickle.dump()
        pickle.dump(error_all, f)
    
    
    model_list = [m1_deep, m1_rcl, m1_mnl, m1_single]
    model_file_name = 'model_'+ 'J' + str(J) + 'K' +  str(K) +'M' + str(M) + '_'+str(seed) + str(dg).split()[1] +".pickle"
    
    with open(model_file_name, "wb") as f:
        # dump the dictionary into the file using pickle.dump()
        pickle.dump(model_list, f)
    
    print(model_file_name)

    ### step 5: elasticity
    J = data_train['J']
    large_record = pd.DataFrame(columns = ['market_id','i','j','old_price','new_price','type',
                                           'true_elasticity','pred_elasticity_deep','pred_elasticity_rcl','pred_elasticity_mnl','pred_elasticity_single','price_i'])
    for prod_id in range(J):
        data_train = cal_true_elasticity(data_train, dg, prod_id, delta, seed)
        record = cal_elasticity_record(data_train, pred_deep, model_list[0], prod_id, delta)
        record = record.rename(columns={'pred_elasticity': 'pred_elasticity_deep'})
        record['pred_elasticity_rcl'] = cal_elasticity_record(data_train, pred_rcl, model_list[1], prod_id, delta)['pred_elasticity']
        record['pred_elasticity_mnl'] = cal_elasticity_record(data_train, pred_logit2, model_list[2], prod_id, delta)['pred_elasticity']
        record['pred_elasticity_single'] = cal_elasticity_record(data_train, pred_single, model_list[3], prod_id, delta)['pred_elasticity']
        record['i'] = prod_id
        record['j'] = record.index % data_train['J']
        large_record = pd.concat([large_record, record]).reset_index(drop=True)

    large_record['error_deep'] = large_record['pred_elasticity_deep'] - large_record['true_elasticity']
    large_record['error_rcl'] = large_record['pred_elasticity_rcl'] - large_record['true_elasticity']
    large_record['error_mnl'] = large_record['pred_elasticity_mnl'] - large_record['true_elasticity']
    large_record['error_single'] = large_record['pred_elasticity_single'] - large_record['true_elasticity']
    
    large_record.to_csv('elas_record_'+ 'J' + str(J) + 'K' +  str(K) +'M' + str(M) + '_'+str(seed) + str(dg).split()[1] +".csv", index = False)
    
    elas_all = large_record.copy()
    elas_all['ae_deep'] = np.abs(elas_all['error_deep'])
    elas_all['ae_mnl'] = np.abs(elas_all['error_mnl'])
    elas_all['ae_rcl'] = np.abs(elas_all['error_rcl'])
    elas_all['ae_single'] = np.abs(elas_all['error_single'])
    
    elas_all.to_csv('elas_all' + 'J' + str(J) + 'K' +  str(K) +'M' + str(M)+str(dg).split()[1] +'.csv', index = False)
    print('elas_all' + 'J' + str(J) + 'K' +  str(K) +'M' + str(M)+str(dg).split()[1] +'.csv')
    elas_valid = elas_all.loc[(elas_all.true_elasticity.isna()==False) & (np.isfinite(elas_all.true_elasticity) == True)].copy()
    
    elas_rslt_median = elas_valid.groupby(['type'])[['ae_deep','ae_mnl','ae_rcl','ae_single']].median().reset_index()
    elas_rslt_mean = elas_valid.groupby(['type'])[['ae_deep','ae_mnl','ae_rcl','ae_single']].mean().reset_index()
    
    print('elas_mean', elas_rslt_mean)
    print('elas_median', elas_rslt_median)
    
    return 1 



def full_one_iteration_keepprice(J, M, K, seed, dg, params, hyper_params, delta):
    setup = [J, K, M, str(dg).split()[1], seed]
    ### step 1: data generation 
    datax = data_generation_keepprice(params, J, K, M, seed, dg)
    
    data_file_name = 'data_'+ 'J' + str(J) + 'K' +  str(K) +'M' + str(M) + '_'+str(seed) + str(dg).split()[1] +".pickle"
    with open(data_file_name, "wb") as f:
    # dump the dictionary into the file using pickle.dump()
        pickle.dump(datax, f)
    
    data_train, data_test = split_train_test(datax, p = 0.8)
    ### step 2: train models 
    m1_deep, loss_deep = train_deep(data_train)
    m1_rcl, losses_rcl = train_rcl(data_train)
    m1_mnl, losses_mnl = train_logit(data_train)
    m1_single,losses_single = train_single(data_train, hyper_params)
    
    ### step 3: evaluation
    data = data_train.copy()

    y_pred_deep1 = pred_deep(data, m1_deep)
    error_deep1 = get_errors(y_pred_deep1, data['Y'])
    y_pred_rcl1 = pred_rcl(data, m1_rcl)
    error_rcl1 = get_errors(y_pred_rcl1, data['Y'])
    y_pred_logit1 = pred_logit2(data, m1_mnl)
    error_logit1 = get_errors(y_pred_logit1, data['Y'])
    y_pred_single1 = pred_single(data, m1_single)
    error_single1 = get_errors(y_pred_single1, data['Y'])
    
    train_pred_df = pd.DataFrame({'true':data['Y'], 'pred_deep':y_pred_deep1, 'pred_rcl': y_pred_rcl1, 'pred_mnl': y_pred_logit1, 'pred_single': y_pred_single1})
    train_pred_df['pred_random'] = np.mean(data_train['Y'])
    
    train_pred_df.to_csv("train"+ 'J' + str(J) + 'K' +  str(K) +'M' + str(M) + '_'+str(seed) + str(dg).split()[1] +".csv")
    print("train"+ 'J' + str(J) + 'K' +  str(K) +'M' + str(M) + '_'+str(seed) + str(dg).split()[1] +".csv")
    print("J=", data['J'],"K=", data['K'],"M=", data['M'], "\n", str(dg).split()[1], 'training error',
          "rcl error: ", error_rcl1[1], "\n",
          "mnl error: ", error_logit1[1], "\n",
          "deep error: ", error_deep1[1], "\n",
         "single error: ", error_single1[1], "\n") 
    
    data = data_test.copy()

    y_pred_deep1 = pred_deep(data, m1_deep)
    error_deep1 = get_errors(y_pred_deep1, data['Y'])
    y_pred_rcl1 = pred_rcl(data, m1_rcl)
    error_rcl1 = get_errors(y_pred_rcl1, data['Y'])
    y_pred_logit1 = pred_logit2(data, m1_mnl)
    error_logit1 = get_errors(y_pred_logit1, data['Y'])
    y_pred_single1 = pred_single(data, m1_single)
    error_single1 = get_errors(y_pred_single1, data['Y'])
    y_pred_random1 = np.mean(data_train['Y']) * np.ones(y_pred_deep1.shape)
    error_random1 = get_errors(y_pred_random1, data['Y'])
    
    print("J=", data['J'],"K=", data['K'],"M=", data['M'], "\n",  str(dg).split()[1], 'test error'
          "rcl error: ", error_rcl1[1], "\n",
          "mnl error: ", error_logit1[1], "\n",
          "deep error: ", error_deep1[1], "\n",
         "single error: ", error_single1[1], "\n") 
    
    test_pred_df = pd.DataFrame({'true':data['Y'], 'pred_deep':y_pred_deep1, 'pred_rcl': y_pred_rcl1, 'pred_mnl': y_pred_logit1, 'pred_single': y_pred_single1})
    test_pred_df['pred_random'] = np.mean(data_train['Y'])
    
    test_pred_df.to_csv("test"+ 'J' + str(J) + 'K' +  str(K) +'M' + str(M) + '_'+str(seed) + str(dg).split()[1] +".csv")
    
    error_all = setup + error_deep1 + error_rcl1 + error_logit1 + error_single1 + error_random1
    error_file_name = 'error_'+ 'J' + str(J) + 'K' +  str(K) +'M' + str(M) + '_'+str(seed) + str(dg).split()[1] +".pickle"
    
    with open(error_file_name, "wb") as f:
        # dump the dictionary into the file using pickle.dump()
        pickle.dump(error_all, f)
    
    
    model_list = [m1_deep, m1_rcl, m1_mnl, m1_single]
    model_file_name = 'model_'+ 'J' + str(J) + 'K' +  str(K) +'M' + str(M) + '_'+str(seed) + str(dg).split()[1] +".pickle"
    
    with open(model_file_name, "wb") as f:
        # dump the dictionary into the file using pickle.dump()
        pickle.dump(model_list, f)
    
    print(model_file_name)

    ### step 5: elasticity
    J = data_train['J']
    large_record = pd.DataFrame(columns = ['market_id','i','j','old_price','new_price','type',
                                           'true_elasticity','pred_elasticity_deep','pred_elasticity_rcl','pred_elasticity_mnl','pred_elasticity_single','price_i'])
    for prod_id in range(J):
        data_train = cal_true_elasticity(data_train, dg, prod_id, delta, seed)
        record = cal_elasticity_record(data_train, pred_deep, model_list[0], prod_id, delta)
        record = record.rename(columns={'pred_elasticity': 'pred_elasticity_deep'})
        record['pred_elasticity_rcl'] = cal_elasticity_record(data_train, pred_rcl, model_list[1], prod_id, delta)['pred_elasticity']
        record['pred_elasticity_mnl'] = cal_elasticity_record(data_train, pred_logit2, model_list[2], prod_id, delta)['pred_elasticity']
        record['pred_elasticity_single'] = cal_elasticity_record(data_train, pred_single, model_list[3], prod_id, delta)['pred_elasticity']
        record['i'] = prod_id
        record['j'] = record.index % data_train['J']
        large_record = pd.concat([large_record, record]).reset_index(drop=True)

    large_record['error_deep'] = large_record['pred_elasticity_deep'] - large_record['true_elasticity']
    large_record['error_rcl'] = large_record['pred_elasticity_rcl'] - large_record['true_elasticity']
    large_record['error_mnl'] = large_record['pred_elasticity_mnl'] - large_record['true_elasticity']
    large_record['error_single'] = large_record['pred_elasticity_single'] - large_record['true_elasticity']
    
    large_record.to_csv('elas_record_'+ 'J' + str(J) + 'K' +  str(K) +'M' + str(M) + '_'+str(seed) + str(dg).split()[1] +".csv", index = False)
    
    elas_all = large_record.copy()
    elas_all['ae_deep'] = np.abs(elas_all['error_deep'])
    elas_all['ae_mnl'] = np.abs(elas_all['error_mnl'])
    elas_all['ae_rcl'] = np.abs(elas_all['error_rcl'])
    elas_all['ae_single'] = np.abs(elas_all['error_single'])
    
    elas_all.to_csv('elas_all' + 'J' + str(J) + 'K' +  str(K) +'M' + str(M)+str(dg).split()[1] +'.csv', index = False)
    print('elas_all' + 'J' + str(J) + 'K' +  str(K) +'M' + str(M)+str(dg).split()[1] +'.csv')
    elas_valid = elas_all.loc[(elas_all.true_elasticity.isna()==False) & (np.isfinite(elas_all.true_elasticity) == True)].copy()
    
    elas_rslt_median = elas_valid.groupby(['type'])[['ae_deep','ae_mnl','ae_rcl','ae_single']].median().reset_index()
    elas_rslt_mean = elas_valid.groupby(['type'])[['ae_deep','ae_mnl','ae_rcl','ae_single']].mean().reset_index()
    
    print('elas_mean', elas_rslt_mean)
    print('elas_median', elas_rslt_median)
    
    return 1 



def full_one_iteration_newproduct(J, M, K, seed, dg, params, hyper_params, delta):
    setup = [J, K, M, str(dg).split()[1], seed]
    ### step 1: data generation 
    datax = data_generation(params, J, K, M, seed, dg)
    data_file_name = 'data_'+ 'J' + str(J) + 'K' +  str(K) +'M' + str(M) + '_'+str(seed) + str(dg).split()[1] +".pickle"
    #with open(data_file_name, "wb") as f:
    # dump the dictionary into the file using pickle.dump()
    #    pickle.dump(datax, f)
    
    data_train, data_test = split_train_test(datax, p = 0.8)
    print('Data is generated.')
    ### step 2: train models 
    m1_deep, loss_deep = train_deep(data_train)
    print("Main model is trained.")
    m1_rcl, losses_rcl = train_rcl(data_train)
    print("RCL is estimated.")
    m1_mnl, losses_mnl = train_logit(data_train)
    print("MNL is estimated.")
    m1_single,losses_single = train_single(data_train, hyper_params)
    print("NP is estimated.")
    
    model_list = [m1_deep, m1_rcl, m1_mnl, m1_single]
    error_all = new_prod_eval4(datax, dg, model_list, seed)
    
    return error_all

