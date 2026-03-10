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


def feature_generation(J, K, M, seed):
    # J is number of products per markets (assume each market has the same number of products but different products)
    # K is number of features (excluding price)
    # M is the number of markets (assume each market has different sets of products)
    np.random.seed(seed)

    Total_J = J * M
    x_1 = np.random.normal(loc=0, scale=1, size=(Total_J , K))
    price = np.random.uniform(0,4,size = Total_J)
    #price = np.abs(np.random.normal(loc=0, scale=1, size = Total_J))
    
    X = pd.DataFrame(x_1)
    X['price'] = price

    return X 

def market_id_gen(J, M):
    Total_J = J * M
    J_list = [J] * M
    ## create a column as market id
    market_id = np.ones(Total_J)
    start = 0 
    for m in range(M):
        j_m = J_list[m]
        market_id[start:start + j_m] = m
        start = start + j_m
        
    return market_id   

def mnl(X, b, J, K, M, seed):
    # seed is not used in this function
    # b are the parameters 
    Total_J = J * M
    
    ## get the share of each product (y)
    u = b[0] * np.ones(Total_J)
    for i in range(1,K+2):
        u = u + b[i] * X.iloc[:,i-1] 

    X['u'] = np.exp(u)    
    market_id = market_id_gen(J, M)
    X['market_id'] = market_id
    X['sum_u'] = X.groupby(['market_id'])['u'].transform('sum') 

    #Y = np.log(X['u'] / (X['sum_u'] + 1))
    Y = X['u'] / (X['sum_u'] + 1)

    del X['u']
    del X['market_id']
    del X['sum_u']

    data = {'X': X, 'Y': Y.to_numpy(),  'M': M, "J": J, "K": K, 'params':b, 'generation_seed': seed,'market_id':market_id}

    return data

def rcl(X, params, J, K, M, seed, N=10000):
    J_list = [J] * M
    Total_J = J * M

    b = params[0]
    sigma = params[1]

    ## generate random coeffcient for each individual
    np.random.seed(seed)
    b_random= []
    
    st1 = time.time()
    for i in range(len(b)):
        b_random.append(np.repeat(np.random.normal(loc = b[i], scale = sigma[i], size = (M, N)), repeats = J, axis=0))
    st2 = time.time()
    
    ## get the utility of each user, the output is (M * J) * N
    u_i = b_random[0] * np.ones((Total_J, N))

    for k in range(1,K+2):
        u_i = u_i + b_random[k] * (X.iloc[:,k-1].to_numpy().reshape(Total_J,1))
    
    #u_i_max = np.max(u_i, axis=0, keepdims=True)  # shape: (1, N)
    #u_i_stable = u_i - u_i_max  
    exp_u_i = np.exp(u_i)  
    u_m = exp_u_i.reshape(M, J, N)
    sum_u_m = np.sum(u_m, axis=1, keepdims=True) 
    ccp_m = u_m / (sum_u_m + 1 )
    
    # Aggregate individual choice by market
    share_m = np.sum(ccp_m, axis=2) / N

    # Flatten share_m from (M, J) to (M*J,)
    Y = share_m.flatten()
    
    market_id = market_id_gen(J, M)
    data = {'X': X, 'Y': Y,  'M': M, "J": J, "K": K, 'params': params, 'generation_seed': seed, 
            'b_random':b_random, 'market_id':market_id}

    return data



def mnl_choice(X, params, J, K, M, seed, N=10000):
    J_list = [J] * M
    Total_J = J * M

    b = params

    ## generate random coeffcient for each individual
    np.random.seed(seed)
    b_random= []
    
    for i in range(len(b)):
        b_random.append(np.repeat(np.ones((M, N)) *  b[i], repeats = J, axis=0))
    
    ## get the utility of each user, the output is (M * J) * N
    u_i = b_random[0] * np.ones((Total_J, N))

    for k in range(1,K+2):
        u_i = u_i + b_random[k] * (X.iloc[:,k-1].to_numpy().reshape(Total_J,1))
        
    e_i = np.random.gumbel(0, 1, size=u_i.shape)
    u_i = u_i + e_i

    Y = np.zeros((M* J))
    for m in range(M):
        ## get the sub-matrix (J * N) of this market
        u_m = u_i[m * J:(m+1)*J,: ]
        u_m_max = np.max(u_m, axis = 0)
        choice = np.where(u_m_max > 0, np.argmax(u_m, axis = 0), J)
        share_m = np.bincount(choice, minlength=J) / N
        Y[m * J:(m+1)*J] = share_m[0:J]
    
    
    market_id = market_id_gen(J, M)
    data = {'X': X, 'Y': Y,  'M': M, "J": J, "K": K, 'params': params, 'generation_seed': seed, 
            'b_random':b_random, 'market_id':market_id,'e_random':e_i}

    return data



def rcl_log(X, params, J, K, M, seed, N=10000):
    J_list = [J] * M
    Total_J = J * M
    
    b = params[0]
    sigma = params[1]

    ## generate random coeffcient for each individual
    np.random.seed(seed)
    b_random= []
    
    st1 = time.time()
    for i in range(len(b)):
        b_random.append(np.repeat(np.random.normal(loc = b[i], scale = sigma[i], size = (M, N)), repeats = J, axis=0))
    st2 = time.time()
    #print('b_random', st2-st1)
    
    ## get the utility of each user, the output is (M * J) * N
    u_i = b_random[0] * np.ones((Total_J, N))

    for k in range(1,K+2):
        #if k < (K+2)/3:
        u_i = u_i + b_random[k] * (np.log( np.abs(16*X.iloc[:,k-1].to_numpy().reshape(Total_J,1) - 8) + 1 ) * np.sign(16*X.iloc[:,k-1].to_numpy().reshape(Total_J,1) - 8))
        #else:
            #u_i = u_i + b_random[k] * (X.iloc[:,k-1].to_numpy().reshape(Total_J,1))

    exp_u_i = np.exp(u_i)
    
    st3 = time.time()
    #print('u', st3-st2)
    
    Y = np.zeros((M* J))
    for m in range(M):
        ## get the sub-matrix (J * N) of this market
        u_m = exp_u_i[m * J:(m+1)*J,: ]
        ## calculate the probability of purchasing for each individual
        sum_u_m = np.sum(u_m, axis = 0).reshape((N,1))
        ccp_m = (u_m.T / (sum_u_m + 1)).T
        ## aggregate individual choice by market
        share_m = ccp_m.sum(axis = 1)/N
        Y[m * J:(m+1)*J] = share_m
    
    st4 = time.time()
    #print('y', st4-st3)
    market_id = market_id_gen(J, M)
    data = {'X': X, 'Y': Y,  'M': M, "J": J, "K": K, 'params': params, 'generation_seed': seed, 
            'b_random':b_random, 'market_id':market_id}

    return data


def rcl_sin(X, params, J, K, M, seed, N=10000):
    J_list = [J] * M
    Total_J = J * M
    
    b = params[0]
    sigma = params[1]

    ## generate random coeffcient for each individual
    np.random.seed(seed)
    b_random= []
    
    st1 = time.time()
    for i in range(len(b)):
        b_random.append(np.repeat(np.random.normal(loc = b[i], scale = sigma[i], size = (M, N)), repeats = J, axis=0))
    st2 = time.time()
    #print('b_random', st2-st1)
    
    ## get the utility of each user, the output is (M * J) * N
    u_i = b_random[0] * np.ones((Total_J, N))

    for k in range(1,K+2):
        #if k < (K+2)/3:
        u_i = u_i + b_random[k] * (np.sin(X.iloc[:,k-1].to_numpy().reshape(Total_J,1)))
        #else:
            #u_i = u_i + b_random[k] * (X.iloc[:,k-1].to_numpy().reshape(Total_J,1))

    exp_u_i = np.exp(u_i)
    
    st3 = time.time()
    #print('u', st3-st2)
    
    Y = np.zeros((M* J))
    for m in range(M):
        ## get the sub-matrix (J * N) of this market
        u_m = exp_u_i[m * J:(m+1)*J,: ]
        ## calculate the probability of purchasing for each individual
        sum_u_m = np.sum(u_m, axis = 0).reshape((N,1))
        ccp_m = (u_m.T / (sum_u_m + 1)).T
        ## aggregate individual choice by market
        share_m = ccp_m.sum(axis = 1)/N
        Y[m * J:(m+1)*J] = share_m
    
    st4 = time.time()
    #print('y', st4-st3)
    market_id = market_id_gen(J, M)
    data = {'X': X, 'Y': Y,  'M': M, "J": J, "K": K, 'params': params, 'generation_seed': seed, 
            'b_random':b_random, 'market_id':market_id}


    return data




def rcl_regenerate(X, data, N=10000):
    J = data['J']
    K = data['K']
    M = data['M']
    b_random = data['b_random']
    
    Total_J = J * M
    
    ## get the utility of each user, the output is (M * J) * N
    u_i = b_random[0] * np.ones((Total_J, N))

    for k in range(1,K+2):
        u_i = u_i + b_random[k] * (X.iloc[:,k-1].to_numpy().reshape(Total_J,1))
    
    exp_u_i = np.exp(u_i)
    
    Y = np.zeros((M* J))
    
    # Reshape exp_u_i from (M*J, N) to (M, J, N)
    u_m = exp_u_i.reshape(M, J, N)

    # Calculate the probability of purchasing for each individual
    sum_u_m = np.sum(u_m, axis=1, keepdims=True)
    ccp_m = (u_m / (sum_u_m + 1))

    # Aggregate individual choice by market
    share_m = np.sum(ccp_m, axis=2) / N

    # Flatten share_m from (M, J) to (M*J,)
    Y = share_m.flatten()
    

    return Y


def rcl_regenerate_in3(X, data, N=10000):
    
    J = data['J']
    K = data['K']
    M = data['M']
    b_random = data['b_random']
    J_list = [J] * M
    Total_J = J * M
    
    ## get the utility of each user, the output is (M * J) * N
    u_i = b_random[0] * np.ones((Total_J, N))

    for k in range(1,K+2):
        u_i = u_i + b_random[k] * (X.iloc[:,k-1].to_numpy().reshape(Total_J,1))

    exp_u_i = np.exp(u_i)
    
    u_df = pd.DataFrame(exp_u_i)
    market_id = market_id_gen(J, M)
    u_df['market_id'] = market_id
    u_df['price'] = X['price'].copy()
    
    u_df['highest'] = 1 - u_df.groupby(['market_id'], group_keys=False)['price'].apply(lambda x: x == x.max()) * 1
    
    highest_price = u_df.groupby(['market_id'])['price'].max()
    
    Y = np.zeros((M* J))
    avg_inattention = []
    for m in range(M):
        max_price = highest_price[m]
        if max_price < 0:
            inattention_share =  0
        elif 0 <= max_price <= 5:
            inattention_share = 1- 1/(1+max_price)#0.2 * max_price  #1/4 * max_price 
        else:
            inattention_share = 1
        avg_inattention.append(inattention_share)
        
        attention_share = 1 - inattention_share

        u_m = u_df.iloc[m * J:(m+1)*J,: ]
        ## 1 is attentioned consumer, 2 is the inattention consumer
        u_df_sum1 = u_m.iloc[:,list(range(int(attention_share*N))) + [-1, -2, -3]].groupby(['market_id']).transform('sum') + 1        
        u_df_sum2 = u_m.iloc[:,list(range(int(attention_share*N), N)) + [-1, -2, -3]].groupby(['market_id','highest']).transform('sum') + 1
        
        p1 = u_m.iloc[:,list(range(int(attention_share*N)))] / u_df_sum1.iloc[:,list(range(int(attention_share*N)))]
        p2 = u_m.iloc[:,list(range(int(attention_share*N), N))] / u_df_sum2.iloc[:,list(range(N-int(attention_share*N)))]
        p2 = p2.mul(u_m['highest'],axis =0)

        ## Get Y
        Y[m * J:(m+1)*J] = (np.sum(p1,axis = 1) + np.sum(p2,axis=1))/N

    return Y




def rcl_regenerate_log(X, data, N=10000):
    J = data['J']
    K = data['K']
    M = data['M']
    b_random = data['b_random']
    
    Total_J = J * M
    
    ## get the utility of each user, the output is (M * J) * N
    u_i = b_random[0] * np.ones((Total_J, N))

    for k in range(1,K+2):
        #if k < K/3:
        u_i = u_i + b_random[k] * (np.log( np.abs(16*X.iloc[:,k-1].to_numpy().reshape(Total_J,1) - 8) + 1 ) * np.sign(16*X.iloc[:,k-1].to_numpy().reshape(Total_J,1) - 8))
        #else:
        #    u_i = u_i + b_random[k] * (X.iloc[:,k-1].to_numpy().reshape(Total_J,1))
            
    exp_u_i = np.exp(u_i)
    
    Y = np.zeros((M* J))
    for m in range(M):
        ## get the sub-matrix (J * N) of this market
        u_m = exp_u_i[m * J:(m+1)*J,: ]
        ## calculate the probability of purchasing for each individual
        sum_u_m = np.sum(u_m, axis = 0).reshape((N,1))
        ccp_m = (u_m.T / (sum_u_m + 1)).T
        ## aggregate individual choice by market
        share_m = ccp_m.sum(axis = 1)/N
        Y[m * J:(m+1)*J] = share_m
    
    st4 = time.time()
    #print('y', st4-st3)

    return Y


def rcl_regenerate_sin(X, data, N=10000):
    J = data['J']
    K = data['K']
    M = data['M']
    b_random = data['b_random']
    
    Total_J = J * M
    
    ## get the utility of each user, the output is (M * J) * N
    u_i = b_random[0] * np.ones((Total_J, N))

    for k in range(1,K+2):
        #if k < K/3:
        u_i = u_i + b_random[k] * (np.sin(X.iloc[:,k-1].to_numpy().reshape(Total_J,1)))
        #else:
        #    u_i = u_i + b_random[k] * (X.iloc[:,k-1].to_numpy().reshape(Total_J,1))
            
    exp_u_i = np.exp(u_i)
    
    Y = np.zeros((M* J))
    for m in range(M):
        ## get the sub-matrix (J * N) of this market
        u_m = exp_u_i[m * J:(m+1)*J,: ]
        ## calculate the probability of purchasing for each individual
        sum_u_m = np.sum(u_m, axis = 0).reshape((N,1))
        ccp_m = (u_m.T / (sum_u_m + 1)).T
        ## aggregate individual choice by market
        share_m = ccp_m.sum(axis = 1)/N
        Y[m * J:(m+1)*J] = share_m
    
    st4 = time.time()
    #print('y', st4-st3)

    return Y


def rcl_in3(X, params, J, K, M, seed, N=10000):
    ## consumer inattention
    # b are the parameters 
    J_list = [J] * M
    Total_J = J * M

    b = params[0]
    sigma = params[1]

    ## generate random coeffcient for each individual
    np.random.seed(seed)
    b_random= []
    
    for i in range(len(b)):
        b_random.append(np.repeat(np.random.normal(loc = b[i], scale = sigma[i], size = (M, N)), repeats = J, axis=0))
    #print('b_random', st2-st1)
    
    ## get the utility of each user, the output is (M * J) * N
    u_i = b_random[0] * np.ones((Total_J, N))

    for k in range(1,K+2):
        u_i = u_i + b_random[k] * (X.iloc[:,k-1].to_numpy().reshape(Total_J,1))

    exp_u_i = np.exp(u_i)
    
    u_df = pd.DataFrame(exp_u_i)
    market_id = market_id_gen(J, M)
    u_df['market_id'] = market_id
    u_df['price'] = X['price'].copy()
    
    ## highest =0
    u_df['highest'] = 1 - u_df.groupby(['market_id'], group_keys=False)['price'].apply(lambda x: x == x.max()) * 1
    
    highest_price = u_df.groupby(['market_id'])['price'].max()
    
    Y = np.zeros((M* J))
    avg_inattention = []
    for m in range(M):
        max_price = highest_price[m]
        if max_price < 0:
            inattention_share =  0
        elif 0 <= max_price <= 5:
            inattention_share = 1 - 1/(1+max_price) # 0.2 * max_price  ## 1/4 * max_price 
        else:
            inattention_share = 1
        avg_inattention.append(inattention_share)
        
        attention_share = 1 - inattention_share
        #print(max_price, attention_share)

        u_m = u_df.iloc[m * J:(m+1)*J,: ]
        ## 1 is attentioned consumer, 2 is the inattention consumer
        u_df_sum1 = u_m.iloc[:,list(range(int(attention_share*N))) + [-1, -2, -3]].groupby(['market_id']).transform('sum') + 1        
        u_df_sum2 = u_m.iloc[:,list(range(int(attention_share*N), N)) + [-1, -2, -3]].groupby(['market_id','highest']).transform('sum') + 1
        p1 = u_m.iloc[:,list(range(int(attention_share*N)))] / u_df_sum1.iloc[:,list(range(int(attention_share*N)))]
        p2 = u_m.iloc[:,list(range(int(attention_share*N), N))] / u_df_sum2.iloc[:,list(range(N-int(attention_share*N)))]
        p2 = p2.mul(u_m['highest'],axis =0)

        ## Get Y
        Y[m * J:(m+1)*J] = (np.sum(p1,axis = 1) + np.sum(p2,axis=1))/N
    
    data = {'X': X, 'Y': Y,  'M': M, "J": J, "K": K, 'params': params, 'generation_seed': seed, 'b_random':b_random,
            'inattention': np.mean(avg_inattention), 'market_id': market_id}
    return data




def rcl_mix(X, params, J, K, M, seed, N=10000):
    J_list = [J] * M
    Total_J = J * M

    b1 = params[0]
    b2 = params[1]
    sigma1 = params[2]
    sigma2 = params[3]
    pi = params[4]

    ## generate random coeffcient for each individual
    np.random.seed(seed)
    b_random= []
    
    ## sigma is the same for two normal distributions
    for i in range(len(b1)):
        pi_real = np.random.binomial(1, pi,size = (M, N))
        b_random.append(np.repeat(pi_real * np.random.normal(loc = b1[i], scale = sigma1[i], size = (M, N))
                                  + (1-pi_real) * np.random.normal(loc = b2[i], scale = sigma2[i], size = (M, N)), 
                                  repeats = J, axis=0))
    
    ## get the utility of each user, the output is (M * J) * N
    u_i = b_random[0] * np.ones((Total_J, N))

    for k in range(1,K+2):
        u_i = u_i + b_random[k] * (X.iloc[:,k-1].to_numpy().reshape(Total_J,1))

    exp_u_i = np.exp(u_i)
    u_m = exp_u_i.reshape(M, J, N)

    # Calculate the probability of purchasing for each individual
    sum_u_m = np.sum(u_m, axis=1, keepdims=True)
    ccp_m = (u_m / (sum_u_m + 1))

    # Aggregate individual choice by market
    share_m = np.sum(ccp_m, axis=2) / N

    # Flatten share_m from (M, J) to (M*J,)
    Y = share_m.flatten()
    
    market_id = market_id_gen(J, M)
    data = {'X': X, 'Y': Y,  'M': M, "J": J, "K": K, 'params': params, 'generation_seed': seed, 
            'b_random':b_random, 'market_id':market_id}

    return data


def data_generation(params, J, K, M, seed, x_to_y):
    X = feature_generation(J, K, M, seed)
    data = x_to_y(X, params, J, K, M, seed)
    #X_test = feature_generation(J, K, M, 1234)
    #data_test = x_to_y(X_test, params, J, K, M, 1234)
    
    #data['X_test'] = data_test['X'].copy()
    #data['Y_test'] = data_test['Y'].copy()
    return data



### fixed effects (fe) 
def feature_generation_fe(J, K, M, seed):
    # J is number of products per markets (assume each market has the same number of products but different products)
    # K is number of features (excluding price)
    # M is the number of markets (assume each market has different sets of products)
    np.random.seed(seed)

    Total_J = J * M
    x_1 = np.random.normal(loc=0, scale=1, size=(Total_J , K))
    price = np.random.uniform(0,4,size = Total_J)
    #price = np.abs(np.random.normal(loc=0, scale=1, size = Total_J))
    
    # Generate product dummies
    product_dummies = np.zeros((Total_J, J))
    for m in range(M):
        for j in range(J):
            product_dummies[m * J + j, j] = 1  # Set the corresponding dummy to 1 for each product in each market
    
    # Combine all features into a DataFrame
    X = pd.DataFrame(x_1)
    X = pd.concat([X, pd.DataFrame(product_dummies, columns=[f'product_{j+1}' for j in range(J)])], axis=1)
    
    # put price as the last feature 
    X['price'] = price
    
    return X 

def rcl_fe(X, params, J, K, M, seed, N=10000):
    J_list = [J] * M
    Total_J = J * M

    b = params[0]
    sigma = params[1]

    ## generate random coeffcient for each individual
    np.random.seed(seed)
    b_random= []
    
    st1 = time.time()
    ## intercept is still estimatable in RCL, because it is random effect 
    for i in range(K+1):
        b_random.append(np.repeat(np.random.normal(loc = b[i], scale = sigma[i], size = (M, N)), repeats = J, axis=0))
    
    ## fixed effect, we still use the random normal but just set scale to 0 
    for i in range(K+1, K+1+J):
        b_random.append(np.repeat(np.random.normal(loc = b[i], scale = 0, size = (M, N)), repeats = J, axis=0))
    
    ## random coefficient for price (the last feature in X) 
    b_random.append(np.repeat(np.random.normal(loc = b[-1], scale = sigma[-1], size = (M, N)), repeats = J, axis=0))
    
    ## get the utility of each user, the output is (M * J) * N
    u_i = b_random[0] * np.ones((Total_J, N))
    
    ## total number of coefficients are K+J+2
    for k in range(1,K+J+2):
        u_i = u_i + b_random[k] * (X.iloc[:,k-1].to_numpy().reshape(Total_J,1))

    exp_u_i = np.exp(u_i)
    u_m = exp_u_i.reshape(M, J, N)

    # Calculate the probability of purchasing for each individual
    sum_u_m = np.sum(u_m, axis=1, keepdims=True)
    ccp_m = (u_m / (sum_u_m + 1))

    # Aggregate individual choice by market
    share_m = np.sum(ccp_m, axis=2) / N

    # Flatten share_m from (M, J) to (M*J,)
    Y = share_m.flatten()
    
    market_id = market_id_gen(J, M)
    data = {'X': X, 'Y': Y,  'M': M, "J": J, "K": K, 'params': params, 'generation_seed': seed, 
            'b_random':b_random, 'market_id':market_id}

    return data


def mnl_choice_fe(X, params, J, K, M, seed, N=10000):
    J_list = [J] * M
    Total_J = J * M

    b = params

    ## generate random coeffcient for each individual
    np.random.seed(seed)
    b_random= []
    
    # original (except price)
    for i in range(K+1):
        b_random.append(np.repeat(np.ones((M, N)) *  b[i], repeats = J, axis=0))
        
    # fixed effects
    for i in range(K+1, K+1+J):
        b_random.append(np.repeat(np.ones((M, N)) *  b[i], repeats = J, axis=0))
        
    ## price
    b_random.append(np.repeat(np.ones((M, N)) *  b[-1], repeats = J, axis=0))
    
    
    ## get the utility of each user, the output is (M * J) * N
    u_i = b_random[0] * np.ones((Total_J, N))

    for k in range(1,K+2):
        u_i = u_i + b_random[k] * (X.iloc[:,k-1].to_numpy().reshape(Total_J,1))
        
    e_i = np.random.gumbel(0, 1, size=u_i.shape)
    u_i = u_i + e_i

    Y = np.zeros((M* J))
    for m in range(M):
        ## get the sub-matrix (J * N) of this market
        u_m = u_i[m * J:(m+1)*J,: ]
        u_m_max = np.max(u_m, axis = 0)
        choice = np.where(u_m_max > 0, np.argmax(u_m, axis = 0), J)
        share_m = np.bincount(choice, minlength=J) / N
        Y[m * J:(m+1)*J] = share_m[0:J]
    
    
    market_id = market_id_gen(J, M)
    data = {'X': X, 'Y': Y,  'M': M, "J": J, "K": K, 'params': params, 'generation_seed': seed, 
            'b_random':b_random, 'market_id':market_id,'e_random':e_i}

    return data


def mnl_choice_regenerate(X, data, N=10000):
    J = data['J']
    K = data['K']
    M = data['M']
    b_random = data['b_random']
    e_i = data['e_random']
    
    Total_J = J * M
    
    ## get the utility of each user, the output is (M * J) * N
    u_i = b_random[0] * np.ones((Total_J, N))

    for k in range(1,K+2):
        u_i = u_i + b_random[k] * (X.iloc[:,k-1].to_numpy().reshape(Total_J,1))

    u_i = u_i + e_i

    Y = np.zeros((M* J))
    for m in range(M):
        ## get the sub-matrix (J * N) of this market
        u_m = u_i[m * J:(m+1)*J,: ]
        u_m_max = np.max(u_m, axis = 0)
        choice = np.where(u_m_max > 0, np.argmax(u_m, axis = 0), J)
        share_m = np.bincount(choice, minlength=J) / N
        Y[m * J:(m+1)*J] = share_m[0:J]
    

    return Y


def rcl_log_fe(X, params, J, K, M, seed, N=10000):
    J_list = [J] * M
    Total_J = J * M

    b = params[0]
    sigma = params[1]

    ## generate random coeffcient for each individual
    np.random.seed(seed)
    b_random= []
    
    st1 = time.time()
    
    ### generate b_random 
    ## intercept is still estimatable in RCL, because it is random effect 
    for i in range(K+1):
        b_random.append(np.repeat(np.random.normal(loc = b[i], scale = sigma[i], size = (M, N)), repeats = J, axis=0))
    
    ## fixed effect, we still use the random normal but just set scale to 0 
    for i in range(K+1, K+1+J):
        b_random.append(np.repeat(np.random.normal(loc = b[i], scale = 0, size = (M, N)), repeats = J, axis=0))
    
    ## random coefficient for price (the last feature in X) 
    b_random.append(np.repeat(np.random.normal(loc = b[-1], scale = sigma[-1], size = (M, N)), repeats = J, axis=0))
    
    
    ## get the utility of each user, the output is (M * J) * N
    u_i = b_random[0] * np.ones((Total_J, N))
    
    ## non-linear features (non-price)
    for k in range(1,K+1):
        u_i = u_i + b_random[k] * (np.log( np.abs(16*X.iloc[:,k-1].to_numpy().reshape(Total_J,1) - 8) + 1 ) * np.sign(16*X.iloc[:,k-1].to_numpy().reshape(Total_J,1) - 8))
        
    ## fixed effects 
    for k in range(K+1,K+J+1):
        u_i = u_i + b_random[k] * (X.iloc[:,k-1].to_numpy().reshape(Total_J,1))
    
    ## non-linear feature (Price)
    for k in range(K+J+1,K+J+2):
        u_i = u_i + b_random[k] * (np.log( np.abs(16*X.iloc[:,k-1].to_numpy().reshape(Total_J,1) - 8) + 1 ) * np.sign(16*X.iloc[:,k-1].to_numpy().reshape(Total_J,1) - 8))

    exp_u_i = np.exp(u_i)
    u_m = exp_u_i.reshape(M, J, N)

    # Calculate the probability of purchasing for each individual
    sum_u_m = np.sum(u_m, axis=1, keepdims=True)
    ccp_m = (u_m / (sum_u_m + 1))

    # Aggregate individual choice by market
    share_m = np.sum(ccp_m, axis=2) / N

    # Flatten share_m from (M, J) to (M*J,)
    Y = share_m.flatten()
    
    market_id = market_id_gen(J, M)
    data = {'X': X, 'Y': Y,  'M': M, "J": J, "K": K, 'params': params, 'generation_seed': seed, 
            'b_random':b_random, 'market_id':market_id}

    return data

def rcl_regenerate_log_fe(X, data, N=10000):
    J = data['J']
    K = data['K'] - J
    M = data['M']
    b_random = data['b_random']
    
    Total_J = J * M
    
    ## get the utility of each user, the output is (M * J) * N
    u_i = b_random[0] * np.ones((Total_J, N))
    
    ## non-linear features (non-price)
    for k in range(1,K+1):
        u_i = u_i + b_random[k] * (np.log( np.abs(16*X.iloc[:,k-1].to_numpy().reshape(Total_J,1) - 8) + 1 ) * np.sign(16*X.iloc[:,k-1].to_numpy().reshape(Total_J,1) - 8))
        
    ## fixed effects 
    for k in range(K+1,K+J+1):
        u_i = u_i + b_random[k] * (X.iloc[:,k-1].to_numpy().reshape(Total_J,1))
    
    ## non-linear feature (Price)
    for k in range(K+J+1,K+J+2):
        u_i = u_i + b_random[k] * (np.log( np.abs(16*X.iloc[:,k-1].to_numpy().reshape(Total_J,1) - 8) + 1 ) * np.sign(16*X.iloc[:,k-1].to_numpy().reshape(Total_J,1) - 8))
       
    exp_u_i = np.exp(u_i)
    u_m = exp_u_i.reshape(M, J, N)

    # Calculate the probability of purchasing for each individual
    sum_u_m = np.sum(u_m, axis=1, keepdims=True)
    ccp_m = (u_m / (sum_u_m + 1))

    # Aggregate individual choice by market
    share_m = np.sum(ccp_m, axis=2) / N

    # Flatten share_m from (M, J) to (M*J,)
    Y = share_m.flatten()
    
    
    return Y

def rcl_sin_fe(X, params, J, K, M, seed, N=10000):
    J_list = [J] * M
    Total_J = J * M

    b = params[0]
    sigma = params[1]

    ## generate random coeffcient for each individual
    np.random.seed(seed)
    b_random= []
        
    ### generate b_random 
    ## intercept is still estimatable in RCL, because it is random effect 
    for i in range(K+1):
        b_random.append(np.repeat(np.random.normal(loc = b[i], scale = sigma[i], size = (M, N)), repeats = J, axis=0))
    
    ## fixed effect, we still use the random normal but just set scale to 0 
    for i in range(K+1, K+1+J):
        b_random.append(np.repeat(np.random.normal(loc = b[i], scale = 0, size = (M, N)), repeats = J, axis=0))
    
    ## random coefficient for price (the last feature in X) 
    b_random.append(np.repeat(np.random.normal(loc = b[-1], scale = sigma[-1], size = (M, N)), repeats = J, axis=0))
    
    
    ## get the utility of each user, the output is (M * J) * N
    u_i = b_random[0] * np.ones((Total_J, N))
    
    for k in range(1,K+1):
        u_i = u_i + b_random[k] * (np.sin(X.iloc[:,k-1].to_numpy().reshape(Total_J,1)))
    ## fixed effects 
    for k in range(K+1,K+J+1):
        u_i = u_i + b_random[k] * (X.iloc[:,k-1].to_numpy().reshape(Total_J,1))
    
    ## non-linear feature (Price)
    for k in range(K+J+1,K+J+2):
        u_i = u_i + b_random[k] * (np.sin(X.iloc[:,k-1].to_numpy().reshape(Total_J,1)))
    
    exp_u_i = np.exp(u_i)
    u_m = exp_u_i.reshape(M, J, N)

    # Calculate the probability of purchasing for each individual
    sum_u_m = np.sum(u_m, axis=1, keepdims=True)
    ccp_m = (u_m / (sum_u_m + 1))

    # Aggregate individual choice by market
    share_m = np.sum(ccp_m, axis=2) / N

    # Flatten share_m from (M, J) to (M*J,)
    Y = share_m.flatten()
    
    market_id = market_id_gen(J, M)
    data = {'X': X, 'Y': Y,  'M': M, "J": J, "K": K, 'params': params, 'generation_seed': seed, 
            'b_random':b_random, 'market_id':market_id}

    return data


def rcl_regenerate_sin_fe(X, data, N=10000):
    J = data['J']
    K = data['K'] - J 
    M = data['M']
    b_random = data['b_random']
    
    Total_J = J * M
    
    ## get the utility of each user, the output is (M * J) * N
    u_i = b_random[0] * np.ones((Total_J, N))
    
    ## non-linear features (non-price)
    for k in range(1,K+1):
        u_i = u_i + b_random[k] * (np.sin(X.iloc[:,k-1].to_numpy().reshape(Total_J,1)))
    ## fixed effects 
    for k in range(K+1,K+J+1):
        u_i = u_i + b_random[k] * (X.iloc[:,k-1].to_numpy().reshape(Total_J,1))
    
    ## non-linear feature (Price)
    for k in range(K+J+1,K+J+2):
        u_i = u_i + b_random[k] * (np.sin(X.iloc[:,k-1].to_numpy().reshape(Total_J,1)))
    
    exp_u_i = np.exp(u_i)
    u_m = exp_u_i.reshape(M, J, N)

    # Calculate the probability of purchasing for each individual
    sum_u_m = np.sum(u_m, axis=1, keepdims=True)
    ccp_m = (u_m / (sum_u_m + 1))

    # Aggregate individual choice by market
    share_m = np.sum(ccp_m, axis=2) / N

    # Flatten share_m from (M, J) to (M*J,)
    Y = share_m.flatten()
    
    
    return Y


def rcl_in3_fe(X, params, J, K, M, seed, N=10000):
    ## consumer inattention
    # seed is not used in this function
    # b are the parameters 
    J_list = [J] * M
    Total_J = J * M

    b = params[0]
    sigma = params[1]

    ## generate random coeffcient for each individual
    np.random.seed(seed)
    b_random= []
    
    ## intercept is still estimatable in RCL, because it is random effect 
    for i in range(K+1):
        b_random.append(np.repeat(np.random.normal(loc = b[i], scale = sigma[i], size = (M, N)), repeats = J, axis=0))
    
    ## fixed effect, we still use the random normal but just set scale to 0 
    for i in range(K+1, K+1+J):
        b_random.append(np.repeat(np.random.normal(loc = b[i], scale = 0, size = (M, N)), repeats = J, axis=0))
    
    ## random coefficient for price (the last feature in X) 
    b_random.append(np.repeat(np.random.normal(loc = b[-1], scale = sigma[-1], size = (M, N)), repeats = J, axis=0))
    
    ## get the utility of each user, the output is (M * J) * N
    u_i = b_random[0] * np.ones((Total_J, N))
    
    ## total number of coefficients are K+J+2
    for k in range(1,K+J+2):
        u_i = u_i + b_random[k] * (X.iloc[:,k-1].to_numpy().reshape(Total_J,1))


    exp_u_i = np.exp(u_i)
    
    u_df = pd.DataFrame(exp_u_i)
    market_id = market_id_gen(J, M)
    u_df['market_id'] = market_id
    u_df['price'] = X['price'].copy()
    
    ## highest =0
    u_df['highest'] = 1 - u_df.groupby(['market_id'], group_keys=False)['price'].apply(lambda x: x == x.max()) * 1
    
    highest_price = u_df.groupby(['market_id'])['price'].max()
    
    Y = np.zeros((M* J))
    avg_inattention = []
    for m in range(M):
        max_price = highest_price[m]
        if max_price < 0:
            inattention_share =  0
        elif 0 <= max_price <= 5:
            inattention_share = 1 - 1/(1+max_price) # 0.2 * max_price  ## 1/4 * max_price 
        else:
            inattention_share = 1
        avg_inattention.append(inattention_share)
        
        attention_share = 1 - inattention_share
        #print(max_price, attention_share)

        u_m = u_df.iloc[m * J:(m+1)*J,: ]
        ## 1 is attentioned consumer, 2 is the inattention consumer
        u_df_sum1 = u_m.iloc[:,list(range(int(attention_share*N))) + [-1, -2, -3]].groupby(['market_id']).transform('sum') + 1        
        u_df_sum2 = u_m.iloc[:,list(range(int(attention_share*N), N)) + [-1, -2, -3]].groupby(['market_id','highest']).transform('sum') + 1
        p1 = u_m.iloc[:,list(range(int(attention_share*N)))] / u_df_sum1.iloc[:,list(range(int(attention_share*N)))]
        p2 = u_m.iloc[:,list(range(int(attention_share*N), N))] / u_df_sum2.iloc[:,list(range(N-int(attention_share*N)))]
        p2 = p2.mul(u_m['highest'],axis =0)

        ## Get Y
        Y[m * J:(m+1)*J] = (np.sum(p1,axis = 1) + np.sum(p2,axis=1))/N
    
    data = {'X': X, 'Y': Y,  'M': M, "J": J, "K": K, 'params': params, 'generation_seed': seed, 'b_random':b_random,
            'inattention': np.mean(avg_inattention), 'market_id': market_id}
    return data






def data_generation_fe(params, J, K, M, seed, x_to_y):
    X = feature_generation_fe(J, K, M, seed)
    data = x_to_y(X, params, J, K, M, seed)
    data['K'] = K + J
    
    return data



def feature_generation_keepprice(J, K, M, seed):
    # J is number of products per markets (assume each market has the same number of products but different products)
    # K is number of features (excluding price)
    # M is the number of markets (assume each market has different sets of products)
    np.random.seed(seed)

    Total_J = J * M
    x_1 = np.random.normal(loc=0, scale=1, size=(Total_J , K))
    price = np.tile(np.random.uniform(0,4,size = int(Total_J/5)), 5)
    
    #price = np.abs(np.random.normal(loc=0, scale=1, size = Total_J))
    
    X = pd.DataFrame(x_1)
    X['price'] = price
    
    return X 

def data_generation_keepprice(params, J, K, M, seed, x_to_y):
    X = feature_generation_keepprice(J, K, M, seed)
    data = x_to_y(X, params, J, K, M, seed)
    #X_test = feature_generation(J, K, M, 1234)
    #data_test = x_to_y(X_test, params, J, K, M, 1234)
    
    #data['X_test'] = data_test['X'].copy()
    #data['Y_test'] = data_test['Y'].copy()
    return data



def feature_generation_fix1(J, K, M, seed):
    # J is number of products per markets (assume each market has the same number of products but different products)
    # K is number of features (excluding price)
    # M is the number of markets (assume each market has different sets of products)
    np.random.seed(seed)

    Total_J = J * M
    x_1 = np.random.normal(loc=0, scale=1, size=(Total_J , K))
    price = np.random.uniform(0,4,size = Total_J)
    
    #price = np.abs(np.random.normal(loc=0, scale=1, size = Total_J))
    
    X = pd.DataFrame(x_1)
    X['price'] = price
    
    fixed_values = X.iloc[0].copy()
    
    # Fix features and price of the first product in each market
    for m in range(M):
        idx = m * J  # index of the first product in market m
        X.iloc[idx, :-1] = fixed_values[:-1]  # only set features, not price

    return X 


def data_generation_fix1(params, J, K, M, seed, x_to_y):
    X = feature_generation_fix1(J, K, M, seed)
    data = x_to_y(X, params, J, K, M, seed)
    #X_test = feature_generation(J, K, M, 1234)
    #data_test = x_to_y(X_test, params, J, K, M, 1234)
    
    #data['X_test'] = data_test['X'].copy()
    #data['Y_test'] = data_test['Y'].copy()
    return data
