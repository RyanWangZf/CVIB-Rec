# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import torch
import pdb
import os
from sklearn.metrics import roc_auc_score
np.random.seed(2020)
torch.manual_seed(2020)

from dataset import load_data
from matrix_factorization import MF, MF_CVIB, MF_IPS, MF_SNIPS

from matrix_factorization import NCF, NCF_CVIB, NCF_IPS, NCF_SNIPS

from matrix_factorization import MF_DR, NCF_DR

from utils import gini_index, get_user_wise_ctr, rating_mat_to_sample, binarize, shuffle, minU
mse_func = lambda x,y: np.mean((x-y)**2)
acc_func = lambda x,y: np.sum(x == y) / len(x)

dataset_name = "coat"
# dataset_name = "yahoo"

if dataset_name == "coat":
    train_mat, test_mat = load_data("coat")        
    x_train, y_train = rating_mat_to_sample(train_mat)
    x_test, y_test = rating_mat_to_sample(test_mat)
    num_user = train_mat.shape[0]
    num_item = train_mat.shape[1]

elif dataset_name == "yahoo":
    x_train, y_train, x_test, y_test = load_data("yahoo")
    x_train, y_train = shuffle(x_train, y_train)
    num_user = x_train[:,0].max() + 1
    num_item = x_train[:,1].max() + 1

print("# user: {}, # item: {}".format(num_user, num_item))

# binarize
y_train = binarize(y_train)
y_test = binarize(y_test)


def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

"MF-CVIB parameter invariance"
output_prefix = "./demo/"

# alpha_list = [0.1,1e-2,1e-3,1e-4]
# gamma_list = [1,0.1,1e-2,1e-3,1e-4]
alpha_list = [2,1,0.5,0.1]
gamma_list = [1,0.1,1e-2,1e-3]

alpha_result = []
col_name = []
for alpha in alpha_list:
    auc_result = []
    for j in range(10):
        setup_seed(2020+j)
        mf_cvib = MF_CVIB(num_user, num_item)
        mf_cvib.fit(x_train, y_train,
            lr=0.01,
            batch_size=128,
            lamb=1e-4,
            alpha=alpha,
            gamma=1e-3,
            tol=1e-5,
            verbose=False)

        test_pred = mf_cvib.predict(x_test)
        auc_mf = roc_auc_score(y_test, test_pred)
        print("{} [alpha]: {} [gamma]: {} auc: {}".format(j,alpha, 1e-3, auc_mf))
        auc_result.append(auc_mf)
    alpha_result.append(auc_result)

output_name = os.path.join(output_prefix,"mf_{}_alpha.csv".format(dataset_name))
df = pd.DataFrame(np.array(alpha_result).T)
df.columns = alpha_list
df.to_csv(output_name)
print("Done.")