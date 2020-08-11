# -*- coding: utf-8 -*-
import numpy as np
import torch
import pdb
from sklearn.metrics import roc_auc_score
np.random.seed(2020)
torch.manual_seed(2020)
import pdb

from dataset import load_data
from matrix_factorization import MF, MF_CVIB, MF_IPS, MF_SNIPS

from matrix_factorization import NCF, NCF_CVIB, NCF_IPS, NCF_SNIPS

from matrix_factorization import MF_DR, NCF_DR

from utils import gini_index, get_user_wise_ctr, rating_mat_to_sample, binarize, shuffle, minU, ndcg_func
mse_func = lambda x,y: np.mean((x-y)**2)
acc_func = lambda x,y: np.sum(x == y) / len(x)

# dataset_name = "coat"
dataset_name = "yahoo"

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


# "NCF CVIB"
# ncf_cvib = NCF_CVIB(num_user, num_item)
# ncf_cvib.fit(x_train, y_train, lr=0.05, 
#     alpha=1, gamma=1e-5, lamb=1e-3, tol=1e-6, 
#     batch_size = 512, verbose=1)

# test_pred = ncf_cvib.predict(x_test)
# mse_ncf = mse_func(y_test, test_pred)
# auc_ncf = roc_auc_score(y_test, test_pred)
# ndcg_res = ndcg_func(ncf_cvib, x_test, y_test)

# print("***"*5 + "[NCF-CVIB]" + "***"*5)
# print("[NCF-CVIB] test mse:",mse_ncf)
# print("[NCF-CVIB] test auc:", auc_ncf)
# print("ndcg@5:{:.6f}, ndcg@10:{:.6f}".format(
#     np.mean(ndcg_res["ndcg_5"]), np.mean(ndcg_res["ndcg_10"])))
# user_wise_ctr = get_user_wise_ctr(x_test,y_test,test_pred)
# gi,gu = gini_index(user_wise_ctr)
# print("***"*5 + "[NCF-CVIB]" + "***"*5)

"NCF CVIB"
ncf_cvib = NCF_CVIB(num_user, num_item)
ncf_cvib.fit(x_train, y_train, lr=0.01, 
    alpha=1.0, gamma=1e-2, lamb=1e-4, tol=1e-6, 
    batch_size = 2048, verbose=1)

test_pred = ncf_cvib.predict(x_test)
mse_ncf = mse_func(y_test, test_pred)
auc_ncf = roc_auc_score(y_test, test_pred)
ndcg_res = ndcg_func(ncf_cvib, x_test, y_test)

print("***"*5 + "[NCF-CVIB]" + "***"*5)
print("[NCF-CVIB] test mse:", mse_ncf)
print("[NCF-CVIB] test auc:", auc_ncf)
print("ndcg@5:{:.6f}, ndcg@10:{:.6f}".format(
    np.mean(ndcg_res["ndcg_5"]), np.mean(ndcg_res["ndcg_10"])))
user_wise_ctr = get_user_wise_ctr(x_test,y_test,test_pred)
gi,gu = gini_index(user_wise_ctr)
print("***"*5 + "[NCF-CVIB]" + "***"*5)

pdb.set_trace()

pass
