# -*- coding: utf-8 -*-
import numpy as np
import torch
import pdb
from sklearn.metrics import roc_auc_score
np.random.seed(2020)
torch.manual_seed(2020)
import pdb

from dataset import load_data
from matrix_factorization import MF, MF_VITA, MF_IPS, MF_SNIPS
from utils import gini_index, get_user_wise_ctr, rating_mat_to_sample, binarize, shuffle, minU
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


"MF-SNIPS"
mf_snips = MF_SNIPS(num_user, num_item)

ips_idxs = np.arange(len(y_test))
np.random.shuffle(ips_idxs)
y_ips = y_test[ips_idxs[:int(0.05 * len(ips_idxs))]]

mf_snips.fit(x_train, y_train, y_ips=y_ips,lr=0.05,
        batch_size=2048, lamb=1e-5,
        verbose=1)

test_pred = mf_snips.predict(x_test)
mse_mfsnips = mse_func(y_test, test_pred)
auc_mfsnips = roc_auc_score(y_test, test_pred)
print("***"*5 + "[MF-SNIPS]" + "***"*5)
print("[MF-SNIPS] test mse:", mse_mfsnips)
print("[MF-SNIPS] test auc:", auc_mfsnips)
user_wise_ctr = get_user_wise_ctr(x_test,y_test,test_pred)
gi,gu = gini_index(user_wise_ctr)
print("***"*5 + "[MF-SNIPS]" + "***"*5)


# "MF naive"
# mf = MF(num_user, num_item)
# mf.fit(x_train, y_train, lr=0.01, batch_size=2048, lamb=1e-4,
#     verbose=1)
# test_pred = mf.predict(x_test)
# mse_mf = mse_func(y_test, test_pred)
# auc_mf = roc_auc_score(y_test, test_pred)
# print("***"*5 + "[MF]" + "***"*5)
# print("[MF] test mse:", mse_func(y_test, test_pred))
# print("[MF] test auc:", auc_mf)
# user_wise_ctr = get_user_wise_ctr(x_test,y_test,test_pred)
# gi,gu = gini_index(user_wise_ctr)
# print("***"*5 + "[MF]" + "***"*5)


# "MF VITA"
# mf_vita = MF_VITA(num_user, num_item)
# mf_vita.fit(x_train, y_train, 
#     lr=0.01,
#     batch_size=2048,
#     lamb=1e-5,
#     alpha=0.1,
#     tol=1e-6,
#     verbose=True)

# test_pred = mf_vita.predict(x_test)
# mse_mf = mse_func(y_test, test_pred)
# auc_mf = roc_auc_score(y_test, test_pred)
# print("***"*5 + "[MF-VITA]" + "***"*5)
# print("[MF-VITA] test mse:", mse_func(y_test, test_pred))
# print("[MF-VITA] test auc:", auc_mf)
# user_wise_ctr = get_user_wise_ctr(x_test,y_test,test_pred)
# gi,gu = gini_index(user_wise_ctr)
# print("***"*5 + "[MF-VITA]" + "***"*5)


# "MF VITA"
# mf_vita = MF_VITA(num_user, num_item)

# # This parameter is for the COAT.
# mf_vita.fit(x_train, y_train, 
#     lr=0.01,
#     batch_size=128,
#     lamb=1e-4,
#     alpha=0.1,
#     tol=1e-5,
#     verbose=True)

# mf_vita.fit(x_train, y_train, 
#     lr=0.01,
#     batch_size=128,
#     lamb=1e-4,
#     alpha=0.1,
#     tol=1e-5,
#     verbose=True)

# This is for YAHOO MF-VITA
# mf_vita.fit(x_train, y_train,
#     lr=0.01,
#     batch_size=2048,
#     lamb=1e-5,
#     alpha=0.1,
#     tol=1e-6,
#     verbose=True)

# This is for YAHOO MF-Naive
# mf_vita.fit(x_train, y_train, 
#     lr=0.01,
#     batch_size=2048,
#     lamb=1e-5,
#     alpha=0.0,
#     tol=1e-6,
#     verbose=True)

# test_pred = mf_vita.predict(x_test)
# mse_mf = mse_func(y_test, test_pred)
# auc_mf = roc_auc_score(y_test, test_pred)
# print("***"*5 + "[MF-VITA]" + "***"*5)
# print("[MF-VITA] test mse:", mse_func(y_test, test_pred))
# print("[MF-VITA] test auc:", auc_mf)
# user_wise_ctr = get_user_wise_ctr(x_test,y_test,test_pred)
# gi,gu = gini_index(user_wise_ctr)
# print("***"*5 + "[MF-VITA]" + "***"*5)

# "MF-SNIPS"
# mf_snips = MF_SNIPS(num_user, num_item)

# ips_idxs = np.arange(len(y_test))
# np.random.shuffle(ips_idxs)
# y_ips = y_test[ips_idxs[:int(0.05 * len(ips_idxs))]]

# mf_snips.fit(x_train, y_train, y_ips=y_ips, lr=0.05, lamb=1e-4,verbose=0)
# test_pred = mf_snips.predict(x_test)
# mse_mfsnips = mse_func(y_test, test_pred)
# auc_mfsnips = roc_auc_score(y_test, test_pred)
# print("***"*5 + "[MF-SNIPS]" + "***"*5)
# print("[MF-SNIPS] test mse:", mse_mfsnips)
# print("[MF-SNIPS] test auc:", auc_mfsnips)
# user_wise_ctr = get_user_wise_ctr(x_test,y_test,test_pred)
# gi,gu = gini_index(user_wise_ctr)
# print("***"*5 + "[MF-SNIPS]" + "***"*5)

pdb.set_trace()

pass
