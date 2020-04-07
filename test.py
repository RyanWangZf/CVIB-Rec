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

from matrix_factorization import NCF, NCF_VITA, NCF_IPS, NCF_SNIPS

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


# "NCF IPS"
# ncf_ips = NCF_IPS(num_user, num_item)

# ips_idxs = np.arange(len(y_test))
# np.random.shuffle(ips_idxs)
# y_ips = y_test[ips_idxs[:int(0.05 * len(ips_idxs))]]

# ncf_ips.fit(x_train, y_train, 
#     y_ips=y_ips,
#     lr=0.01,
#     batch_size=2048,
#     lamb=1e-4,tol=1e-6, verbose=1)

# test_pred = ncf_ips.predict(x_test)
# mse_ncfips = mse_func(y_test, test_pred)
# auc_ncfips = roc_auc_score(y_test, test_pred)
# print("***"*5 + "[NCF-IPS]" + "***"*5)
# print("[NCF-IPS] test mse:", mse_ncfips)
# print("[NCF-IPS] test auc:", auc_ncfips)
# user_wise_ctr = get_user_wise_ctr(x_test,y_test,test_pred)
# gi,gu = gini_index(user_wise_ctr)
# print("***"*5 + "[NCF-IPS]" + "***"*5)

# "NCF SNIPS"
# ncf_snips = NCF_SNIPS(num_user, num_item)

# ips_idxs = np.arange(len(y_test))
# np.random.shuffle(ips_idxs)
# y_ips = y_test[ips_idxs[:int(0.05 * len(ips_idxs))]]

# ncf_snips.fit(x_train, y_train, 
#     y_ips=y_ips,
#     lr=0.01,
#     batch_size=2048,
#     lamb=1e-4,tol=1e-6, verbose=1)

# test_pred = ncf_snips.predict(x_test)
# mse_ncfips = mse_func(y_test, test_pred)
# auc_ncfips = roc_auc_score(y_test, test_pred)
# print("***"*5 + "[NCF-SNIPS]" + "***"*5)
# print("[NCF-SNIPS] test mse:", mse_ncfips)
# print("[NCF-SNIPS] test auc:", auc_ncfips)
# user_wise_ctr = get_user_wise_ctr(x_test,y_test,test_pred)
# gi,gu = gini_index(user_wise_ctr)
# print("***"*5 + "[NCF-SNIPS]" + "***"*5)

"NCF VITA"
ncf_vita = NCF_VITA(num_user, num_item)
ncf_vita.fit(x_train, y_train, lr=0.01, 
    alpha=1e-3, gamma=1e-3, lamb=1e-4, tol=1e-6, 
    batch_size = 2048, verbose=1)

test_pred = ncf_vita.predict(x_test)
mse_ncf_vita = mse_func(y_test, test_pred)
auc_ncf_vita = roc_auc_score(y_test, test_pred)
print("***"*5 + "[NCF-VITA]" + "***"*5)
print("[NCF] test mse:", mse_func(y_test, test_pred))
print("[NCF] test auc:", auc_ncf_vita)
user_wise_ctr = get_user_wise_ctr(x_test,y_test,test_pred)
gi,gu = gini_index(user_wise_ctr)
print("***"*5 + "[NCF-VITA]" + "***"*5)


# "NCF naive"
# ncf = NCF(num_user, num_item)
# ncf.fit(x_train, y_train, lr=0.01, lamb=1e-4, tol=1e-6, 
#     batch_size = 2048, verbose=1)
# test_pred = ncf.predict(x_test)
# mse_ncf = mse_func(y_test, test_pred)
# auc_ncf = roc_auc_score(y_test, test_pred)
# print("***"*5 + "[NCF]" + "***"*5)
# print("[NCF] test mse:", mse_func(y_test, test_pred))
# print("[NCF] test auc:", auc_ncf)
# user_wise_ctr = get_user_wise_ctr(x_test,y_test,test_pred)
# gi,gu = gini_index(user_wise_ctr)
# print("***"*5 + "[NCF]" + "***"*5)


pdb.set_trace()

pass
