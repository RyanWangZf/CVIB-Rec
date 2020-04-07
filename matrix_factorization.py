# -*- coding: utf-8 -*-
import numpy as np
import torch
torch.manual_seed(2020)
from torch import nn
import torch.nn.functional as F

import pdb

def generate_total_sample(num_user, num_item):
    sample = []
    for i in range(num_user):
        sample.extend([[i,j] for j in range(num_item)])
    return np.array(sample)

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

class MF_VITA(nn.Module):
    def __init__(self, num_users, num_items, embedding_k=4, *args, **kwargs):
        super(MF_VITA, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k)

        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def forward(self, x, is_training=False):
        user_idx = torch.LongTensor(x[:,0])
        item_idx = torch.LongTensor(x[:,1])
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)

        out = torch.sum(U_emb.mul(V_emb), 1)

        if is_training:
            return out, U_emb, V_emb
        else:
            return out

    def fit(self, x, y, 
        num_epoch=1000, batch_size=128, lr=0.05, lamb=0, 
        alpha=0.1,
        tol=1e-4, verbose=True):

        self.alpha = alpha

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=lamb)
        last_loss = 1e9

        # generate all counterfactuals and factuals for info reg
        x_all = generate_total_sample(self.num_users, self.num_items)

        num_sample = len(x)
        total_batch = num_sample // batch_size

        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)

            # sampling counterfactuals
            ul_idxs = np.arange(x_all.shape[0])
            np.random.shuffle(ul_idxs)

            epoch_loss = 0
            for idx in range(total_batch):
                # mini-batch training
                selected_idx = all_idx[batch_size*idx:(idx+1)*batch_size]
                sub_x = x[selected_idx]
                sub_y = y[selected_idx]
                sub_y = torch.Tensor(sub_y)

                pred, u_emb, v_emb = self.forward(sub_x, True)
                pred = self.sigmoid(pred)
                xent_loss = self.xent_func(pred,sub_y)

                # pair wise loss
                x_sampled = x_all[ul_idxs[idx* batch_size:(idx+1)*batch_size]]

                pred_ul,_,_ = self.forward(x_sampled, True)
                pred_ul = self.sigmoid(pred_ul)

                # log_p = sub_y.log()
                # log_pul = F.log_softmax(pred_ul, dim=0)
                
                logp_hat = F.log_softmax(pred, dim=0)
                info_loss = torch.mean(-pred_ul * logp_hat) + 0.01 * torch.mean(pred * logp_hat)

                # info_loss = torch.mean((pred_ul-pred)**2)

                loss = xent_loss + self.alpha * info_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += xent_loss.detach().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                print("[MF-VITA] epoch:{}, xent:{}".format(epoch, epoch_loss))
                break
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-VITA] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-VITA] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        pred = self.forward(x)
        pred = self.sigmoid(pred)
        return pred.detach().numpy()


class MF(nn.Module):
    def __init__(self, num_users, num_items, embedding_k=4, *args, **kwargs):
        super(MF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k)

        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def forward(self, x, is_training=False):
        user_idx = torch.LongTensor(x[:,0])
        item_idx = torch.LongTensor(x[:,1])
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)

        out = torch.sum(U_emb.mul(V_emb), 1)

        if is_training:
            return out, U_emb, V_emb
        else:
            return out

    def fit(self, x, y, 
        num_epoch=1000, batch_size=128, lr=0.05, lamb=0, 
        tol=1e-4, verbose=True):

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=lamb)
        last_loss = 1e9

        num_sample = len(x)
        total_batch = num_sample // batch_size

        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)
            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                selected_idx = all_idx[batch_size*idx:(idx+1)*batch_size]
                sub_x = x[selected_idx]
                sub_y = y[selected_idx]
                sub_y = torch.Tensor(sub_y)

                pred, u_emb, v_emb = self.forward(sub_x, True)
                pred = self.sigmoid(pred)
                xent_loss = self.xent_func(pred,sub_y)

                loss = xent_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += xent_loss.detach().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                print("[MF] epoch:{}, xent:{}".format(epoch, epoch_loss))
                break
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        pred = self.forward(x)
        pred = self.sigmoid(pred)
        return pred.detach().numpy()


class MF_IPS(nn.Module):
    def __init__(self, num_users, num_items, embedding_k=4, *args, **kwargs):
        super(MF_IPS, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k)

        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def forward(self, x, is_training=False):
        user_idx = torch.LongTensor(x[:,0])
        item_idx = torch.LongTensor(x[:,1])
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)

        out = torch.sum(U_emb.mul(V_emb), 1)

        if is_training:
            return out, U_emb, V_emb
        else:
            return out

    def fit(self, x, y, y_ips=None,
        num_epoch=1000, batch_size=128, lr=0.05, lamb=0, 
        tol=1e-4, verbose=True):

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=lamb)
        last_loss = 1e9

        num_sample = len(x)
        total_batch = num_sample // batch_size


        if y_ips is None:
            one_over_zl = self._compute_IPS(x, y)
        else:
            one_over_zl = self._compute_IPS(x, y, y_ips)

        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)
            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                selected_idx = all_idx[batch_size*idx:(idx+1)*batch_size]
                sub_x = x[selected_idx]
                sub_y = y[selected_idx]

                # propensity score
                inv_prop = one_over_zl[selected_idx]

                sub_y = torch.Tensor(sub_y)

                pred, u_emb, v_emb = self.forward(sub_x, True)
                pred = self.sigmoid(pred)

                xent_loss = F.binary_cross_entropy(pred, sub_y,
                    weight=inv_prop)

                loss = xent_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += xent_loss.detach().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                print("[MF-IPS] epoch:{}, xent:{}".format(epoch, epoch_loss))
                break
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-IPS] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-IPS] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        pred = self.forward(x)
        pred = self.sigmoid(pred)
        return pred.detach().numpy()

    def _compute_IPS(self,x,y,y_ips=None):
        if y_ips is None:
            one_over_zl = np.ones(len(y))
        else:
            py1 = y_ips.sum() / len(y_ips)
            py0 = 1 - py1
            po1 = len(x) / (x[:,0].max() * x[:,1].max())
            py1o1 = y.sum() / len(y)
            py0o1 = 1 - py1o1

            propensity = np.zeros(len(y))

            propensity[y == 0] = (py0o1 * po1) / py0
            propensity[y == 1] = (py1o1 * po1) / py1
            one_over_zl = 1 / propensity

        one_over_zl = torch.Tensor(one_over_zl)
        return one_over_zl


class MF_SNIPS(nn.Module):
    def __init__(self, num_users, num_items, embedding_k=4, *args, **kwargs):
        super(MF_SNIPS, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k)

        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def forward(self, x, is_training=False):
        user_idx = torch.LongTensor(x[:,0])
        item_idx = torch.LongTensor(x[:,1])
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)

        out = torch.sum(U_emb.mul(V_emb), 1)

        if is_training:
            return out, U_emb, V_emb
        else:
            return out

    def fit(self, x, y, y_ips=None,
        num_epoch=1000, batch_size=128, lr=0.05, lamb=0, 
        tol=1e-4, verbose=True):

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=lamb)
        last_loss = 1e9

        num_sample = len(x)
        total_batch = num_sample // batch_size


        if y_ips is None:
            one_over_zl = self._compute_IPS(x, y)
        else:
            one_over_zl = self._compute_IPS(x, y, y_ips)

        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)
            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                selected_idx = all_idx[batch_size*idx:(idx+1)*batch_size]
                sub_x = x[selected_idx]
                sub_y = y[selected_idx]

                # propensity score
                inv_prop = one_over_zl[selected_idx]
                sum_inv_prop = torch.sum(inv_prop)

                sub_y = torch.Tensor(sub_y)

                pred, u_emb, v_emb = self.forward(sub_x, True)
                pred = self.sigmoid(pred)
                

                xent_loss = F.binary_cross_entropy(pred, sub_y,
                    weight=inv_prop, reduction="sum")

                xent_loss = xent_loss / sum_inv_prop

                loss = xent_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += xent_loss.detach().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                print("[MF-SNIPS] epoch:{}, xent:{}".format(epoch, epoch_loss))
                break
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-SNIPS] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-SNIPS] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        pred = self.forward(x)
        pred = self.sigmoid(pred)
        return pred.detach().numpy()

    def _compute_IPS(self,x,y,y_ips=None):
        if y_ips is None:
            one_over_zl = np.ones(len(y))
        else:
            py1 = y_ips.sum() / len(y_ips)
            py0 = 1 - py1
            po1 = len(x) / (x[:,0].max() * x[:,1].max())
            py1o1 = y.sum() / len(y)
            py0o1 = 1 - py1o1

            propensity = np.zeros(len(y))

            propensity[y == 0] = (py0o1 * po1) / py0
            propensity[y == 1] = (py1o1 * po1) / py1
            one_over_zl = 1 / propensity

        one_over_zl = torch.Tensor(one_over_zl)
        return one_over_zl


def one_hot(x):
    out = torch.cat([torch.unsqueeze(1-x,1),torch.unsqueeze(x,1)],axis=1)
    return out

def sharpen(x, T):
    temp = x**(1/T)
    return temp / temp.sum(1, keepdim=True)










