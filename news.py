import torch
import numpy as np
import scipy.special
import time
import news_utils

import pandas as pd
from pandas import Series, DataFrame
#import feature_based_preds

def myPhi(x):
  return 0.5 * (1 + torch.erf(x/1.4142135623730951))
#  return 0.5 * (1 + torch.erf(x/np.sqrt(2)))
def myphi(x):
  return 0.3989422804014327 * torch.exp(-x*x/2)
#  return 1.0 / np.sqrt(2*np.pi) * torch.exp(-x*x/2)

def init_myfunc2_arr(device):
  # We hardcode 2000 bins from x=0 until x=20
  x = torch.linspace(-20, 20, 2001, device=device)
  mP = myPhi(x).to(device)
  mp = myphi(x).to(device)
  V = (x * mP + mp).to(device)
  return V, mP
  
class MyFunc(torch.autograd.Function):
  V, mP = None, None  # We will call init_myfunc2_arr to initialize these
  @staticmethod
  def forward(ctx, x):
    this_bin = torch.clamp(((x / 20 + 1) * 1000).to(torch.long), min=0, max=2000)
    this_val, this_mP = MyFunc.V[this_bin], MyFunc.mP[this_bin]
    ctx.save_for_backward(this_mP)
    return this_val

  @staticmethod
  def backward(ctx, grad_output):
    this_mP, = ctx.saved_tensors
    return grad_output * this_mP

def do_multiple_ls(W_nodes, W_nbrs, all_num_nbrs, default_t, device):
  # W_nbrs is (n_1 + n_2 + ...) * p, where all_num_nbrs = [n_1, n_2, ...]
  # We will assume that p is small

  num_nodes, p = W_nodes.shape
  nodes_to_consider = torch.nonzero(all_num_nbrs-1).flatten() # We only consider nodes with at least 2 nbrs
  num_nodes_to_consider = len(nodes_to_consider)

  if num_nodes_to_consider == 0:
    all_t = torch.ones(num_nodes, device=device) * default_t
    return all_t

  # First, demean W_nbrs for nodes_to_consider
  all_num_nbrs_list = all_num_nbrs.tolist()
  toconsider_num_nbrs_list = [all_num_nbrs_list[i] for i in nodes_to_consider]

  Xdm = []
  sumsq_W_nodes_dot_Xdm = []
  for i, W_nbrs_onenode in enumerate(torch.split(W_nbrs, all_num_nbrs_list, dim=0)):
    if all_num_nbrs_list[i] > 1:
      this_Xdm = W_nbrs_onenode - W_nbrs_onenode.mean(dim=0)
      this_sumsq_W_nodes_dot_Xdm = torch.sum(torch.square(torch.matmul(this_Xdm, W_nodes[i])))
      Xdm.append(this_Xdm)
      sumsq_W_nodes_dot_Xdm.append(this_sumsq_W_nodes_dot_Xdm)
  Xdm = torch.cat(Xdm)
  sumsq_W_nodes_dot_Xdm = torch.stack(sumsq_W_nodes_dot_Xdm)

  Xdm = Xdm.unsqueeze(-1) # (sum n_i) * p * 1, where n_i is only for the nodes_to_consider
  XXT = torch.matmul(Xdm, torch.transpose(Xdm, 1, 2)) # (sum n_i) * p * p
  all_diag_E = torch.sum(torch.diagonal(XXT, dim1=1, dim2=2), axis=1) # (sum n_i)


  Z = torch.stack([torch.sum(thisXXT, dim=0) for thisXXT in torch.split(XXT, toconsider_num_nbrs_list, dim=0)]).to(device)
  sum_diag_E = torch.stack([torch.sum(this_diag_E) for this_diag_E in torch.split(all_diag_E, toconsider_num_nbrs_list)], dim=0).to(device)
  sum_sq_diag_E = torch.stack([torch.sum(torch.square(this_diag_E)) for this_diag_E in torch.split(all_diag_E, toconsider_num_nbrs_list)], dim=0).to(device)
  
  nminus1_safe = all_num_nbrs[nodes_to_consider] - 1
  trace_Z2 = torch.sum(torch.square(Z), dim=(1,2))
  m = sum_diag_E/(nminus1_safe * p)
  tr_St_S = trace_Z2 / torch.square(nminus1_safe)
  d2 = tr_St_S / p - torch.square(m)
  d2clamped = torch.clamp(d2, min=1e-7)
  b_bar2 = (sum_sq_diag_E - (all_num_nbrs[nodes_to_consider]-2) * tr_St_S) / (nminus1_safe * nminus1_safe * p)
  b2 = torch.min(d2, b_bar2)
  a2 = d2 - b2
  alphas = b2/d2clamped * m
  qs = a2/d2clamped / nminus1_safe

  t_for_nodes_to_consider = torch.clamp(torch.sqrt(alphas * sumsq_W_nodes_dot_Xdm + qs * torch.sum(torch.square(W_nodes[nodes_to_consider]), axis=1)), 1e-7)

  all_t = torch.ones(num_nodes, device=device) * default_t
  all_t[nodes_to_consider] = t_for_nodes_to_consider
  return all_t


def all_matmuls(W_nodes, W_nbrs, all_num_nbrs, W_neg):
  all_neg_matmuls = torch.mm(W_nodes, W_neg.T) # nodes * num_neg
  W_pos = torch.vstack([W_nodes[i].expand(all_num_nbrs[i], -1) for i in range(len(all_num_nbrs))])
  all_pos_matmuls = torch.sum(W_pos * W_nbrs, axis=1) # len = \sum all_num_nbrs
  return all_pos_matmuls, all_neg_matmuls
  

def loss_on_batch(c, w, node_ids, all_nbr_ids, all_num_nbrs, neg_ids, default_t, device, loss_out):
  W_nodes = w(node_ids)
  W_nbrs = w(all_nbr_ids)
  W_neg = w(neg_ids)

  all_pos_matmuls, all_neg_matmuls = all_matmuls(W_nodes, W_nbrs, all_num_nbrs, W_neg)
  my_c = torch.cat([c[this_id].expand(all_num_nbrs[i]) for i, this_id in enumerate(node_ids)])
  s_pos = 1 - (my_c + c[all_nbr_ids] + all_pos_matmuls)
  s_neg = 1 + (c[node_ids][:, None] + c[neg_ids][None, :] + all_neg_matmuls)

  all_t = do_multiple_ls(W_nodes=W_nodes, W_nbrs=W_nbrs, all_num_nbrs=all_num_nbrs, default_t=torch.tensor(default_t, dtype=W_nodes.dtype, device=device), device=device)

  all_num_nbrs_list = all_num_nbrs.tolist()
  MF = MyFunc.apply
  
  loss_out[:, 0] = torch.mean(torch.clamp(s_neg, min=0), dim=1)
  loss_out[:, 1] = torch.stack([all_t[i] * torch.mean(MF(this_s_pos/all_t[i])) for i, this_s_pos in enumerate(torch.split(s_pos, all_num_nbrs_list, dim=0))])


def do_train(adjlist, node_ids, dim=4, epochs=1, batch_size=100, seed=0, num_neg_samples=300, default_t=1.0, lr=0.1, save_every=None, embfilebase=None):
  dtype = torch.float
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  torch.manual_seed(seed)

  MyFunc.V, MyFunc.mP = init_myfunc2_arr(device)  # initialize fast calculators

  num_nodes = len(adjlist)
  effdim = dim-1
  w = torch.nn.Embedding(num_nodes, effdim, device=device, dtype=dtype)
  c = torch.zeros(num_nodes, requires_grad=True, device=device)
  
  degrees = torch.tensor([len(x) for x in adjlist])
  neg_probs = torch.pow(degrees, 0.75)
  optimizer = torch.optim.Adam([w.weight, c], lr=lr)

  start_time = time.time()
  for epoch in range(epochs):
    ordering = torch.randperm(num_nodes)
    num_batches = int(np.ceil(num_nodes / batch_size))
    losses_in_epoch = torch.zeros(num_batches, 2)
    for i in np.arange(num_batches):
      start_idx = i * batch_size
      end_idx = min(start_idx + batch_size, num_nodes)
      node_ids = ordering[start_idx:end_idx]
      all_nbr_ids = []
      all_num_nbrs = np.zeros(end_idx-start_idx).astype(int)
      for j, x in enumerate(adjlist.iloc[node_ids].values):
        all_nbr_ids.append(torch.tensor(x))
        all_num_nbrs[j] = len(x)
      neg_ids = torch.multinomial(neg_probs, min(num_nodes, num_neg_samples))

      all_nbr_ids = torch.cat(all_nbr_ids).to(device)
      all_num_nbrs = torch.tensor(all_num_nbrs, device=device)
      node_ids = node_ids.to(device)
      neg_ids = neg_ids.to(device)

      loss_out = torch.zeros((len(node_ids), 2), device=device)
      loss_on_batch(c=c,
                    w=w,
                    node_ids=node_ids,
                    all_nbr_ids=all_nbr_ids,
                    all_num_nbrs=all_num_nbrs,
                    neg_ids=neg_ids,
                    default_t=default_t,
                    device=device,
                    loss_out=loss_out)

      this_loss = torch.mean(torch.sum(loss_out, axis=1))
      losses_in_epoch[i] = torch.mean(loss_out, axis=0)

      optimizer.zero_grad()
      this_loss.backward()
      optimizer.step()
    
    expected_time_per_epoch = (time.time() - start_time) / (epoch + 1)
    print('epoch={}, loss={:3.3f} ({:3.3f}, {:3.3f}), expected time: per-epoch={:1.1f}s, total={:1.1f}s'.format(epoch, torch.mean(torch.sum(losses_in_epoch, axis=1)), torch.mean(losses_in_epoch[:,0]), torch.mean(losses_in_epoch[:,1]), expected_time_per_epoch, expected_time_per_epoch * epochs))
    if save_every is not None and embfilebase is not None and (epoch+1) % save_every == 0:
      news_utils.save_embedding(w, c, embfilebase+'-epoch-{}'.format(epoch+1), node_ids=node_ids)

  return w, c
