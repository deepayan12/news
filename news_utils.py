import numpy as np
import pandas as pd
import torch

def create_adjlist(filename):
  df = pd.read_csv(filename, header=None)
  nodes = pd.concat([df[0], df[1]]).drop_duplicates()
  node_ids = nodes.values
  nodemap = dict(zip(node_ids, np.arange(len(nodes))))
  adjlist = df.replace(nodemap).groupby(0)[1].apply(lambda s: s.values)
  return adjlist, node_ids

def save_embedding(w, c, embedding_filebase, node_ids):
  emb = torch.cat([c.reshape(-1,1), w.weight], axis=1).detach().cpu().numpy()
  np.savez_compressed(embedding_filebase, node_ids=node_ids, emb=emb)
  print(f'Saved to {embedding_filebase}.npz')

def get_embedding(embedding_file):
  F = np.load(embedding_file)
  node_ids = F['node_ids']
  emb = F['emb']
  return emb, node_ids
