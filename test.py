import argparse
import time
import torch
import news
import news_utils


def parse_args():
  parser = argparse.ArgumentParser(description='Run NEWS')
  parser.add_argument('--dataset', type=str, default='airports_train.csv')
  parser.add_argument('--dim', type=int, default=32, help='Embedding dimension')
  parser.add_argument('--batch_size', type=int, default=500, help='Number of nodes in each mini-batch')
  parser.add_argument('--epochs', type=int, default=75, help='Number of epochs')
  parser.add_argument('--save_every', type=int, default=40, help='Number of epochs after which you save embedding as checkpoint')
  parser.add_argument('--num_neg_samples', type=int, default=300, help='Number of negative samples')
  parser.add_argument('--lr', type=float, default=0.05, help='Learning rate')

  args = parser.parse_args()
  print(args)
  return args


if __name__ == '__main__':
  args = parse_args()
  embfilebase = '{}_dim_{}_bs_{}_lr_{:3.3f}_news'.format(args.dataset.split('.csv')[0], args.dim, args.batch_size, args.lr)
  adjlist, node_ids = news_utils.create_adjlist(args.dataset)

  start_time = time.time()
  w, c = news.do_train(adjlist=adjlist,
                       node_ids=node_ids,
                       dim=args.dim,
                       epochs=args.epochs,
                       batch_size=args.batch_size,
                       num_neg_samples=args.num_neg_samples, 
                       lr=args.lr,
                       save_every=args.save_every,
                       embfilebase=embfilebase,
                       )
  news_utils.save_embedding(w, c, embfilebase, node_ids)
  print('Time taken: dataset={}, dim={}, method=NEWS: {:2.2f} secs'.format(args.dataset, args.dim, time.time() - start_time))
