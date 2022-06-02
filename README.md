# Node Embeddings Without Similarity assumptions (NEWS)
Given a network, create embedding vectors for each node without assuming any similarity measure between unconnected nodes.

Sample run:
<pre>
$ /usr/bin/python3 -u test.py --dataset test_network.csv --dim 32 --batch_size 500 --lr 0.05 --epochs 75 --save_every=40
</pre>

This runs 75 epochs and saves the embedding in <i>test_network_dim_32_bs_500_lr_0.050_news.npz</i>.
Similar files are created after every 40 epochs.

To load the embedding:
<pre>
$ jupyter console
> emb, node_ids = news_utils.get_embedding('test_network_dim_32_bs_500_lr_0.050_news.npz')
</pre>

_emb_ is a matrix where <i>emb[i]</i> is the embedding vector for node <i>node_ids[i]</i>.
