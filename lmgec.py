import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
import numpy as np


def update_rule_F(XW, G, k):
    F = tf.math.unsorted_segment_mean(XW, G, k)
    return F


def update_rule_W(X, F, G):
    _, U, V = tf.linalg.svd(tf.transpose(X) @ tf.gather(F, G), full_matrices=False)
    W = U @ tf.transpose(V)
    return W


def update_rule_G(XW, F):
    centroids_expanded = F[:, None, ...]
    distances = tf.reduce_mean(tf.math.squared_difference(XW, centroids_expanded), 2)
    G = tf.math.argmin(distances, 0, output_type=tf.dtypes.int32)
    return G


def init_G_F(XW, k):
    km = KMeans(k).fit(XW)
    G = km.labels_
    F = km.cluster_centers_
    return G, F


def init_W(X, f):
    svd = TruncatedSVD(f).fit(X)
    W = svd.components_.T
    return W



@tf.function
def train_loop(Xs, F, G, alphas, k, max_iter, tolerance):
    n_views = len(Xs)
    losses = tf.TensorArray(tf.float64, size=0, dynamic_size=True)
    prev_loss = tf.float64.max

    for i in tf.range(max_iter):
        loss = 0
        XW_consensus = 0
        for v in range(n_views):
          Wv = update_rule_W(Xs[v], F, G)
          XWv = Xs[v]@Wv
          XW_consensus += alphas[v] * XWv
          loss_v = tf.linalg.norm(Xs[v] - tf.gather(F @ tf.transpose(Wv), G))
          loss += alphas[v] * loss_v

        G = update_rule_G(XW_consensus, F)
        F = update_rule_F(XW_consensus, G, k)

        # if tf.abs(prev_loss - loss) < tolerance:
        #   break

        losses = losses.write(i, loss)
        prev_loss = loss

    return G, F, XW_consensus, losses.stack()


def lmgec(Xs, k, m, temperature, max_iter=30, tolerance=10e-7):
    n_views = len(Xs)

    # init G and F
    alphas = np.zeros(n_views)

    XW_consensus = 0

    # init G and F
    for v in range(n_views):
      Wv = init_W(Xs[v], m)
      XWv = Xs[v]@Wv
      Gv, Fv = init_G_F(XWv, k)
      # alphas[v] = 1
      inertia = np.linalg.norm(XWv - Fv[Gv])
      alphas[v] = np.exp(-inertia / temperature)
      XW_consensus += alphas[v] * XWv

    XW_consensus = XW_consensus / alphas.sum()
    G, F = init_G_F(XW_consensus, k)

    G, F, XW_consensus, loss_history = train_loop(Xs, F, G, alphas, k, max_iter, tolerance)

    return G, F, XW_consensus, loss_history
