from scipy import io
import numpy as np
import scipy.sparse as sp
from sklearn import metrics
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfTransformer
import os
from sklearn.neighbors import kneighbors_graph

def acm():
  dataset = "data/ACM"
  data = io.loadmat('{}.mat'.format(dataset))

  X = data['features']
  A = data['PAP']
  B = data['PLP']

  Xs = []
  As = []

  Xs.append(X.toarray())
  As.append(A.toarray())
  As.append(B.toarray())

  labels = data['label']
  labels = labels.reshape(-1)

  return As, Xs, labels

def dblp():
  dataset = "data/DBLP"
  data = io.loadmat('{}.mat'.format(dataset))

  X = data['features']
  A = data['net_APTPA']
  B = data['net_APCPA']
  C = data['net_APA']

  Xs = []
  As = []

  Xs.append(X.toarray())
  As.append(A.toarray())
  As.append(B.toarray())
  As.append(C.toarray())

  labels = data['label']
  labels = labels.reshape(-1)

  return As, Xs, labels


def imdb():
  dataset = "data/IMDB"
  data = io.loadmat('{}.mat'.format(dataset))
  
  X = data['features']
  A = data['MAM']
  B = data['MDM']
  
  Xs = []
  As = []

  Xs.append(X.toarray())
  As.append(A.toarray())
  As.append(B.toarray())

  labels = data['label']
  labels = labels.reshape(-1)

  return As, Xs, labels


def photos():
  dataset = 'data/Amazon_photos'
  data = io.loadmat('{}.mat'.format(dataset))
  
  X = data['features']
  A = data.get('adj')
  labels = data['label']
  labels = labels.reshape(-1)
  
  As = [A]
  
  Xs = [X, X @ X.T]

  return As, Xs, labels
  
  

def wiki():
  data = io.loadmat(os.path.join('data', f'wiki.mat'))
  X = data['fea'].toarray().astype(float)
  A = data.get('W')
  labels = data['gnd'].reshape(-1)
  
  As = [A, kneighbors_graph(X, 5, metric='cosine')]
  Xs = [X, np.log2(1+X)]

  return As, Xs, labels

def datagen(dataset):
  if dataset == 'imdb': return imdb()
  if dataset == 'dblp': return dblp()
  if dataset == 'acm': return acm()
  if dataset == 'photos': return photos()
  if dataset == 'wiki': return wiki()


def preprocess_dataset(adj, features, tf_idf=False, beta=1):
  adj = adj + beta * sp.eye(adj.shape[0])
  rowsum = np.array(adj.sum(1))
  r_inv = np.power(rowsum, -1).flatten()
  r_inv[np.isinf(r_inv)] = 0.
  r_mat_inv = sp.diags(r_inv)
  adj = r_mat_inv.dot(adj)

  if tf_idf:
      features = TfidfTransformer(norm='l2').fit_transform(features)
  else:
      features = normalize(features, norm='l2')

  return adj, features


def ordered_confusion_matrix(y_true, y_pred):
    conf_mat = metrics.confusion_matrix(y_true, y_pred)
    w = np.max(conf_mat) - conf_mat
    row_ind, col_ind = linear_sum_assignment(w)
    conf_mat = conf_mat[row_ind, :]
    conf_mat = conf_mat[:, col_ind]
    return conf_mat


def cmat_to_psuedo_y_true_and_y_pred(cmat):
        y_true = []
        y_pred = []
        for true_class, row in enumerate(cmat):
            for pred_class, elm in enumerate(row):
                y_true.extend([true_class] * elm)
                y_pred.extend([pred_class] * elm)
        return y_true, y_pred

def clustering_accuracy(y_true, y_pred):
    conf_mat = ordered_confusion_matrix(y_true, y_pred)
    return np.trace(conf_mat) / np.sum(conf_mat)


def clustering_f1_score(y_true, y_pred, **kwargs):
    conf_mat = ordered_confusion_matrix(y_true, y_pred)
    pseudo_y_true, pseudo_y_pred = cmat_to_psuedo_y_true_and_y_pred(conf_mat)
    return metrics.f1_score(pseudo_y_true, pseudo_y_pred, **kwargs)

