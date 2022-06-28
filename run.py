from utils import clustering_accuracy, clustering_f1_score, preprocess_dataset, datagen
from lmgec import lmgec
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
import numpy as np
from sklearn.metrics import adjusted_rand_score as ari
from time import time
from sklearn.preprocessing import StandardScaler
from itertools import product
import tensorflow as tf

flags = tf.compat.v1.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('dataset', 'cora', 'Dataset to use (acm, dblp, imdb, photos or wiki).')
flags.DEFINE_float('temperature', 1, 'Temprature for view importance softmax.')
flags.DEFINE_float('beta', 1, 'Self-loop parameter.')
flags.DEFINE_integer('max_iter', 10, 'Number of iterations of the algorithm.')
flags.DEFINE_float('tol', 1e-7, 'Tolerance threshold of convergence.')
flags.DEFINE_integer('runs', 1, 'Number of runs.')

dataset = FLAGS.dataset
temperature = FLAGS.temperature
beta = FLAGS.beta
runs = FLAGS.runs
max_iter = FLAGS.max_iter
tolerance = FLAGS.tol


print('-----------------', dataset, '-----------------')
As, Xs, labels = datagen(dataset)
k = len(np.unique(labels))
views = list(product(As, Xs))

for v in range(len(views)):
  A, X = views[v]
  tf_idf = dataset in ['acm', 'dblp', 'imdb', 'photos']
  norm_adj, features = preprocess_dataset(A, X, tf_idf=tf_idf, beta=beta)

  if type(features) != np.ndarray:
    features = features.toarray()

  if type(norm_adj) == np.matrix:
    norm_adj = np.asarray(norm_adj)
  
  views[v] = (norm_adj, features)

metrics = {}
metrics['acc'] = []
metrics['nmi'] = []
metrics['ari'] = []
metrics['f1'] = []
metrics['loss'] = []
metrics['time'] = []


for run in range(runs):
  t0 = time()
  
  Hs = []
  
  for v, (S, X) in enumerate(views):
    features = S @ X
    x = features
    x = StandardScaler(with_std=False).fit_transform(x)
    Hs.append(x)
  
  Z, F, XW_consensus, losses = lmgec(Hs, k, k+1, temperature=temperature, max_iter=10, tolerance=0)

  metrics['time'].append(time()-t0)
  metrics['acc'].append(clustering_accuracy(labels, Z))
  metrics['nmi'].append(nmi(labels, Z))
  metrics['ari'].append(ari(labels, Z))
  metrics['f1'].append(clustering_f1_score(labels, Z, average='macro'))
  metrics['loss'].append(losses[-1])
    

results = {
    'mean': {k:np.mean(v).round(4) for k,v in metrics.items()}, 
    'std': {k:np.std(v).round(4) for k,v in metrics.items()}
}
means = results['mean']
  
print(means['acc'], means['f1'], means['nmi'], means['ari'], sep='&')