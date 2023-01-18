# LMGEC

This reporsitoty provides the implementation of the experiments for "Simultaneous Linear Multi-view Attributed Graph Representation Learning and Clustering" accepted in WSDM '23.

## Running the code 
To run experiments on a specific dataset use the correspoding command


#### Run on DBLP:
```bash
python run.py --dataset dblp --beta 2 --temperature 10 --runs 3
```

#### Run on ACM:
```bash
python run.py --dataset acm --beta 2 --temperature 100 --runs 3
```

#### Run on IMDB:
```bash
python run.py --dataset imdb --beta 0.2 --temperature 10 --runs 3
```

#### Run on Amazon Photos:
```bash
python run.py --dataset photos --beta 2 --temperature 10 --runs 3

```

#### Run on Wiki:
```bash
python run.py --dataset wiki --beta 1 --temperature 1 --runs 3
```

If you the code please do cite :

```BibTeX
@inproceedings{fettal2022efficient,
  author = {Fettal, Chakib and Labiod, Lazhar and Nadif, Mohamed},
  title = {Simultaneous Linear Multi-view Attributed Graph Representation Learning and Clustering},
  year = {2023},
  publisher = {Association for Computing Machinery},
  doi = {10.1145/3539597.3570367},
  booktitle = {Proceedings of the Sixteenth ACM International Conference on Web Search and Data Mining}
}
```

