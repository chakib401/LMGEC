# LMGEC

## Getting the data
To download the data, use command

```bash
gdown "1DWtBaW0qF41KTZwGcfJe6vbrZy6MS8Uw"
```

or download them manually from [here](https://drive.google.com/file/d/1DWtBaW0qF41KTZwGcfJe6vbrZy6MS8Uw/view?usp=sharing).
 

```bash
python run.py --dataset dblp --beta 2 --temperature 10 --runs 3
```

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
