# fake_news_data
Cleans and creates training and test files for Gossipcop and Politifact datasets. The same training and test sets are used used across all adversary and classifier training scripts. 

### Create conda env "data_clean"
`conda env create -f env.yml`

### Clean Data
`python clean_data.py -dataset [gossipcop, politifact]`
