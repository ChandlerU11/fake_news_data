import sys
import csv
import re
import argparse
from langdetect import detect
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from bs4 import BeautifulSoup
from nltk import tokenize
import numpy as np

parser = argparse.ArgumentParser(description='Train Text CNN classificer')

parser.add_argument(
    '-dataset',
    type=str,
    default="gossipcop",
    help='dataset to use')

csv.field_size_limit(sys.maxsize)
newsdic = {}


def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = str(string)
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()

if __name__ == '__main__':
    # dataset used for training
    args = parser.parse_args()

    filename = 'data_files/' + args.dataset + '_comment_no_ignore.tsv'
    with open(filename) as tsvfile:
        reader = csv.reader(tsvfile, delimiter=str(u'\t'))
        next(reader, None) # skip header row
    
        for row in reader:
            id = row[0]
            comments = row[1].split("::")
            comments = [ comment for comment in comments if isinstance(comment, str) ]
            newsdic[id] = comments

    for id in tqdm(newsdic):
        clean_comments = []
        for comment in newsdic[id]:
            if re.search('[a-zA-Z]', comment): # skips comments that are only numbers and spaces or non-ascii
                try:
                    language = detect(comment)
                except Exception:
                    print('Comment: "' + comment + '" not included')
                    continue
                comment_words = [ word for word in comment.split() if not re.match("^http", word) ] # strip out URLs
                num_words = len(comment_words)
                if language == "en" and num_words > 4 and num_words < 21:
                    comment = " ".join(comment_words)
                    clean_comments.append(comment)
        newsdic[id] = clean_comments

    # write to file
    filename = 'data_files/' + args.dataset + '_clean_comments.tsv'
    with open(filename, 'wt') as tsvfile:
        writer = csv.writer(tsvfile, delimiter='\t')
        writer.writerow(['id', 'comments'])
        for id, comments in newsdic.items():
            comments = "::".join(comments)
            writer.writerow([id, comments])

    
    
    
    data_train = pd.read_csv('data_files/' + args.dataset + '_content_no_ignore.tsv', sep='\t')
    VALIDATION_SPLIT = 0.1 # train on 90%, test on 10%
    contents = []
    labels = []
    texts = []
    ids = []

    for idx in range(data_train.content.shape[0]):
        text = BeautifulSoup(data_train.content[idx], features="html.parser")
        text = clean_str(text.get_text())
        texts.append(text)
        sentences = tokenize.sent_tokenize(text)
        contents.append(sentences)
        ids.append(data_train.id[idx])
        labels.append(data_train.label[idx])

    labels = np.asarray(labels)

    # load user comments
    comments = []
    comments_train = pd.read_csv('data_files/' + args.dataset + '_clean_comments.tsv', sep='\t')
    print(comments_train.shape)

    content_ids = set(ids)

    for idx in range(comments_train.comments.shape[0]):
        if comments_train.id[idx] in content_ids:
            com_text = comments_train.comments[idx]
            if isinstance(com_text, str):
                com_text = BeautifulSoup(com_text, features="html.parser")
                com_text = clean_str(com_text.get_text())
            else:
                com_text = None
            entries = [ comment for comment in com_text.split('::') ] if com_text else []
            comments.append(entries[:10])

    df = pd.DataFrame()
    df['id'] = ids
    df['content'] = contents
    df['comments'] = comments
    print(labels)
    df['label'] = labels
    
    df_titles = pd.read_csv('data_files/' + args.dataset + '_title_no_ignore.tsv', sep = '\t', names = ['id', 'title'])
    print(df_titles)
    df_titles['title'] = df_titles['title'].fillna('')
    df_titles['title'] = [[clean_str(BeautifulSoup(each, features="html.parser").get_text())] for each in df_titles['title']]
    print(df_titles)
    df = df.merge(df_titles, how = 'inner', on = 'id')
    print(df)
    df_train, df_test = train_test_split(df, test_size=VALIDATION_SPLIT, random_state=42, stratify=labels)

    df_train.to_csv(args.dataset + '_train.csv', index = False)
    df_test.to_csv(args.dataset + '_test.csv', index = False)
