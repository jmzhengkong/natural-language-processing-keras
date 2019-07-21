""" A data helper to process data. """
import os
import sys
import collections
import pandas as pd
import re
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from gensim.models import KeyedVectors


def clean_document(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"(\@[A-Za-z0-9_(),!?]+)", "", string)
    # Remove emoji
    string = re.sub(r"(\&[A-Za-z0-9(),!?;\#]+)", "", string)
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"RT", "", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", "", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.lower().strip()


def truncate_vocab(vocab, nb_words):
    new_vocab = collections.defaultdict(float)
    i = 0
    for w in sorted(vocab, key=vocab.get, reverse=True):
        new_vocab[w] = vocab[w]
        i += 1
        if i >= nb_words:
            break
    return new_vocab


def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def load_text_vec(fname, vocab, splitter=' ', ext_num=10000):
    """
    Loads dx1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    ext_count = 0
    with open(fname, "r") as f:
        vocab_size = file_len(fname)
        layer1_size = None

        for line in f:
            ss = line.split(' ')
            if len(ss) <= 3:
                continue
            #word = ss[0].decode('utf-8', 'ignore')
            word = ss[0]
            dims = ' '.join(ss[1:]).strip().split(splitter)
            if layer1_size is None:
                layer1_size = len(dims)
                print("reading word2vec at vocab_size:{:d}, dimension:{:d}".format(
                    vocab_size, layer1_size))
            if word in vocab:
                word_vecs[word] = np.fromstring(
                    ' '.join(dims), dtype='float32', count=layer1_size, sep=' ')

            elif ext_count < ext_num:  # add this word to vocabulary
                ext_count += 1
                word_vecs[word] = np.fromstring(
                    ' '.join(dims), dtype='float32', count=layer1_size, sep=' ')

    return vocab_size, word_vecs, layer1_size


def load_bin_vec(fname, vocab):
    """
    Loads word2vec from
    Quanzhi Li, Sameena Shah, Xiaomo Liu, Armineh Nourbakhsh,
    Data Set: Word Embeddings Learned from Tweets and General Data,
    The 11th International AAAI Conference on Web and Social Media (ICWSM-17).
    """
    word_vecs = {}
    wv_from_bin = KeyedVectors.load_word2vec_format(fname, binary=True)
    for word in vocab:
        if word in wv_from_bin:
            word_vecs[word] = wv_from_bin[word]
    return word_vecs, wv_from_bin["computer"].shape[0]


def add_unknown_words(word_vecs, vocab, min_df=1, k=300):
    """
    For words that occur in at least min_df documents, create a separate word
    vector.
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            #word_vecs[word] = np.random.uniform(-0.25, 0.25, k)
            word_vecs[word] = np.zeros(k)


def get_W(word_vecs, k=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size + 1, k), dtype='float32')
    W[0] = np.zeros(k, dtype='float32')
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map


def doc2idx(docs, word_idx_map, maxlen=400, num_clas=2):
    y = np.zeros((len(docs), num_clas))
    X = []
    truncate_num = 0
    for t, datum in enumerate(docs):
        for k in datum['category']:
            y[t, k] = 1
        x = []
        for i, w in enumerate(datum['text'].split()):
            if i >= maxlen:
                truncate_num += 1
                break
            if w in word_idx_map:
                x.append(word_idx_map[w])
            else:
                x.append(0)
        X.append(x)
    print("Truncate {:d} docs.".format(truncate_num))
    return pad_sequences(X, maxlen, dtype='int32'), y


def process_tweets(csv_path, w2v_file, maxlen=400, nb_words=20000):
    vocab = collections.defaultdict(float)
    documents = []

    dataframe = pd.read_csv("hateSpeech.csv", index_col=0)
    for index, row in dataframe.iterrows():
        document_id = index
        document = clean_document(row.tweet)
        category = row["class"]

        words = set(document.split())
        for word in words:
            vocab[word] += 1
        datum = {"text": document,
                 "num_words": len(document.split()),
                 "category": [category],
                 "id": document_id}
        documents.append(datum)

    num_words = [datum["num_words"] for datum in documents]
    max_num_words = np.max(num_words)
    mean_num_words = np.mean(num_words)
    vocab = truncate_vocab(vocab, nb_words)

    print("Data loaded!")
    print("The number of documents: {:d}".format(len(documents)))
    print("The vocabulary size: {:d}".format(len(vocab)))
    print("The maximum sentence length: {:d}".format(max_num_words))
    print("The average sentence length: {:.3f}".format(mean_num_words))

    # Load w2v_file
    if w2v_file.endswith("txt"):
        _, w2v, layer1_size = load_text_vec(w2v_file, vocab)
    elif w2v_file.endswith("bin"):
        w2v, layer1_size = load_bin_vec(w2v_file, vocab)
    print("The word2vec loaded!")
    print("The number of words already in word2vec: {:d}".format(len(w2v)))
    add_unknown_words(w2v, vocab, k=layer1_size)
    W, word_idx_map = get_W(w2v, k=layer1_size)

    X, y = doc2idx(documents, word_idx_map, maxlen, 3)

    np.savez("hateSpeech.{:d}.npz".format(maxlen), X=X, y=y, W=W, maxlen=maxlen)


if __name__ == "__main__":
    #process_tweets("hateSpeech.csv", "all.review.vec.txt", 40)
    process_tweets("hateSpeech.csv",
                   "Set10_TweetDataWithSpam_GeneralData_Word_Phrase.bin", 24)
