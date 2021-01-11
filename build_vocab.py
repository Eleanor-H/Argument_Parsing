# -*- coding: utf-8 -*-

import nltk
from nltk import word_tokenize, sent_tokenize
import string
import pickle
import json
import argparse
from collections import Counter

class Vocabulary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

def build_vocab(train_file, max_num):
    with open(train_file) as f:
        data = json.load(f)

    counter = Counter()
    paragraph = [ str(item['paragraph']) for item in data ]
    para_words = []
    for item in paragraph:
        tokens = word_tokenize(item.lower())
        para_words.extend(tokens)
    
    counter.update(para_words)
    print(counter)

    words = [ word for (word, count) in counter.most_common(max_num)]
#    print(len(words))
#    assert 1 == 0

    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<unk>')

    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab

def main(args):
    vocab = build_vocab(train_file=args.train_file, max_num=args.max_num)
    vocab_path = args.vocab_path
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print("total vocabulary size: %d" %len(vocab))
    print("Saved teh vocabulary wrapper to '%s'" %vocab_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str, default='./corpus/train.json')
    parser.add_argument('--max_num', type=int, default=10000)
    parser.add_argument('--vocab_path', type=str, default='./vocab.pkl')
    args = parser.parse_args()
    main(args)
