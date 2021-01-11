import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, backward
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import os
import pickle
from nltk import sent_tokenize
from data_loader import get_loader
from build_vocab import Vocabulary
from preprocess import get_embed_weights
from model import Toulmin_RNN, Toulmin_GRU, Toulmin_LSTM, Toulmin_CNNOnly, Toulmin_singleLSTM

def to_var(x, volatile=False):
    if torch.cuda.is_available(): x = x.cuda()
    return Variable(x, volatile=volatile)

def load_txt(txt):
    #udata = open(text_file).read().decode('utf-8')   # unicode string 
    #text = unicodedata.normalize('NFD', udata).encode('ascii', 'ignore')
    text_file = './corpus/raw/criminal_law/' + txt + '.txt'
    text = open(text_file).read()#.decode('utf-8')   # unicode string 
    para_list = sent_tokenize(text)

    return para_list


def main(args):

    idx2label = { 1: "Not Toulmin Component", 2: "Claim", 3: "Datum",
                  4: "Warrant", 5: "Backing", 0: "Padding"}

#    idx2label = { 0: "Not Toulmin Component", 1: "Claim", 2: "Datum",
#                  3: "Warrant", 4: "Backing" }

    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    test_loader = get_loader(anno_file=args.test_file, vocab=vocab,
                             max_sent_seq=args.max_sent_seq, max_word_seq=args.max_word_seq,
                             batch_size=args.test_batch_size,
                             shuffle=False, num_workers=args.num_workers)

    # embed_weights is a numpy matrix of shape (len(vocab), args.embed_size)
    embed_weights = get_embed_weights(args.word2vec_path, vocab)
    if args.config == 'RNN':
        model = Toulmin_RNN(len(vocab), args.embed_size, embed_weights, args.hidden_size)
    elif args.config == 'GRU':
        model = Toulmin_GRU(len(vocab), args.embed_size, embed_weights, args.hidden_size)
    elif args.config == 'LSTM':
        model = Toulmin_LSTM(len(vocab), args.embed_size, embed_weights, args.hidden_size)
    elif args.config == 'CNN':
        model = Toulmin_CNNOnly(len(vocab), args.embed_size, embed_weights, args.hidden_size)
    elif args.config == 'singleLSTM':
        model = Toulmin_singleLSTM(len(vocab), args.embed_size, embed_weights, args.hidden_size)
    model.eval()
    model.load_state_dict(torch.load(args.snapshot_path)['state_dict'])
    if torch.cuda.is_available(): model.cuda()

    for para, label in test_loader:
        para = to_var(para, volatile=True)
        label = to_var(label)

        preds = model(para)  # (batch_size, max_sent_seq, 6)

        pred_labels = []    
        for i in range(preds.size(1)):
            pred_label = torch.topk(preds[:,i,:], 1)[1].data
            pred_labels.append(pred_label)
        pred_labels = torch.cat(pred_labels, dim=1)

#        print(pred_labels)
#        assert 1 == 0
        
#        paragraph = para[0] # (10, 50)
#        para_list = []
#        for sent_idxs in paragraph:
#            sent_list = []
#            for word_idx in sent_idxs:
#                if (word_idx.data == 0).all(): break
#                word = vocab.idx2word[word_idx.data[0]]
#                sent_list.append(word)
#            sent = ' '.join(sent_list)
#           para_list.append(sent)
#        print(para_list) # len=10, list of sents
        
        para_list = load_txt(args.txt)
        print('--- Paragraph ---')
        print(para_list)

        print('--- Prediction ---')
        pred_label = pred_labels[2] # [0]:CR230, [1]:CR240, [2]:CR380, [3]: CR420, [4]: any
        for i in range(len(para_list)):
#            if para_list[i] == '': break
            label_name = idx2label[pred_label[i]]
            print('%s: %s\n' %(para_list[i], label_name))

        print('--- Ground Truth ---')
        gt_labels = []
        for i in range(10):
            if label.data[2,i] == 0: break # [0]:CR230, [1]:CR240, [2]:CR380, [3]: CR420, [4]: any
            gt_label = idx2label[label.data[2,i]] # [0]:CR230, [1]:CR240, [2]:CR380, [3]: CR420, [4]: any
            gt_labels.append(gt_label)
        gt_labels = '  '.join(gt_labels)
        print(gt_labels)
        print('-----------------------------')
        print('-----------------------------')



    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_file', type=str, default='./corpus/test.json')
    parser.add_argument('--txt', type=str, default='CR380')
    parser.add_argument('--word2vec_path', type=str, 
        default='../sent-conv-torch/GoogleNews-vectors-negative300.bin')
    parser.add_argument('--vocab_path', type=str, default='./vocab.pkl')
    parser.add_argument('--max_sent_seq', type=int, default=10)
    parser.add_argument('--max_word_seq', type=int, default=50)
    parser.add_argument('--embed_size', type=int, default=300)
    parser.add_argument('--hidden_size', type=int, default=256)

    parser.add_argument('--test_batch_size', type=int, default=5)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--snapshot_path', type=str,
                            default='./models/nobacking-3rd-LSTM-81.pth.tar')
#                            default='./models/nobacking-4th-CNN-119.pth.tar')
    parser.add_argument('--config', type=str, default='LSTM')

    args = parser.parse_args()
    print(args)
    main(args)    