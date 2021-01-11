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
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)

def main(args):

    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)


    val_loader = get_loader(anno_file=args.val_file, vocab=vocab,
     						max_sent_seq=args.max_sent_seq, max_word_seq=args.max_word_seq,
     						batch_size=args.val_batch_size,
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

    for para, label in val_loader:
        para = to_var(para, volatile=True)
        #label = to_var(label)
        label = label.cuda()

        preds = model(para) # (batch_size, max_sent_seq, 6)

        pred_labels = []
        for i in range(preds.size(1)):
            pred_label = torch.topk(preds[:,i,:], 1)[1].data
            pred_labels.append(pred_label)
        pred_labels = torch.cat(pred_labels, dim=1)

        #print(pred_labels)
        #print(label)

        selected_NTC = pred_labels == 1 # Byte Tensor, NOT VARIABLE!!
        selected_Claim = pred_labels == 2
        selected_Datum = pred_labels == 3
        selected_Warrant = pred_labels == 4
        selected_Backing = pred_labels == 5

        relevant_NTC = label == 1 # Byte Tensor, NOT VARIABLE
        relevant_Claim = label == 2
        relevant_Datum = label == 3
        relevant_Warrant = label == 4
        relevant_Backing = label == 5

        not_selected_NTC = selected_NTC == 0 # Byte Tensor, 1's represent not_selected
        not_selected_Claim = selected_Claim == 0
        not_selected_Datum = selected_Datum == 0
        not_selected_Warrant = selected_Warrant == 0
        not_selected_Backing = selected_Backing == 0

        not_relevant_NTC = relevant_NTC == 0 # Byte Tensor, 1's represent not_selected
        not_relevant_Claim = relevant_Claim == 0
        not_relevant_Datum = relevant_Datum == 0
        not_relevant_Warrant = relevant_Warrant == 0
        not_relevant_Backing = relevant_Backing == 0

        # Fetch coordinate of 1's
        selected_NTC_coor = set([tuple(item) for item in torch.nonzero(selected_NTC)])
        selected_Claim_coor = set([tuple(item) for item in torch.nonzero(selected_Claim)])
        selected_Datum_coor = set([tuple(item) for item in torch.nonzero(selected_Datum)])
        selected_Warrant_coor = set([tuple(item) for item in torch.nonzero(selected_Warrant)])
        selected_Backing_coor = set([tuple(item) for item in torch.nonzero(selected_Backing)])

        relevant_NTC_coor = set([tuple(item) for item in torch.nonzero(relevant_NTC)])
        relevant_Claim_coor = set([tuple(item) for item in torch.nonzero(relevant_Claim)])
        relevant_Datum_coor = set([tuple(item) for item in torch.nonzero(relevant_Datum)])
        relevant_Warrant_coor = set([tuple(item) for item in torch.nonzero(relevant_Warrant)])
        relevant_Backing_coor = set([tuple(item) for item in torch.nonzero(relevant_Backing)])

        not_selected_NTC_coor = set([tuple(item) for item in torch.nonzero(not_selected_NTC)])
        not_selected_Claim_coor = set([tuple(item) for item in torch.nonzero(not_selected_Claim)])
        not_selected_Datum_coor = set([tuple(item) for item in torch.nonzero(not_selected_Datum)])
        not_selected_Warrant_coor = set([tuple(item) for item in torch.nonzero(not_selected_Warrant)])
        not_selected_Backing_coor = set([tuple(item) for item in torch.nonzero(not_selected_Backing)])

        not_relevant_NTC_coor = set([tuple(item) for item in torch.nonzero(not_relevant_NTC)])
        not_relevant_Claim_coor = set([tuple(item) for item in torch.nonzero(not_relevant_Claim)])
        not_relevant_Datum_coor = set([tuple(item) for item in torch.nonzero(not_relevant_Datum)])
        not_relevant_Warrant_coor = set([tuple(item) for item in torch.nonzero(not_relevant_Warrant)])
        not_relevant_Backing_coor = set([tuple(item) for item in torch.nonzero(not_relevant_Backing)])


        tp_NTC = len(selected_NTC_coor & relevant_NTC_coor)
        tp_Claim = len(selected_Claim_coor & relevant_Claim_coor) 
        tp_Datum = len(selected_Datum_coor & relevant_Datum_coor)
        tp_Warrant = len(selected_Warrant_coor & relevant_Warrant_coor)
        tp_Backing = len(selected_Backing_coor & relevant_Backing_coor)
        tp_all = tp_NTC + tp_Claim + tp_Datum + tp_Warrant + tp_Backing

        tn_NTC = len(not_selected_NTC_coor & not_relevant_NTC_coor)
        tn_Claim = len(not_selected_Claim_coor & not_relevant_Claim_coor)
        tn_Datum = len(not_selected_Datum_coor & not_relevant_Datum_coor)
        tn_Warrant = len(not_selected_Warrant_coor & not_relevant_Warrant_coor)        
        tn_Backing = len(not_selected_Backing_coor & not_relevant_Backing_coor)
        tn_all = tn_NTC + tn_Claim + tn_Datum + tn_Warrant + tn_Backing

        fp_NTC = len(selected_NTC_coor & not_relevant_NTC_coor)
        fp_Claim = len(selected_Claim_coor & not_relevant_Claim_coor)
        fp_Datum = len(selected_Datum_coor & not_relevant_Datum_coor)
        fp_Warrant = len(selected_Warrant_coor & not_relevant_Warrant_coor)
        fp_Backing = len(selected_Backing_coor & not_relevant_Backing_coor)
        fp_all = fp_NTC + fp_Claim + fp_Datum + fp_Warrant + fp_Backing

        fn_NTC = len(not_selected_NTC_coor & relevant_NTC_coor)
        fn_Claim = len(not_selected_Claim_coor & relevant_Claim_coor)
        fn_Datum = len(not_selected_Datum_coor & relevant_Datum_coor)
        fn_Warrant = len(not_selected_Warrant_coor & relevant_Warrant_coor)
        fn_Backing = len(not_selected_Backing_coor & relevant_Backing_coor)
        fn_all = fn_NTC + fn_Claim + fn_Datum + fn_Warrant + fn_Backing

        epsilon = 1e-3

        precision_NTC = tp_NTC / (len(selected_NTC_coor))
        precision_Claim = tp_Claim / (len(selected_Claim_coor))
        precision_Datum = tp_Datum / (len(selected_Datum_coor))
        precision_Warrant = tp_Warrant / (len(selected_Warrant_coor))
        precision_Backing = tp_Backing / (len(selected_Backing_coor) + epsilon)

        recall_NTC = tp_NTC / (len(relevant_NTC_coor))
        recall_Claim = tp_Claim / (len(relevant_Claim_coor))
        recall_Datum = tp_Datum / (len(relevant_Datum_coor))
        recall_Warrant = tp_Warrant / (len(relevant_Warrant_coor))
        recall_Backing = tp_Backing / (len(relevant_Backing_coor))

        F1_NTC = 2 * precision_NTC * recall_NTC / (precision_NTC + recall_NTC)
        F1_Claim = 2 * precision_Claim * recall_Claim / (precision_Claim + recall_Claim)
        F1_Datum = 2 * precision_Datum * recall_Datum  / (precision_Datum + recall_Datum)
        F1_Warrant = 2 * precision_Warrant * recall_Warrant / (precision_Warrant + recall_Warrant)
        F1_Backing = 2 * precision_Backing * recall_Backing / (precision_Backing + recall_Backing + epsilon)

        macro_avg = (F1_NTC + F1_Claim + F1_Datum + F1_Warrant + F1_Backing) / 5
#        micro_avg = torch.sum(pred_labels == label) / label.numel()
        micro_avg = (tp_all + tn_all) / (tp_all + tn_all + fp_all + fn_all)

        
        print('-----------------------------------------------------------------------------------------------')
        print('Model: %s' % args.snapshot_path)
        print('-----------------------------------------------------------------------------------------------')
        print('--Components--||---TP---||---TN---||---FP---||---FN---||---Selected---||---Relevant---||||||--Precision--||--Recall--||--F1_Score--')
        print('NTC:            %d           %d          %d           %d          %d           %d          %.4f         %.4f         %.4f' 
            % (tp_NTC, tn_NTC, fp_NTC, fn_NTC, torch.sum(selected_NTC), torch.sum(relevant_NTC), precision_NTC, recall_NTC, F1_NTC))
        print('Claim:          %d          %d          %d          %d          %d           %d          %.4f          %.4f          %.4f' 
            % (tp_Claim, tn_Claim, fp_Claim, fn_Claim, torch.sum(selected_Claim), torch.sum(relevant_Claim), precision_Claim, recall_Claim, F1_Claim))
        print('Datum:          %d          %d          %d           %d          %d           %d         %.4f          %.4f          %.4f' 
            % (tp_Datum, tn_Datum, fp_Datum, fn_Datum, torch.sum(selected_Datum), torch.sum(relevant_Datum), precision_Datum, recall_Datum, F1_Datum))
        print('Warrant:        %d          %d          %d           %d          %d            %d         %.4f          %.4f          %.4f' 
            % (tp_Warrant, tn_Warrant, fp_Warrant, fn_Warrant, torch.sum(selected_Warrant), torch.sum(relevant_Warrant), precision_Warrant, recall_Warrant, F1_Warrant))
        print('Backing:        %d           %d          %d           %d          %d            %d         %.4f          %.4f          %.4f' 
            % (tp_Backing, tn_Backing, fp_Backing, fn_Backing, torch.sum(selected_Backing), torch.sum(relevant_Backing), precision_Backing, recall_Backing, F1_Backing))
        print('-----------------------------------------------------------------------------------------------')
        print('-----------------------------------------------------------------------------------------------')
        print('--Macro_Avg--||--Micro_Avg--')
        print('  %.4f        %.4f      ' % (macro_avg,micro_avg))
        print('-----------------------------------------------------------------------------------------------')
        
        


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--val_file', type=str, default='./corpus/val.json')
    parser.add_argument('--word2vec_path', type=str, 
        default='../sent-conv-torch/GoogleNews-vectors-negative300.bin')
    parser.add_argument('--vocab_path', type=str, default='./vocab.pkl')
    parser.add_argument('--max_sent_seq', type=int, default=10)
    parser.add_argument('--max_word_seq', type=int, default=50)
    parser.add_argument('--embed_size', type=int, default=300)
    parser.add_argument('--hidden_size', type=int, default=256)

    parser.add_argument('--val_batch_size', type=int, default=43)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--snapshot_path', type=str, 
                            default='./models/backing-9th-LSTM-83.pth.tar')
    parser.add_argument('--config', type=str, default='LSTM')
    parser.add_argument('--loss_weight', type=tuple, default=(1,3,10,5,10,1))

    args = parser.parse_args()
    print(args)
    main(args)    