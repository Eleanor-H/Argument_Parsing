# -*- coding: utf-8 -*-

import torch
import torch.utils.data as data
import re
import pickle
import json
import numpy as np
import nltk
from nltk import word_tokenize, sent_tokenize

class CaseBriefDataset(data.Dataset):
    def __init__(self, anno_file, vocab, max_sent_seq, max_word_seq):

        with open(anno_file) as f: 
            self.anno_file = json.load(f)
        self.vocab = vocab
        self.max_sent_seq = max_sent_seq
        self.max_word_seq = max_word_seq

    def __getitem__(self, index):
        """
        Returns one data tuple (paragraph, label)
            -paragraph: long type tensor of size (max_sent_seq, max_word_seq), containing word idx.
            -labels: long type tensor of size (max_sent_seq), containing labels to the corresponding sentence.
            -para_mask: float type tensor of size(max_sent_seq, 6), containing 1 or 0.
        """
        max_sent_seq = self.max_sent_seq
        max_word_seq = self.max_word_seq
        anno_file = self.anno_file
        vocab = self.vocab

        idx = anno_file[index]['index']
        paragraph = anno_file[index]['paragraph']
        label = anno_file[index]['labels'] # list whose elements within [1, 6], pad=0

        # cleaning paragraph
        paragraph = re.sub(r"(?<=[a-zA-Z])\’s", "\'s", paragraph)
        paragraph = re.sub(r"i\.e\.", "ie", paragraph)
        paragraph = re.sub(r"e\.g\.", "eg", paragraph)
        paragraph = re.sub(r"\.\”", "”.", paragraph)
        paragraph = re.sub(r"\?\”", "”?", paragraph)
        paragraph = re.sub(r"\!\”", "”!", paragraph)
        paragraph = re.sub(r"\,\”", "”,", paragraph)
        paragraph = re.sub(r"\/", " / ", paragraph)
        paragraph = re.sub(r"U\.S\.C\.", "USC", paragraph)
        paragraph = re.sub(r"UY\.S\.C\.", "UYSC", paragraph)
        paragraph = re.sub(r"Fed\.", "Fed", paragraph)
        paragraph = re.sub(r"Cir\.", "Cir", paragraph)
        paragraph = re.sub(r"Dr\.", "Dr", paragraph)
        paragraph = re.sub(r"C\.J\.", "CJ", paragraph)
        paragraph = re.sub(r".J\.", "J", paragraph)
        paragraph = re.sub(r"V\.T\.", "VT", paragraph)
        paragraph = re.sub(r"Pa\.", "Pa", paragraph)
        paragraph = re.sub(r"Super\.", "Super", paragraph)
        paragraph = re.sub(r"Cal\.", "Cal", paragraph)
        paragraph = re.sub(r"\s{2,}", " ", paragraph)

        para_sents = sent_tokenize(paragraph.lower())
        para_sent_lists = [ word_tokenize(sent) for sent in para_sents ]

        #for item in para_sents:
        #    print(item+'\n')
        #print(para_sent_lists)

        #print('%d sentences'% len(para_sents))
        #print('word counts:', count)
#        cal_log = open('cal_log.txt', 'a')
#        cal_log.write('Index: %s, Sentences: %d, words: ' %(idx, len(para_sents)))
#        for item in para_sent_lists:
#            cal_log.write('%s, ' %len(item))
#        cal_log.write('\n')

#        anno_name = './corpera/anno/criminal_law/CR' + str(idx) + '.txt'
#        anno_file = open(anno_name, 'w')
#        for item in para_sents:
#            anno_file.write(item)
#            anno_file.write('\n\n\n\n')
#        anno_file.write('------------------------------------------------------------------------\n')
#        anno_file.write('Sentences: %d, Words: '%len(para_sents))
#        for item in para_sent_lists:
#            anno_file.write('%s, ' %len(item))
#        anno_file.write('\n\n\n\n')
#        assert 1 == 0

        para_tensor = torch.zeros(max_sent_seq, max_word_seq).long()
        count_sent = 0        
        for sent_idx in range(len(para_sent_lists)):
            if count_sent == max_sent_seq: break
#           if len(para_sent_lists[sent_idx]) > max_word_seq:
#                for i in range(max_word_seq):
#                    para[sent_idx, i] = vocab(para_sent_lists[sent_idx][i])
#            else:
#                for i in range(len(para_sent_lists[sent_idx]))

            for i in range(min(len(para_sent_lists[sent_idx]), max_word_seq)):
                para_tensor[sent_idx, i] = vocab(para_sent_lists[sent_idx][i])
            count_sent += 1

        #label = torch.from_numpy(np.array(label))
        label_tensor = torch.zeros(max_sent_seq).long()
        for i in range(min(len(label),max_sent_seq)):
            label_tensor[i] = label[i] 

#        para_mask = torch.ones(max_sent_seq, 6)
#        for i in range(max_sent_seq):
#            if para_tensor[i,0] == 0: para_mask[i:,:] = 0


        return para_tensor, label_tensor#, para_mask
        

    def __len__(self):
        return len(self.anno_file)


def collate_fn(data):
    """
    Args:
        data: list of tuple (paragraph, label)
            -paragraph: torch tensor of shape (max_sent_seq, max_word_seq)
            -label: torch tensor of shape (max_sent_seq)
            -para_mask: torch tensor of shape (max_sent_seq, 6)
    Returns:
        -paragraphs: long type torch tensor of shape (batch_size, max_sent_seq, max_word_seq)
        -labels: long type torch tensor of shape (batch_size, max_sent_seq)
        -para_masks: float tensor of shape (batch_size, max_sent_seq, 6)
    """

    paragraphs, labels = zip(*data)


    # Merge para_tensors (from tuple of 2D tensor to 3D tensor).
    paragraphs = torch.stack(paragraphs, 0)

    # Merge labels (from tuple of 1d tensor to 2D tensor).
    labels = torch.stack(labels, 0)

    # Merge para_bps (from tuple of 2D tensor to 3D tensor).
#    para_masks = torch.stack(para_masks, 0)


    return paragraphs, labels#, para_masks


def get_loader(anno_file, vocab, max_sent_seq, max_word_seq, batch_size, shuffle, num_workers):

    casebriefs = CaseBriefDataset(anno_file=anno_file, vocab=vocab,
                                  max_sent_seq=max_sent_seq,
                                  max_word_seq=max_word_seq)

    data_loader = torch.utils.data.DataLoader(dataset=casebriefs,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)

    return data_loader
