import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import json
import numpy as np
import nltk
from nltk import word_tokenize, sent_tokenize
from PIL import Image
#from build_sent_vocab import SentVocabulary
#from build_tags_vocab import TagsVocabulary

class ReportsDataset(data.Dataset):
    def __init__(self, root, tags, json_file, sent_vocab, max_sent_seq, max_word_seq, train=True, test=False, transform=None):
        """Set the path for imgs, captions and vocabulary wrapper.

        Args:
            root: image directory.
            tags: tags file path.
            json_file: annotation file path.
            sent_vocab: sent vocabulary wrapper.
            transform: image transformer.
        """
        self.train = train
        self.test = test
        
        self.root = root
        self.tags = tags
        with open(json_file) as f: self.anno = json.load(f)
        self.sent_vocab = sent_vocab
        self.max_sent_seq = max_sent_seq
        self.max_word_seq = max_word_seq
        self.transform = transform

        tags_file = torch.load(self.tags)
        self.img_id = tags_file['ix2id']
        self.tags = tags_file['tags']

        imgs = [os.path.join(self.root, self.img_id[_]) for _ in range(len(self.img_id))]
        if self.train:
#            self.train_data = imgs[:2]
            self.train_data = imgs[:6470]
        elif self.test:
            self.test_data = imgs[6971:]
        else:
            self.val_data = imgs[6471:6970]

    def __getitem__(self, index):
        """
        Returns one data tuple (image, tag, stop_target, cap).
            - image: 
            - tag: long type tensor of size (1, 151), containing 1 or 0.
            - stop_target: long type tensor of size (1, max_sent_seq), containing 1 or 0.
            - cap: long type tensor of size (max_sent_seq, max_word_seq) containing word idx.
            
        """
        if self.train:
            img = Image.open(self.train_data[index]).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
        elif self.test:
            img = Image.open(self.test_data[index]).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
        else:
            img = Image.open(self.val_data[index]).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)

#        tags = self.tags[index]
#        print(torch.nonzero(tags))
        
        max_sent_seq = self.max_sent_seq
        max_word_seq = self.max_word_seq
        anno = self.anno
        sent_vocab = self.sent_vocab
        with open('tags_vocab.pkl', 'rb') as g: tags_vocab = pickle.load(g)

        if self.train: image_id = str(self.train_data[index]).split('/')[-1]
        elif self.test: image_id = str(self.test_data[index]).split('/')[-1]
        else: image_id = str(self.val_data[index]).split('/')[-1]
 #       print(image_id)

        tags = anno[index]['tags']
        tags = [tag.lower() for tag in tags]
#        print(tags)
        tag_ids = [tags_vocab.tag2idx[tag] for tag in tags]
#        print(tag_ids)
        tag = torch.zeros(151)
        for id in tag_ids: tag[id-1] = 1
#        print(tag)

        impression = anno[index]['impression']
        findings = anno[index]['findings']
        caption = impression + ' ' + findings
#        print(caption)
#        print('------------------------')
#        assert 1 == 0

        # Convert caption to long type tensor of size (max_sent_seq, max_word_seq) containing word idx.
        caption_sents = sent_tokenize(caption.lower()) # list of sentences
        caption_sent_lists = [ word_tokenize(sent) for sent in caption_sents ]

        stop_target = torch.zeros(1, max_sent_seq).long()
        if not len(caption_sent_lists) > max_sent_seq:
            stop_idx = len(caption_sent_lists) - 1
            stop_target[:, stop_idx:] = 1

        cap = torch.zeros(max_sent_seq, max_word_seq).long()
        count_sent = 0
        for sent_idx in range(len(caption_sent_lists)):
            if count_sent == max_sent_seq: break
            cap[sent_idx,0] = sent_vocab('<start>')
            if len(caption_sent_lists[sent_idx]) > (max_word_seq - 2):
                for i in range(max_word_seq):
                    if i == (max_word_seq-1):
                        cap[sent_idx, i] = sent_vocab('<end>')
                    elif i == 0:
                        pass
                    else:
                        cap[sent_idx, i] = sent_vocab(caption_sent_lists[sent_idx][i-1])
            else:
                for i in range(1, len(caption_sent_lists[sent_idx]) + 2):
                    if i == (len(caption_sent_lists[sent_idx]) + 1):
                        cap[sent_idx, i] = sent_vocab('<end>')
                        break
                    elif i == 0:
                        pass
                    else:
                        cap[sent_idx, i] = sent_vocab(caption_sent_lists[sent_idx][i-1])
            count_sent += 1
        
        return image_id, img, tag, stop_target, cap

    def __len__(self):
        if self.train:
            return len(self.train_data)
        elif self.test:
            return len(self.test_data)
        else:
            return len(self.val_data)


def collate_fn(data):
    """
    Args:
        data: list of tuple (img, tags, stop_targets, caps).
            - img: torch tensor of shape (3, 224, 224).
            - tag: torch tensor of shape (151).
            - stop_targets: torch tensor of shape (1, max_sent_seq)
            - caps: torch tensor of shape (max_sent_seq, max_word_seq)

    Returns:
        - imgs: torch tensor of shape (batch_size, 3, 224, 224).
        - tags: torch tensor of shape (batch_size, 151).
        - stop_targets: torch tensor of shape (batch_size, max_sent_seq)
        - caps: torch tensor of shape (batch_size, max_sent_seq, max_word_seq)
        - lengths: list; valid length for each padded caption. (deprecated)
    """
    
    img_ids, imgs, tags, stop_targets, caps = zip(*data)
#    imgs = tuple(data[i][0] for i in range(len(data)))
#    tags = tuple(data[i][1] for i in range(len(data)))
#    stop_targets = tuple(data[i][2] for i in range(len(data)))
#    caps = tuple(data[i][3] for i in range(len(data)))

    # Sort a data list by caption length (descending order). (deprecated)
    #data.sort(key=lambda x: len(x[2]), reverse=True)

    # Merge imgs (from tuple of 3D tensor to 4D tensor).
    imgs = torch.stack(imgs, 0)

    # Merge tags (from tuple of 1D tensor to 2D tensor).
    tags = torch.stack(tags, 0).squeeze()

    # Merge stop_targets (from tuple of 2D tensor to 3D tensor).
    stop_targets = torch.stack(stop_targets, 0).squeeze()

    # Merge caps (from tuple of 2D tensor to 3D tensor).
    #lengths = [len(cap) for cap in caps]
    #caps = torch.zeros(len(caps), max(lengths)).long()
    #for i,cap in enumerate(caps):
    #    end = lengths[i]
    #    caps[i, :end] = cap[:end]
    caps = torch.stack(caps, 0)

    return img_ids, imgs, tags, stop_targets, caps 

def get_loader(root, tags, json_file, sent_vocab, max_sent_seq, max_word_seq, train, test, transform, batch_size, shuffle, num_workers):
    """Returns torch.utils.data.Dataloader for medical imaging reports dataset."""

    # medical imaging reports dataset
    reports = ReportsDataset(root=root,
                             tags=tags,
                             json_file=json_file,
                             sent_vocab=sent_vocab,
                             max_sent_seq=max_sent_seq,
                             max_word_seq=max_word_seq,
                             train=train,
                             test=test,
                             transform=transform)
    
    # Data loader for medical imaging reports dataset
    # This will return (imgs, tags, caps, lengths) for every iteration.
    # - imgs: tensor of shape (batch_size, 3, 224, 224).
    # - tags: tensor of shape (batch_size, 151).
    # - caps: tensor of shape (batch_size, max_sent_seq, max_word_seq-1)
    # - lengths: list indicating valid length for each caption. length is (batch_size) (deprecated)
    data_loader = torch.utils.data.DataLoader(dataset=reports,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader
