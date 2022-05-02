from doctest import OutputChecker
import sys
import cv2
import torch
import numpy as np
import os
import torch.utils.data as data
from ..utils import Corpus
import os.path as osp

class OurImgDataset(data.Dataset):
    def __init__(self, root, transforms=None, max_query_len=20):
        
        self.root = root
        self.transform = transforms
        self.query_len = max_query_len
        self.img_path = os.path.join(self.root, 'imgs')
        self.phrases_path = os.path.join(self.root, 'phrases.txt')

        self.imgs = os.listdir(self.img_path)
        
        self.img_queries = {}
        with open(self.phrases_path, 'r') as f:
            for line in f:
                clean_line = line.split()
                self.img_queries[int(clean_line[0])] = ' '.join(clean_line[2:])
        
        self.corpus = Corpus()
        corpus_path = osp.join(dataset_path, 'corpus.pth')
        self.corpus = torch.load(corpus_path)

    def __len__(self):
        return len(self.img_queries)

    def __getitem__(self, i):
        img_path = os.path.join(self.img_path, self.imgs[i])
        img_num = self.imgs[i][4:8]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        ret_img = img

        if self.transform is not None:
            img = self.transform(img)

        phrase = self.img_queries[img_num]
        phrase = phrase.lower()

        word_id = self.corpus.tokenize(phrase, self.query_len)
        word_mask = np.array(word_id > 0, dtype=int)

        return img, np.array(word_id, dtype=int), np.array(word_mask, dtype=int), phrase, ret_img

