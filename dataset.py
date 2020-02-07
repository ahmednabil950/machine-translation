#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 21:48:22 2020

@author: ahmednabil
"""

import tools as tools
import torch
import pandas as pd

SOS = 0
EOS = 1
FILES = {'ar': 'UNv1.0.ar-en.ar.10k', 'en': 'UNv1.0.ar-en.en.10k'}


class Dictionary:
    def __init__(self, name):
        print('Building Dictionary for lang %s' %(name))
        self.name = name
        self.word_to_idx = {}
        self.idx_to_word = {'<S>': SOS, '</S>': EOS}
        self.words_count = {}
        self.n_words = 2  # Count SOS, EOS special tokens

    def add_word(self, word):
        if word not in self.word_to_idx:
            self.word_to_idx[word] = self.n_words
            self.words_count[word] = 1
            self.idx_to_word[self.n_words] = word
            self.n_words += 1
        else:
            self.words_count[word] += 1

    def add_sentence(self, sentence):
        for word in sentence.split():
            self.add_word(word)



class SentenceReader:
    """
        * Use Set to iterate over instead of list according to this thread 
        * Python Garbage Collector Bug when list is going bigger

    https://stackoverflow.com/questions/2473783/is-there-a-way-to-circumvent-python-list-append-becoming-progressively-slower
    
    """
    to_remove = set()
    to_have = set()
    
    def __init__(self, lang, path, max_len=100, min_len=40):
        self.lang = lang
        self.path = path
        self.sentences = []
        self.max_len = max_len
        self.min_len = min_len
        
    def __filter_sentences(self):
        max_len, min_len = self.max_len, self.min_len
        to_have = self.__class__.to_have
        to_remove = self.__class__.to_remove

        if not len(to_remove):
            for i, sentence in enumerate(self.sentences):
                if len(sentence.split()) > 25:
                    to_remove.add(i)
                    continue
                if len(sentence) > max_len:
                    to_remove.add(i)
                elif len(sentence) < min_len:
                    to_remove.add(i)
                if i not in to_remove:
                    to_have.add(i)

        ######################################################################
        ## TODO: too slow should be refactored
        # s = len(self.sentences)
        # for i in sorted(range(s), reverse=True):
        #     if i in to_remove:
        #         del self.sentences[i]
        #     else:
        #         self.sentences[i] = tools.cleaner_job(self.sentences[i])
        ######################################################################

        sent_series_obj = pd.Series(self.sentences)
        self.sentences = sent_series_obj[to_have].tolist()

    def read_sentences(self):
        print('Reading Sentences for language %s ...' %(self.lang))
        with open(self.path, 'r') as reader:
            self.sentences += list(map(tools.cleaner_job, reader.readlines()))

        self.__filter_sentences()
        
        return self.sentences

    def get_tokenized(self):
        print('Tokenizing %s ...' %(self.lang))
        return [[w for w in s.split()] for s in self.sentences]


def prepare_dataset(in_lang, out_lang):
    print('Preparing Dataset')
    
    in_sentences_obj = SentenceReader('en', in_lang)
    out_sentences_obj = SentenceReader('ar', out_lang)

    pairs = [(in_sent, out_sent) for (in_sent, out_sent) in zip(in_sentences_obj.read_sentences(), 
                                                                out_sentences_obj.read_sentences())]

    print('Building Language Dictionary')

    in_dictionary = Dictionary('english')
    ou_dictionary = Dictionary('arabic')
    for in_sent, out_sent in pairs:
        in_dictionary.add_sentence(in_sent)
        ou_dictionary.add_sentence(out_sent)

    print('Encoding Tokenized Sentences to unique identifiers')
    in_word_to_idx = in_dictionary.word_to_idx
    ou_word_to_idx = ou_dictionary.word_to_idx
    pairs_encoded = [([in_word_to_idx[in_w]  for in_w in in_sent], 
                      [ou_word_to_idx[out_w] for out_w in out_sent]) 
                                             for (in_sent, out_sent) 
                                             in zip(in_sentences_obj.get_tokenized(), 
                                                    out_sentences_obj.get_tokenized())]
    
    return {
        'pairs': pairs, 
        'pairs_index': pairs_encoded, 
        'in_dictionary': in_dictionary, 
        'out_dictionary': ou_dictionary,
    }
    
def pad_sequence(indices):
    max_len = 0
    for seq in indices:
        if len(seq) > max_len:
            max_len = len(seq)
            
    for seq in indices:
        seq += [EOS] * (max_len - len(seq))

def index_to_tensor(pairs):
    input_idxs = []
    output_idxs = []
    tensors = {}
    
    for pair in pairs.copy():
        input_idxs += [pair[0] + [EOS]]
        output_idxs += [[SOS] + pair[1] + [EOS]]
        
    for indices in [input_idxs, output_idxs]:
        pad_sequence(indices)
        
    print(input_idxs[:10])
    print(output_idxs[:10])
        
    tensors['input'] = torch.tensor(input_idxs)
    tensors['target'] = torch.tensor(output_idxs)
    
    return tensors

FILES_ABS_PATH = ('./data/'+FILES['en'], './data/'+FILES['ar'])
dataset = prepare_dataset(*FILES_ABS_PATH)
tensors = index_to_tensor(dataset['pairs_index'])
