#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 21:48:22 2020

@author: ahmednabil
"""

import tools as tools
import torch
import pandas as pd
from network import EncoderRNN, DecoderRNN
from torch import optim

SOS = 0
EOS = 1
FILES = {'ar': 'ara.txt', 'en': 'eng.txt'}


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



class Reader:
    
    def __init__(self, lang, path, max_len=10, min_len=1, max_chars=40):
        self.lang = lang
        self.path = path
        self.sentences = []
        self.max_len = max_len
        self.min_len = min_len
        self.max_chars = max_chars

    def read(self):
        print('Reading Sentences for language %s ...' %(self.lang))
        with open(self.path, 'r') as reader:
            self.sentences += list(map(tools.cleaner_job, reader.readlines()))
        return self.sentences

    def get_tokenized(self):
        print('Tokenizing %s ...' %(self.lang))
        return [[w for w in s.split()] for s in self.sentences]



class SentenceReader(Reader):
    """
        * Use Set to iterate over instead of list according to this thread 
        * Python Garbage Collector Bug when list is going bigger

    https://stackoverflow.com/questions/2473783/is-there-a-way-to-circumvent-python-list-append-becoming-progressively-slower
    
    """
    def __init__(self, input_lang_map, outpt_lang_map):
        super(SentenceReader, self).__init__(input_lang_map, outpt_lang_map)
        in_lang, in_path = tuple(*(input_lang_map.items()))
        out_lang, out_path = tuple(*(outpt_lang_map.items()))
        self.input = Reader(lang=in_lang, path=in_path)
        self.outpt = Reader(lang=out_lang, path=out_path)
        self.lang_map = {in_lang: self.input, out_lang: self.outpt}
        self.input.read()
        self.outpt.read()
        self.to_remove = set()
        self.to_have = set()
        self.__filter_sentences()
        
    def read_sentences(self, lang):
        to_have = self.to_have
        lang = self.lang_map.get(lang)
        if lang:
            return lang.sentences
        else:
            raise AttributeError("Invalid Language Attribute")
        
    def get_tokenized(self, lang):
        print('Tokenizing %s ...' %(lang))
        sentences = self.lang_map.get(lang).sentences
        return [[w for w in s.split()] for s in sentences]
    
    def __filter_sentences(self):
        max_len, min_len, max_chars = self.max_len, self.min_len, self.max_chars
        to_have = self.to_have
        to_remove = self.to_remove
        
        ######################################################################
        ## TODO: too slow should be refactored
        # s = len(self.sentences)
        # for i in sorted(range(s), reverse=True):
        #     if i in to_remove:
        #         del self.sentences[i]
        #     else:
        #         self.sentences[i] = tools.cleaner_job(self.sentences[i])
        ######################################################################
        
        for reader in [self.input, self.outpt]:
            for i, sentence in enumerate(reader.sentences):
                if len(sentence.split()) > max_len:
                    to_remove.add(i)
                elif len(sentence.split()) < min_len:
                    to_remove.add(i)
                if i not in to_remove:
                    to_have.add(i)
            to_have = [i for i in to_have if i not in to_remove]
            sent_series_obj = pd.Series(reader.sentences)
            reader.sentences = sent_series_obj[[i for i in to_have]].tolist()
            to_have = set(to_have)

def prepare_dataset(in_lang_path, out_lang_path):
    print('Preparing Dataset')
    
    reader = SentenceReader({'en': in_lang_path}, {'ar': out_lang_path})

    pairs = [(in_sent, out_sent) for (in_sent, out_sent) in zip(reader.read_sentences('en'), 
                                                                reader.read_sentences('ar'))]

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
                                             in zip(reader.get_tokenized('en'), 
                                                    reader.get_tokenized('ar'))]
    
    return {
        'pairs': pairs, 
        'pairs_encoded': pairs_encoded, 
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

# FILES_ABS_PATH = ('./data/'+FILES['en'], './data/'+FILES['ar'])
# dataset = prepare_dataset(*FILES_ABS_PATH)
# tensors = index_to_tensor(dataset['pairs_index'])
