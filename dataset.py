#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 21:48:22 2020

@author: ahmednabil
"""

import tools as tools

SOS = 0
EOS = 1
FILES = {'ar': 'UNv1.0.ar-en.ar.500k', 'en': 'UNv1.0.ar-en.en.500k'}


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
    def __init__(self, lang, path):
        self.lang = lang
        self.path = path
        self.sentences = []

    def read_sentences(self):
        print('Reading Sentences for language %s ...' %(self.lang))
        with open(self.path, 'r') as reader:
            sent = map(lambda s: tools.cleaner_job(s), reader.readlines())
            self.sentences += list(sent)
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
                                             for (in_sent, out_sent) in zip(in_sentences_obj.get_tokenized(), 
                                                                            out_sentences_obj.get_tokenized())]
    
    return pairs, pairs_encoded, in_dictionary, ou_dictionary


FILES_ABS_PATH = ('./data/'+FILES['en'], './data/'+FILES['ar'])
dataset = prepare_dataset(*FILES_ABS_PATH)
