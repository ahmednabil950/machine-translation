#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  16 15:36:22 2020

@author: ahmednabil
"""

import dataset as ds
from network import *
from dataset import SOS, EOS, prepare_dataset
from torch import optim
import torch
import math
import random
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

plt.switch_backend('agg')

PATHS = {
    'eng': './data/eng.txt',
    'ara': './data/ara.txt'
}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Model:
    
    def train(self, **kw):
        raise NotImplementedError()
    
    def evaluate(self, **kw):
        raise NotImplementedError()
    
    def plot(self, **kw):
        raise NotImplementedError()
    
    def _as_minutes(self, s):
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

    def _time_since(self, since, percent):
        now = time.time()
        s = now - since
        es = s / (percent)
        rs = es - s
        return '%s (- %s)' % (self._as_minutes(s), self._as_minutes(rs))


class Seq2seqModel(Model):
    
    def __init__(self, **kw):
        self.lr = kw['lr']
        self.max_length = kw['max_length']
        self.dictionaries = kw['dictionaries']
        self.iterations = kw['iterations']
        self.encoder = EncoderRNN(kw['input_size'], kw['hidden_size']).to(DEVICE)
        self.decoder = DecoderRNN(kw['hidden_size'], kw['output_size']).to(DEVICE)
        self.en_optimizer = optim.SGD(self.encoder.parameters(), lr=self.lr)
        self.de_optimizer = optim.SGD(self.decoder.parameters(), lr=self.lr)
        self.criterion = nn.NLLLoss()
        self.encoder.initHidden()
        self.decoder.initHidden()
    
    def train(self, **kw):
        tensors = [random.choice(kw['tensors']) for i in range(self.iterations)]
                
        plot_losses = []
        print_loss_total = 0
        plot_loss_total = 0
        print_every = 5000
        plot_every = 5000
        
        # tik tok
        start = time.time()
                
        for iter in range(1, self.iterations):
            random_sample = tensors[iter-1]
            input_tensor = random_sample[0] + [EOS]
            output_tensor = random_sample[1] + [EOS]
            
            loss = self.steps_over_sequence(input_tensor, output_tensor)
            print_loss_total += loss
            plot_loss_total += loss
            
            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (self._time_since(start, iter / self.iterations),
                                        iter, iter / self.iterations * 100, print_loss_avg))

            if iter % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0 # restart loss value
                
        self.plot(plot_losses)
            
    def steps_over_sequence(self, input_tensor, output_tensor):
        ht_encoder = self.encoder.initHidden()
        
        # initial gradients
        self.en_optimizer.zero_grad()
        self.de_optimizer.zero_grad()
        
        input_tensor = torch.tensor([input_tensor], device=DEVICE).view(-1, 1)
        output_tensor = torch.tensor([output_tensor], device=DEVICE).view(-1, 1)
        input_length = input_tensor.size(0)
        target_length = output_tensor.size(0)
        
        loss = 0
        
        ## Encoder feed forward
        encoder_outputs = torch.zeros(self.max_length, self.encoder.hidden_size, device=DEVICE)
        for i_step in range(input_length):
            yt_encoder, ht_encoder = self.encoder(input_tensor[i_step], ht_encoder)
            encoder_outputs[i_step] = yt_encoder[0, 0]
            
        ## Decode feed forward
        ht_decoder = ht_encoder
        decoder_input = torch.tensor([[SOS]], device=DEVICE)
        for i_step in range(target_length):
            output, ht_decoder = self.decoder(decoder_input, ht_decoder)
            loss +=  self.criterion(output, output_tensor[i_step])
            topv, topi = output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input
            
            loss += self.criterion(output, output_tensor[i_step])
            if decoder_input.item() == EOS:
                break
            
        # back propagation
        loss.backward()

        # gradient steps parameters upgrade
        self.en_optimizer.step()
        self.de_optimizer.step()

        return loss.item() / target_length

    def plot(self, history):
        plt.figure()
        fig, ax = plt.subplots()
        # this locator puts ticks at regular intervals
        loc = ticker.MultipleLocator(base=0.2)
        ax.yaxis.set_major_locator(loc)
        plt.plot(history)

    def evaluate(self, tensors, n=10):
        word2ix = self.dictionaries['input'].word_to_idx
        for i in range(n):
            pair = random.choice(tensors)
            print('Input  |', pair[0])
            print('Output |', pair[1])
            sentence = [word2ix[w] for w in pair[0].split()] + [EOS]
            output_words = self.predict(sentence)
            output_sentence = '  '.join(output_words)            
            print('Pred   |', output_sentence)
            print('===='*20)
    
    def predict(self, sentence_encoded, **kw):
        ix2word = self.dictionaries['output'].idx_to_word
        with torch.no_grad():
            input_tensor = torch.tensor(sentence_encoded, device=DEVICE).view(-1, 1)
            input_length = input_tensor.size()[0]
            ht_encoder = self.encoder.initHidden()

            encoder_outputs = torch.zeros(self.max_length, self.encoder.hidden_size, device=DEVICE)

            for i_step in range(input_length):
                encoder_output, ht_encoder = self.encoder(input_tensor[i_step], ht_encoder)
                encoder_outputs[i_step] += encoder_output[0, 0]

            decoder_input = torch.tensor([[SOS]], device=DEVICE)  # SOS

            ht_decoder = ht_encoder

            decoded_words = []

            for _ in range(self.max_length):
                decoder_output, ht_decoder = self.decoder(decoder_input, ht_decoder)
                topv, topi = decoder_output.data.topk(1)
                if topi.item() == EOS:
                    decoded_words.append(ix2word[EOS])
                    break
                else:
                    decoded_words.append(ix2word[topi.item()])

                decoder_input = topi.squeeze().detach()
    
        return decoded_words

class AttentionModel(Model):
    
    def __init__(self):
        super().__init__()
        
        
if __name__ == "__main__":
    data = prepare_dataset(PATHS['eng'], PATHS['ara'])
    
    HIDDEN_SIZE = 128
    INPUT_SIZE = data.get('in_dictionary').n_words
    OUTPUT_SIZE = data.get('out_dictionary').n_words
    
    tensors = data.get('pairs_encoded')
    dictionaries = {'output': data.get('out_dictionary'),
                    'input' : data.get('in_dictionary')}
    
    model = Seq2seqModel(iterations=100, lr=0.01, hidden_size=HIDDEN_SIZE, dictionaries=dictionaries,
                         input_size=INPUT_SIZE, output_size=OUTPUT_SIZE, max_length=7)
    
    model.train(tensors=tensors[:5000])