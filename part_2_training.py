import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
#import jsonlines
#import json
import re
import pickle
import os

#from tqdm import tqdm_notebook as tqdm
from tqdm import tqdm

from random import randrange

from sklearn.metrics import confusion_matrix

import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from scipy import sparse

#%matplotlib inline
seed = 42

class_to_index = {"contradiction": 0, "entailment": 1, "neutral": 2}
index_to_class = {v:k for k, v in class_to_index.items()}

if os.path.exists("processed_data.pickle"):
    with open('processed_data.pickle', 'rb') as handle:
        processed_lists = pickle.load(handle)
    vocab_to_index = processed_lists[0]
    p_train = processed_lists[1]
    p_train_lens = processed_lists[2]
    h_train = processed_lists[3]
    h_train_lens = processed_lists[4]
    y_train = processed_lists[5]
    p_validate = processed_lists[6]
    p_validate_lens = processed_lists[7]
    h_validate = processed_lists[8]
    h_validate_lens = processed_lists[9]
    y_validate = processed_lists[10]
    p_test = processed_lists[11]
    p_test_lens = processed_lists[12]
    h_test = processed_lists[13]
    h_test_lens = processed_lists[14]
    y_test = processed_lists[15]
    
else: 
	print("preprocessed input not found!")
	exit(1)

# Merge train and validate lists
p_train = p_train + p_validate
h_train = h_train + h_validate
p_train_lens = p_train_lens + p_validate_lens
h_train_lens = h_train_lens + h_validate_lens
y_train = y_train + y_validate

n_vocab = len(vocab_to_index)

index_to_vocab = {v:k for k, v in vocab_to_index.items()}

# Need to pad the sentences to max length
p_max_length = max(max(p_train_lens), max(p_validate_lens), max(p_test_lens))
h_max_length = max(max(h_train_lens), max(h_validate_lens), max(h_test_lens))

def get_train_batches(batch_size):
    num_batches = len(p_train)//batch_size
    last_batch_size = len(p_train) - num_batches*batch_size
    for b in range(0, len(p_train), batch_size):
        padded_p = np.ones((batch_size, p_max_length), dtype=int) * vocab_to_index['<PAD>']
        padded_h = np.ones((batch_size, h_max_length), dtype=int) * vocab_to_index['<PAD>']
        for i, p_len in enumerate(p_train_lens[b:b+batch_size]):
            sequence = p_train[i+b]
            padded_p[i, 0:p_len] = sequence[:p_len]
        for i, h_len in enumerate(h_train_lens[b:b+batch_size]):
            sequence = h_train[i+b]
            padded_h[i, 0:h_len] = sequence[:h_len]
        
        if(b + batch_size <= len(p_train)):
            yield padded_p, padded_h, p_train_lens[b:b+batch_size], h_train_lens[b:b+batch_size], \
                np.array(y_train[b:b+batch_size])
        else:
            yield padded_p[:last_batch_size-1], padded_h[:last_batch_size-1], \
                p_train_lens[b:b+last_batch_size-1], h_train_lens[b:b+last_batch_size-1], \
                np.array(y_train[b:b+last_batch_size-1])

def get_test_batches(batch_size):
    num_batches = len(p_test)//batch_size
    last_batch_size = len(p_test) - num_batches*batch_size
    for b in range(0, len(p_test), batch_size):
        padded_p = np.ones((batch_size, p_max_length), dtype=int) * vocab_to_index['<PAD>']
        padded_h = np.ones((batch_size, h_max_length), dtype=int) * vocab_to_index['<PAD>']
        for i, p_len in enumerate(p_test_lens[b:b+batch_size]):
            sequence = p_test[i+b]
            padded_p[i, 0:p_len] = sequence[:p_len]
        for i, h_len in enumerate(h_test_lens[b:b+batch_size]):
            sequence = h_test[i+b]
            padded_h[i, 0:h_len] = sequence[:h_len]
        
        if(b + batch_size <= len(p_test)):
            yield padded_p, padded_h, p_test_lens[b:b+batch_size], h_test_lens[b:b+batch_size], \
                np.array(y_test[b:b+batch_size])
        else:
            yield padded_p[:last_batch_size-1], padded_h[:last_batch_size-1], \
                p_test_lens[b:b+last_batch_size-1], h_test_lens[b:b+last_batch_size-1], \
                np.array(y_test[b:b+last_batch_size-1])

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = 'cpu'
print("Running on", device)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class LSTM(nn.Module):
    def __init__(self, hyperparams):
        super(LSTM, self).__init__()
        self.hyperparams = hyperparams
        self.emb = nn.Embedding(hyperparams['vocab_size'], hyperparams['emb_dim'], padding_idx=0)
        #self.proj = nn.Linear(hyperparams['emb_dim'], hyperparams['proj_dim'])
        self.word_dropout = nn.Dropout(p=hyperparams['word_dropout'])
        self.dropout = nn.Dropout(p=hyperparams['dropout'])
        self.lstm = nn.LSTM(hyperparams['emb_dim'], 
                            hyperparams['lstm_hidden_dim'], 
                            num_layers=hyperparams['num_lstm_layers'],
                            dropout=hyperparams['lstm_dropout'],
                            bidirectional = hyperparams['bidirectional'])
        # self.lstm_h = nn.LSTM(hyperparams['emb_dim'], 
        #                     hyperparams['hidden_dim'], 
        #                     num_layers=hyperparams['num_lstm_layers'],
        #                     bidirectional = hyperparams['bidirectional'])
        self.relu = nn.ReLU()
        self.fc = nn.Sequential(
                        nn.Linear(2*hyperparams['lstm_hidden_dim'], hyperparams['hidden_dim']),
                        self.relu,
                        self.dropout,
                        nn.Linear(hyperparams['hidden_dim'], hyperparams['hidden_dim']),
                        self.relu,
                        self.dropout,
                        # nn.Linear(hyperparams['hidden_dim'], hyperparams['hidden_dim']),
                        # self.relu,
                        # self.dropout,
                        nn.Linear(hyperparams['hidden_dim'], hyperparams['num_classes'])
                    )
        
    def forward(self, premise, hypothesis, p_seq_lens, h_seq_lens):
        premise_emb = self.emb(premise)
        hypothesis_emb = self.emb(hypothesis)
        
        premise_emb = torch.transpose(premise_emb, 0, 1)
        hypothesis_emb = torch.transpose(hypothesis_emb, 0, 1)
        
        premise_emb = self.word_dropout(premise_emb)
        hypothesis_emb = self.word_dropout(hypothesis_emb)

        lstm_input_p = nn.utils.rnn.pack_padded_sequence(premise_emb, p_seq_lens, batch_first=False, enforce_sorted = False)
        lstm_input_h = nn.utils.rnn.pack_padded_sequence(hypothesis_emb, h_seq_lens, batch_first=False, enforce_sorted = False)

        p_lstm_out, (p_ht, p_hc) = self.lstm(lstm_input_p)    #inital hidden weights by default will be zero
        h_lstm_out, (h_ht, h_hc) = self.lstm(lstm_input_h)    #inital hidden weights by default will be zero
        
        #p_out = nn.utils.rnn.pad_packed_sequence(p_lstm_out, batch_first=False)
        #h_out = nn.utils.rnn.pad_packed_sequence(h_lstm_out, batch_first=False)
        #print(len(p_out))
        #print(p_out[0].shape)
        
        if(self.hyperparams['bidirectional']):
            p_ht_forward = p_ht.view(self.hyperparams['num_lstm_layers'], 2, -1, self.hyperparams['lstm_hidden_dim'])[-1, 0]
            p_ht_backward = p_ht.view(self.hyperparams['num_lstm_layers'], 2, -1, self.hyperparams['lstm_hidden_dim'])[-1, 1]
            p_fc_in = torch.stack([p_ht_forward, p_ht_backward], dim=0).sum(dim=0)
            
            h_ht_forward = h_ht.view(self.hyperparams['num_lstm_layers'], 2, -1, self.hyperparams['lstm_hidden_dim'])[-1, 0]
            h_ht_backward = h_ht.view(self.hyperparams['num_lstm_layers'], 2, -1, self.hyperparams['lstm_hidden_dim'])[-1, 1]
            h_fc_in = torch.stack([h_ht_forward, h_ht_backward], dim=0).sum(dim=0)

        else:
            p_fc_in = p_ht.view(self.hyperparams['num_lstm_layers'], 1, -1, self.hyperparams['lstm_hidden_dim'])[-1][0]
            h_fc_in = h_ht.view(self.hyperparams['num_lstm_layers'], 1, -1, self.hyperparams['lstm_hidden_dim'])[-1][0]

        #last layer
        combined = torch.cat((p_fc_in, h_fc_in), dim=1)
        output = self.fc(combined)
        return F.log_softmax(output, dim=1)

model_num = 6
model_hyperparams = {
    'vocab_size': n_vocab,
    'word_dropout': 0.1,
    'emb_dim': 512,

    'lstm_dropout': 0.3,
    'num_lstm_layers': 2,
    'lstm_hidden_dim': 256,
    'bidirectional': True,
    
    'dropout': 0.5,
    'hidden_dim': 256,

    'num_classes': 3
}
model = LSTM(model_hyperparams).to(device)

print("Parameters:", count_parameters(model))

num_epochs = 20
batch_size = 128

lr = 0.001

loss_function = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Function to get loss and accuracy on validate/test/train data
def evaluate(batch_size, data='test'):
    model.eval()
    
    if(data == 'train'):
        batches = get_train_batches
    else:
        batches = get_test_batches
    losses = []
    correct = 0
    for p, h, p_lens, h_lens, y in batches(batch_size):
        # move data to device
        p = torch.LongTensor(p).to(device)
        h = torch.LongTensor(h).to(device)
        
        # forward pass
        scores = model(p, h, p_lens, h_lens)
        
        # Get loss
        loss = loss_function(scores, torch.LongTensor(y).to(device))
        loss_value = loss.item()
        losses.append(loss_value)
        
        preds = torch.argmax(scores, dim=1)
        correct += len([i for i in range(len(preds)) if preds[i] == y[i]])
        
    if(data == 'train'):
        accuracy = correct/len(y_train)
    else:
        accuracy = correct/len(y_test)
        
    return np.mean(losses), accuracy


epoch_losses = []
test_losses = []
iteration = 0
for epoch in range(1, num_epochs+1):
    running_loss = []
    for p, h, p_lens, h_lens, y in get_train_batches(batch_size):
        iteration += 1
        # training mode
        model.train()
        # reset gradients
        model.zero_grad()  
        #optimizer.zero_grad() - can also be used. When we have multiple optimizers on same model, better to use model.zero_grad()

        # move data to device
        p = torch.LongTensor(p).to(device)
        h = torch.LongTensor(h).to(device)

        # forward pass
        scores = model(p, h, p_lens, h_lens)

        # Get loss
        loss = loss_function(scores, torch.LongTensor(y).to(device))
        loss_value = loss.item()
        running_loss.append(loss_value)

        # Backpropagation
        loss.backward()

        # Optimization
        optimizer.step()

        if iteration % 300 == 0:
            test_loss, test_accuracy = evaluate(batch_size, data='test')
            print('Epoch: {}/{}'.format(epoch, num_epochs),
                      'Iteration: {}'.format(iteration),
                      'Current Loss: {:.4f}'.format(loss_value),
                      'Test Loss: {:.4f}'.format(test_loss),
                      'Test Accuracy: {:.2f}%'.format(test_accuracy*100))
    
    # Losses after this epoch 
    # Average trainin loss
    epoch_losses.append(np.mean(running_loss))
    # Average test loss
    test_loss, test_accuracy = evaluate(batch_size, data='test')
    test_losses.append(test_loss)
    print("------------------------------------------------------------------------------------------------------")
    print('Epoch: {}/{}'.format(epoch, num_epochs),
              'Iterations: {}'.format(iteration),
              'Current Loss: {:.4f}'.format(loss_value),
              'Test Loss: {:.4f}'.format(test_loss),
              'Test Accuracy: {:.2f}%'.format(test_accuracy*100))
    print("______________________________________________________________________________________________________")

train_loss, train_accuracy = evaluate(batch_size, data='train')
print("Train:")
print("Loss:", train_loss, "Accuracy: ",  train_accuracy*100)

test_loss, test_accuracy = evaluate(batch_size, data='test')
print("Test:")
print("Loss:", test_loss, "Accuracy: ", test_accuracy*100)

# Save model
torch.save(model.state_dict(), "saved_model_" + str(model_num) + "_f")

# Save losses for plotting
fileName = "model_" + str(model_num)
with open(fileName + "_train.txt", "w") as output:
    output.write(str(epoch_losses))
with open(fileName + "_test.txt", "w") as output:
    output.write(str(test_losses))
