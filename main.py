import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import jsonlines
import json
import re

from random import randrange

from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix

import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from itertools import chain
from scipy import sparse
seed = 42

import pickle

# Load test file
path = './snli_1.0/'
test_file = path + 'snli_1.0_test.jsonl'
# Load test data
test = []
with jsonlines.open(test_file) as f:
    for line in f.iter():
        test.append(line)
test = pd.DataFrame(test)

# Remove samples without gold_label
#test = test[test.gold_label != "-"]

# For preprocessing
nltk.download('stopwords')
nltk.download('punkt')
STOP_WORDS = nltk.corpus.stopwords.words('english') + list(string.punctuation)
snowBallStemmer = SnowballStemmer("english")
#print(len(STOP_WORDS))
#print(STOP_WORDS)

# Text Preprocessing
def preprocess(sentence):
    # Tokenize
    sentence = sentence.replace('\n', ' ').replace('\t', ' ').lower()
    sentence = re.sub('[^a-z ]', ' ', sentence)
    tokens = sentence.split(' ')
    tokens = [token for token in tokens if len(token) > 0]
    # Remove stop words
    #words = [i for i in tokens if i not in STOP_WORDS]
    words = [i for i in tokens if len(i) > 0]
    # Stemming
    stemWords = [snowBallStemmer.stem(word) for word in words]
    return stemWords

class_to_index = {"contradiction": 0, "entailment": 1, "neutral": 2}
index_to_class = {v:k for k, v in class_to_index.items()}

################## Part 1: tfidf using Logistic Regression #############################

# Load model and tfidf
with open('model/logistic_regression.pkl', 'rb') as f:
    log_reg = pickle.load(f) 
    
with open('model/tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f) 

tfidf_test = tfidf.transform(list(chain(*zip(test['sentence1'], test['sentence2']))))

X_test = tfidf_test.reshape((len(test), 2*len(tfidf.get_feature_names())))
y_test = list(test['gold_label'])

preds = log_reg.predict(X_test)
#print(preds)
log_reg_acc = log_reg.score(X_test, y_test)*100
print("Logistic Regression accuracy on test set:", log_reg_acc, "%")

with open("tfidf.txt", "w") as f:
	f.write("\n".join(preds))


############################## Part 2: Deep Model ######################################

with open('model/vocab_to_index.pkl', 'rb') as f:
    vocab_to_index = pickle.load(f) 

p_test = []
p_test_lens = []
h_test = []
h_test_lens = []
y_test = []
cnt = 0
def prepare_sample(row):
    sentence1 = row['sentence1']
    sentence2 = row['sentence2']
    sentence1_tokens = preprocess(sentence1)
    sentence2_tokens = preprocess(sentence2)
    input = ([], [])
    for token in sentence1_tokens:
        if(token in vocab_to_index):
            input[0].append(vocab_to_index[token])
    for token in sentence2_tokens:
        if(token in vocab_to_index):
            input[1].append(vocab_to_index[token])
    if(row['gold_label'] == '-'):
        output = -1
    else:
        output = class_to_index[row['gold_label']]
    return input, output

# Prepare test samples
for index, row in test.iterrows():
    input, output = prepare_sample(row) 
    if (len(input[0]) > 0 and len(input[1]) > 0):
        p_test.append(input[0])
        p_test_lens.append(len(input[0]))
        h_test.append(input[1])
        h_test_lens.append(len(input[1]))
        y_test.append(output)
        if(output == -1):
            cnt += 1

index_to_vocab = {v:k for k, v in vocab_to_index.items()}

n_vocab = len(vocab_to_index)
#print(n_vocab)

# Need to pad the sentences to max length
p_max_length = max(p_test_lens)
h_max_length = max(h_test_lens)

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Running on", device)

# Final Model
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

# Load saved model
model.load_state_dict(torch.load('model/deep_model'))

batch_size = 128

# Function to get loss and accuracy on test data
def evaluate(batch_size):
    model.eval()
    
    batches = get_test_batches
    predictions = []
    correct = 0
    for p, h, p_lens, h_lens, y in batches(batch_size):
        # move data to device
        p = torch.LongTensor(p).to(device)
        h = torch.LongTensor(h).to(device)
        
        # forward pass
        scores = model(p, h, p_lens, h_lens)
        
        preds = torch.argmax(scores, dim=1)
        predictions += list(preds.cpu().detach().numpy())
        correct += len([i for i in range(len(preds)) if preds[i] == y[i]])
        
    #accuracy = correct/len(y_test)
    return predictions, correct

predictions, correct = evaluate(batch_size)
lstm_acc = correct/len(y_test)*100
print("LSTM accuracy on test set:", lstm_acc, "%")

with open("deep_model.txt", "w") as f:
   f.write("\n".join([index_to_class[i] for i in predictions]))

print("Total test samples without gold label:", cnt)
print("Accuracies when not considering them:")
log_reg_acc2 = (log_reg_acc*len(y_test))/(len(y_test) - cnt)
print("Logistic regression on TFIDF: ", log_reg_acc2, "%")
lstm_acc2 = (100*correct)/(len(y_test) - cnt)
print("LSTM: ", lstm_acc2, "%")