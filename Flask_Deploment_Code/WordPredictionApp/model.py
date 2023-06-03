"""# Importing required packages"""
import tensorflow as tf
import string
import requests
import numpy as np
import keras
from keras import backend as K
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

"""# Getting data from the site suing Http request"""
response = requests.get('https://www.gutenberg.org/files/98/98-0.txt')
data = response.text.split('\n')
data = data[108:]
data = " ".join(data)

"""# Clean function to clean the text file"""
def clean_text(doc):
  tokens = doc.split()
  table = str.maketrans('', '', string.punctuation)
  tokens = [w.translate(table) for w in tokens]
  tokens = [word for word in tokens if word.isalpha()]
  tokens = [word.lower() for word in tokens]
  return tokens

"""# Cleaning the data and getting tokens means words"""
tokens = clean_text(data)

"""# Setting the train length and getting sequences"""
train_len = 5+1
text_sequences = []
for i in range(train_len,len(tokens)):
    seq = tokens[i-train_len:i]
    text_sequences.append(seq)

sequences = {}
count = 1
for i in range(len(tokens)):
    if tokens[i] not in sequences:
        sequences[tokens[i]] = count
        count += 1

"""# Now we actually use the tokenizer from nltk library to convert the words or tokens in numeric values"""
tokenizer = Tokenizer()
tokenizer.fit_on_texts(text_sequences)
sequences = tokenizer.texts_to_sequences(text_sequences)

"""# Now we get the vocabulary size and set train inputs and targets"""
vocabulary_size = len(tokenizer.word_counts)+1

n_sequences = np.empty([len(sequences),train_len], dtype='int32')
for i in range(len(sequences)):
    n_sequences[i] = sequences[i]

train_inputs = n_sequences[:,:-1]
train_targets = n_sequences[:,-1]
train_targets = to_categorical(train_targets, num_classes=vocabulary_size)
seq_len = train_inputs.shape[1]


def recall_m(train_inputs, train_targets):
    true_positives = K.sum(K.round(K.clip(train_inputs * train_targets, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(train_inputs, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(train_inputs, train_targets):
    true_positives = K.sum(K.round(K.clip(train_inputs * train_targets, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(train_targets, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(train_inputs, train_targets):
    precision = precision_m(train_inputs, train_targets)
    recall = recall_m(train_inputs, train_targets)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


dependencies={
    'f1_m':f1_m,
    'precision_m':precision_m,
    'recall_m':recall_m}
    
"""Loading our previously trained model which is saved as nxtwordmodel.h5"""
nextwmodel=keras.models.load_model("nextwordmodel1.h5",custom_objects=dependencies)
