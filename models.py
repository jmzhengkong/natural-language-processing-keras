import numpy as np
np.random.seed(1337)  # for reproducibility
import sys
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Input, Bidirectional, Activation, Reshape
from keras.layers import Conv1D, MaxPooling1D, Flatten
from sklearn.metrics import f1_score, accuracy_score
import os
import json


def rnn(json_hp_path, embeding, label_dims, maxlen):
    max_features = embeding.shape[0]
    embedding_dims = embeding.shape[1]
    with open(json_hp_path) as json_hp_file:
        hp = json.load(json_hp_file)
    dropout_rate = hp['dropout_rate']
    dense_dims = hp['dense1']
    lstm_dims = hp['lstm']

    model = Sequential()
    model.add(Embedding(max_features, embedding_dims,
                        input_length=maxlen, weights=[embeding]))
    model.add(Bidirectional(LSTM(lstm_dims)))
    #model.add(LSTM(lstm_dims))
    model.add(Dense(dense_dims))
    model.add(Dropout(dropout_rate))
    model.add(Activation('relu'))
    model.add(Dense(label_dims))
    model.add(Activation('softmax'))
    loss_type = 'categorical_crossentropy'

    model.compile(loss=loss_type, optimizer='adam', metrics=['accuracy'])
    return model


def cnn(json_hp_path, embeding, label_dims, maxlen):
    max_features = embeding.shape[0]
    embedding_dims = embeding.shape[1]
    with open(json_hp_path) as json_hp_file:
        hp = json.load(json_hp_file)

    model = Sequential()
    model.add(Embedding(max_features, embedding_dims,
                        input_length=maxlen, weights=[embeding]))
    model.add(Conv1D(
        filters=hp['filters'],
        kernel_size=hp['kernel_size'],
        padding='valid',
        activation='relu',
        strides=1))
    model.add(MaxPooling1D(pool_size=int(model.output_shape[1])))
    model.add(Flatten())
    model.add(Dense(hp['hidden_dim']))
    model.add(Dropout(hp['dropout_rate']))
    model.add(Activation('relu'))
    model.add(Dense(label_dims))
    model.add(Activation('softmax'))
    loss_type = 'categorical_crossentropy'

    model.compile(loss=loss_type, optimizer='adam', metrics=['accuracy'])
    return model
