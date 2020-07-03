import numpy as np
import tensorflow as tf
from tensorflow import keras
from MyCorpus import MyCorpus
import sys, os, traceback
from gensim.models import word2vec
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Flatten
from keras.layers import Embedding, LSTM, GRU, GlobalMaxPool1D
import pandas as pd
from tensorflow.keras import regularizers
from gensim.models import Word2Vec
from keras.utils import np_utils
import pickle
from keras.models import load_model


class main(object):

    def __init__(self, path=None):
        '''
        :param path: directory path contains 8 data splits
        '''

        # data folder
        self.__path = path
        if (not path):
            # if no path is given, use default
            self.__path = os.path.join(os.getcwd(), 'data')

        self.__data = MyCorpus(self.__path).get_raw_data()
        # path of pretrained word2vec model
        self.__model_path = os.path.join(self.__path, 'w2v.model')


    def classifier(self):

        # prepare input data
        X = pd.concat([self.__data['train'], self.__data['test'], self.__data['validate']])['data'].values
        X_train = self.__data['train']['data'].values
        y_train = self.__data['train']['target'].to_numpy()
        y_train = np_utils.to_categorical(y_train, 2)

        X_val = self.__data['validate']['data'].values
        y_val = self.__data['validate']['target'].to_numpy()
        y_val = np_utils.to_categorical(y_val, 2)

        X_test = self.__data['test']['data'].values
        y_test = self.__data['test']['target'].to_numpy()
        y_test = np_utils.to_categorical(y_test, 2)


        # load pre-trained word2vec model
        word2vec_model = word2vec.Word2Vec.load(self.__model_path)
        # words = list(word2vec_model.wv.vocab)
        # print('vocabulary size: %d' % len(words))

        # pad sequence
        maxlen = 27
        # word embedding dimension; for word2vec.model, the default dimension is 100, which is used previously in a3
        embedding_dim = 100

        '----------------------------convert words to vectors-----------------------------'
        # initialize tokenizer, setting the num_words = vocabulary size
        tokenizer = Tokenizer(80000)
        tokenizer.fit_on_texts(X)
        # how many words are tokenized
        word_index = tokenizer.word_index
        # print('found %s unique tokens.' %len(word_index))

        # save tokenzier
        with open(os.path.join(self.__path, 'tokenizer.pickle'), 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # vectorize a text corpus into a list of integers. Each integer maps to a value in a dictionary
        # that encodes the entire corpus, with the keys in the dictionary being the vocabulary terms themselves
        X_train = tokenizer.texts_to_sequences(X_train)
        X_val = tokenizer.texts_to_sequences(X_val)
        X_test = tokenizer.texts_to_sequences(X_test)
        # each text sequence has in most cases different length of words. simply pads the sequence of words with zeros
        X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
        X_val = pad_sequences(X_val, padding='post', maxlen=maxlen)
        X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

        '------------------------------END-----------------------------'

        '------------------------------map words into pretrained embeddings-------------------------------'
        num_words = len(word_index) + 1
        embedding_matrix = np.random.uniform(-0.05, 0.05, size=(num_words, embedding_dim))
        for word, i in word_index.items():
            try:
                embeddings_vector = word2vec_model[word]
            except KeyError:
                embeddings_vector = None
            if embeddings_vector is not None:
                embedding_matrix[i] = embeddings_vector
        # convert the wv word vectors into a numpy matrix that is suitable for insertion
        # into our TensorFlow and Keras models
        # embeddings_index ={}
        # f = open(os.path.join(self.__path, 'w2v_embedding.txt'), encoding="utf-8")
        # for line in f:
        #     values = line.split()
        #     word = values[0]
        #     coefs = np.asarray(values[1:])
        #     embeddings_index[word] = coefs
        # f.close()
        #
        # num_words = len(word_index)+1
        # embedding_matrix = np.zeros((num_words, embedding_dim))
        # for word, i in word_index.items():
        #     if i > num_words:
        #         continue
        #     embedding_vector = embeddings_index.get(word)
        #     if embedding_vector is not None:
        #         embedding_matrix[i] = embedding_vector

        '---------------------------------Train NN----------------------------------'
        model = Sequential()
        model.add(Embedding(input_dim=num_words,
                            output_dim=embedding_dim,
                            weights=[embedding_matrix],
                            trainable=False,
                            input_length=maxlen))
        model.add(Flatten())
        model.add(Dropout(0.1))
        model.add(Dense(units=520, activation="relu", kernel_regularizer=regularizers.l2(l=0.001)))
        # model.add(Dense(units=520, activation="sigmoid", kernel_regularizer=regularizers.l2(l=0.001)))
        # model.add(Dense(units=520, activation="tanh", kernel_regularizer=regularizers.l2(l=0.001)))
        model.add(Dropout(0.4))
        model.add(Dense(2, activation='softmax',  kernel_regularizer=regularizers.l2(l=0.001)))
        opt = keras.optimizers.Adam(learning_rate=0.0001)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        model.summary()
        model.fit(X_train, y_train, batch_size=500, epochs=10, validation_data=(X_val, y_val), verbose=2)
        loss, accuracy = model.evaluate(X_test, y_test)
        print('Accuracy: %f' % (accuracy * 100))
        # save the model
        model.save(os.path.join(self.__path, 'nn_relu.model'))
        print('Model save at: >', self.__path, '<')





if __name__ == '__main__':

    files_path = ''
    try:
        params = sys.argv

        # if no path given, use default
        if (len(sys.argv) == 1):
            files_path = os.path.join(os.getcwd(), 'data')
            if not os.path.exists(files_path):
                raise Exception("Illegal path. QUITING...")
            print('Using >', files_path, '< as input files path.')

        elif len(sys.argv) > 1 and len(sys.argv) <= 2:
            # path contains corpus
            files_path = params[1]
            if not os.path.exists(files_path):
                raise Exception("Illegal path. QUITING...")
        else:
            raise Exception("Illegal path. QUITING...")

        main(files_path).classifier()

    except Exception as ex:
        print('Pass your data folder path, and only one argument is needed.')
        traceback.print_exc()
