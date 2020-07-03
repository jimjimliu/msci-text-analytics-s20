import os
import pandas as pd
import sys
import traceback
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import array
import pickle

class Inference():

    def __init__(self, path=None, act_func='sigmoid'):
        '''
        :param path: (String) input txt file path
        :param act_func: (String) activation function of sigmoid, relu, and tanh; default=sigmoid
        '''

        self.__path = path
        # if no path is given, use default
        if (not path):
            self.__path = os.path.join(os.getcwd(), 'data', 'sentences.txt')

        # if input activation function is not one of the three, raise exception
        try:
            self.__act_func = act_func
            if self.__act_func.lower() not in ['sigmoid', 'tanh', 'relu']:
                raise Exception("The activation function should be one of the three: [sigmoid, tanh, relu]")
        except Exception as ex:
            traceback.print_exc()

        self.__path_dict = {
            'sigmoid': os.path.join(os.getcwd(), 'data', 'nn_sigmoid.model'),
            'relu': os.path.join(os.getcwd(), 'data', 'nn_relu.model'),
            'tanh': os.path.join(os.getcwd(), 'data', 'nn_tanh.model')
        }
        self.__model_path = self.__path_dict[self.__act_func]


    def classifier(self):

        sentences = pd.read_csv(r'' + self.__path, sep='\t', names=['data'])

        # load model
        model = load_model(self.__model_path)
        # model.summary()

        # load tokenizer
        with open(os.path.join(os.getcwd(), 'data', 'tokenizer.pickle'), 'rb') as handle:
            tokenizer = pickle.load(handle)
        # print(len(tokenizer.word_index))

        instance = tokenizer.texts_to_sequences(sentences['data'].values)
        instance = pad_sequences(instance, padding='post', maxlen=27)
        print(model.predict(instance))
        print(np.argmax(model.predict(instance), axis=-1))




if __name__ == '__main__':

    try:
        params = sys.argv

        # if no path given, use default
        if (len(sys.argv) == 1):
            files_path = os.path.join(os.getcwd(), 'data', 'sentences.txt')
            act_func = 'relu'
            if not os.path.exists(files_path):
                raise Exception("Illegal path. QUITING...")
            print('Using >', files_path, '< as input files path.')

        elif len(sys.argv) > 1 and len(sys.argv)<=3:
            # path contains corpus
            files_path = params[1]
            act_func = params[2]
            if not os.path.exists(files_path):
                raise Exception("Illegal path. QUITING...")

        else:
            raise Exception("Illegal path. QUITING...")

        Inference(files_path, act_func).classifier()

    except Exception as ex:
        traceback.print_exc()