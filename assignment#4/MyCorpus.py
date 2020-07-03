import pandas as pd
import os;
from nltk.corpus import stopwords
import re

STOP_WORDS = set(stopwords.words('english'));
OUTPUT_PATH = os.getcwd()+"/data"


class MyCorpus():

    def __init__(self, path=None):

        # data folder
        self.path = path
        if (not path):
            # if no path is given, use default
            self.path = os.path.join(os.getcwd(), 'data')

        # data files with stopwords
        self.train_csv = os.path.join(self.path, 'train_with_sw.csv')
        self.val_csv = os.path.join(self.path, 'validate_with_sw.csv')
        self.test_csv = os.path.join(self.path, 'test_with_sw.csv')
        # data files without stopwords
        self.train_ns_csv = os.path.join(self.path, 'train_wo_sw.csv')
        self.val_ns_csv = os.path.join(self.path, 'validate_wo_sw.csv')
        self.test_ns_csv = os.path.join(self.path, 'test_wo_sw.csv')

        self.__sets = self.__reader()

    def __reader(self):

        '''
        readin 8 preprocessed csv files, and return a dictionary contains dataframes of 8 data splits files
        :return:
            (Dictionary) result_dic={string, DataFrame}
        '''


        train_df = pd.read_csv(r'' + self.train_csv, header=None, sep='\t', names=['data'])
        # split the last character, it is a digit, which is the label of each row
        train_df['target'] = train_df['data'].map(lambda x: int(x[-1]))
        # delete the last character from the content
        train_df['data'] = train_df['data'].map(lambda x: x[:-1])
        train_df['data'] = train_df['data'].str.replace(',', ' ')

        val_df = pd.read_csv(r'' + self.val_csv, header=None, sep='\t', names=['data'])
        val_df['target'] = val_df['data'].map(lambda x: int(x[-1]))
        val_df['data'] = val_df['data'].map(lambda x: x[:-1])
        val_df['data'] = val_df['data'].str.replace(',', ' ')

        test_df = pd.read_csv(r'' + self.test_csv, header=None, sep='\t', names=['data'])
        test_df['target'] = test_df['data'].map(lambda x: int(x[-1]))
        test_df['data'] = test_df['data'].map(lambda x: x[:-1])
        test_df['data'] = test_df['data'].str.replace(',', ' ')

        train_ns_df = pd.read_csv(r'' + self.train_ns_csv, header=None, sep='\t', names=['data'])
        train_ns_df['target'] = train_ns_df['data'].map(lambda x: int(x[-1]))
        train_ns_df['data'] = train_ns_df['data'].map(lambda x: x[:-1])
        train_ns_df['data'] = train_ns_df['data'].str.replace(',', ' ')

        val_ns_df = pd.read_csv(r'' + self.val_ns_csv, header=None, sep='\t', names=['data'])
        val_ns_df['target'] = val_ns_df['data'].map(lambda x: int(x[-1]))
        val_ns_df['data'] = val_ns_df['data'].map(lambda x: x[:-1])
        val_ns_df['data'] = val_ns_df['data'].str.replace(',', ' ')

        test_ns_df = pd.read_csv(r'' + self.test_ns_csv, header=None, sep='\t', names=['data'])
        test_ns_df['target'] = test_ns_df['data'].map(lambda x: int(x[-1]))
        test_ns_df['data'] = test_ns_df['data'].map(lambda x: x[:-1])
        test_ns_df['data'] = test_ns_df['data'].str.replace(',', ' ')


        result_dic = {
            'train': train_df,
            'validate': val_df,
            'test': test_df,
            'train_ns': train_ns_df,
            'val_ns': val_ns_df,
            'test_ns': test_ns_df
        }

        return result_dic

    def get_raw_data(self):
        return self.__sets

    def get_max_len(self):
        data = self.__sets
        X = pd.concat([data['train'], data['test'], data['validate']])
        max_len = 0
        index = 0
        for i, row in X.iterrows():
            s = row['data'].split()
            if len(s) > max_len:
                max_len = len(s)
                index = i
        # print(X.iloc[index])
        return max_len



if __name__ == '__main__':
    path = os.getcwd()+'/data'
    corpus = MyCorpus(path)
    df = corpus.get_raw_data()









