from gensim.test.utils import datapath
from gensim import utils
import pandas as pd
import os
import re

class MyCorpus(object):
    """An interator that yields sentences (lists of str)."""

    def __init__(self, path=None):

        self.path = path
        if (not path):
            self.path = os.getcwd() + '/data'

        self.data = self.__reader()

    def __iter__(self):

        for index, row in self.data.iterrows():
            # assume there's one document per line, tokens separated by whitespace
            yield utils.simple_preprocess(row['sentence'])


    def __reader(self):

        path1 = os.path.join(self.path, 'pos.txt')
        path2 = os.path.join(self.path, 'neg.txt')

        # target name is stance
        pos_df = pd.read_csv(r'' + path1, delimiter=None,  header=None, sep='\t', names=['sentence'])
        pos_df['target_name'] = 'POS'
        pos_df['target'] = 1
        neg_df = pd.read_csv(r'' + path2, delimiter=None, header=None, sep='\t', names=['sentence'])
        neg_df['target_name'] = 'NEG'
        neg_df['target'] = 0

        df = pd.concat([pos_df, neg_df])
        # drop rows contain missing value
        df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
        # clean the data
        df['sentence'] = df.apply(self.__preprocess, axis=1, column='sentence')

        return df

    def __preprocess(self, dataframe, column):

        data = dataframe[column]
        str = re.sub('[!"#$%&()*+/:;<=>@[^`{|}~\t\n’!"\]#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\^_`{|}~]+', "", data)
        # remove numbers
        str = re.sub('[0-9]', '', str)

        return str.lower()


    def get_df(self):
        return self.data

# if __name__ == '__main__':
#     sentences = MyCorpus()
#     df = sentences.get_df()
#     print(df[7:15])
