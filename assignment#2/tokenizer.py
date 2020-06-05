# download nltk data first if not installed before

import nltk
# import ssl
#
# try:
#     _create_unverified_https_context = ssl._create_unverified_context
# except AttributeError:
#     pass
# else:
#     ssl._create_default_https_context = _create_unverified_https_context
#
# nltk.download()

import pandas as pd
import nltk
import re;
import csv;
import os;
import words_generator as wg;
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np

STOP_WORDS = set(stopwords.words('english'));
OUTPUT_PATH = os.getcwd()+"/output"


class Tokenizer():

    def __init__(self):

        self.sets = self.tokenize()

    def tokenize(self):

        df = wg.Generator().generator();
        df['data'] = df['data'].str.lower()
        '''
        remove special characters and filter out/ or not the stopwords
        '''
        df['tokens_with_sw'] = df.apply(self.remove_stops, axis=1, flag_remove_sw=False)
        df['tokens_without_sw'] = df.apply(self.remove_stops, axis=1, flag_remove_sw=True)

        # split the dataframe
        # use numpy to split the set into first 80%, 80%-90%(10%), and last 90%
        train_df, validate_df, test_df = np.split(df.sample(frac=1), [int(.8 * len(df)), int(.9 * len(df))])

        # tokened sets with stopwords
        train_set = train_df['tokens_with_sw']
        validate_set = validate_df['tokens_with_sw']
        test_set = test_df['tokens_with_sw']
        # tokened sets without stopwords
        train_set_wo = train_df['tokens_without_sw']
        validate_set_wo = validate_df['tokens_without_sw']
        test_set_wo = test_df['tokens_without_sw']

        result_dic = {
            'train_set': train_df,
            'validate_set': validate_df,
            'test_set': test_df
        }

        return result_dic

    def remove_stops(slef, row, flag_remove_sw):
        '''

        :param row:
        :param flag_remove_sw: remove stopwords when true, flase otherwise
        :return: a set contains the tokenized sentence of <row>
        '''

        my_list = row['data']
        str = re.sub('[!"#$%&()*+/:;<=>@[^`{|}~\t\n’!"\]#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\^_`{|}~]+', "",my_list)
        word_tokens = word_tokenize(str)
        if (flag_remove_sw):
            result = [w for w in word_tokens if not w in STOP_WORDS]
        else:
            result = word_tokens

        return result

    def write_csv(self, file_name, datas):
        if not os.path.exists(os.getcwd()+"/output"):
            os.mkdir(os.getcwd()+"/output")

        file_csv = open(file_name, 'w')
        writer = csv.writer(file_csv, delimiter=',', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
        for data in datas:
            writer.writerow(data)

        file_csv.close()
        print("File saved in "+file_name+" , QUITING...")





#
# if __name__ == '__main__':
#     token = Tokenizer()
#     df = token.tokenize()








