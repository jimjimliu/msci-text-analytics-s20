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

    def tokenize(self):

        df = wg.Generator().generator();
        df['sentence'] = df['sentence'].str.lower()
        df['tokens_with_sw'] = df.apply(self.remove_stops, axis=1, flag_remove_sw=False)
        df['tokens_without_sw'] = df.apply(self.remove_stops, axis=1, flag_remove_sw=True)

        # split the dataframe
        # use numpy to split the set into first 80%, 80%-90%(10%), and last 90%
        train_df, validate_df, test_df = np.split(df.sample(frac=1), [int(.8 * len(df)), int(.9 * len(df))])

        # tokened sets with stopwords
        train_set = train_df.apply(self.remove_stops, axis=1, flag_remove_sw=False)
        validate_set = validate_df.apply(self.remove_stops, axis=1, flag_remove_sw=False)
        test_set = test_df.apply(self.remove_stops, axis=1, flag_remove_sw=False)
        # tokened sets without stopwords
        train_set_wo = train_df.apply(self.remove_stops, axis=1, flag_remove_sw=True)
        validate_set_wo = validate_df.apply(self.remove_stops, axis=1, flag_remove_sw=True)
        test_set_wo = test_df.apply(self.remove_stops, axis=1, flag_remove_sw=True)

        self.write_csv(os.path.join(OUTPUT_PATH, 'with_stop_words.csv'), df['tokens_with_sw'])
        self.write_csv(os.path.join(OUTPUT_PATH, 'without_stop_words.csv'), df['tokens_without_sw'])
        self.write_csv(os.path.join(OUTPUT_PATH, 'train_with_sw.csv'), train_set)
        self.write_csv(os.path.join(OUTPUT_PATH, 'validate_with_sw.csv'), validate_set)
        self.write_csv(os.path.join(OUTPUT_PATH, 'test_with_sw.csv'), test_set)
        self.write_csv(os.path.join(OUTPUT_PATH, 'train_wo_sw.csv'), train_set_wo)
        self.write_csv(os.path.join(OUTPUT_PATH, 'validate_wo_sw.csv'), validate_set_wo)
        self.write_csv(os.path.join(OUTPUT_PATH, 'test_wo_sw.csv'), test_set_wo)


        return df

    def remove_stops(slef, row, flag_remove_sw):
        '''

        :param row:
        :param flag_remove_sw: remove stopwords when true, flase otherwise
        :return: a set contains the tokenized sentence of <row>
        '''

        label = row['target']
        my_list = row['sentence']
        str = re.sub('[!"#$%&()*+/:;<=>@[^`{|}~\t\n’!"\]#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\^_`{|}~]+', "",my_list)
        word_tokens = word_tokenize(str)
        if (flag_remove_sw):
            result = [w for w in word_tokens if not w in STOP_WORDS]
        else:
            result = word_tokens

        result.append(label)
        return result

    def write_csv(self, file_name, datas):
        if not os.path.exists(os.getcwd()+"/output"):
            os.mkdir(os.getcwd()+"/output")

        # field_names = ['data', 'target']
        file_csv = open(file_name, 'w')
        writer = csv.writer(file_csv, delimiter=',', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
        # writer.writerow(field_names)
        for data in datas:
            writer.writerow(data)

        file_csv.close()
        print("File saved in "+file_name+" , QUITING...")






if __name__ == '__main__':
    token = Tokenizer()
    df = token.tokenize()
    # print(df['tokens'])








