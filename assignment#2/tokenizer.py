import pandas as pd
import os;
from nltk.corpus import stopwords

STOP_WORDS = set(stopwords.words('english'));
OUTPUT_PATH = os.getcwd()+"/data"


class Tokenizer():

    def __init__(self, path):

        self.path = path
        # data files with stopwords
        self.train_csv = os.path.join(self.path, 'train_with_sw.csv')
        self.val_csv = os.path.join(self.path, 'validate_with_sw.csv')
        self.test_csv = os.path.join(self.path, 'test_with_sw.csv')
        # data files without stopwords
        self.train_ns_csv = os.path.join(self.path, 'train_wo_sw.csv')
        self.val_ns_csv = os.path.join(self.path, 'validate_wo_sw.csv')
        self.test_ns_csv = os.path.join(self.path, 'test_wo_sw.csv')

        self.sets = self.tokenize()

    def tokenize(self):

        '''
        readin 8 preprocessed csv files, and return a dictionary contains dataframes of 8 data splits files
        :return:
        '''


        train_df = pd.read_csv(r'' + self.train_csv, header=None, sep='\t', names=['data'])
        train_df['target'] = train_df['data'].map(lambda x: x[-1])
        train_df['data'] = train_df['data'].map(lambda x: x[:-1])
        train_df['data'] = train_df['data'].str.replace(',', ' ')

        val_df = pd.read_csv(r'' + self.val_csv, header=None, sep='\t', names=['data'])
        val_df['target'] = val_df['data'].map(lambda x: x[-1])
        val_df['data'] = val_df['data'].map(lambda x: x[:-1])
        val_df['data'] = val_df['data'].str.replace(',', ' ')

        test_df = pd.read_csv(r'' + self.test_csv, header=None, sep='\t', names=['data'])
        test_df['target'] = test_df['data'].map(lambda x: x[-1])
        test_df['data'] = test_df['data'].map(lambda x: x[:-1])
        test_df['data'] = test_df['data'].str.replace(',', ' ')

        train_ns_df = pd.read_csv(r'' + self.train_ns_csv, header=None, sep='\t', names=['data'])
        train_ns_df['target'] = train_ns_df['data'].map(lambda x: x[-1])
        train_ns_df['data'] = train_ns_df['data'].map(lambda x: x[:-1])
        train_ns_df['data'] = train_ns_df['data'].str.replace(',', ' ')

        val_ns_df = pd.read_csv(r'' + self.val_ns_csv, header=None, sep='\t', names=['data'])
        val_ns_df['target'] = val_ns_df['data'].map(lambda x: x[-1])
        val_ns_df['data'] = val_ns_df['data'].map(lambda x: x[:-1])
        val_ns_df['data'] = val_ns_df['data'].str.replace(',', ' ')

        test_ns_df = pd.read_csv(r'' + self.test_ns_csv, header=None, sep='\t', names=['data'])
        test_ns_df['target'] = test_ns_df['data'].map(lambda x: x[-1])
        test_ns_df['data'] = test_ns_df['data'].map(lambda x: x[:-1])
        test_ns_df['data'] = test_ns_df['data'].str.replace(',', ' ')


        # df = wg.Generator().generator()[:100];
        # df['data'] = df['data'].str.lower()
        #
        # # remove special characters and filter out/ or not the stopwords
        # df['tokens_with_sw'] = df.apply(self.remove_stops, axis=1, flag_remove_sw=False)
        # df['tokens_without_sw'] = df.apply(self.remove_stops, axis=1, flag_remove_sw=True)
        #
        # # split the dataframe
        # # use numpy to split the set into first 80%, 80%-90%(10%), and last 90%
        # train_df, validate_df, test_df = np.split(df.sample(frac=1), [int(.8 * len(df)), int(.9 * len(df))])



        result_dic = {
            'train': train_df,
            'validate': val_df,
            'test': test_df,
            'train_ns': train_ns_df,
            'val_ns': val_ns_df,
            'test_ns': test_ns_df
        }

        return result_dic





# if __name__ == '__main__':
#     path = os.getcwd()+'/data'
#     token = Tokenizer(path)
#     df = token.tokenize()








