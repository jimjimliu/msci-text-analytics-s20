import pandas as pd;
import os;

INPUT_PATH1 = os.getcwd()+'/sentiment/neg.txt';
INPUT_PATH2 = os.getcwd()+'/sentiment/pos.txt';

class Generator():

    '''
    read in input files and concatenate two/several files into one
    '''

    def generator(self):

        df_neg = pd.read_csv(r''+INPUT_PATH1,header=None,sep='\t', names=['data', 'target_name'])
        df_neg['target_name'] = 'NEG';
        df_neg['target'] = 0;

        df_pos = pd.read_csv(r'' + INPUT_PATH2, header=None, sep='\t', names=['data', 'target_name'])
        df_pos['target_name'] = 'POS';
        df_pos['target'] = 1;

        df = pd.concat([df_neg, df_pos])
        return df




# if __name__ == '__main__':
#     gen = Generator();
#     data = gen.generator()


