import os
from gensim.models import word2vec
import pandas as pd
import sys
import traceback



class Inference():

    def __init__(self, path):

        self.model_path = os.path.join(os.getcwd(), 'data', 'w2v.model')
        self.words_path = path

    def cal_similar_token(self):

        print('Using model saved at >', self.model_path, '<')
        # load model
        model = word2vec.Word2Vec.load(self.model_path)

        pos_df = pd.read_csv(r'' + self.words_path, delimiter=None, header=None, sep='\t', names=['word'])

        result_df = pd.DataFrame(columns=['word' ,'most_similar', 'similarity'])

        for index, row in pos_df.iterrows():
            for e in model.wv.most_similar(row['word'], topn=20):
            # for e in model.wv.most_similar(positive=[row['word'], 'good'], negative=['bad'], topn=20):
                print(e[0], e[1])
                result_df = result_df.append([{'word':row['word'], 'most_similar': e[0], 'similarity': e[1]}], ignore_index=False)

        result_df.to_csv(os.path.join(os.getcwd(), 'data', 'similar_terms.csv'), index=False)




if __name__ == '__main__':

    try:
        params = sys.argv

        # if no path given, use default
        if (len(sys.argv) == 1):
            files_path = os.path.join(os.getcwd(), 'data', 'words.txt')
            if not os.path.exists(files_path):
                raise Exception("Illegal path. QUITING...")
            print('Using >', files_path, '< as input files path.')
            Inference(files_path).cal_similar_token()
        elif len(sys.argv) > 1 and len(sys.argv)<=2:
            # path contains corpus
            files_path = params[1]
            if not os.path.exists(files_path):
                raise Exception("Illegal path. QUITING...")
            Inference(files_path).cal_similar_token()
        else:
            raise Exception("Illegal path. QUITING...")

    except Exception as ex:
        print('Pass your data folder path, and only one argument is needed.')
        traceback.print_exc()