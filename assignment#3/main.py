import os
import sys
import traceback
from gensim.models import word2vec
import MyCorpus


class MyModel():

    def __init__(self, path):
        self.path = path
        self.output_path = os.path.join(os.getcwd(), 'data')
        self.model = self.__trainner()

    def __trainner(self):
        data = MyCorpus.MyCorpus(self.path).get_df()
        sentences = MyCorpus.MyCorpus()

        model = word2vec.Word2Vec(sentences=sentences, sg=1)
        model.save(os.path.join(self.output_path, 'w2v.model'))
        print('Model saved at >', os.path.join(self.output_path, 'w2v.model'), '<')

        return model


if __name__ == '__main__':

    try:
        params = sys.argv

        # if no path given, use default
        if (len(sys.argv) == 1):
            files_path = os.getcwd() + '/data'
            if not os.path.exists(files_path):
                raise Exception("Illegal path. QUITING...")
            print('Using >', files_path, '< as input files path.')
            MyModel(files_path)
        elif len(sys.argv) > 1 and len(sys.argv) <= 2:
            # path contains corpus
            files_path = params[1]
            if not os.path.exists(files_path):
                raise Exception("Illegal path. QUITING...")
            print('Using >', files_path, '< as input files path.')
            MyModel(files_path)
        else:
            raise Exception("Illegal path. QUITING...")

    except Exception as ex:
        print('Pass your data folder path, and only one argument is needed.')
        traceback.print_exc()
