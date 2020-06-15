import tokenizer as tokens
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import preprocessing
import os
from multiprocessing import Process, Manager
import pickle
import sys
import traceback

le = preprocessing.LabelEncoder()


INPUT_PATH = os.getcwd()+'/data'



class MNB():

    def __init__(self, path):
        # path of corpus splits
        self.path = path

    def mnb(self):

        '''

        :param iter: iteration times. default =10. Repeat calculating the model accuracy for 10 times.
        :return:
        '''

        '''prepare train/test data'''
        raw_data = tokens.Tokenizer(self.path).sets

        # initialize a dictionary store the result returned by functions in each Process
        result_dic = Manager().dict()
        keys = ['uni', 'bi', 'uni_bi', 'uni_ns', 'bi_ns', 'uni_bi_ns']
        for item in keys:
            result_dic[item] = 0


        print('Parent process %s.' % os.getpid())

        '''using data with stopwords'''
        # using unigram mnb
        p1 = Process(target=self.classify, args=(raw_data, True, result_dic, 'uni',  1, 1, 'mnb_uni', ))
        # using bigram mnb
        p2 = Process(target=self.classify, args=(raw_data, True, result_dic, 'bi', 2, 2, 'mnb_bi', ))
        # using unigram + bigram mnb
        p3 = Process(target=self.classify, args=(raw_data, True, result_dic, 'uni_bi', 1, 2, 'mnb_uni_bi',))

        '''using data without stopwords'''
        # unigram
        p4 = Process(target=self.classify, args=(raw_data, False, result_dic, 'uni_ns', 1, 1, 'mnb_uni_ns', ))
        # bigram
        p5 = Process(target=self.classify, args=(raw_data, False, result_dic, 'bi_ns', 2, 2, 'mnb_bi_ns',))
        # unigram + bigram
        p6 = Process(target=self.classify, args=(raw_data, False, result_dic, 'uni_bi_ns', 1, 2, 'mnb_uni_bi_ns',))

        print('Child process will start.')
        proc_arr = [p1, p2, p3, p4, p5, p6]
        for item in proc_arr:
            item.start()
        for item in proc_arr:
            item.join()
        print('Child process end.')


        result_dic = {k : v for k, v in result_dic.items()}
        print(result_dic)
        return result_dic



    def classify(self, raw_data, stopwords, result_dic, key, n_head=1, n_tail=1, model_name='mnb_default'):

        '''
         using naive bayes: multinomial NB to train and test model using the data set provided by Tokenizer.py.
        The parameters are the boundary of the range of different n-grams the model is using.

        :param n_head: The lower boundary of the range of n-values for different word n-grams or char n-grams to be extracted
        :param n_tail: The upper boundary of the range of n-values for different word n-grams or char n-grams to be extracted
        :param analyzer: string, {‘word’, ‘char’, ‘char_wb’} or callable, default=’word’.
                        Whether the feature should be made of word n-gram or character n-grams.
                        [https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.
                        html#sklearn.feature_extraction.text.CountVectorizer]
        :param stopwords: string {‘english’}, list, default=None.
                    If ‘english’, a built-in stop word list for English is used.
                    [https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html#sklearn.
                    feature_extraction.text.CountVectorizer]
        :param result_dic: a dictionary stores the accumulative accuracy results.
        :param key: dictionary key
        :return:
        '''



        # using n gram
        count_vect = CountVectorizer(analyzer='word', ngram_range=(n_head, n_tail))

        mnb_classifier = MultinomialNB()

        '''
        train the model
        '''
        # if stopwords true, use set contains stopwords
        if(stopwords):
            train_data = raw_data['train']['data']
            y = raw_data['train']['target']
            test_data = raw_data['test']['data']
            test_y = raw_data['test']['target']
        else:
            train_data = raw_data['train_ns']['data']
            y = raw_data['train_ns']['target']
            test_data = raw_data['test_ns']['data']
            test_y = raw_data['test_ns']['target']

        # transform text to number vectors
        X_train_counts = count_vect.fit_transform(train_data)

        # using tf-idf
        X_train_tfidf = TfidfTransformer().fit_transform(X_train_counts)

        x_clf = mnb_classifier.fit(X_train_tfidf, y)
        # store classifier
        self.store_model(x_clf, model_name)

        '''
        using the trained model to test data
        '''
        X_test_counts = count_vect.transform(test_data)
        X_test_tfidf = TfidfTransformer().fit_transform(X_test_counts)

        # predict the test set
        predicted = mnb_classifier.predict(X_test_tfidf)
        accracy = np.mean(predicted == test_y) * 100
        result_dic[key] = accracy


    def store_model(self, clf, model_name):
        if not os.path.exists(os.getcwd()+"/data"):
            os.mkdir(os.getcwd()+"/data")

        path = os.path.join(os.getcwd(), 'data', model_name)

        with open(path+'.pkl', 'wb') as f:
            pickle.dump(clf, f)

        # with open(os.getcwd()+'/data/mnb_uni.pkl', 'rb') as f:
        # # 以读取的方式 读取模型存储的pickle文件，并放在变量f里
        #     clf_load = pickle.load(f)  # 将模型存储在变量clf_load中





if __name__ == '__main__':

    # path = os.getcwd()+'/data'
    # mnb = MNB(path).mnb()
    # print(mnb)

    try:
        params = sys.argv

        # if no path given, use default
        if (len(sys.argv) == 1):
            files_path = os.getcwd()+'/data'
            if not os.path.exists(files_path):
                raise Exception("Illegal path. QUITING...")
            print('Using >', files_path, '< as input files path.')
            MNB(files_path).mnb()
        elif len(sys.argv) > 1 and len(sys.argv)<=2:
            # path contains corpus
            files_path = params[1]
            if not os.path.exists(files_path):
                raise Exception("Illegal path. QUITING...")
            MNB(files_path).mnb()
        else:
            raise Exception("Illegal path. QUITING...")

    except Exception as ex:
        print('Pass your data folder path, and only one argument is needed.')
        traceback.print_exc()


