import tokenizer as tokens
import sklearn as sklearn
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn import datasets
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import preprocessing
import os
from multiprocessing import Process, Manager


class MNB():

    def mnb(self, iter=10):

        '''

        :param iter: iteration times. default =10. Repeat calculating the model accuracy for 10 times.
        :return:
        '''

        # initialize a dictionary store the result returned by functions in each Process
        result_dic = Manager().dict()
        keys = ['uni_sw', 'bi_sw', 'uni_bi_sw', 'uni_wo_sw', 'bi_wo_sw', 'uni_bi_wo_sw']
        for item in keys:
            result_dic[item] = 0

        for i in range(iter):
            print('Parent process %s.' % os.getpid())

            '''using data with stopwords'''
            # using unigram mnb
            p1 = Process(target=self.classify, args=(result_dic, 'uni_sw',  1, 1,))
            # using bigram mnb
            p2 = Process(target=self.classify, args=(result_dic, 'bi_sw', 2, 2,))
            # using unigram + bigram mnb
            p3 = Process(target=self.classify, args=(result_dic, 'uni_bi_sw', 1, 2,))

            '''using data without stopwords'''
            # unigram
            p4 = Process(target=self.classify, args=(result_dic, 'uni_wo_sw', 1, 1, 'english', ))
            # bigram
            p5 = Process(target=self.classify, args=(result_dic, 'bi_wo_sw', 2, 2, 'english',))
            # unigram + bigram
            p6 = Process(target=self.classify, args=(result_dic, 'uni_bi_wo_sw', 1, 2, 'english',))

            print('Child process will start.')
            proc_arr = [p1, p2, p3, p4, p5, p6]
            for item in proc_arr:
                item.start()
            for item in proc_arr:
                item.join()
            print('Child process end.')


        result_dic = {k : v/iter for k, v in result_dic.items()}
        # print(result_dic)
        return result_dic



    def classify(self, result_dic, key, n_head=1, n_tail=1, stopwords=None, analyzer='word',):

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


        '''prepare train/test data'''
        raw_data = tokens.Tokenizer().sets

        # using n gram
        count_vect = CountVectorizer(analyzer=analyzer, ngram_range=(n_head, n_tail), stop_words=stopwords)

        mnb_classifier = MultinomialNB()

        '''
        train the model
        '''
        train_data = raw_data['train_set']['data']

        # transform text to number vectors
        X_train_counts = count_vect.fit_transform(train_data)
        # using tf-idf
        X_train_tfidf = TfidfTransformer().fit_transform(X_train_counts)
        y = raw_data['train_set']['target']

        x_clf = mnb_classifier.fit(X_train_tfidf, y)

        '''
        using the trained model to test data
        '''
        test_data = raw_data['test_set']['data']
        X_test_counts = count_vect.transform(test_data)
        X_test_tfidf = TfidfTransformer().fit_transform(X_test_counts)

        # predict the test set
        predicted = mnb_classifier.predict(X_test_tfidf)
        accracy = np.mean(predicted == raw_data['test_set']['target']) * 100
        result_dic[key] += accracy
        # print(np.mean(predicted == raw_data['test_set']['target'])*100)




if __name__ == '__main__':

    mnb = MNB().mnb()
    print(mnb)

