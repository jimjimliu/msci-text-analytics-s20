import tokenizer as tokens
import sklearn as sklearn
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn import datasets
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import preprocessing
import os
from multiprocessing import Process


class MNB():

    def mnb(self):

        print('Parent process %s.' % os.getpid())
        # using unigram mnb
        p1 = Process(target=self.classify, args=(1, 1,))
        # using bigram mnb
        p2 = Process(target=self.classify, args=(2, 2,))
        # using unigram + bigram mnb
        p3 = Process(target=self.classify, args=(1, 2,))
        print('Child process will start.')
        p1.start()
        p2.start()
        p3.start()
        p1.join()
        p2.join()
        p3.join()
        print('Child process end.')


    def classify(self, n_head=1, n_tail=1, analyzer='word'):

        '''
        using naive bayes: multinomial NB to train and test model.
        one could always set the boundary of the range of different n-grams.

        :param n_head: The lower boundary of the range of n-values for different word n-grams or char n-grams to be extracted
        :param n_tail: The upper boundary of the range of n-values for different word n-grams or char n-grams to be extracted
        :param analyzer: string, {‘word’, ‘char’, ‘char_wb’} or callable, default=’word’.
                        Whether the feature should be made of word n-gram or character n-grams.
                        [https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.
                        html#sklearn.feature_extraction.text.CountVectorizer]
        :return:
        '''

        '''prepare train/test data'''
        raw_data = tokens.Tokenizer().sets

        # using n gram
        count_vect = CountVectorizer(analyzer=analyzer, ngram_range=(n_head, n_tail))

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
        print(np.mean(predicted == raw_data['test_set']['target']))




if __name__ == '__main__':

    mnb = MNB().mnb()

