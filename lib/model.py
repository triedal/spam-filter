from __future__ import print_function, division
import os
from collections import Counter
from string import punctuation
from random import shuffle, randint
from nltk import WordNetLemmatizer, word_tokenize
from nltk.corpus import stopwords
from nltk import NaiveBayesClassifier, classify

class Model(object):
    stoplist = stopwords.words('english')

    def __init__(self):
        self.trainDir = self.getRandomDir()
        strTrainDir = str(self.trainDir)
        print('Training with enron' + strTrainDir + ' dataset.')
        hamPath = 'data/enron' + strTrainDir + '/ham/'
        spamPath = 'data/enron' + strTrainDir + '/spam/'
        spamFiles = os.listdir(spamPath)
        hamFiles = os.listdir(hamPath)
        hamCorpus = self.buildCorpus(hamFiles, hamPath)
        spamCorpus = self.buildCorpus(spamFiles, spamPath)
        self.emails = self.labelEmails(spamCorpus, hamCorpus)
        print ('CORPUS SIZE: ' + str(len(self.emails)) + ' emails\n')
        shuffle(self.emails)

    @staticmethod
    def buildCorpus(fileList, path):
        corpus = []
        for file in fileList:
            if not file.startswith('.'):
                f = open(path + file, 'r')
                corpus.append(f.read())
        f.close()
        return corpus

    @staticmethod
    def getRandomDir():
        numDirs = len([f for f in os.listdir('data') if not f.startswith('.')])
        return randint(1, numDirs)

    @staticmethod
    def labelEmails(spam, ham):
        emails = [(email, 'ham') for email in ham]
        emails.extend([(email, 'spam') for email in spam])
        return emails

    @staticmethod
    def preprocess(email):
        lemmatizer = WordNetLemmatizer()
        # Tokenize email
        tokens = word_tokenize(unicode(email, errors='ignore'), language='english')
        # Remove punctuation from token list
        tokens = [token for token in tokens if token not in punctuation]
        # Lemmatize the tokens
        lemmatized = [lemmatizer.lemmatize(token.lower()) for token in tokens]
        return lemmatized

    def getFeatures(self, email):
        return { word: count for word, count in Counter(self.preprocess(email)).items() if not word in self.stoplist }

    def getAccuracy(self, testSet):
        return classify.accuracy(self.classifier, testSet)

    def train(self, proportion=0.8):
        print('Generating feature sets...')
        features = [(self.getFeatures(email), label) for (email, label) in self.emails]
        trainSize = int(len(features) * proportion)
        # Training set is first 80% of features
        self.trainSet, self.testSet = features[:trainSize], features[trainSize:]
        # Train the classifier
        print('Training predictive model...')
        self.classifier = NaiveBayesClassifier.train(self.trainSet)

    def evaluate(self):
        # Check how the classifier performs the test set
        print('\n----------------- RESULTS -----------------')
        print ('TEST SET ACCURACY:     ' + '{0:.1%}'.format(classify.accuracy(self.classifier, self.testSet)) + '\n')

        self.classifier.show_most_informative_features(20)
        print('\n----------------- END RESULTS -------------')
