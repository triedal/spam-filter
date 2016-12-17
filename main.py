from __future__ import print_function, division
import os
from collections import Counter
from string import punctuation
from random import shuffle
from nltk import WordNetLemmatizer, word_tokenize
from nltk.corpus import stopwords
from nltk import NaiveBayesClassifier, classify

stoplist = stopwords.words('english')

def evaluate(trainSet, testSet, classifier):
    # check how the classifier performs on the training and test sets
    print('\n----------------- RESULTS -----------------')
    print ('TRAINING SET ACCURACY: ' + '{0:.1%}'.format(classify.accuracy(classifier, trainSet)))
    print ('TEST SET ACCURACY:     ' + '{0:.1%}'.format(classify.accuracy(classifier, testSet)) + '\n')

    classifier.show_most_informative_features(20)
    print('\n----------------- END RESULTS -------------')
    return

def train(features, proportion=0.8):
    trainSize = int(len(features) * proportion)
    # Training set is first 80% of features
    trainSet, testSet = features[:trainSize], features[trainSize:]
    # Train the classifier
    classifier = NaiveBayesClassifier.train(trainSet)
    return trainSet, testSet, classifier

def labelEmails(spam, ham):
    emails = [(email, 'ham') for email in ham]
    emails.extend([(email, 'spam') for email in spam])
    return emails

def getFeatures(email):
    return { word: count for word, count in Counter(preprocess(email)).items() if not word in stoplist }

def preprocess(email):
    lemmatizer = WordNetLemmatizer()
    # Tokenize email
    tokens = word_tokenize(unicode(email, errors='ignore'), language='english')
    # Remove punctuation from token list
    tokens = [token for token in tokens if token not in punctuation]
    # Lemmatize the tokens
    lemmatized = [lemmatizer.lemmatize(token.lower()) for token in tokens]
    return lemmatized

def buildCorpus(fileList, path):
    corpus = []
    for file in fileList:
        if not file.startswith('.'):
            f = open(path + file, 'r')
            corpus.append(f.read())
    f.close()
    return corpus

def init(path):
    hamPath = path + 'ham/'
    spamPath = path + 'spam/'
    spamFiles = os.listdir(spamPath)
    hamFiles = os.listdir(hamPath)
    hamCorpus = buildCorpus(hamFiles, hamPath)
    spamCorpus = buildCorpus(spamFiles, spamPath)
    return (spamCorpus, hamCorpus)

if __name__ == '__main__':
    spam, ham = init('data/enron1/')
    emails = labelEmails(spam, ham)
    shuffle(emails)
    print ('CORPUS SIZE: ' + str(len(emails)) + ' emails\n')

    print('Generating feature sets...')
    features = [(getFeatures(email), label) for (email, label) in emails]

    print('Training predictive model...')
    trainSet, testSet, classifier = train(features)

    evaluate(trainSet, testSet, classifier)
    # # extract the features
    # all_features = [(get_features(email, ''), label) for (email, label) in all_emails]
    # print ('Collected ' + str(len(all_features)) + ' feature sets')
