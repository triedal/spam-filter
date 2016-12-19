from __future__ import print_function
from random import choice
from cmd import Cmd
import os

class Shell(Cmd):

    prompt = '> '

    def __init__(self, args):
        Cmd.__init__(self)
        self.model = args
        testDirIdx = self.getRandomTestDir()
        hamPath = 'data/enron' + str(testDirIdx) + '/ham/'
        spamPath = 'data/enron' + str(testDirIdx) + '/spam/'
        spamFiles = os.listdir(spamPath)
        hamFiles = os.listdir(hamPath)
        hamCorpus = self.model.buildCorpus(hamFiles, hamPath)
        spamCorpus = self.model.buildCorpus(spamFiles, spamPath)
        self.testEmails = self.model.labelEmails(spamCorpus, hamCorpus)

    def getRandomTestDir(self):
        numDirs = len([f for f in os.listdir('data') if not f.startswith('.')])
        possTestDirs = range(1, numDirs + 1)
        possTestDirs.remove(self.model.trainDir)
        return choice(possTestDirs)

    @staticmethod
    def printEmail(email):
        print('------- EMAIL -------\n')
        print(email)
        print('\n------- END EMAIL ---\n')

    @staticmethod
    def printResults(prediction, actual):
        print('PREDICT: ' + prediction)
        print('ACTUAL:  ' + actual + '\n')


    def do_random(self, args):
        '''Tests random email against spam filter.'''
        randomEmail = choice(self.testEmails)
        self.printEmail(randomEmail[0])
        # import pdb; pdb.set_trace()
        features = self.model.getFeatures(randomEmail[0])
        result = self.model.classifier.classify(features)
        self.printResults(result, randomEmail[1])


    def do_quit(self, args):
        '''Quits the program.'''
        print("Bye.")
        return True
