import sys
from collections import defaultdict
import math
import random
import os
import os.path 

#This function takes in a text file and returns a python generator
#by which to parse a corpus.
def corpus_reader(corpusfile, lexicon=None): 
    with open(corpusfile,'r') as corpus: 
        for line in corpus: 
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon: 
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else: 
                    yield sequence
                    
#Returns a set of words that appear in a corpus more than once.
def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence: 
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)  


#This function returns padded n-grams in the form of touples
#given a string sequence. The symbols 'START' and 'END' are
#used to represent the start and end of a sentence.
def get_ngrams(sequence, n):

    paddedNGrams = []
    
    #The first touple added will hold the 'START' symbol
    #n times.
    firstTouple = ()
    for i in range(n):
        firstTouple += ('START',)
    
    paddedNGrams.append(firstTouple)
        
    #For n = 1, simply iterate though each word and add a touple
    #containing only the word to the list.
    if n == 1:
    
        paddedNGrams.append(('START',))
        
        for word in sequence:
            paddedNGrams.append((word,))
    
        paddedNGrams.append(('END',))
    
        return paddedNGrams

    
    for i in range(len(sequence)):
        
        #This portion creates all touples that require the 'START' symbol.
        if i < n - 1:
        
            newTouple = ('START',)
            
            for j in range(n-i-2):
                newTouple += ('START',)
            
            k = 0
            while len(newTouple) < n :
                newTouple += (sequence[k],)
                k += 1
                
            paddedNGrams.append(newTouple)
        
        #This portion is used to create touples that will only contain
        #words in the sequence.
        elif i != len(sequence):
        
            newTouple = ()
            
            for j in range(i-n+1, i + 1):
                newTouple += (sequence[j],)
            
            paddedNGrams.append(newTouple)
            
    
    #Creates the last touple ending in the 'END' symbol.
    lastTouple = ()
    last_index = len(sequence) - 1
    
    #For cases where n - 1 is larger than the sentence length,
    #'START' symbols must be appended to the beginning of the
    #touple to make the touple have length n.
    if len(sequence) + 1 < n :
        for i in range(n - len(sequence) - 1):
            lastTouple += ('START',)
            
        for j in range(len(sequence)):
            lastTouple += (sequence[j],)
        
        lastTouple += (('END',))
        
        paddedNGrams.append(lastTouple)
    #Here, the sentence is long enough, so no 'START' symbols
    #need to be appended.
    else:
        for j in range(last_index - n + 2, last_index + 1):
            lastTouple += (sequence[j],)
    
        lastTouple += ('END',)
        
        paddedNGrams.append(lastTouple)
    
    return paddedNGrams

class TrigramModel(object):
    
    def __init__(self, corpusfile):
    
        #Iterates through the corpus to build a lexicon.
        #The symbol 'UNK' is used for words that appear
        #only once.
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")
    
        #Iterates through the corpus again and count the ngrams.
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)
        self.calculateNumberOfWords(corpusfile)

    #Counts the number of uni/bi/tri-grams that appear in a corpus
    #using dictionaries.
    def count_ngrams(self, corpus):
   
        self.unigramcounts = {}
        self.bigramcounts = {} 
        self.trigramcounts = {} 

        for sentence in corpus:
            unigrams = get_ngrams(sentence, 1)
            bigrams = get_ngrams(sentence, 2)
            trigrams = get_ngrams(sentence, 3)
            
            for touple in unigrams:
                if touple in self.unigramcounts:
                    self.unigramcounts[touple] += 1
                else:
                    self.unigramcounts[touple] = 1
                    
            for touple in bigrams:
                if touple in self.bigramcounts:
                    self.bigramcounts[touple] += 1
                else:
                    self.bigramcounts[touple] = 1
                    
            for touple in trigrams:
                if touple in self.trigramcounts:
                    self.trigramcounts[touple] += 1
                else:
                    self.trigramcounts[touple] = 1
                    
        return

    #Returns the raw probability of a trigram.
    def raw_trigram_probability(self,trigram):
        
        if trigram not in self.trigramcounts:
            return 0
        
        return (self.trigramcounts[trigram])/self.bigramcounts[(trigram[0],trigram[1])]

    #Returns the raw probability of a brigram.
    def raw_bigram_probability(self, bigram):

        if bigram not in self.bigramcounts:
            return 0
        
        return (self.bigramcounts[bigram])/self.unigramcounts[(bigram[0],)]
    
    #Simple function that calculates the number of words
    #in the corupus and saves it in the instance variable
    #called wordCount
    def calculateNumberOfWords(self, fileName):

        total = 0
        with open(fileName,"r") as f: 
            for line in f: 
                words = line.split(" ")
                total += len(words)
                
        self.wordCount = total
        
        return
    #Returns the raw probability of a unigram.
    def raw_unigram_probability(self, unigram):
        
        if unigram not in self.unigramcounts:
            return 0
        
        return (self.unigramcounts[unigram]/self.wordCount)

    #Returns the smoothed probability of a trigram using
    #linear interpolation.
    def smoothed_trigram_probability(self, trigram):

        lambda1 = 1/3.0
        lambda2 = 1/3.0
        lambda3 = 1/3.0
        
        smoothedProbability = (lambda1 * self.raw_trigram_probability(trigram)) + \
            (lambda2 * self.raw_bigram_probability((trigram[1],trigram[2]))) + \
            (lambda3 * self.raw_unigram_probability((trigram[2],)))
        
        return smoothedProbability
        
    #Returns the log probability of an entire sequence.
    def sentence_logprob(self, sentence):
        
        logProbability = 0
        
        trigrams = get_ngrams(sentence, 3)
        
        for trigram in trigrams:
            logProbability += math.log2(self.smoothed_trigram_probability(trigram))
        
        return logProbability

    #Returns the perplexity of a corpus.
    def perplexity(self, corpus):
    
        logSentenceProbabilitySum = 0
        
        for sentence in corpus:
            logSentenceProbabilitySum += self.sentence_logprob(sentence)

        l = (1/self.wordCount) * logSentenceProbabilitySum
        
        return 2**(-l)


#Conducts the esssay classification experiment.
def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):
        
    #This model will be used to train a model using essays that were
    #classified as "high," or otherwise written by individuals
    #who are fluent in English.
    model1 = TrigramModel(training_file1)
        
    #This model is trained using essays that were classified as "low,"
    #in other words written by individuals not fluent in English.
    model2 = TrigramModel(training_file2)


    total = 0
    correct = 0
 
    #This will iterate through unseen essays in testdirect1, which holds
    #essays classified as "high." The perplexity is calculated using both
    #the high and low models, and the model with the lower complexity
    #is used to categorize the essay. Correct gets incremented if the essay
    #is correctly categorized as "high."
    for f in os.listdir(testdir1):
        high_model_perplexity = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
        low_model_perplexity = model2.perplexity(corpus_reader(os.path.join(testdir1, f), model2.lexicon))
        total += 1
            
        if high_model_perplexity < low_model_perplexity:
            correct += 1
    
    #This will iterate through the essays classified as low, and calculates
    #how many essays were correctly classified as low using the model.
    for f in os.listdir(testdir2):
        high_model_perplexity = model1.perplexity(corpus_reader(os.path.join(testdir2, f), model1.lexicon))
        low_model_perplexity = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))

        total += 1
            
        if low_model_perplexity < high_model_perplexity:
            correct += 1
        
    #Returns the percentage of how many essays were correctly categorized
    #using the models.
    return (correct/total)

if __name__ == "__main__":

    #Essay scoring experiment: 
    accuracy = 100 * essay_scoring_experiment('train_high.txt', 'train_low.txt', 'test_high', 'test_low')
    print("This text classifier, when tested on unseen essays, has an accuracy of " + str(round(accuracy,4)) + "%")
