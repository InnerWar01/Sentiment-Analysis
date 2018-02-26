try:
    # for Python 2.x
    from StringIO import StringIO
except ImportError:
    # for Python 3.x
    from io import StringIO
import csv
import pandas as pd
import re as regex
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics import f1_score
import numpy as np
from sklearn.model_selection import train_test_split
import random

def remove_by_regex(review, regexp):
    p = regex.compile(regexp)
    return p.sub(' ', review)

def remove_data_id(review):
    return remove_by_regex(review, '"review_id":"[A-Za-z0-9_\-/]+","user_id":"[A-Za-z0-9_\-/]+","business_id":"[A-Za-z0-9_\-/]+",')

def remove_date(review):
    return remove_by_regex(review, '"date":"[A-Za-z0-9_\-/]+",')

def remove_unecessary_adjectives(review):
    return remove_by_regex(review, ',"useful":[0-9],"funny":[0-9],"cool":[0-9]')

def remove_new_line(review):
    return remove_by_regex(review, '\\[a-z]')

'''
def remove_urls(review):
    return remove_by_regex(review, 'href="[A-Za-z0-9_ :./]+"')

def remove_tags(review):
    return remove_by_regex(review, '<[A-Za-z0-9_ /]+>')

def remove_symbol_codes(review):
    return remove_by_regex(review, '&[A-Za-z0-9_ /]+;')

def remove_short_is(review): #it's
    return remove_by_regex(review, "'[A-Za-z0-9_ /]")

def remove_special_chars(review):
    return remove_by_regex(review, "[,:\=&;%$@^*(){}[\]|/><'#.!?\\-]")
'''

#Remove all the stop words, the punctuation, change uppercase characters to lowercase
#and remove words that are smaller than 3 characters
def data_cleaning(X):
    Z = []
    stop_words = set(stopwords.words('english'))
    for comment in X:
        comment = remove_data_id(comment)
        comment = remove_date(comment)
        comment = remove_unecessary_adjectives(comment)
        comment = remove_new_line(comment)
        '''
        comment = remove_urls(comment)
        comment = remove_tags(comment)
        comment = remove_symbol_codes(comment)
        comment = remove_short_is(comment)
        comment = remove_special_chars(comment)
        '''
        comment = comment.lower()
        word_tokens = word_tokenize(comment)
        filtered_comment = [w for w in word_tokens if not w in stop_words]
        filtered_comment = [w for w in filtered_comment if len(w) > 3]
        Z.append(filtered_comment)
    return Z

#Get a list of all the words in the reviews
def get_words_in_reviews(reviews):
    all_words = []
    for comment in reviews:
        for word in comment:
            all_words.append(word)
    return all_words

#List of word features: list with every distinct word ordered by frequency of appearance
def get_word_features(wordlist):
    cnt = Counter()
    for word in wordlist:
        cnt[word] += 1
    with open('common_words.csv', 'w') as myfile:
        wr = csv.writer(myfile, delimiter=';')
        wr.writerows(cnt.most_common(10))
    wordlist = nltk.FreqDist(wordlist)
    word_features = wordlist.keys()
    return word_features

#Returns a dictionary indicating what words are contained in the input passed - a review
def extract_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features
               
def get_sentiment_distribution(reviews):
    sent1 = 0
    sent2 = 0
    sent3 = 0

    for tuple in reviews:
        if tuple[1] == "negative":
            sent1 += 1
        elif tuple[1] == "neutral":
            sent2 += 1
        elif tuple[1] == "positive":
            sent3 += 1
    
    score = []
    score.append("negative")
    score.append("neutral")
    score.append("positive")
    sentiment = []
    sentiment.append(sent1)
    sentiment.append(sent2)
    sentiment.append(sent3)

    with open('sentiment.csv', 'w') as myfile:
        wr = csv.writer(myfile, delimiter=';')
        wr.writerows(zip(score, sentiment))

def get_sentiment_from_score(Y):
    new_Y = []
    for review in Y:
        if review <= 2 and review >= 0:
            new_Y.append("negative")
        elif review == 3:
            new_Y.append("neutral")
        elif review > 3 and review <= 5:
            new_Y.append("positive")
    return new_Y;  

reviews_train = pd.read_csv("lab_train.csv", sep=";", usecols=["review", "score"])

#### Test our classfier by splitting the train data set into 2
X = reviews_train["review"]
Y = get_sentiment_from_score(reviews_train["score"])
X = data_cleaning(X)
size1 = (len(X)*70)/100
size2 = len(X) - size1
random_train = random.sample(range(0, len(X)), int(size1))
new_X_1 = []
new_Y_1 = []
new_X_2 = []
new_Y_2 = []
for number in random_train:
    new_X_1.append(X[number])
    new_Y_1.append(Y[number])

for index, comment in enumerate(Y):
    if index not in random_train:
        new_X_2.append(X[index])
        new_Y_2.append(Y[index])

word_features = get_word_features(get_words_in_reviews(new_X_1))
reviews_train_set = []
for index, comment in enumerate(new_X_1):
    tup = (new_X_1[index], new_Y_1[index])
    reviews_train_set.append(tup)
training_set = nltk.classify.apply_features(extract_features, reviews_train_set)
classifier = nltk.NaiveBayesClassifier.train(training_set)

reviews_test_set = []
for index, comment in enumerate(new_X_2):
    tup = (new_X_2[index], new_Y_2[index])
    reviews_test_set.append(tup)
test_set = nltk.classify.apply_features(extract_features, reviews_test_set)

Y_pred = []
for index, comment in enumerate(new_X_2):
	pred = classifier.classify(extract_features(new_X_2[index]))
	Y_pred.append(pred)

print("Prediction:\nPositive reviews: {}\nNeutral reviews: {}\nNegative reviews: {}\n".format(Y_pred.count("positive"),Y_pred.count("neutral"), Y_pred.count("negative")))
print("F1 score for each label: ")
print("Negative: {:.3},  Neutral: {:.3},  Positive: {:.3}".format(f1_score(new_Y_2, Y_pred, average=None)[0], f1_score(new_Y_2, Y_pred, average=None)[1], f1_score(new_Y_2, Y_pred, average=None)[2]))
print("F1 score: {}".format(f1_score(new_Y_2, Y_pred, average='micro')*100))

accuracy = nltk.classify.util.accuracy(classifier, test_set)
#print(accuracy * 100)

##############################################################
X_train = reviews_train["review"]
Y_train = get_sentiment_from_score(reviews_train["score"])
X_train = data_cleaning(X_train)
word_features = get_word_features(get_words_in_reviews(X_train))
reviews = []
for index, comment in enumerate(X_train):
    tup = (X_train[index], Y_train[index])
    reviews.append(tup)
#words_sentiments = get_common_words_across_sentiments(reviews)
training_set = nltk.classify.apply_features(extract_features, reviews)
classifier = nltk.NaiveBayesClassifier.train(training_set)
#get_sentiment_distribution(reviews) #Sentiment type distribution in training set
#print(classifier.show_most_informative_features(32))

reviews_test = pd.read_csv("lab_test.csv", sep=";")
X_test = reviews_test["review"]
Y_test = get_sentiment_from_score(reviews_test["score"])
X_test = data_cleaning(X_test)
reviews = []
for index, comment in enumerate(X_test):
    tup = (X_test[index], Y_test[index])
    reviews.append(tup)
test_set = nltk.classify.apply_features(extract_features, reviews)

Y_pred = []
for index, comment in enumerate(X_test):
	pred = classifier.classify(extract_features(X_test[index]))
	Y_pred.append(pred)	

print(f1_score(Y_test, Y_pred, average=None)[0])
print("F1 score for each label: ")
print("Negative: {:.3},  Neutral: {:.3},  Positive: {:.3}".format(f1_score(Y_test, Y_pred, average=None)[0], f1_score(Y_test, Y_pred, average=None)[1], f1_score(Y_test, Y_pred, average=None)[2]))
print("F1 score: {}".format(f1_score(Y_test, Y_pred, average='micro')*100))

accuracy = nltk.classify.util.accuracy(classifier, test_set)
#print(accuracy * 100)
#get_sentiment_distribution(reviews) #Sentiment type distribution in testing set