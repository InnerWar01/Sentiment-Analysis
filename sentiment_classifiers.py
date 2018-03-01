from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
import re as regex
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import random
import json
import pandas as pd
import csv
from sklearn import svm
from collections import Counter
from sklearn.metrics import f1_score
import time

def remove_by_regex(review, regexp):
    p = regex.compile(regexp)
    return p.sub(' ', review)

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

#Remove all the stop words, the punctuation, change uppercase characters to lowercase
#and remove words that are smaller than 3 characters
def data_cleaning(X):
    Z = []
    stop_words = set(stopwords.words('english'))
    for comment in X:
        comment = remove_urls(comment)
        comment = remove_tags(comment)
        comment = remove_symbol_codes(comment)
        comment = remove_short_is(comment)
        comment = remove_special_chars(comment)
        comment = comment.lower()
        word_tokens = word_tokenize(comment)
        filtered_comment = [w for w in word_tokens if not w in stop_words]
        filtered_comment = [w for w in filtered_comment if len(w) > 3]
        Z.append(filtered_comment)
    return Z

def feature_names(vectorizer):
    idf = vectorizer.idf_
    my_dict = dict(zip(vectorizer.get_feature_names(), idf))
    with open('idf_features.csv', 'w') as myfile:
        w = csv.writer(myfile)
        w.writerows(my_dict.items())

def get_sentiment_from_score(Y):
    new_Y = []
    for review in Y:
        if review <= 2 and review >= 0:
            new_Y.append("negative")
        elif review > 2 and review <= 5:
            new_Y.append("positive")
    return new_Y;

def get_sentiment_from_score_neutral(Y):
    new_Y = []
    for review in Y:
        if review <= 2 and review >= 0:
            new_Y.append("negative")
        elif review == 3:
            new_Y.append("neutral")
        elif review > 3 and review <= 5:
            new_Y.append("positive")
    return new_Y;

# Start a time counter
start_time = time.time()

# read the entire file into an array of json elements
array_reviews = []
with open('reviews_test.json','rb') as file:
	allData = file.readlines()
error_count = 0
for review in allData:
    try:
	    array_reviews.append(json.loads(review))	
    except:
	# Forget that line
	error_count +=1
print("File reading completed!")
print("Number of lines with errors: ", error_count)
print("--- %s seconds ---" % (time.time() - start_time))
step_time = time.time() - start_time

# Extract scores and text from reviews
array_scores = []
array_text = []
for review in array_reviews:
	array_scores.append(review["stars"])
	array_text.append(review["text"])

'''X_train = array_text
Y_train = get_sentiment_from_score(array_scores)
X_train = data_cleaning(X_train)'''

X = array_text
Y = get_sentiment_from_score(array_scores) # Only positive and negative labels
Y_neutral = get_sentiment_from_score_neutral(array_scores) # With neutral label
X = data_cleaning(X)

print("Extract scores and text from reviews completed! --- %s minutes ---" % ((time.time() - step_time)/60))
step_time = time.time() - start_time

# Split data into train (70%) and test (30%) sets
# (We need to include the validation set)

size1 = (len(X)*70)/100
size2 = len(X) - size1
random_train = random.sample(range(0, len(X)), int(size1))
X_train = []
Y_train = []
Y_train_neutral = []
X_test = []
Y_test = []
Y_test_neutral = []
for number in random_train:
    X_train.append(X[number])
    Y_train.append(Y[number])
    Y_train_neutral.append(Y_neutral[number])

for index, comment in enumerate(Y):
    if index not in random_train:
        X_test.append(X[index])
        Y_test.append(Y[index])
        Y_test_neutral.append(Y_neutral[index])

print("Data divided into train and test sets --- %s minutes ---" % ((time.time() - step_time)/60))
step_time = time.time() - start_time

# Create feature vectors
vectorizer = TfidfVectorizer(sublinear_tf=True, use_idf=False)
X_train = [" ".join(review) for review in X_train]
X_test = [" ".join(review) for review in X_test]

train_vectors = vectorizer.fit_transform(X_train)
test_vectors = vectorizer.transform(X_test) 

print("Create feature vectors completed! --- %s minutes ---" % ((time.time() - step_time)/60))
step_time = time.time() - start_time

############################################
# Perform classification with Neural Network

clf = MLPClassifier(solver='lbfgs', alpha=1e-3, hidden_layer_sizes=(10,), random_state=1)
clf.fit(train_vectors, Y_train)
print("MLP Classifier trained --- %s minutes ---" % ((time.time() - step_time)/60))
step_time = time.time() - start_time

prediction_nn = clf.predict(test_vectors)
print("Predictions completed! --- %s minutes ---" % ((time.time() - step_time)/60))
step_time = time.time() - start_time

clf.fit(train_vectors, Y_train_neutral)
print("MLP Classifier with neutral label trained --- %s minutes ---" % ((time.time() - step_time)/60))
step_time = time.time() - start_time

prediction_nn_neutral = clf.predict(test_vectors)
print("Predictions with neutral label completed! --- %s minutes ---" % ((time.time() - step_time)/60))
step_time = time.time() - start_time

print("Results for Neural Network")
print(classification_report(Y_test, prediction_nn))
# Good news! We finally have a f1 score for the negative reviews greater than 0.5!

print("Results for Neural Network with neutral label")
print(classification_report(Y_test_neutral, prediction_nn_neutral))
# Neutral reviews are not being detected, we have more wrong than right guesses (the f1 score is lower than 0.5)

#############################################

# Perform classification with SVM, kernel=linear
classifier_linear = svm.SVC(kernel='linear')
classifier_linear.fit(train_vectors, Y_train)
print("SVM Classifier trained --- %s minutes ---" % ((time.time() - step_time)/60))

prediction_linear = classifier_linear.predict(test_vectors)
print("Predictions completed! --- %s minutes ---" % ((time.time() - step_time)/60))
step_time = time.time() - start_time

print("Results for SVC(kernel=linear)")
print(classification_report(Y_test, prediction_linear))

classifier_linear.fit(train_vectors, Y_train_neutral)
print("SVM Classifier with neutral label trained --- %s minutes ---" % ((time.time() - step_time)/60))
prediction_linear = classifier_linear.predict(test_vectors)
print("Predictions with neutral label completed! --- %s minutes ---" % ((time.time() - step_time)/60))
step_time = time.time() - start_time

print("Results for SVC(kernel=linear) with neutral label")
print(classification_report(Y_test_neutral, prediction_linear))


##############################################

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

print("Preparing data for Naive BayesClassifier")
word_features = get_word_features(get_words_in_reviews(X_train))
reviews = []
for index, comment in enumerate(X_train):
    tup = (X_train[index], Y_train[index])
    reviews.append(tup)
training_set = nltk.classify.apply_features(extract_features, reviews)
classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Naive BayesClassifier trained --- %s minutes ---" % ((time.time() - step_time)/60))
step_time = time.time() - start_time

reviews = []
for index, comment in enumerate(X_test):
    tup = (X_test[index], Y_test[index])
    reviews.append(tup)
test_set = nltk.classify.apply_features(extract_features, reviews)
print("Extract features from test data complete! --- %s minutes ---" % ((time.time() - step_time)/60))
step_time = time.time() - start_time
Y_pred = []
for index, comment in enumerate(X_test):
	pred = classifier.classify(extract_features(X_test[index]))
	Y_pred.append(pred)	
print("Predictions complete! --- %s minutes ---" % ((time.time() - step_time)/60))
step_time = time.time() - start_time

print("Results for Naive Bayes Classifier")
print("F1 score for each label: ")
print("Negative: {:.3},  Positive: {:.3}".format(f1_score(Y_test, Y_pred, average=None)[0], f1_score(Y_test, Y_pred, average=None)[1]))
print("F1 score: {:.3}".format(f1_score(Y_test, Y_pred, average='micro')*100))

# Adding neutral label
print("Preparing data for Naive BayesClassifier with neutral label")

reviews = []
for index, comment in enumerate(X_train):
    tup = (X_train[index], Y_train_neutral[index])
    reviews.append(tup)
training_set = nltk.classify.apply_features(extract_features, reviews)
classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Naive BayesClassifier with neutral label trained --- %s minutes ---" % ((time.time() - step_time)/60))
step_time = time.time() - start_time

reviews = []
for index, comment in enumerate(X_test):
    tup = (X_test[index], Y_test_neutral[index])
    reviews.append(tup)
test_set = nltk.classify.apply_features(extract_features, reviews)
print("Extract features from test data with neutral label complete! --- %s minutes ---" % ((time.time() - step_time)/60))
step_time = time.time() - start_time

Y_pred = []
for index, comment in enumerate(X_test):
	pred = classifier.classify(extract_features(X_test[index]))
	Y_pred.append(pred)
print("Predictions with neutral label complete! --- %s minutes ---" % ((time.time() - step_time)/60))
step_time = time.time() - start_time

print("Results for Naive Bayes Classifier with neutral label")

print("F1 score for each label: ")
print("Negative: {:.3},  Neutral: {:.3},  Positive: {:.3}".format(f1_score(Y_test_neutral, Y_pred, average=None)[0], f1_score(Y_test_neutral, Y_pred, average=None)[1], f1_score(Y_test_neutral, Y_pred, average=None)[2]))
print("F1 score: {:.3}".format(f1_score(Y_test_neutral, Y_pred, average='micro')*100))

print("Total time of execution --- %s minutes ---" % ((time.time() - start_time)/60))