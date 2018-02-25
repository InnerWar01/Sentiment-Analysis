from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import classification_report
import re as regex
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
import random
import csv

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

reviews_train = pd.read_csv("lab_train.csv", sep=";", usecols=["review", "score"])
X_train = reviews_train["review"]
Y_train = reviews_train["score"] > 2
X_train = data_cleaning(X_train)

train_data = []
for index, comment in enumerate(X_train):
    tup = (X_train[index], Y_train[index])
    train_data.append(tup)

reviews_test = pd.read_csv("lab_test.csv", sep=";")
X_test = reviews_test["review"]
Y_test = reviews_test["score"] > 2
X_test = data_cleaning(X_test)

test_data = []
for index, comment in enumerate(X_test):
    tup = (X_test[index], Y_test[index])
    test_data.append(tup)

# Test our classfier by splitting the train data set into 2
X = reviews_train["review"]
Y = reviews_train["score"] > 2
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

new_X_1 = [" ".join(review) for review in new_X_1]
new_X_2 = [" ".join(review) for review in new_X_2]

# Create feature vectors
vectorizer = TfidfVectorizer(sublinear_tf=True, use_idf=True)
new_train_vectors = vectorizer.fit_transform(new_X_1)
new_test_vectors = vectorizer.transform(new_X_2)

# Perform classification with SVM, kernel=linear
new_classifier_rbf = svm.SVC(kernel='linear')
new_classifier_rbf.fit(new_train_vectors, new_Y_1)
new_prediction_rbf = new_classifier_rbf.predict(new_test_vectors)
##########################

# Create feature vectors
vectorizer = TfidfVectorizer(sublinear_tf=True, use_idf=True)
X_train = [" ".join(review) for review in X_train]
X_test = [" ".join(review) for review in X_test]
train_vectors = vectorizer.fit_transform(X_train)
#feature_names(vectorizer)

test_vectors = vectorizer.transform(X_test)

# Perform classification with SVM, kernel=rbf
classifier_rbf = svm.SVC()
classifier_rbf.fit(train_vectors, Y_train)
prediction_rbf = classifier_rbf.predict(test_vectors)

# Perform classification with SVM, kernel=linear
classifier_linear = svm.SVC(kernel='linear')
classifier_linear.fit(train_vectors, Y_train)
prediction_linear = classifier_linear.predict(test_vectors)

# Perform classification with SVM, kernel=linear
classifier_liblinear = svm.LinearSVC()
classifier_liblinear.fit(train_vectors, Y_train)
prediction_liblinear = classifier_liblinear.predict(test_vectors)

print("Results for SVC(kernel=linear) with the divided training dataset")
print(classification_report(new_Y_2, new_prediction_rbf))

'''
print("Results for SVC(kernel=rbf)")
print(classification_report(Y_test, prediction_rbf))
'''

print("Results for SVC(kernel=linear)")
print(classification_report(Y_test, prediction_linear))

'''
print("Results for SVC(kernel=linear)")
print(classification_report(Y_test, prediction_liblinear))
'''