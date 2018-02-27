from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
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
X_train = reviews_train["review"]
Y_train = get_sentiment_from_score(reviews_train["score"])
X_train = data_cleaning(X_train)

'''train_data = []
for index, comment in enumerate(X_train):
    tup = (X_train[index], Y_train[index])
    train_data.append(tup)'''

reviews_test = pd.read_csv("lab_test.csv", sep=";")
X_test = reviews_test["review"]
Y_test = get_sentiment_from_score(reviews_test["score"])
X_test = data_cleaning(X_test)

'''test_data = []
for index, comment in enumerate(X_test):
    tup = (X_test[index], Y_test[index])
    test_data.append(tup)'''

# Create feature vectors
vectorizer = TfidfVectorizer(sublinear_tf=True, use_idf=False)
X_train = [" ".join(review) for review in X_train]
X_test = [" ".join(review) for review in X_test]
#print("Xtrain: ", X_train)
train_vectors = vectorizer.fit_transform(X_train)
#feature_names(vectorizer)

test_vectors = vectorizer.transform(X_test)

# Scale and normalize the vectors
from sklearn.preprocessing import StandardScaler

'''scaler = StandardScaler()  
# Don't cheat - fit only on training data
scaler.fit(train_vectors)  
train_vectors = scaler.transform(train_vectors)  
# apply same transformation to test data
test_vectors = scaler.transform(test_vectors) ''' 

# Perform classification with Neural Network
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(20,), random_state=1)
clf.fit(train_vectors, Y_train)
print(train_vectors[0])
prediction_nn = clf.predict(test_vectors)

print("Results for Neural Network")
print(classification_report(Y_test, prediction_nn))