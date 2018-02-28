from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
import re as regex
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import random
import json

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

# read the entire file into an array of json elements
array_reviews = []
with open('reviews_test.json','rb') as file:
	allData = file.readlines()
for review in allData:
	array_reviews.append(json.loads(review))	

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

# Test our classfier by splitting the data into train (70%) and test (30%) sets
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

# Create feature vectors
vectorizer = TfidfVectorizer(sublinear_tf=True, use_idf=False)
X_train = [" ".join(review) for review in X_train]
X_test = [" ".join(review) for review in X_test]

train_vectors = vectorizer.fit_transform(X_train)
test_vectors = vectorizer.transform(X_test) 

# Perform classification with Neural Network
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(20,), random_state=1)
clf.fit(train_vectors, Y_train)
prediction_nn = clf.predict(test_vectors)

clf.fit(train_vectors, Y_train_neutral)
prediction_nn_neutral = clf.predict(test_vectors)

print("Results for Neural Network")
print(classification_report(Y_test, prediction_nn))
# Good news! We finally have a f1 score for the negative reviews greater than 0.5!

print("Results for Neural Network with neutral label")
print(classification_report(Y_test_neutral, prediction_nn_neutral))
# Neutral reviews are not being detected, we have more wrong than right guesses (the f1 score is lower than 0.5)