import pickle
from sklearn.neural_network import MLPClassifier
import numpy
from numpy import random
from sklearn import metrics
from sklearn.metrics import adjusted_rand_score
import pandas as pd
import re
import gensim.downloader as api
import pandas as pd
import re
from sklearn.neural_network import _multilayer_perceptron
import sklearn.neural_network
from sklearn.neural_network import MLPClassifier
from gensim.models import word2vec as wc
from gensim.models import tfidfmodel as tfidf
from gensim.models import KeyedVectors as kv
import pickle
import gensim.downloader as api
import numpy
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer

def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return numpy.sum(numpy.amax(contingency_matrix, axis=0)) / numpy.sum(contingency_matrix)



#Getting and processing input

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
train_genres_strings = df_train['genres'].tolist()
test_genres_strings = df_test['genres'].tolist()
trainlabels = []
testlabels = []

for i in range(len(train_genres_strings)):
    trainlabel = re.findall(r"\d+", train_genres_strings[i])
    trainlabel = str(trainlabel)
    trainlabel = trainlabel[1:len(trainlabel)-1]
    trainlabels.append(trainlabel)
df_train['labels'] = trainlabels

for i in range(len(test_genres_strings)):
    testlabel = re.findall(r"\d+", test_genres_strings[i])
    testlabel = str(testlabel)
    testlabel = testlabel[1:len(testlabel)-1]
    testlabels.append(testlabel)
df_test['labels'] = testlabels

new_traindf = df_train.drop('labels', axis=1).join(df_train['labels'].str.split(', ', expand=True).stack().
                                              reset_index(level=1, drop=True).rename('labels'))
new_testdf = df_test.drop('labels', axis=1).join(df_test['labels'].str.split(', ', expand=True).stack().
                                              reset_index(level=1, drop=True).rename('labels'))


trainlabels = new_traindf['labels'].tolist()
trainoverviews = new_traindf['overview'].tolist()
testlabels = new_traindf['labels'].tolist()
testoverviews = new_traindf['overview'].tolist()

#Done processing input

for i in range(len(trainoverviews)):
    trainoverviews[i] = re.sub('[,?!"().]', '', str(trainoverviews[i]))

for p in range(len(testoverviews)):
    testoverviews[p] = re.sub('[,?!"().]', '', str(testoverviews[p]))

train_list_of_lists = []
test_list_of_lists = []

for e in trainoverviews:
    train_list_of_lists.append(str(e).split(' '))
for e in testoverviews:
    test_list_of_lists.append(str(e).split(' '))

vectorizer = TfidfVectorizer(max_df = 0.99,min_df = 0.01)
X_train_Tfidf = vectorizer.fit_transform(trainoverviews).toarray()

feature_names = vectorizer.get_feature_names()
vocab ={}
for value,key in enumerate(feature_names):
    vocab[key] = value

vectorizer =  TfidfVectorizer(vocabulary = vocab)
X_test_Tfidf = vectorizer.fit_transform(testoverviews).toarray()

mlp = MLPClassifier(hidden_layer_sizes=(12,4), learning_rate_init=0.04, max_iter=1000)

mlp.fit(X_train_Tfidf, trainlabels)

print(adjusted_rand_score(X_test_Tfidf, testlabels))
print(purity_score(testlabels, X_test_Tfidf))
