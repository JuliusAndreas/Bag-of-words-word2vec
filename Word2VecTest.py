import pickle
from sklearn.neural_network import MLPClassifier
import numpy
from numpy import random
from sklearn import metrics
from sklearn.metrics import adjusted_rand_score
import pandas as pd
import re
import gensim.downloader as api

def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return numpy.sum(numpy.amax(contingency_matrix, axis=0)) / numpy.sum(contingency_matrix)


# load the model from disk
filename = 'mlp_model_word2vec.sav'
mlp_model = pickle.load(open(filename, 'rb'))
df_test = pd.read_csv('test.csv')
test_genres_strings = df_test['genres'].tolist()
labels = []

for i in range(len(test_genres_strings)):
    label = re.findall(r"\d+", test_genres_strings[i])
    label = str(label)
    label = label[1:len(label)-1]
    labels.append(label)

#Getting and processing input

df_test['labels'] = labels

new_df = df_test.drop('labels', axis=1).join(df_test['labels'].str.split(', ', expand=True).stack().
                                              reset_index(level=1, drop=True).rename('labels'))
labels = new_df['labels'].tolist()
overviews = new_df['overview'].tolist()

#Done processing input

for i in range(len(overviews)):
    overviews[i] = re.sub('[,?!"().]', '', str(overviews[i]))

list_of_lists = []
for e in overviews:
    list_of_lists.append(str(e).split(' '))

model = api.load("glove-wiki-gigaword-100")
print("model loaded successfully")
overviews_vectors = []
print(len(overviews_vectors))
for i in range(len(list_of_lists)):
    vectors = []
    for word in list_of_lists[i]:
        if word in model.vocab:
            vectors.append(model[word])
    sum = 0
    for j in range(len(vectors)):
        sum += vectors[j]
    for h in range(len(sum)):
        sum[h] = sum[h]/len(vectors)
    overviews_vectors.append(sum)


print("length of all overviews vectors list: "+str(len(overviews_vectors)))

predicted_labels = mlp_model.predict(overviews_vectors)

print(adjusted_rand_score(predicted_labels, labels))
print(purity_score(labels, predicted_labels))




