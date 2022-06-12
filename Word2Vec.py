import pandas as pd
import re
from sklearn.neural_network import _multilayer_perceptron
import sklearn.neural_network
from sklearn.neural_network import MLPClassifier
from gensim.models import word2vec as wc
from gensim.models import KeyedVectors as kv
import pickle
import gensim.downloader as api
import numpy

#Getting and processing input

df_train = pd.read_csv('train.csv')
train_genres_strings = df_train['genres'].tolist()
labels = []

for i in range(len(train_genres_strings)):
    label = re.findall(r"\d+", train_genres_strings[i])
    label = str(label)
    label = label[1:len(label)-1]
    labels.append(label)
df_train['labels'] = labels

new_df = df_train.drop('labels', axis=1).join(df_train['labels'].str.split(', ', expand=True).stack().
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


mlp = MLPClassifier(hidden_layer_sizes=(12,4), learning_rate_init=0.04, max_iter=1000)

mlp.fit(overviews_vectors, labels)

# save the model to disk
filename = 'mlp_model_word2vec.sav'
pickle.dump(mlp, open(filename, 'wb'))