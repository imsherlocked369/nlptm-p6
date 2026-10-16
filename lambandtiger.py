import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer as wnl
from urllib import request
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import numpy as np

titles= ["The Lamb", "The Tiger", "Sonnets"]
poems = [
	"./poems/thelamb.txt",
	"./poems/thetiger.txt",
	"./poems/sonnets.txt"
]



poems_dict = {}
documents = []
vocabulary = []

for poem in poems:
	file = open(poem, "r")
	tokens = word_tokenize(file.read())
	documents.append(tokens)
	vocabulary.extend(tokens)

vocabulary = sorted(list(set(vocabulary)))


tfidf_scores = np.zeros((len(documents), len(vocabulary)))
bm25_scores = np.zeros((len(documents), len(vocabulary)))
idfs = []

k_1 = 1.5
b = 0.75
avgdl = sum([len(document) for document in documents]) / len(documents)


for x in range(len(vocabulary)):
	term = vocabulary[x]
	count = 1
	for document in documents:
		count += 1 if term in document else 0
	idf = np.log10((len(documents) + 1)/ count)
	idfs.append(idf)
	
	for y in range(len(documents)):
		tf = documents[y].count(term) / len(documents[y])
		
		tfidf_scores[y][x] = tf * idf
		bm25_scores[y][x] = idf * tf * (k_1 + 1) / (tf + k_1 * (1 - b + b * (len(documents[y]) / avgdl)))



print("done")


features = vocabulary

stops = []
normal = []
count = len(poems)
for i in range(len(features)):
	if features[i] in stopwords.words():
		stops.append( [tfidf_scores[n][i] for n in range(count)] )
	else:
		normal.append( [tfidf_scores[n][i] for n in range(count)] )


fig, ax = plt.subplots(count, 1)
fig.set_size_inches(20, 10)

for i in range(count):
	data_stops = [x[i] for x in stops if x[i] > 0.0]
	data_normal = [x[i] for x in normal if x[i] > 0.0]
	ax[i].eventplot([data_stops, data_normal], lineoffsets=[1, 2], colors=['tab:blue', 'tab:orange'])
	ax[i].eventplot([[np.mean(data_stops)], [np.mean(data_normal)]], lineoffsets=[1, 2], colors=['tab:red', 'tab:green'])
	ax[i].set(
		title=titles[i],
		xlabel="tf-idf",
		ylabel="word type",
		xlim=(0, None),
		ylim=(0, 3), 
		yticks=np.arange(1, 3),
		yticklabels=["stop", "normal"]
	)

plt.savefig("tfidf_lambtiger.png")
plt.close()


stops = []
normal = []
count = len(poems)
for i in range(len(features)):
	if features[i] in stopwords.words():
		stops.append( [bm25_scores[n][i] for n in range(count)] )
	else:
		normal.append( [bm25_scores[n][i] for n in range(count)] )


fig, ax = plt.subplots(count, 1)
fig.set_size_inches(20, 10)

for i in range(count):
	data_stops = [x[i] for x in stops if x[i] > 0.0]
	data_normal = [x[i] for x in normal if x[i] > 0.0]
	ax[i].eventplot([data_stops, data_normal], lineoffsets=[1, 2], colors=['tab:blue', 'tab:orange'])
	ax[i].eventplot([[np.mean(data_stops)], [np.mean(data_normal)]], lineoffsets=[1, 2], colors=['tab:red', 'tab:green'])
	ax[i].set(
		title=titles[i],
		xlabel="tf-idf",
		ylabel="word type",
		xlim=(0, None),
		ylim=(0, 3), 
		yticks=np.arange(1, 3),
		yticklabels=["stop", "normal"]
	)

plt.savefig("bm25_lambtiger.png")
plt.close()

