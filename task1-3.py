
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer as wnl
from nltk.corpus import stopwords

from urllib import request
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import numpy as np

poems = {
	"The Lamb": "./poems/thelamb.txt",
	"The Tiger": "./poems/thetiger.txt",
	"Sonnets": "./poems/sonnets.txt"
}

# Tokenization
documents = []
features = []

for key in poems.keys():
	file = open(poems[key], "r")
	tokens = word_tokenize(file.read())
	documents.append(tokens)
	features.extend(tokens)


# Remove duplicates and sort
features = sorted(list(set(features)))


# Calculating tf-idf and bm25 scores
tfidf_scores = np.zeros((len(documents), len(features)))
bm25_scores = np.zeros((len(documents), len(features)))
idfs = []

k_1 = 1.5
b = 0.75
avgdl = sum([len(document) for document in documents]) / len(documents)

for x in range(len(features)):
	term = features[x]
	count = 1
	for document in documents:
		count += 1 if term in document else 0
	idf = np.log10((len(documents) + 1)/ count)
	idfs.append(idf)
	
	for y in range(len(documents)):
		tf = documents[y].count(term) / len(documents[y])
		
		tfidf_scores[y][x] = tf * idf
		bm25_scores[y][x] = idf * tf * (k_1 + 1) / (tf + k_1 * (1 - b + b * (len(documents[y]) / avgdl)))


# Sort score to stopword and normal
stops_tfidf = []
stops_bm25 = []
normal_tfidf = []
normal_bm25 = []
count = len(poems)

for i in range(len(features)):
	if wnl().lemmatize(features[i]) in stopwords.words():
		stops_tfidf.append( [tfidf_scores[n][i] for n in range(count)] )
		stops_bm25.append( [bm25_scores[n][i] for n in range(count)] )
	else:
		normal_tfidf.append( [tfidf_scores[n][i] for n in range(count)] )
		normal_bm25.append( [bm25_scores[n][i] for n in range(count)] )


# Plot scores
fig, ax = plt.subplots(count, 1)
fig.set_size_inches(20, 10)

for i in range(count):
	data_stops = [x[i] for x in stops_tfidf if x[i] != 0.0]
	data_normal = [x[i] for x in normal_tfidf if x[i] != 0.0]
	ax[i].eventplot([data_stops, data_normal], lineoffsets=[1, 2], colors=['tab:blue', 'tab:olive'])
	ax[i].eventplot([[np.mean(data_stops)], [np.mean(data_normal)]], lineoffsets=[1, 2], colors=['tab:orange', 'tab:purple'])
	ax[i].set(
		title=list(poems)[i],
		xlabel="tf-idf",
		ylabel="word type",
		xlim=(0, None),
		ylim=(0, 3), 
		yticks=np.arange(1, 3),
		yticklabels=["stop", "normal"]
	)

plt.savefig("./output/task1-3/fig_tfidf_lambtiger.png")
plt.close()

fig, ax = plt.subplots(count, 1)
fig.set_size_inches(20, 10)

for i in range(count):
	data_stops = [x[i] for x in stops_bm25 if x[i] != 0.0]
	data_normal = [x[i] for x in normal_bm25 if x[i] != 0.0]
	ax[i].eventplot([data_stops, data_normal], lineoffsets=[1, 2], colors=['tab:blue', 'tab:olive'])
	ax[i].eventplot([[np.mean(data_stops)], [np.mean(data_normal)]], lineoffsets=[1, 2],colors=['tab:orange', 'tab:purple'])
	ax[i].set(
		title=list(poems)[i],
		xlabel="tf-idf",
		ylabel="word type",
		xlim=(0, None),
		ylim=(0, 3), 
		yticks=np.arange(1, 3),
		yticklabels=["stop", "normal"]
	)

plt.savefig("./output/task1-3/fig_bm25_lambtiger.png")
plt.close()
