
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer as wnl
from nltk.corpus import stopwords

from urllib import request
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

poems = {
	"The Lamb": "./poems/thelamb.txt",
	"The Tiger": "./poems/thetiger.txt",
	"Sonnets": "./poems/sonnets.txt"
}

poems = {
	"The Lamb": "./poems/thelamb.txt",
	"The Tiger": "./poems/thetiger.txt",
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
def calculate_tfidf_and_bm25(documents, features):
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
	
	return tfidf_scores, bm25_scores

tfidf_scores, bm25_scores = calculate_tfidf_and_bm25(documents, features)


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
	# Filter out zero values
	data_stops = [x[i] for x in stops_tfidf if x[i] != 0.0]
	data_normal = [x[i] for x in normal_tfidf if x[i] != 0.0]
	
	# Calculate pearsonr
	data_x = data_stops + data_normal
	data_y = [0 for _ in range(len(data_stops))] + [1 for _ in range(len(data_normal))]
	print(f"{list(poems)[i]} - tfidf correlation: {pearsonr(data_x, data_y)}")
	
	ax[i].eventplot([data_stops, data_normal], lineoffsets=[1, 2], colors=['tab:blue', 'tab:orange'])
	ax[i].eventplot([[np.mean(data_stops)], [np.mean(data_normal)]], lineoffsets=[1, 2], colors=['tab:green', 'tab:red'])
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

	# Calculate pearsonr
	data_x = data_stops + data_normal
	data_y = [0 for _ in range(len(data_stops))] + [1 for _ in range(len(data_normal))]
	print(f"{list(poems)[i]} - bm25 correlation: {pearsonr(data_x, data_y)}")

	ax[i].eventplot([data_stops, data_normal], lineoffsets=[1, 2], colors=['tab:blue', 'tab:olive'])
	ax[i].eventplot([[np.mean(data_stops)], [np.mean(data_normal)]], lineoffsets=[1, 2],colors=['tab:orange', 'tab:purple'])
	ax[i].set(
		title=list(poems)[i],
		xlabel="bm25",
		ylabel="word type",
		xlim=(0, None),
		ylim=(0, 3), 
		yticks=np.arange(1, 3),
		yticklabels=["stop", "normal"]
	)

plt.savefig("./output/task1-3/fig_bm25_lambtiger.png")
plt.close()



## Extras 
# Analyze sonnets for stopword recognition, where each sonnet is regarded as its own document
sonnets_regex = r"[\d]+\n\n([\D]+).+" * 154

file = open("./poems/sonnets.txt")
sonnets_raw = file.read()
sonnets = re.search(sonnets_regex, sonnets_raw).groups()

documents = []
features = []
for sonnet in sonnets:
	tokens = word_tokenize(sonnet)
	documents.append(tokens)
	features.extend(tokens)

features = sorted(list(set(features)))

tfidf_scores, bm25_scores = calculate_tfidf_and_bm25(documents, features)

stops_tfidf = []
stops_bm25 = []
normal_tfidf = []
normal_bm25 = []
count = len(documents)

for i in range(len(features)):
	if wnl().lemmatize(features[i]) in stopwords.words():
		stops_tfidf.extend( [tfidf_scores[n][i] for n in range(count)] )
		stops_bm25.extend( [bm25_scores[n][i] for n in range(count)] )
	else:
		normal_tfidf.extend( [tfidf_scores[n][i] for n in range(count)] )
		normal_bm25.extend( [bm25_scores[n][i] for n in range(count)] )


fig, ax = plt.subplots()
fig.set_size_inches(15, 5)

data_stops = [x for x in stops_tfidf if x != 0.0]
data_normal = [x for x in normal_tfidf if x != 0.0]

data_x = data_stops + data_normal
data_y = [0 for _ in range(len(data_stops))] + [1 for _ in range(len(data_normal))]
print(f"Sonnets - tfidf correlation: {pearsonr(data_x, data_y)}")

ax.eventplot([data_stops, data_normal], lineoffsets=[1, 2], colors=['tab:blue', 'tab:olive'])
ax.eventplot([[np.mean(data_stops)], [np.mean(data_normal)]], lineoffsets=[1, 2], colors=['tab:orange', 'tab:purple'])
ax.set(
	title="Sonnets",
	xlabel="tf-idf",
	ylabel="word type",
	xlim=(0, None),
	ylim=(0, 3), 
	yticks=np.arange(1, 3),
	yticklabels=["stop", "normal"]
)

plt.savefig("./output/task1-3/fig_tfidf_sonnets.png")
plt.close()

fig, ax = plt.subplots()
fig.set_size_inches(15, 5)

data_stops = [x for x in stops_bm25 if x != 0.0]
data_normal = [x for x in normal_bm25 if x != 0.0]


data_x = data_stops + data_normal
data_y = [0 for _ in range(len(data_stops))] + [1 for _ in range(len(data_normal))]
print(f"Sonnets - bm25 correlation: {pearsonr(data_x, data_y)}")

ax.eventplot([data_stops, data_normal], lineoffsets=[1, 2], colors=['tab:blue', 'tab:olive'])
ax.eventplot([[np.mean(data_stops)], [np.mean(data_normal)]], lineoffsets=[1, 2], colors=['tab:orange', 'tab:purple'])
ax.set(
	title="Sonnets",
	xlabel="bm25",
	ylabel="word type",
	xlim=(0, None),
	ylim=(0, 3), 
	yticks=np.arange(1, 3),
	yticklabels=["stop", "normal"]
)

plt.savefig("./output/task1-3/fig_bm25_sonnets.png")
plt.close()