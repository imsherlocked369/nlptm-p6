
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer as wnl
from nltk.corpus import stopwords

from urllib import request
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from nltk.stem.snowball import SnowballStemmer


poems = {
	"The Lamb": "./poems/thelamb.txt",
	"The Tiger": "./poems/thetiger.txt",
}

# Tokenization
documents = []
features = []

for key in poems.keys():
	file = open(poems[key], "r")
	tokens = [token.lower() for token in word_tokenize(file.read()) if token.isalpha()]
	documents.append(tokens)
	features.extend(tokens)


# Remove duplicates and sort
features = sorted(list(set(features)))


# Calculating tf-idf and bm25 scores
def calculate_scores(documents, features):
	N = len(documents)
	M = len(features)

	tf_scores = np.zeros((N, M))
	bm25_scores = np.zeros((N, M))
	idf_scores = np.zeros((M))
	ent_scores = np.zeros((M))

	k_1 = 1.5
	b = 0.75
	gl = sum([len(d) for d in documents])
	avgdl = gl / N

	for x in range(M):
		term = features[x]
		df = 0
		gf = 0

		for d in documents:
			df += 1 if term in d else 0
			gf += d.count(term)
		gf /= gl
		
		idf = 1 + np.log2(N / df)
		ent = 0
		
		idf_scores[x] = idf

		for y in range(N):
			tf = documents[y].count(term) / len(documents[y])
			ent += (tf / gf) * np.log2(tf / gf) if tf else 0
			tf_scores[y][x] = tf
			bm25_scores[y][x] = tf * (k_1 + 1) / (tf + k_1 * (1 - b + b * (len(documents[y]) / avgdl)))

		ent = ent / np.log2(N) + 1.0
		ent_scores[x] = ent

	
	return tf_scores, bm25_scores, idf_scores, ent_scores

tf_scores, bm25_scores, idf_scores, ent_scores = calculate_scores(documents, features)


stop_list = [SnowballStemmer("english").stem(feature) in stopwords.words() for feature in features]



def plot_locglo(local_weight, global_weight, stops, name=""):
	N, M = np.shape(local_weight)

	fig, ax = plt.subplots(N, 1)
	fig.set_size_inches(12, 9)

	data = np.zeros((N, M))
	for n in range(N):
		for m in range(M):
			data[n][m] = local_weight[n][m] * global_weight[m]
	
	correlations = []

	for i in range(N):
		data_stops = [data[i][j] for j in range(M) if stops[j] and data[i][j] != 0.0]
		data_normal = [data[i][j] for j in range(M) if not stops[j] and data[i][j] != 0.0]
		
		data_x = data_stops + data_normal
		data_y = [0 for _ in range(len(data_stops))] + [1 for _ in range(len(data_normal))]
		cor = pearsonr(data_x, data_y)
		correlations.append(cor.statistic)
		print(f"{list(poems)[i]} {name}: {cor}")

		ax[i].eventplot([data_stops, data_normal], lineoffsets=[1, 2], colors=['tab:blue', 'tab:orange'])
		ax[i].eventplot([[np.mean(data_stops)], [np.mean(data_normal)]], lineoffsets=[1, 2], colors=['tab:green', 'tab:red'])
		ax[i].set(
			title=list(poems)[i],
			xlabel=name,
			ylabel="word type",
			xlim=(0, None),
			ylim=(0, 3),
			yticks=np.arange(1, 3),
			yticklabels=["stop", "normal"]
		)

	# Calculate pearsonr

	plt.savefig(f"./output/task1-3/fig_{name}.png")
	plt.close()

	return correlations


# Plot scores
cors = np.transpose([
	plot_locglo(tf_scores, idf_scores, stop_list, "tf-idf"),
	plot_locglo(tf_scores, ent_scores, stop_list, "tf-ent"),
	plot_locglo(bm25_scores, idf_scores, stop_list, "bm-idf"),
	plot_locglo(bm25_scores, ent_scores, stop_list, "bm-ent")
])




names = list(poems)
types = ["tf-idf", "tf-ent", "bm-idf", "bm-ent"]


fig, ax = plt.subplots()
fig.set_size_inches(12, 6)
im = ax.imshow(cors, cmap='viridis', vmin=-1.0, vmax=1.0)
ax.set(
	title="Low weight correlation with stopword",
	xticks=range(len(types)),
	xticklabels=types,
	yticks=range(len(names)),
	yticklabels=names
)

ax.figure.colorbar(im, ax=ax)

for i in range(len(names)):
	for j in range(len(types)):
		text = ax.text(j, i, f"{cors[i, j]:.03f}", ha="center", va="center", color="w")

plt.savefig(f"./output/task1-3/fig_cors.png")
plt.close()


print("Done")