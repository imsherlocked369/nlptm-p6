from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer as wnl
from nltk.corpus import stopwords

from urllib import request
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

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

def sonnets_correlations(local_weight, global_weight, stops):
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

	return correlations

sonnet_max = 154

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
	if len(documents) == sonnet_max:
		break

features = sorted(list(set(features)))

tf_scores, bm25_scores, idf_scores, ent_scores = calculate_scores(documents, features)


stop_list = [wnl().lemmatize(feature) in stopwords.words() for feature in features]


# Plot scores
cors = np.array([
	sonnets_correlations(tf_scores, idf_scores, stop_list),
	sonnets_correlations(tf_scores, ent_scores, stop_list),
	sonnets_correlations(bm25_scores, idf_scores, stop_list),
	sonnets_correlations(bm25_scores, ent_scores, stop_list)
])





names = len(documents)
types = ["tf-idf", "tf-ent", "bm-idf", "bm-ent"]

for i in range(len(cors)):
	avg = sum(cors[i]) / len(cors[i])
	print(f"{types[i]}: {avg:.03f}")


fig, ax = plt.subplots()
fig.set_size_inches(25, 5)
im = ax.imshow(cors, cmap='viridis', vmin=-1.0, vmax=1.0)
ax.set(
	title="Sonnets - Low weight correlation with stopword",
	xlabel="Sonnet number",
	ylabel="weight function",
	yticks=range(len(types)),
	yticklabels=types,
	xticks=range(0, names, 5),
	aspect=4
)

ax.figure.colorbar(im, ax=ax)

for j in range(names):
	for i in range(len(types)):
		text = ax.text(j, i, f"{cors[i, j]:.02f}", ha="center", va="center", color="w", rotation=90)

plt.tight_layout()
plt.savefig(f"./output/task1-3/fig_sonnets_cors.png")
plt.close()


print("Done")
exit()






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