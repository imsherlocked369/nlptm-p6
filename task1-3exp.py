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
	tokens = [token.lower() for token in word_tokenize(sonnet) if token.isalpha()]
	documents.append(tokens)
	features.extend(tokens)
	if len(documents) == sonnet_max:
		break

features = sorted(list(set(features)))
stop_list = [SnowballStemmer("english").stem(feature) in stopwords.words() for feature in features]

tf_scores, bm25_scores, idf_scores, ent_scores = calculate_scores(documents, features)


N, M = np.shape(tf_scores)
print(N, M)

tf_idf_stop = [ ]
tf_ent_stop = [ ]
bm_idf_stop = [ ]
bm_ent_stop = [ ]

tf_idf_norm = [ ]
tf_ent_norm = [ ]
bm_idf_norm = [ ]
bm_ent_norm = [ ]

for n in range(N):
	for m in range(M):
		if stop_list[m]:
			tf_idf_stop.append(tf_scores[n][m] * idf_scores[m])
			tf_ent_stop.append(tf_scores[n][m] * ent_scores[m])
			bm_idf_stop.append(bm25_scores[n][m] * idf_scores[m])
			bm_ent_stop.append(bm25_scores[n][m] * ent_scores[m])
		else:
			tf_idf_norm.append(tf_scores[n][m] * idf_scores[m])
			tf_ent_norm.append(tf_scores[n][m] * ent_scores[m])
			bm_idf_norm.append(bm25_scores[n][m] * idf_scores[m])
			bm_ent_norm.append(bm25_scores[n][m] * ent_scores[m])



combined_stop = [
	tf_idf_stop,
	tf_ent_stop,
	bm_idf_stop,
	bm_ent_stop
]

combined_norm = [
	tf_idf_norm,
	tf_ent_norm,
	bm_idf_norm,
	bm_ent_norm
]


types = ["tf-idf", "tf-ent", "bm-idf", "bm-ent"]
fig, ax = plt.subplots(4, 1)
fig.set_size_inches(15, 15)
fig.suptitle("Sonnets weights") 

mean_of_means = []

for i in range(4):
	data_stops = [data for data in combined_stop[i] if data != 0.0]
	data_normal = [data for data in combined_norm[i] if data != 0.0]
	mean = [[np.mean(data_stops)], [np.mean(data_normal)]]
	mean_of_means.append((mean[0][0] + mean[1][0]) / 2)

	ax[i].eventplot([data_stops, data_normal], lineoffsets=[1, 2], colors=['tab:blue', 'tab:orange'])
	ax[i].eventplot(mean, lineoffsets=[1, 2], colors=['tab:green', 'tab:red'])
	ax[i].set(
		xlabel=types[i],
		ylabel="word type",
		xlim=(0, None),
		ylim=(0, 3),
		yticks=np.arange(1, 3),
		yticklabels=["stop", "normal"]
	)

print(mean_of_means)
plt.savefig(f"./output/task1-3/fig_sonnets_eventplot.png")
plt.close()


for i in range(4):
	tp = 0
	tn = 0
	fp = 0
	fn = 0
	for n in range(len(combined_stop[i])):
		if combined_stop[i][n] == 0:
			pass
		elif combined_stop[i][n] <= mean_of_means[i]:
			tp += 1
		else:
			fn += 1
	for n in range(len(combined_norm[i])):
		if combined_norm[i][n] == 0:
			pass
		elif combined_norm[i][n] > mean_of_means[i]:
			tn += 1
		else:
			fp += 1
	
	print(f"{types[i]}\nacc\t{(tp+tn)/(tp+tn+fn+fp)}\npre\t{tp/(tp+fp)}\n")




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
