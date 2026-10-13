import nltk
from urllib import request
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import numpy as np

poems = [
	"./poems/thelamb.txt",
	"./poems/thetiger.txt"
]


poems_dict = {}
documents = []

for poem in poems:
	file = open(poem, "r")
	documents.append(file.read())


tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
features = tfidf_vectorizer.get_feature_names_out()


scores = []
tfidf_scores = tfidf_matrix.toarray()



stops = []
normal = []

count = len(poems)
for i in range(len(features)):
	if features[i] in stopwords.words():
		stops.append( [tfidf_scores[n][i] for n in range(count)] )
	else:
		normal.append( [tfidf_scores[n][i] for n in range(count)] )


fig, ax = plt.subplots(1, 2)
fig.set_size_inches(20, 10)

ax[0].set_xlabel("tf-idf")
ax[0].set_ylabel("stop (1) or normal (3) ")
ax[0].eventplot([[x[0] for x in stops], [x[0] for x in normal]], lineoffsets=[1, 2])
ax[0].set(
	title="The Lamb",
	xlabel="tf-idf",
	ylabel="word type",
	ylim=(0, 3), 
	yticks=np.arange(1, 3),
	yticklabels=["stop", "normal"]
)


ax[1].set_xlabel("tf-idf")
ax[1].set_ylabel("stop (1) or normal (3) ")
ax[1].eventplot([[x[1] for x in stops], [x[1] for x in normal]], lineoffsets=[1, 2])
ax[1].set(
	title="The Tiger",
	xlabel="tf-idf",
	ylabel="word type",
	ylim=(0, 3), 
	yticks=np.arange(1, 3),
	yticklabels=["stop", "normal"]
)

plt.savefig("lambtiger.png")
plt.close()

