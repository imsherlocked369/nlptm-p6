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
from nltk.corpus import wordnet as wn

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

def wn_link(word1, lang1, word2, lang2):
	synsets1 = wn.synsets(word1, lang=lang1)
	synsets2 = wn.synsets(word2, lang=lang2)

	set1 = set(synsets1)
	set2 = set(synsets2)

	common_synsets = set1.intersection(set2)
	return len(common_synsets) > 0


def get_score(scores, key, m, n, weight_type="tf-idf"):

	tf_scores, bm25_scores, idf_scores, ent_scores = scores[key]
	match weight_type:
		case "tf-idf":
			return tf_scores[n][m] * idf_scores[m]
		case "tf-ent":
			return tf_scores[n][m] * ent_scores[m]
		case "bm-idf":
			return bm_scores[n][m] * idf_scores[m]
		case "bm-ent":
			return bm_scores[n][m] * ent_scores[m]




poems = [
	("The Lamb", "./poems/thelamb.txt", "english"),
	("The Tiger", "./poems/thetiger.txt", "english"),
	("Sonnets", "./poems/sonnets.txt", "english"),

	("Ensimmainen lempi", "./poems/ensimmainenlempi.txt", "finnish"),
	("Oli mulla kulta kaunoinen", "./poems/olimullakultakaunoinen.txt", "finnish")
]


documents = {}
features = {}
langs = list(set([lang for _, _, lang in poems]))

for name, path, lang in poems:
	documents[lang] = []
	features[lang] = []

for name, path, lang in poems:
	file = open(path, "r")
	tokens = [token.lower() for token in word_tokenize(file.read()) if token.isalpha()]
	documents[lang].append(tokens)
	features[lang].extend(tokens)

for key in langs:
	features[key] = sorted(list(set(features[key])))
	print(f"{key}: {len(features[key])}")


scores = {}
for key in langs:
	scores[key] = (calculate_scores(documents[key], features[key]))


stop_list = {}
for lang in langs:
	stop_list[lang] = [SnowballStemmer(lang).stem(feature) in stopwords.words(lang) for feature in features[lang]]
	print(f"{lang}: stops {stop_list[lang].count(True) / len(features[lang]):.02f}")


for i in range(len(langs) - 1):
	for j in range(i + 1, len(langs)):
		lang1 = langs[i]
		lang2 = langs[j]
		
		both_stops = []
		both_normal = []
		different = []

		shape1 = np.shape(scores[lang1][0])
		shape2 = np.shape(scores[lang2][0])

		for u1 in range(len(features[lang1])):
			for u2 in range(len(features[lang2])):

				lang1_feature = features[lang1][u1]
				lang2_feature = features[lang2][u2]

				lang1_stop = stop_list[lang1][u1]
				lang2_stop = stop_list[lang2][u2]

				if wn_link(lang1_feature, lang1[0:3], lang2_feature, lang2[0:3]):
					sum1 = [get_score(scores, lang1, u1, m) for m in range(len(documents[lang1]))]
					sum2 = [get_score(scores, lang2, u2, m) for m in range(len(documents[lang2]))]
					avg1 = sum(sum1) / len(documents[lang1])
					avg2 = sum(sum2) / len(documents[lang2])
					if lang1_stop and lang2_stop:
						both_stops.append((lang1_feature, lang2_feature, avg1, avg2))
					elif not lang1_stop and not lang1_stop:
						both_normal.append((lang1_feature, lang2_feature, avg1, avg2))
					else: 
						different.append((lang1_feature, lang2_feature, avg1, avg2))

		spt = sum([[a, b] for _, _, a, b in both_stops], [])
		npt = sum([[a, b] for _, _, a, b in both_normal], [])
		mpt = sum([[a, b] for _, _, a, b in different], [])
		stop_plot = [np.mean(spt), np.std(spt), max(spt), min(spt)]
		normal_plot = [np.mean(npt), np.std(npt), max(npt), min(npt)]
		mixed_plot = [np.mean(mpt), np.std(mpt), max(mpt), min(mpt)]



		print("\n\\begin{table}\\centering\\begin{tabular}{|c|c|c|c|c|}\\hline \n\t\t\t& Mean & Std & Max & min \\\\ \\hline \\hline")
		print(f"\t\tStop & {stop_plot[0]:.3e} & {stop_plot[1]:.3e} & {stop_plot[2]:.3e} & {stop_plot[3]:.3e} \\\\ \\hline")
		print(f"\t\tNormal & {normal_plot[0]:.3e} & {normal_plot[1]:.3e} & {normal_plot[2]:.3e} & {normal_plot[3]:.3e} \\\\ \\hline")
		print(f"\t\tMixed & {mixed_plot[0]:.3e} & {mixed_plot[1]:.3e} & {mixed_plot[2]:.3e} & {mixed_plot[3]:.3e} \\\\ \\hline")
		print(f"\\end{{tabular}}\\caption{{{lang1} {lang2}}}\\label{{tab:{lang1} {lang2}}}\\end{{table}}")






		


