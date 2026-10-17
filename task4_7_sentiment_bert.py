import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


#task4
poems = [("The Lamb", "./poems/thelamb.txt"),("The Tiger", "./poems/thetiger.txt")]
stops = set(stopwords.words("english"))
sia = SentimentIntensityAnalyzer()

def read_poem_lines(filename):
    file = open(filename, "r", encoding="utf-8")
    lines = []

    for line in file:
        line = line.strip()

        if line != "":
            lines.append(line)

    file.close()
    return lines


def contains_stopword(line):
    tokens = word_tokenize(line)

    for token in tokens:
        if token.lower() in stops:
            return True

    return False

results = []


for title, filename in poems:
    lines = read_poem_lines(filename)

    for line_number in range(len(lines)):
        line = lines[line_number]

        if contains_stopword(line):
            sentiment = sia.polarity_scores(line)

            results.append([
                title,
                line_number + 1,
                line,
                sentiment["neg"],
                sentiment["neu"],
                sentiment["pos"],
                sentiment["compound"]
            ])


os.makedirs("./output/task4", exist_ok=True)

output_file = "./output/task4/vader_sentiment.csv"

with open(output_file, "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)

    writer.writerow([
        "poem",
        "line_number",
        "line",
        "negative",
        "neutral",
        "positive",
        "compound"
    ])

    writer.writerows(results)

    
# Task 5

os.makedirs("./output/task5", exist_ok=True)

task5_results = []

fig, ax = plt.subplots(2, 1, figsize=(10, 8))

for i in range(len(poems)):
    title = poems[i][0]

    poem_results = [
        row for row in results
        if row[0] == title
    ]

    line_numbers = [row[1] for row in poem_results]
    compound_scores = [row[6] for row in poem_results]

    rho, p_value = spearmanr(
        line_numbers,
        compound_scores
    )

    task5_results.append([
        title,
        rho,
        p_value
    ])

    ax[i].plot(
        line_numbers,
        compound_scores,
        marker="o"
    )

    ax[i].axhline(
        0,
        linestyle="--"
    )

    ax[i].set_title(title)
    ax[i].set_xlabel("Line number")
    ax[i].set_ylabel("VADER compound score")
    ax[i].set_ylim(-1, 1)


plt.tight_layout()
plt.savefig("./output/task5/sentiment_trend.png")
plt.close()


with open(
    "./output/task5/spearman_results.csv",
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "poem",
        "spearman_rho",
        "p_value"
    ])

    writer.writerows(task5_results)


print("\nTask 5 done")

for row in task5_results:
    print(
        row[0],
        "rho =", round(row[1], 4),
        "p =", round(row[2], 4)
    )