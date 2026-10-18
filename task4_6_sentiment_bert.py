import os
import re
import csv
import numpy as np
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from scipy.stats import spearmanr
from scipy.spatial.distance import jensenshannon
from collections import Counter


base_dir = os.path.dirname(os.path.abspath(__file__))

poems = [
    ("The Lamb", os.path.join(base_dir, "poems", "thelamb.txt")),
    ("The Tiger", os.path.join(base_dir, "poems", "thetiger.txt"))
]

stops = set(stopwords.words("english"))
sia = SentimentIntensityAnalyzer()


def tokenize_line(line):
    line = line.replace("’", "'").replace("‘", "'")
    return re.findall(r"[^\W\d_]+(?:'[^\W\d_]+)?", line)


def read_poem_lines(filename):
    lines = []

    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if line != "":
                lines.append(line)

    return lines


def contains_stopword(line):
    tokens = tokenize_line(line)

    for token in tokens:
        if token.lower() in stops:
            return True

    return False


# Task 4

results = []

for title, filename in poems:
    lines = read_poem_lines(filename)

    for line_number, line in enumerate(lines, start=1):

        if contains_stopword(line):
            sentiment = sia.polarity_scores(line)

            results.append([
                title,
                line_number,
                line,
                sentiment["neg"],
                sentiment["neu"],
                sentiment["pos"],
                sentiment["compound"]
            ])


task4_output = os.path.join(base_dir, "output", "task4")
os.makedirs(task4_output, exist_ok=True)

output_file = os.path.join(task4_output, "vader_sentiment.csv")

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


print("Task 4 done")
print("Number of lines analysed:", len(results))


# Task 5

task5_output = os.path.join(base_dir, "output", "task5")
os.makedirs(task5_output, exist_ok=True)

task5_results = []

fig, ax = plt.subplots(2, 1, figsize=(10, 8))

for i in range(len(poems)):
    title = poems[i][0]

    poem_results = [
        row for row in results
        if row[0] == title
    ]

    line_numbers = [row[1] for row in poem_results]
    compound_values = [row[6] for row in poem_results]

    rho, p_value = spearmanr(line_numbers, compound_values)

    task5_results.append([
        title,
        rho,
        p_value
    ])

    ax[i].plot(
        line_numbers,
        compound_values,
        marker="o"
    )

    ax[i].axhline(0, linestyle="--")

    ax[i].set_title(title)
    ax[i].set_xlabel("Line number")
    ax[i].set_ylabel("VADER compound score")
    ax[i].set_ylim(-1, 1)


plt.tight_layout()

plt.savefig(
    os.path.join(
        task5_output,
        "sentiment_trend.png"
    )
)

plt.close()


spearman_file = os.path.join(
    task5_output,
    "spearman_results.csv"
)

with open(spearman_file, "w", newline="", encoding="utf-8") as file:
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


# Task 6

task6_output = os.path.join(base_dir, "output", "task6")
os.makedirs(task6_output, exist_ok=True)

bin_edges = np.linspace(-1.0, 1.0, 11)

profiles = {}
stopword_line_counts = Counter()

compound_scores = {}

for row in results:
    poem = row[0]
    line_number = row[1]
    compound = row[6]

    compound_scores[(poem, line_number)] = compound


for title, filename in poems:
    lines = read_poem_lines(filename)

    for line_number, line in enumerate(lines, start=1):

        tokens = tokenize_line(line)

        line_stopwords = set()

        for token in tokens:
            word = token.lower()

            if word in stops:
                line_stopwords.add(word)

        if len(line_stopwords) == 0:
            continue

        compound = compound_scores[(title, line_number)]

        bin_number = np.digitize(
            compound,
            bin_edges[1:-1]
        )

        for word in line_stopwords:
            key = (title, word)

            if key not in profiles:
                profiles[key] = np.zeros(10, dtype=int)

            profiles[key][bin_number] += 1
            stopword_line_counts[key] += 1


profile_count_file = os.path.join(
    task6_output,
    "stopword_sentiment_profile_counts.csv"
)

with open(profile_count_file, "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)

    header = [
        "poem",
        "stopword",
        "line_count"
    ]

    for i in range(10):
        header.append("bin_" + str(i + 1))

    writer.writerow(header)

    for key in sorted(profiles):
        title, word = key

        writer.writerow([
            title,
            word,
            stopword_line_counts[key],
            *profiles[key]
        ])


probability_profiles = {}

for key in profiles:
    counts = profiles[key].astype(float)
    total = counts.sum()

    if total > 0:
        probability_profiles[key] = counts / total


probability_file = os.path.join(
    task6_output,
    "stopword_sentiment_profile_probabilities.csv"
)

with open(probability_file, "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)

    header = [
        "poem",
        "stopword"
    ]

    for i in range(10):
        header.append("bin_" + str(i + 1))

    writer.writerow(header)

    for key in sorted(probability_profiles):
        title, word = key

        writer.writerow([
            title,
            word,
            *probability_profiles[key]
        ])


for title, filename in poems:

    words = [
        word
        for poem_title, word in probability_profiles
        if poem_title == title
    ]

    words = sorted(words)

    distance_matrix = np.zeros(
        (len(words), len(words))
    )

    for i in range(len(words)):
        for j in range(len(words)):

            profile_a = probability_profiles[
                (title, words[i])
            ]

            profile_b = probability_profiles[
                (title, words[j])
            ]

            distance_matrix[i][j] = jensenshannon(
                profile_a,
                profile_b,
                base=2
            )


    safe_title = title.lower().replace(" ", "_")

    distance_file = os.path.join(
        task6_output,
        safe_title + "_sentiment_distances.csv"
    )

    with open(distance_file, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow(
            ["stopword"] + words
        )

        for i in range(len(words)):
            writer.writerow(
                [words[i]] + list(distance_matrix[i])
            )


    plt.figure(figsize=(10, 8))

    plt.imshow(
        distance_matrix,
        vmin=0,
        vmax=1
    )

    plt.colorbar(
        label="Jensen-Shannon distance"
    )

    plt.xticks(
        range(len(words)),
        words,
        rotation=90
    )

    plt.yticks(
        range(len(words)),
        words
    )

    plt.title(
        title + " - Stopword sentiment profile distance"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            task6_output,
            safe_title + "_sentiment_distance_heatmap.png"
        )
    )

    plt.close()


print("\nTask 6 done")

for title, filename in poems:

    poem_words = [
        word
        for poem_title, word in profiles
        if poem_title == title
    ]

    print(
        title,
        "number of stopwords analysed:",
        len(poem_words)
    )

print(
    "Histogram counts saved to:",
    profile_count_file
)

print(
    "Probability profiles saved to:",
    probability_file
)