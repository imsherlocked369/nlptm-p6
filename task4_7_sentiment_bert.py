import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from collections import Counter
from scipy.spatial.distance import jensenshannon


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


for row in task5_results:
    print(
        row[0],
        "rho =", round(row[1], 4),
        "p =", round(row[2], 4)
    )

# Task 6 

os.makedirs("./output/task6", exist_ok=True)
bin_edges = np.linspace(-1.0, 1.0, 11)
profiles = {}
stopword_line_counts = Counter()
compound_scores = {}

for row in results:
    poem = row[0]
    line_number = row[1]
    compound = row[6]

    compound_scores[(poem, line_number)] = compound


# sentiment histogram for each stopword


for title, filename in poems:

    lines = read_poem_lines(filename)

    for line_number, line in enumerate(lines, start=1):

        tokens = word_tokenize(line)

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
                profiles[key] = np.zeros(
                    10,
                    dtype=int
                )

            profiles[key][bin_number] += 1
            stopword_line_counts[key] += 1


# Save raw histogram counts

profile_count_file = (
    "./output/task6/"
    "stopword_sentiment_profile_counts.csv"
)

with open(
    profile_count_file,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    header = [
        "poem",
        "stopword",
        "line_count"
    ]

    for i in range(10):
        header.append(f"bin_{i + 1}")

    writer.writerow(header)

    for key in sorted(profiles):

        title, word = key

        writer.writerow([
            title,
            word,
            stopword_line_counts[key],
            *profiles[key]
        ])


# Convert counts to probability distributions

probability_profiles = {}

for key in profiles:

    counts = profiles[key].astype(float)

    total = counts.sum()

    if total > 0:
        probability_profiles[key] = counts / total


probability_file = (
    "./output/task6/"
    "stopword_sentiment_profile_probabilities.csv"
)

with open(
    probability_file,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    header = [
        "poem",
        "stopword"
    ]

    for i in range(10):
        header.append(f"bin_{i + 1}")

    writer.writerow(header)

    for key in sorted(probability_profiles):

        title, word = key

        writer.writerow([
            title,
            word,
            *probability_profiles[key]
        ])


for title, filename in poems:

    words = []

    for poem_title, word in probability_profiles:

        if poem_title == title:
            words.append(word)

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

            distance = jensenshannon(
                profile_a,
                profile_b,
                base=2
            )

            distance_matrix[i][j] = distance


    safe_title = (
        title
        .lower()
        .replace(" ", "_")
    )


    distance_file = (
        "./output/task6/"
        + safe_title
        + "_sentiment_distances.csv"
    )

    with open(
        distance_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            ["stopword"] + words
        )

        for i in range(len(words)):

            writer.writerow(
                [words[i]]
                + list(distance_matrix[i])
            )

    words_by_frequency = sorted(
        words,
        key=lambda word:
            stopword_line_counts[
                (title, word)
            ],
        reverse=True
    )

    plot_words = words_by_frequency[:15]

    plot_matrix = np.zeros(
        (
            len(plot_words),
            len(plot_words)
        )
    )

    for i in range(len(plot_words)):

        for j in range(len(plot_words)):

            profile_a = probability_profiles[
                (title, plot_words[i])
            ]

            profile_b = probability_profiles[
                (title, plot_words[j])
            ]

            plot_matrix[i][j] = (
                jensenshannon(
                    profile_a,
                    profile_b,
                    base=2
                )
            )


    plt.figure(figsize=(10, 8))

    plt.imshow(
        plot_matrix,
        vmin=0,
        vmax=1
    )

    plt.colorbar(
        label="Jensen-Shannon distance"
    )

    plt.xticks(
        range(len(plot_words)),
        plot_words,
        rotation=90
    )

    plt.yticks(
        range(len(plot_words)),
        plot_words
    )

    plt.title(
        title
        + " - Stopword sentiment profile distance"
    )

    plt.tight_layout()

    plt.savefig(
        "./output/task6/"
        + safe_title
        + "_sentiment_distance_heatmap.png"
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

print("Histogram counts saved to:", profile_count_file)

print("Probability profiles saved to:",probability_file)