import os
import re
import csv
import numpy as np
import matplotlib.pyplot as plt
import torch

from nltk.corpus import stopwords
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import jensenshannon


base_dir = os.path.dirname(os.path.abspath(__file__))

poems = [
    (
        "The Lamb",
        os.path.join(base_dir, "poems", "thelamb.txt")
    ),
    (
        "The Tiger",
        os.path.join(base_dir, "poems", "thetiger.txt")
    )
]

output_dir = os.path.join(
    base_dir,
    "output",
    "task7"
)

os.makedirs(output_dir, exist_ok=True)


# stopwords

stops = set(stopwords.words("english"))


# tokenizer used for poem words

def tokenize_line(line):
    line = line.replace("’", "'").replace("‘", "'")

    return re.findall(
        r"[^\W\d_]+(?:'[^\W\d_]+)?",
        line
    )


# read poem and remove empty lines

def read_poem_lines(filename):
    lines = []

    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if line != "":
                lines.append(line)

    return lines


# load BERT

tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-uncased"
)

model = AutoModel.from_pretrained(
    "bert-base-uncased"
)
model.eval()


print("BERT loaded")
def get_bert_embedding(text):
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )

    with torch.no_grad():
        output = model(**encoded)

    hidden = output.last_hidden_state[0]

   
    if hidden.shape[0] > 2:
        hidden = hidden[1:-1]

    embedding = hidden.mean(dim=0)

    return embedding.numpy()

# semantic contribution

contribution_rows = []


for title, filename in poems:

    print("\nProcessing:", title)

    lines = read_poem_lines(filename)

    for line_number, line in enumerate(
        lines,
        start=1
    ):

        words = tokenize_line(line)

        if len(words) == 0:
            continue

        # check if the line contains at least one stopword
        line_stopwords = []

        for word in words:
            if word.lower() in stops:
                line_stopwords.append(
                    word.lower()
                )

        if len(line_stopwords) == 0:
            continue

        # embedding for whole line
        line_embedding = get_bert_embedding(line)

        similarities = []

       
        for word in words:

            word_embedding = get_bert_embedding(
                word
            )

            similarity = cosine_similarity(
                [line_embedding],
                [word_embedding]
            )[0][0]

            similarities.append([
                word,
                similarity
            ])

        similarity_sum = sum(
            item[1]
            for item in similarities
        )

        if abs(similarity_sum) < 1e-12:
            continue

        for word, similarity in similarities:

            word_lower = word.lower()

            if word_lower in stops:

                contribution = (
                    similarity /
                    similarity_sum
                )

                contribution_rows.append([
                    title,
                    line_number,
                    line,
                    word_lower,
                    similarity,
                    contribution
                ])



# save contribution values


contribution_file = os.path.join(
    output_dir,
    "stopword_semantic_contributions.csv"
)

with open(
    contribution_file,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "poem",
        "line_number",
        "line",
        "stopword",
        "cosine_similarity",
        "contribution"
    ])

    writer.writerows(
        contribution_rows
    )



# stopword summary


summary_rows = []


for title, filename in poems:

    poem_words = {}

    for row in contribution_rows:

        if row[0] != title:
            continue

        word = row[3]
        contribution = row[5]

        if word not in poem_words:
            poem_words[word] = []

        poem_words[word].append(
            contribution
        )

    for word in sorted(poem_words):

        values = poem_words[word]

        summary_rows.append([
            title,
            word,
            len(values),
            np.mean(values),
            np.median(values),
            np.std(values),
            np.min(values),
            np.max(values)
        ])


summary_file = os.path.join(
    output_dir,
    "stopword_contribution_summary.csv"
)

with open(
    summary_file,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "poem",
        "stopword",
        "count",
        "mean_contribution",
        "median_contribution",
        "standard_deviation",
        "minimum",
        "maximum"
    ])

    writer.writerows(
        summary_rows
    )



# make 10-bin contribution profiles


contribution_profiles = {}


for title, filename in poems:

    poem_rows = []

    for row in contribution_rows:
        if row[0] == title:
            poem_rows.append(row)

    if len(poem_rows) == 0:
        continue

    values = [
        row[5]
        for row in poem_rows
    ]

    minimum = min(values)
    maximum = max(values)

    if minimum == maximum:
        maximum = minimum + 0.000001

    # use the same bin range for all stopwords
    # inside one poem
    bin_edges = np.linspace(
        minimum,
        maximum,
        11
    )

    for row in poem_rows:

        word = row[3]
        contribution = row[5]

        key = (
            title,
            word
        )

        if key not in contribution_profiles:

            contribution_profiles[key] = np.zeros(
                10,
                dtype=int
            )

        bin_number = np.digitize(
            contribution,
            bin_edges[1:-1]
        )

        contribution_profiles[
            key
        ][bin_number] += 1



# save contribution histograms


profile_file = os.path.join(
    output_dir,
    "stopword_contribution_profiles.csv"
)

with open(
    profile_file,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    header = [
        "poem",
        "stopword",
        "count"
    ]

    for i in range(10):
        header.append(
            "bin_" + str(i + 1)
        )

    writer.writerow(
        header
    )

    for key in sorted(
        contribution_profiles
    ):

        title, word = key

        counts = contribution_profiles[
            key
        ]

        writer.writerow([
            title,
            word,
            int(counts.sum()),
            *counts
        ])



# comparing contribution distributions


for title, filename in poems:

    words = []

    for poem_title, word in contribution_profiles:

        if poem_title == title:
            words.append(word)

    words = sorted(words)

    if len(words) == 0:
        continue

    distance_matrix = np.zeros(
        (
            len(words),
            len(words)
        )
    )

    for i in range(len(words)):

        for j in range(len(words)):

            profile_a = contribution_profiles[
                (title, words[i])
            ].astype(float)

            profile_b = contribution_profiles[
                (title, words[j])
            ].astype(float)

            profile_a = (
                profile_a /
                profile_a.sum()
            )

            profile_b = (
                profile_b /
                profile_b.sum()
            )

            distance = jensenshannon(
                profile_a,
                profile_b,
                base=2
            )

            distance_matrix[
                i,
                j
            ] = distance


    safe_title = (
        title
        .lower()
        .replace(" ", "_")
    )


   

    distance_file = os.path.join(
        output_dir,
        safe_title
        + "_contribution_distances.csv"
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
                + list(
                    distance_matrix[i]
                )
            )


    # heatmap

    plt.figure(
        figsize=(10, 8)
    )

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
        title
        + " - Semantic contribution distance"
    )

    plt.tight_layout()

    heatmap_file = os.path.join(
        output_dir,
        safe_title
        + "_contribution_heatmap.png"
    )

    plt.savefig(
        heatmap_file
    )

    plt.close()



# print stopwords with highest average contribution
print("\nMost contributing stopwords")


for title, filename in poems:

    poem_summary = []

    for row in summary_rows:

        if row[0] == title:
            poem_summary.append(row)

    # only use stopwords appearing at least twice
    repeated_words = [
        row
        for row in poem_summary
        if row[2] >= 2
    ]

    repeated_words = sorted(
        repeated_words,
        key=lambda row: row[3],
        reverse=True
    )

    print("\n" + title)

    for row in repeated_words[:5]:

        print(
            row[1],
            "- mean contribution:",
            round(row[3], 4),
            "- occurrences:",
            row[2]
        )


print("Contribution value:",contribution_file)

print("Summary ",summary_file)

print("Profiles",profile_file)
