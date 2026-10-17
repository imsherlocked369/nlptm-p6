import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
import csv
import os


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


print("Task 4 done")
print("Number of lines analysed:", len(results))
print("Results saved to:", output_file)