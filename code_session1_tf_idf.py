import os
import re
from collections import defaultdict
import numpy as np

# Gather directory of data (test, train and list words)
def gather_20newsgroups_data():
    path = "20news-bydate/"

    dirs = [
        path + dir_name + "/"
        for dir_name in os.listdir(path)
        if not os.path.isfile(path + dir_name)
    ]

    train_dir, test_dir = (
        (dirs[0], dirs[1]) if "train" in dirs[0] else (dirs[1], dirs[0])
    )

    list_newsgroups = [newsgroup for newsgroup in os.listdir(train_dir)]
    list_newsgroups.sort()
    return train_dir, test_dir, list_newsgroups


# Gather stop words
with open("20news-bydate/stop_words.txt") as f:
    stop_words = f.read().splitlines()

from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

# Collecting data from parent_dir and newsgroup_list
def collect_data_from(parent_dir, newsgroup_list):
    data = []
    for group_id, newsgroup in enumerate(newsgroup_list):
        label = group_id
        dir_path = parent_dir + "/" + newsgroup + "/"
        files = [
            (filename, dir_path + filename)
            for filename in os.listdir(dir_path)
            if os.path.isfile(dir_path + filename)
        ]
        files.sort()

        for filename, filepath in files:
            with open(filepath) as f:
                text = f.read().lower()

                words = [
                    stemmer.stem(word)
                    for word in re.split("\W+", text)
                    if word not in stop_words
                ]
                content = " ".join(words)
                assert len(content.splitlines()) == 1
                data.append(str(label) + "<fff>" + filename + "<fff>" + content)

    return data


# Get data
# train_dir, test_dir, list_newsgroups = gather_20newsgroups_data()
# train_data = collect_data_from(parent_dir=train_dir, newsgroup_list=list_newsgroups)
# test_data = collect_data_from(parent_dir=test_dir, newsgroup_list=list_newsgroups)

# full_data = train_data + test_data
# with open("20news-bydate/20news-train-processed.txt", "w") as f:
#     f.write("\n".join(train_data))
# with open("20news-bydate/20news-test-processed.txt", "w") as f:
#     f.write("\n".join(test_data))
# with open("20news-bydate/20news-full-processed.txt", "w") as f:
#     f.write("\n".join(full_data))

# Vocabulary
def generate_vocabulary(data_path):
    def compute_idf(article_appeared, article_total):
        assert article_appeared > 0
        return np.log10(article_total / article_appeared)

    with open(data_path) as f:
        lines = f.read().splitlines()
    doc_count = defaultdict(int)
    article_total = len(lines)
    for line in lines:
        text = line.split("<fff>")[-1]  # line = [label, filename , text]
        words = list(set(text.split()))
        for word in words:
            doc_count[word] += 1

    words_idfs = [
        (word, compute_idf(article_appeared, article_total))
        for word, article_appeared in zip(doc_count.keys(), doc_count.values())
        if article_appeared > 10 and not word.isdigit()
    ]

    words_idfs.sort(key=lambda x: -x[1])
    print("Vocabulary size: {}".format(len(doc_count)))

    with open("20news-bydate/test_words_idfs.txt", "w") as f:
        f.write("\n".join([word + "<fff>" + str(idf) for word, idf in words_idfs]))


# Counting tf-idf score
def get_tf_idf(data_path):
    with open("20news-bydate/test_words_idfs.txt") as f:
        lines = f.read().splitlines()
        words_idfs = [
            (line.split("<fff>")[0], float(line.split("<fff>")[1])) for line in lines
        ]

        words_IDs = dict(
            [(word, index) for index, (word, idf) in enumerate(words_idfs)]
        )

        idfs = dict(words_idfs)
    with open(data_path) as f:
        documents = [
            (
                int(line.split("<fff>")[0]),
                int(line.split("<fff>")[1]),
                line.split("<fff>")[2],
            )
            for line in f.read().splitlines()
        ]

    data_tf_idf = []
    for document in documents:
        label, doc_id, text = document
        words = [word for word in text.split() if word in idfs]
        word_set = list(set(words))
        max_term_freq = max([words.count(word) for word in word_set])
        words_tfidfs = []
        sum_squares = 0.0
        for word in word_set:
            term_freq = words.count(word)
            tf_idf_value = term_freq / max_term_freq * idfs[word]
            words_tfidfs.append((words_IDs[word], tf_idf_value))
            sum_squares += tf_idf_value ** 2

        words_tfidfs_normalized = [
            str(index) + ":" + str(tf_idf_value / np.sqrt(sum_squares))
            for index, tf_idf_value in words_tfidfs
        ]
        sparse_rep = " ".join(words_tfidfs_normalized)
        data_tf_idf.append((label, doc_id, sparse_rep))

    with open("20news-bydate/test_words_tf_idf.txt", "w") as f:
        f.write(
            "\n".join(
                [
                    str(label) + "<fff>" + str(word_id) + "<fff>" + str(tf_idfs)
                    for (label, word_id, tf_idfs) in data_tf_idf
                ]
            )
        )


generate_vocabulary("20news-bydate/20news-test-processed.txt")
get_tf_idf("20news-bydate/20news-test-processed.txt")
