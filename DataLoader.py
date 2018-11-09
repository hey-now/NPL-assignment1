from porter_stemmer import PorterStemmer
import os


def load_data(stem=True):
    ps = PorterStemmer()
    positive = []
    negative = []

    directory = os.fsencode("./NEG")
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        f = open('./NEG/' + filename, 'r')
        text = f.read()
        if stem:
            text = ps.stem(text, 0, len(text) - 1)
        negative.append(text)

    directory = os.fsencode("./POS")
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        f = open('./POS/' + filename, 'r')
        text = f.read()
        if stem:
            text = ps.stem(text, 0, len(text) - 1)
        positive.append(text)

    target_pos = []
    target_neg = []
    for i in range(0, 1000):
        target_pos.append(0)
    for i in range(0, 1000):
        target_neg.append(1)

    X = positive + negative
    y = target_pos + target_neg
    return X, y
