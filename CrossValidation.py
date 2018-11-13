from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
import numpy as np
from DataLoader import *
from SignTest import sign_test

folds_num = 10


def get_folds(X,Y):
    folds = []
    for i in range(folds_num):
        folds.append([[], []])
    for i in range(0, len(X)):
        folds[i % folds_num][0].append(X[i])
        folds[i % folds_num][1].append(Y[i])
    return folds


def cross_validate(X, Y, vectorizer):
    folds = get_folds(X, Y)
    nb_accuracies = []
    svm_accuracies = []
    predictionsA = []
    predictionsB = []

    for i in range(0, folds_num):
        X_train = []
        Y_train = []
        X_test = []
        Y_test = []

        for j in range(0, folds_num):
            if i == j:
                X_test = folds[j][0]
                Y_test = folds[j][1]
            else:
                X_train += folds[j][0]
                Y_train += folds[j][1]

        print("Fold ", i+1)

        X_train_counts = vectorizer.fit_transform(X_train)
        print("Feature number:", X_train_counts.shape[1])
        X_test_counts = vectorizer.transform(X_test)

        predictA = MultinomialNB().fit(X_train_counts, Y_train).predict(X_test_counts)
        nb_accuracy = np.mean(predictA == Y_test)
        predictionsA += predictA.tolist()
        print('NB accuracy:', nb_accuracy)
        nb_accuracies.append(nb_accuracy)

        predictB = svm.SVC(kernel='linear').fit(X_train_counts, Y_train).predict(X_test_counts)
        svm_accuracy = np.mean(predictB == Y_test)
        predictionsB += predictB.tolist()
        print('SVM accuracy:', svm_accuracy)
        svm_accuracies.append(svm_accuracy)

    print('\nNB mean accuracy:', np.mean(nb_accuracies))
    print('SVM mean accuracy:', np.mean(svm_accuracies))

    sign_test_result = sign_test(Y, predictionsA, predictionsB)
    print("Sign test p=", sign_test_result)


def classifier(X_train, X_test, Y_train, model, vectorizer):
    X_train_counts = vectorizer.fit_transform(X_train)
    X_test_counts = vectorizer.transform(X_test)

    clf = model.fit(X_train_counts, Y_train)
    predicted = clf.predict(X_test_counts)
    return predicted


X, Y = load_data()
vectorizer = CountVectorizer(ngram_range=(1, 1), binary=False)
cross_validate(X, Y, vectorizer)

"""
vectorizer = CountVectorizer(ngram_range=(2, 2), binary=True)
cross_validate(X, Y, vectorizer)

vectorizer = CountVectorizer(ngram_range=(1, 2), binary=True)
cross_validate(X, Y, vectorizer)

vectorizer = CountVectorizer(ngram_range=(1, 1), min_df=5, binary=True)
cross_validate(X, Y, vectorizer)"""