# NLP assignment 1

SVM based sentiment classification

### Prerequisites

Place the datasets in the folders "POS" and "NEG" in the main directory (they are referenced using a relative "./POS" path from the .py files)

Install following libraries:
```
pip install sklearn
pip install scipy
pip install numpy
```

### Running

Run CrossValidation.py file with python 3.6 in order to perform the main experiment comparing Naive Bayes and SVM performance. It will also output the sing test results. 
One can also change ngram_range and binary parameters in vectorizer in order to test on different features. 

```
python ./CrossValidation.py
```
