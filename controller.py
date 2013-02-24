import logging
import gc
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cross_validation import ShuffleSplit
from sklearn.linear_model import *
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from datasets import *
from utilities import *

logging.basicConfig(level=logging.INFO)

logging.info('Loading train file...')
h, rows = loadFile('data/Train_rev1.csv')

logging.info('Converting to numPy array...')
rows = np.array(rows)
n            = rows.shape[0]
descriptions = rows[:, h['FullDescription']]
salaries     = rows[:, h['SalaryNormalized']].astype(int)

logging.info('Freeing test set memory...')
del rows
gc.collect()

rs = ShuffleSplit(n, test_size=0.2, random_state=0)
count_vectorizer = CountVectorizer()
tfidf_transformer = TfidfTransformer()

models = [
    MultinomialNB(),
    KNeighborsRegressor(),
    SVR(),
    LinearRegression(),
    Ridge(),
    Lasso(),
    ElasticNet(),
    SGDRegressor(),
  ]

model_scores = {}
for model in models:
  scores = []
  logging.info("Evaluating %(model)s" % {'model': model.__class__.__name__})
  for train_index, cv_index in rs:
    X_train_counts = count_vectorizer.fit_transform(descriptions[train_index])
    X_train = tfidf_transformer.fit_transform(X_train_counts)
    y_train = salaries[train_index]

    X_cv_counts = count_vectorizer.transform(descriptions[cv_index])
    X_cv = tfidf_transformer.transform(X_cv_counts)
    y_cv = salaries[cv_index]

    clf = model.fit(X_train, y_train)
    scores.append(clf.score(X_cv, y_cv))
  model_scores[model.__class__.__name__] = mean(scores)

for (model, score) in model_scores.iteritems():
  print "%(model)s: %(score)f" % \
      {'model': model, 'score': score}

logging.info('Loading test file...')
h, test = loadFile('data/Valid_rev1.csv')
test = np.array(test)
n = test.shape[0]

logging.info('Saving submission...')
submission = np.empty([n, 2])
submission[:, 0] = test[:, h['Id']]
submission[:, 1] = np.median(salaries)
np.savetxt('submission.csv', submission, delimiter=',', fmt='%i')
