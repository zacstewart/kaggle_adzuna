import logging
import gc
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import ShuffleSplit
from datasets import *

logging.basicConfig(level=logging.INFO)

logging.info('Loading train file...')
h, rows = loadFile('data/Train_rev1.csv')

logging.info('Converting to numPy array...')
rows = np.array(rows[0:1000])
n            = rows.shape[0]
descriptions = rows[:, h['FullDescription']]
salaries     = rows[:, h['SalaryNormalized']].astype(int)

logging.info('Freeing test set memory...')
del rows
gc.collect()

logging.info('Loading test file...')
h, test = loadFile('data/Valid_rev1.csv')
test = np.array(test)
n = test.shape[0]

logging.info('Saving submission...')
submission = np.empty([n, 2])
submission[:, 0] = test[:, h['Id']]
submission[:, 1] = np.median(salaries)
np.savetxt('submission.csv', submission, delimiter=',', fmt='%i')
