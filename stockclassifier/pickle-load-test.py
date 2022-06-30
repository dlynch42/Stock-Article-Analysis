import pickle
from vectorizer import vect
import numpy as np

clf = pickle.load(open('classifier.pkl', 'rb'))


label = {0: 'negative', 1: 'positive'}
example = ['Great earnings report']

X = vect.transform(example)

print('Prediction: %s\nProbability: %.2f%%' %
      (label[clf.predict(X)[0]],
       np.max(clf.predict_proba(X)) * 100))