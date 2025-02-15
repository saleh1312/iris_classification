import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle


x = pd.read_csv("data/x.csv")

y = np.load("data/y.npy")

clf = LogisticRegression(random_state=0).fit(x, y)
clf.score(x, y)

with open('models/model.pkl', 'wb') as f:
    pickle.dump(clf, f)
