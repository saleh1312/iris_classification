import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

df = pd.read_csv("data/iris.csv")


x = df.iloc[:, 1:-1]


y = df["Species"]


le = LabelEncoder()
y_enc = le.fit_transform(y)


x.to_csv("data/x.csv", index=False)

np.save("data/y.npy", y_enc)
