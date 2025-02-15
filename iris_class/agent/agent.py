import pickle
import numpy as np

with open('models/model.pkl', 'rb') as f:
    clf = pickle.load(f)

with open('assets/lblencode.pkl', 'rb') as f:
    le = pickle.load(f)


def iris_infer(sepal_lenght: float,
               sepal_width: float,
               petal_length: float,
               petal_width: float):

    input_user = np.array([sepal_lenght,
                           sepal_width,
                           petal_length,
                           petal_width])[None, :]

    pred_enc = clf.predict(input_user)

    prediction = le.inverse_transform(pred_enc).tolist()[0]

    return prediction


if __name__ == "__main__":
    print(iris_infer(11, 5, 7, 0.5))
