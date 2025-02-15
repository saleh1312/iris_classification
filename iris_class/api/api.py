from flask import Flask
from iris_class.agent.agent import iris_infer
from flask import request

app = Flask(__name__)


@app.route('/predict')
def predict():
    sepal_lenght = float(request.args.get('sepal_lenght'))
    sepal_width = float(request.args.get('sepal_width'))
    petal_length = float(request.args.get('petal_length'))
    petal_width = float(request.args.get('petal_width'))
    pred = iris_infer(sepal_lenght, sepal_width, petal_length, petal_width)
    return pred


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
