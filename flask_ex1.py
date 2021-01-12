from flask import Flask, request
import pickle
import sklearn
import pandas as pd
import json

app = Flask(__name__)

model_file = r'C:\Users\RUBENS\Desktop\JUPYTER NOTEBOOK\EXERCISES\heart_disease_model.pkl'
testset_file = r'C:\Users\RUBENS\Desktop\JUPYTER NOTEBOOK\EXERCISES\test_set.pkl'
model = pickle.load(open(model_file, 'rb'))
X, y = pickle.load(open(testset_file, 'rb'))
print(model.score(X, y))


@app.route('/predict')
def predict():
    age = request.args.get('age')
    sex = request.args.get('sex')
    cp = request.args.get('cp')
    trestbps = request.args.get('trestbps')
    chol = request.args.get('chol')
    fbs = request.args.get('fbs')
    restecg = request.args.get('restecg')
    thalach = request.args.get('thalach')
    exang = request.args.get('exang')
    oldpeak = request.args.get('oldpeak')
    slope = request.args.get('slope')
    ca = request.args.get('ca')
    thal = request.args.get('thal')

    # http://127.0.0.1:5000/predict?age=34&sex=0&cp=0&trestbps=130&chol=233&fbs=0&restecg=1&thalach=187&exang=1&oldpeak=2.3&slope=2&ca=0&thal=2

    features = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
    prediction = model.predict([features])

    return str(prediction)


if __name__ == '__main__':
    app.run()