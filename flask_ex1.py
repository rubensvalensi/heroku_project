import os

from flask import Flask, request
import pickle
import sklearn
import pandas as pd
import json

app = Flask(__name__)

model_file = 'heart_disease_model.pkl'
testset_file = 'test_set.pkl'
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
    # Heroku provides environment variable 'PORT' that should be listened on by Flask
    port = os.environ.get('PORT')

    if port:
        # 'PORT' variable exists - running on Heroku, listen on external IP and on given by Heroku port
        app.run(host='0.0.0.0', port=int(port))
    else:
        # 'PORT' variable doesn't exist, running not on Heroku, presumabely running locally, run with default
        #   values for Flask (listening only on localhost on default Flask port)
        app.run()
