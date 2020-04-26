import pickle

import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
model = pickle.load(open('logistic.pickle', 'rb'))
tdidf = pickle.load(open('tdidf.pkl', 'rb'))
n1 = pickle.load(open('label_encoder.pickle', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    features = request.form['news_article']
    features = str(features)
    features = np.array(features).reshape(-1, 1)
    final_features = tdidf.transform(features[0])
    output = n1.inverse_transform(model.predict(final_features))[0]
    output = output.upper()

    return render_template('index.html', prediction_text='{}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)