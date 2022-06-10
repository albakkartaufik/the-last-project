from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import joblib

app = Flask(__name__)


@app.route("/index", methods=['GET', 'POST'])
def index():
    return render_template("index.html")

@app.route("/", methods=['POST', 'GET'])
def prediction():
    if request.method == 'GET':
        return render_template("prediction.html")
    elif request.method == 'POST':
        jenis_kelamin = request.form['Jenis Kelamin']
        usia = request.form['usia']
        demam = request.form['demam']
        batuk = request.form['batuk']
        pilek = request.form['pilek']
        nyeri = request.form['nyeri']
        pneumonia =  request.form['pneumonia']
        diare = request.form['diare']
        infeksiparu = request.form['infeksiparu']
        isolasi = request.form['isolasi']
        model = request.form['model']
        
        sample_data = [jenis_kelamin,usia,demam,batuk,pilek,nyeri,pneumonia,diare,infeksiparu,isolasi]
        clean_data = [float(i) for i in sample_data]
        ex1 = np.array(clean_data).reshape(1,-1)
        
        if model == 'Logistic':
            logit_model = joblib.load('model-development/covid_predictor.pkl')
            result_prediction = logit_model.predict(ex1)
            probabilities = logit_model.predict_proba(ex1)
            if probabilities[0][1]> probabilities[0][0]:
                true_probabilities = probabilities[0][1] * 100
            else:
                true_probabilities = probabilities[0][0] * 100
        elif model == 'Dtree':
            dtree_model = joblib.load('model-development/covid_predictor2.pkl')
            result_prediction = dtree_model.predict(ex1)
            probabilities = dtree_model.predict_proba(ex1)
            if probabilities[0][1]> probabilities[0][0]:
                true_probabilities = probabilities[0][1] * 100
            else:
                true_probabilities = probabilities[0][0] * 100
        elif model == 'SVM':
            SVM_model = joblib.load('model-development/covid_predictor3.pkl')
            result_prediction = SVM_model.predict(ex1)
            probabilities = SVM_model.predict_proba(ex1)
            if probabilities[0][1]> probabilities[0][0]:
                true_probabilities = probabilities[0][1] * 100
            else:
                true_probabilities = probabilities[0][0] * 100    
        return render_template('hasil.html', result=result_prediction,model_selected=model,probabilities=true_probabilities)

if __name__ == '__main__':
    app.run(port=5000, debug=True)