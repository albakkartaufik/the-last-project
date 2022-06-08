from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import joblib

app = Flask(__name__)


@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("jenisKelamin.html")

@app.route("/generic.html", methods=['GET', 'POST'])
def generic():
    if request.method == 'GET':
        return render_template("generic.html")
    elif request.method == 'POST':
        jenis_kelamin = request.form['jeniskelamin']
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
        
        sample_data = [ jenis_kelamin,usia,demam,batuk,pilek,nyeri,pneumonia,diare,infeksiparu,isolasi]
        clean_data = [float(i) for i in sample_data]
        ex1 = np.array(clean_data).reshape(1,-1)
        
        if model == 'Logistic':
            logit_model = joblib.load('model-development/covid_predictor.pkl')
            result_prediction = logit_model.predict(ex1)
        elif model == 'Dtree':
            dtree_model = joblib.load('model-development/covid_predictor2.pkl')
            result_prediction = dtree_model.predict(ex1)
        return render_template('generic.html', result=result_prediction,model_selected=model)
    else:
        return "Unsupported Request Method"


if __name__ == '__main__':
    app.run(port=5000, debug=True)