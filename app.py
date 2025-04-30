

from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__,template_folder='templates')

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():

    return render_template('\\home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Extract values from form
            features = [
                int(request.form['GENDER']),
                int(request.form['AGE']),
                int(request.form['SMOKING']),
                int(request.form['YELLOW_FINGERS']),
                int(request.form['ANXIETY']),
                int(request.form['PEER_PRESSURE']),
                int(request.form['CHRONIC_DISEASE']),
                int(request.form['FATIGUE']),
                int(request.form['ALLERGY']),
                int(request.form['WHEEZING']),
                int(request.form['ALCOHOL_CONSUMING']),
                int(request.form['COUGHING']),
                int(request.form['SHORTNESS_OF_BREATH']),
                int(request.form['SWALLOWING_DIFFICULTY']),
                int(request.form['CHEST_PAIN'])
            ]
            prediction = model.predict([np.array(features)])[0]
            result = 'High Risk of Lung Cancer' if prediction == 1 else 'Low Risk of Lung Cancer'
            return render_template('predict.html', prediction=result)
        except:
            return render_template('predict.html', prediction="Error in input. Please check again.")
    return render_template('predict.html')

@app.route('/symptoms')
def symptoms():
    return render_template('symptoms.html')

@app.route('/faq')
def faq():
    return render_template('faq.html')

@app.route('/riskfactors')
def riskfactors():
    return render_template('riskfactors.html')

@app.route('/diagnosis')
def diagnosis():
    return render_template('diagnosis.html')

@app.route('/treatment')
def treatment():
    return render_template('treatment.html')

if __name__ == '__main__':
    app.run(debug=True)