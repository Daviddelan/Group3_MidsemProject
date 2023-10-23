from flask import Flask, request, render_template
import pickle
import numpy as np
import joblib


app = Flask(__name__)
model = pickle.load(open('/Users/daviddela/Flask/my_flask_project/model3.pkl', 'rb'))

scaler = pickle.load(open('/Users/daviddela/Flask/my_flask_project/scaler1.pkl', 'rb'))

@app.route('/', methods=['GET'])
def index():
    
    return render_template('SportsPrediction.html')

@app.route('/submit', methods = ['POST'])
def submit():
    if request.method == 'POST':
        try:
            data = request.form
            print(data)

            # Extract input values from the form data
            value_eur = float(data['value'])

            age = int(data['age'])
            release_clause_eur = float(data['releaseClause'])
            movement_reactions = int(data['movementReaction'])
            potential = int(data['potential'])

            prediction = model.predict(scaler.transform([[value_eur, age, release_clause_eur, movement_reactions, potential]]))
            print(prediction)
            print(value_eur)
            print(age)
            print(release_clause_eur)
            print(movement_reactions)
            print(potential)
            

            # Calculate prediction score and confidence interval (replace with your logic)
            confidence_interval = "84%"  # Example confidence interval

            return render_template('prediction.html', prediction=prediction, confidence_interval=confidence_interval)
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            return render_template('prediction.html', error=error_message)

if __name__ == '__main__':
    app.run(debug=True)

