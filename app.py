
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('linear_reg_model.sav', 'rb'))

@app.route('/')
def home():
    return render_template('index.html',**locals())

@app.route('/predict',methods=['POST'])
def predict():
    
    if request.method == 'POST':
        sqft = int(request.form['sqft'])
        bath = int(request.form['bathrooms'])
        bhk = int(request.form['bedrooms'])
        location = request.form['location']

        model_list = [sqft, bath, bhk, location]
        x_col = model.feature_names_in_
        model_input = []

        model_input.append(model_list[0])
        model_input.append(model_list[1])
        model_input.append(model_list[2])
        for col in x_col[3:]:
          if col == location:
            model_input.append(1)
          else:
            model_input.append(0)
        data = np.array([model_input])
        my_prediction = model.predict(data)
        output = round(my_prediction[0], 3)
        

    return render_template('index.html',prediction_text='Price of Property: Rs.{} Lakhs'.format(output))

if __name__ == "__main__":
    app.run(debug=True)