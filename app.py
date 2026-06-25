from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__, template_folder='../templates')

model = joblib.load('models/best_crop_model.pkl')
preprocessor = joblib.load('models/preprocessor.pkl')
metrics = joblib.load('models/evaluation_results.pkl')

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        input_data = pd.DataFrame([{
            'rainfall': float(request.form['rainfall']),
            'temperature': float(request.form['temperature']),
            'soil_type': request.form['soil_type'],
            'fertilizer_use': request.form['fertilizer_use'],
            'irrigation': int(request.form['irrigation']),
            'season': request.form['season']
        }])
        transformed_input = preprocessor.transform(input_data)
        pred_val = model.predict(transformed_input)[0]
        prediction = f"{round(pred_val, 2)} tons per hectare"
    return render_template('index.html', metrics=metrics, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
