import os
import sys
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# Add root directory to path so we can import predict.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from predict import predict_diet

app = Flask(__name__, template_folder="../frontend/templates", static_folder="../frontend/static")
# Enable CORS so the separate frontend can communicate with the backend
CORS(app)

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        # Convert numeric fields from strings to floats
        numeric_fields = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
        for field in numeric_fields:
            if field in data and str(data[field]).strip() != "":
                try:
                    data[field] = float(data[field])
                except ValueError:
                    pass # Handled by the model gracefully

        # Extract BMI fields sent by the frontend (not used by ML model)
        bmi = float(data.pop('bmi', 0))
        bmi_category = data.pop('bmi_category', 'Unknown')

        # Call the machine learning pipeline, forwarding BMI context
        result = predict_diet(data, bmi=bmi, bmi_category=bmi_category)
        
        # Check if the predict_diet function threw a handled error
        if "error" in result:
            return jsonify(result), 500
            
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e), "status": "failed"}), 500

if __name__ == '__main__':
    print("========================================")
    print("Starting AI Diet Recommendation API...")
    print("Running on http://127.0.0.1:5000")
    print("========================================")
    app.run(debug=True, port=5000)
