from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)

# Load model once when the app starts
model = joblib.load("emma_salary_predict_model.ml")


@app.route("/")
def home():
    """Landing page for the Salary Prediction API"""
    return (
        "<h1>Salary Prediction API</h1>"
        "<p>BAIS:3300 - Digital Product Development</p>"
        "<p>Emma Chase</p>"
    )


@app.route("/predict", methods=["GET"])
def predict():
    return "<h1> Prediction route is working...</h1>"
    """
    Predict salary based on input JSON payload
    Expected keys: age, gender, country, highest_deg, coding_exp, title, company_size
    """
    print("inside predict")
    try:
        data = request.get_json()

        print(f"data from the user: {data}")

        required_fields = [
            "age",
            "gender",
            "country",
            "highest_deg",
            "coding_exp",
            "title",
            "company_size",
        ]
        if not all(field in data for field in required_fields):
            return jsonify({"error": "Missing one or more required fields"}), 400

        # Ensure correct order and type
        features = [
            int(data["age"]),
            int(data["gender"]),
            int(data["country"]),
            int(data["highest_deg"]),
            int(data["coding_exp"]),
            int(data["title"]),
            int(data["company_size"]),
        ]

        print(f"features before using the model: {data}")

        prediction = model.predict([features])[0]

        print(f"prediction: {prediction}")

        return jsonify({"predicted_salary": prediction})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=True)
