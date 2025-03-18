from flask import Flask, render_template, request, jsonify
import xgboost as xgb
import pandas as pd

# Load trained model
model = xgb.XGBRegressor()
model.load_model("model.json")

# Load feature names and clean spaces
model_features = pd.read_csv("model_features.csv").values.flatten()
model_features = [feature.strip() for feature in model_features]  # Remove spaces

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get user input from form
        data = request.form.to_dict()

        # Convert numerical inputs to float
        data["Crop_Year"] = float(data["Crop_Year"])
        data["Area"] = float(data["Area"])

        # Convert categorical values (Season, Crop) into one-hot encoded format
        encoded_data = {col: 0 for col in model_features}  # Initialize all features to 0

        # Set the user-input values
        for key, value in data.items():
            column_name = key.strip()  # Remove extra spaces
            if column_name in model_features:
                encoded_data[column_name] = 1  # Set the correct feature to 1

        # Convert to DataFrame
        df = pd.DataFrame([encoded_data])

        # Make prediction
        yield_prediction = model.predict(df)[0]
        production_prediction = yield_prediction * data["Area"]

        return render_template(
            "index.html", 
            prediction_text=f"Predicted Yield: {yield_prediction:.2f}, Predicted Production: {production_prediction:.2f}"
        )

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
