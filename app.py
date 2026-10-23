from flask import Flask, render_template, request
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, r2_score

app = Flask(__name__)
DATA_PATH = "parking_data.csv"

CATEGORICAL = ["area", "parking_type", "security", "covered"]
NUMERIC = ["distance_to_centre_km", "demand_level", "hour_of_day", "duration_hours"]
FEATURES = CATEGORICAL + NUMERIC

def train_model():
    data = pd.read_csv(DATA_PATH)
    X = data[FEATURES]
    y = data["hourly_price_inr"]
    preprocessor = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)
    ], remainder="passthrough")
    model = Pipeline([
        ("preprocess", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=180, max_depth=12, random_state=42
        ))
    ])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics = {
        "mae": round(mean_absolute_error(y_test, preds), 2),
        "r2": round(r2_score(y_test, preds), 3)
    }
    return model, metrics, data

model, metrics, parking_data = train_model()

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    estimated_total = None
    form = {
        "area": "Central",
        "parking_type": "Apartment",
        "security": "Yes",
        "covered": "Yes",
        "distance_to_centre_km": "3",
        "demand_level": "3",
        "hour_of_day": "18",
        "duration_hours": "2"
    }
    if request.method == "POST":
        form.update({k: request.form.get(k, form[k]) for k in form})
        row = pd.DataFrame([{
            "area": form["area"],
            "parking_type": form["parking_type"],
            "security": form["security"],
            "covered": form["covered"],
            "distance_to_centre_km": float(form["distance_to_centre_km"]),
            "demand_level": int(form["demand_level"]),
            "hour_of_day": int(form["hour_of_day"]),
            "duration_hours": float(form["duration_hours"])
        }], columns=FEATURES)
        prediction = max(10, round(float(model.predict(row)[0]), 2))
        estimated_total = round(prediction * float(form["duration_hours"]), 2)
    spaces = parking_data.sample(min(6, len(parking_data)), random_state=7).to_dict("records")
    return render_template("index.html", prediction=prediction,
                           estimated_total=estimated_total, form=form,
                           spaces=spaces, metrics=metrics)

if __name__ == "__main__":
    app.run(debug=True)
