# ParkSmart AI — Smart Parking Rental System

A student AI/ML mini-project that demonstrates how temporarily unused apartment or private parking spaces could be listed for hourly rental. A Random Forest regression model estimates an indicative hourly price from area, parking type, security, covered status, distance to the city centre, demand, start hour and duration.

> **Important:** This is an academic prototype. The included dataset is synthetic. Listings are illustrative rather than live availability. The app does not implement authentication, booking, payments, owner permission checks or real-time calendars.

## Features
- Flask web dashboard with responsive UI
- 800-row synthetic demonstration dataset
- Random Forest price-estimation model
- Estimated hourly price and total rental cost
- Test-set MAE and R² displayed in the dashboard
- Example listing cards

## Requirements
- Python 3.10 or newer
- pip
- A modern web browser

## Run locally
1. Download or clone this repository.
2. Open a terminal in the project folder.
3. Create and activate a virtual environment (recommended).
   - Windows: `py -m venv .venv` then `.venv\Scripts\activate`
   - macOS/Linux: `python3 -m venv .venv` then `source .venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Start the app: `python app.py`
6. Open `http://127.0.0.1:5000` in your browser.

The model is trained automatically from `parking_data.csv` whenever the app starts.

## Project structure
```text
ParkSmart_AI_Project/
├── app.py
├── parking_data.csv
├── requirements.txt
├── README.md
├── .gitignore
├── templates/
│   └── index.html
└── static/
    └── style.css
```

## Model
- Algorithm: `RandomForestRegressor`
- Preprocessing: one-hot encoding for categorical features
- Train/test split: 80:20
- Metrics: Mean Absolute Error (MAE) and R²
- Random state: 42 for reproducibility

## Dataset fields
`space_id`, `area`, `parking_type`, `security`, `covered`, `distance_to_centre_km`, `demand_level`, `hour_of_day`, `duration_hours`, `hourly_price_inr`.

The values were generated for demonstration and are not actual market rates.

## Future improvements
- User accounts and owner verification
- Real availability calendar and conflict prevention
- Apartment/owner permission workflow
- Booking confirmation and payment gateway
- Map integration and geolocation
- Reviews, cancellation policy and notifications
- Privacy, security and deployment hardening
