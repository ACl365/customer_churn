# Project Structure for Churn Dashboard

churn_dashboard/
│
├── app.py                 # Main application file
├── layouts.py             # Page layouts for each tab
├── callbacks.py           # Interactive callbacks
├── data_processing.py     # Data preparation functions
├── visualizations.py      # Reusable plotting functions
├── assets/                # CSS, images, etc.
│   ├── custom.css         # Custom styling
│   └── logo.png           # Company logo
│
├── data/                  # Data files
│   ├── churn_predictions.csv     # Prediction results
│   ├── customer_data.csv         # Customer demographics (needed)
│   ├── customer_value.csv        # CLV or revenue data (needed)
│   └── model_metrics.json        # Model performance metrics (needed)
│
└── README.md              # Project documentation
