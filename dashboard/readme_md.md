# Customer Churn Analytics Dashboard

A comprehensive dashboard for visualizing and analyzing customer churn predictions to drive retention strategies.

## Project Overview

This dashboard provides an end-to-end solution for analyzing the outputs of a machine learning model that predicts customer churn. It is designed to help stakeholders understand churn risk patterns, identify high-risk customer segments, and develop targeted intervention strategies.

## Key Features

- **Executive Summary**: High-level overview of churn metrics and KPIs
- **Customer Segmentation**: Interactive analysis of churn risk across different customer segments
- **Churn Drivers**: Visualization of feature importance and key factors influencing churn
- **Intervention Simulator**: Interactive tool to estimate ROI of different retention strategies
- **Model Performance**: Technical evaluation of the predictive model's performance metrics

## Technical Implementation

The dashboard is built using:

- **Dash**: For the interactive web application framework
- **Plotly**: For data visualization
- **Pandas**: For data manipulation and analysis
- **Bootstrap Components**: For responsive UI elements

## Project Structure

```
churn_dashboard/
│
├── app.py                 # Main application file
├── layouts.py             # Page layouts for each tab
├── callbacks.py           # Interactive callbacks
├── data_processing.py     # Data preparation functions
├── visualizations.py      # Reusable plotting functions
├── assets/                # CSS, images, etc.
│   └── custom.css         # Custom styling
│
├── data/                  # Data files
│   ├── churn_predictions.csv     # Prediction results
│   ├── customer_data.csv         # Customer demographics (needed)
│   ├── customer_value.csv        # CLV or revenue data (needed)
│   └── model_metrics.json        # Model performance metrics (needed)
```

## Installation and Setup

1. Clone the repository
2. Install dependencies:
   ```
   pip install dash dash-bootstrap-components plotly pandas flask-caching
   ```
3. Add your data files to the `/data` directory:
   - `churn_predictions.csv` (provided)
   - `customer_data.csv` (required, or will use synthetic data)
   - `customer_value.csv` (required, or will use synthetic data)
   - `model_metrics.json` (required, or will use synthetic data)

4. Run the dashboard:
   ```
   python app.py
   ```

## Data Requirements

### churn_predictions.csv
- **CustomerID**: Unique identifier for each customer
- **predicted_churn**: Binary indicator (1=churn, 0=no churn)
- **churn_probability**: Probability score between 0 and 1

### customer_data.csv (required, or will use synthetic data)
Customer demographic and behavioral data including:
- **CustomerID**: Unique identifier matching churn_predictions.csv
- **tenure**: Number of months as a customer
- **contract_type**: Type of contract (Month-to-Month, One Year, Two Year)
- Additional behavioral and demographic features

### customer_value.csv (required, or will use synthetic data)
Customer revenue and value metrics including:
- **CustomerID**: Unique identifier matching churn_predictions.csv
- **annual_revenue**: Annual revenue from the customer
- **customer_lifetime_value**: Calculated customer lifetime value

### model_metrics.json (required, or will use synthetic data)
Model performance metrics and metadata including:
- Model name and training date
- Validation metrics (accuracy, precision, recall, F1, AUC)
- Feature importance values
- Model comparison data

## Dashboard Customization

The dashboard is designed to be easily customizable:

- Color scheme: Edit the `COLORS` dictionary in `visualizations.py`
- Layout: Modify the component structure in `layouts.py`
- Metrics: Adjust calculations in `data_processing.py`

## Notes for Senior Data Professionals

This dashboard showcases several advanced data science and visualization techniques:

1. **Ensemble Model Integration**: The dashboard is designed to work with stacked ensemble models, showing how different models contribute to the final prediction.

2. **Financial Impact Analysis**: Revenue at risk calculations and ROI projections demonstrate business value beyond just technical metrics.

3. **Interactive Simulation**: The intervention simulator allows stakeholders to evaluate different strategies, demonstrating the practical application of predictive models.

4. **Explainable AI Elements**: Feature importance and partial dependence plots provide insights into the "black box" of machine learning models.

5. **Data-Driven UX**: The dashboard design emphasizes clear data presentation with targeted insights rather than unnecessary visual elements.

## Future Enhancements

Potential areas for expansion:

- Integration with live data sources
- A/B testing framework for intervention strategies
- Automated model retraining pipeline
- Survival analysis for time-to-churn predictions
- Customer journey mapping visualization

## Author

[Your Name] - Senior Data Scientist

## License

This project is licensed under the MIT License
