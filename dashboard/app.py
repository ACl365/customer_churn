import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from flask_caching import Cache

from layouts import (
    create_executive_summary,
    create_customer_segmentation,
    create_churn_drivers,
    create_intervention_simulator,
    create_model_performance
)
from callbacks import register_callbacks
from data_processing import load_data

# Initialize the Dash app with a clean, professional theme
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    title="Customer Churn Analytics Dashboard"
)

# Configure caching for performance optimization
cache = Cache(app.server, config={
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR': 'cache-directory',
    'CACHE_DEFAULT_TIMEOUT': 300
})

# Load data
data = load_data()

# Define the app layout with tabs
app.layout = html.Div([
    # Header
    html.Div([
        html.Img(src="/assets/logo.png", className="logo"),
        html.H1("Customer Churn Analytics Dashboard", className="header-title"),
        html.P("Insights and Recommendations for Reducing Customer Churn", className="header-description")
    ], className="header"),
    
    # Main Content
    html.Div([
        # Tabs
        dcc.Tabs(id="tabs", value="tab-exec-summary", className="custom-tabs", children=[
            dcc.Tab(
                label="Executive Summary",
                value="tab-exec-summary",
                className="custom-tab",
                selected_className="custom-tab--selected"
            ),
            dcc.Tab(
                label="Customer Segmentation",
                value="tab-segmentation",
                className="custom-tab",
                selected_className="custom-tab--selected"
            ),
            dcc.Tab(
                label="Churn Drivers",
                value="tab-drivers",
                className="custom-tab",
                selected_className="custom-tab--selected"
            ),
            dcc.Tab(
                label="Intervention Simulator",
                value="tab-simulator",
                className="custom-tab",
                selected_className="custom-tab--selected"
            ),
            dcc.Tab(
                label="Model Performance",
                value="tab-model",
                className="custom-tab",
                selected_className="custom-tab--selected"
            )
        ]),
        
        # Tab content
        html.Div(id="tab-content", className="tab-content")
        
    ], className="main-content"),
    
    # Footer
    html.Div([
        html.P(f"Data Last Updated: January 15, 2025 | Model Version: Ensemble v2.1", className="footer-text"),
        html.P("Created by: [Your Name] | Senior Data Scientist", className="footer-text")
    ], className="footer")
], className="app-container")

# Register all callbacks
register_callbacks(app, data, cache)

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)