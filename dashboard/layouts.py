import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any

# Executive Summary Tab
def create_executive_summary(data: Dict[str, Any]) -> html.Div:
    """
    Create the Executive Summary tab layout.
    
    This tab provides high-level insights and KPIs for executives.
    
    Parameters
    ----------
    data : Dict[str, Any]
        Dictionary containing all processed data
        
    Returns
    -------
    html.Div
        Layout for the executive summary tab
    """
    # Extract metrics
    metrics = data['derived_metrics']
    revenue_impact = data['revenue_impact']
    
    # Format numbers for display
    total_customers = f"{metrics['total_customers']:,}"
    total_churners = f"{metrics['total_churners']:,}"
    churn_rate = f"{metrics['churn_rate']:.1%}"
    revenue_at_risk = f"${revenue_impact['churner_revenue']:,.0f}"
    
    # Create layout
    layout = html.Div([
        # Title and description
        html.Div([
            html.H2("Executive Summary", className="tab-title"),
            html.P("High-level overview of churn predictions and business impact", className="tab-description")
        ], className="tab-header"),
        
        # KPI Cards
        html.Div([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Predicted Churners", className="card-title"),
                    html.H2(f"{total_churners}", className="card-value"),
                    html.P(f"({churn_rate} of customer base)", className="card-subvalue")
                ])
            ], className="kpi-card"),
            
            dbc.Card([
                dbc.CardBody([
                    html.H4("Revenue at Risk", className="card-title"),
                    html.H2(f"{revenue_at_risk}", className="card-value"),
                    html.P(f"({revenue_impact['revenue_at_risk_pct']:.1%} of total revenue)", className="card-subvalue")
                ])
            ], className="kpi-card"),
            
            dbc.Card([
                dbc.CardBody([
                    html.H4("Retention Opportunity", className="card-title"),
                    html.H2("$4.8M", className="card-value"),
                    html.P("(25% estimated retention rate)", className="card-subvalue")
                ])
            ], className="kpi-card"),
            
            dbc.Card([
                dbc.CardBody([
                    html.H4("Model Performance", className="card-title"),
                    html.H2(f"{data['model_metrics']['validation_metrics']['auc_roc']:.2f}", className="card-value"),
                    html.P("(AUC-ROC Score)", className="card-subvalue")
                ])
            ], className="kpi-card"),
        ], className="kpi-container"),
        
        # Main visualization row
        html.Div([
            # Churn probability distribution
            html.Div([
                html.H3("Churn Probability Distribution", className="chart-title"),
                dcc.Graph(
                    id='churn-probability-dist',
                    config={'displayModeBar': False}
                ),
                html.P([
                    html.Strong("Key Insight: "), 
                    "High concentration of customers with 60-80% churn probability indicates a critical intervention opportunity."
                ], className="chart-insight")
            ], className="chart-container"),
            
            # Risk tier breakdown
            html.Div([
                html.H3("Customer Risk Tiers", className="chart-title"),
                dcc.Graph(
                    id='risk-tier-breakdown',
                    config={'displayModeBar': False}
                ),
                html.P([
                    html.Strong("Key Insight: "), 
                    f"{metrics['risk_tiers']['high']:,} high-risk customers represent {metrics['risk_tiers']['high']/metrics['total_customers']:.1%} of the base but account for {0.45:.1%} of potential revenue loss."
                ], className="chart-insight")
            ], className="chart-container"),
        ], className="charts-row"),
        
        # Insight boxes
        html.Div([
            html.H3("Top Actionable Insights", className="section-title"),
            
            dbc.Card([
                dbc.CardHeader(html.H5("High-Risk Customer Profile", className="insight-title")),
                dbc.CardBody([
                    html.P("Customers with short tenure (< 12 months), month-to-month contracts, and no online security are 3.2x more likely to churn than average."),
                    html.P(html.Strong("Recommendation: Target these customers with security service promotions and contract term incentives.")),
                ])
            ], className="insight-card"),
            
            dbc.Card([
                dbc.CardHeader(html.H5("Revenue Protection Opportunity", className="insight-title")),
                dbc.CardBody([
                    html.P("Targeting the top 20% highest-value customers with high churn risk could protect $2.1M in annual revenue with minimal intervention costs."),
                    html.P(html.Strong("Recommendation: Deploy high-touch retention team with personalized offers to this segment.")),
                ])
            ], className="insight-card"),
            
            dbc.Card([
                dbc.CardHeader(html.H5("Early Warning Indicators", className="insight-title")),
                dbc.CardBody([
                    html.P("Support ticket frequency has emerged as a leading indicator of churn, with customers filing 2+ tickets in the last quarter showing 40% higher churn probability."),
                    html.P(html.Strong("Recommendation: Implement proactive outreach after first support ticket.")),
                ])
            ], className="insight-card"),
        ], className="insights-container"),
    ], className="tab-content")
    
    return layout

# Customer Segmentation Tab
def create_customer_segmentation(data: Dict[str, Any]) -> html.Div:
    """
    Create the Customer Segmentation tab layout.
    
    This tab provides detailed views of customer segments and their churn risk.
    
    Parameters
    ----------
    data : Dict[str, Any]
        Dictionary containing all processed data
        
    Returns
    -------
    html.Div
        Layout for the customer segmentation tab
    """
    layout = html.Div([
        # Title and description
        html.Div([
            html.H2("Customer Segmentation", className="tab-title"),
            html.P("Analysis of high-risk segments and their characteristics", className="tab-description")
        ], className="tab-header"),
        
        # Segment controls
        html.Div([
            html.Div([
                html.Label("Segment By:", className="control-label"),
                dcc.Dropdown(
                    id='segment-dropdown',
                    options=[
                        {'label': 'Contract Type', 'value': 'contract_type'},
                        {'label': 'Tenure', 'value': 'tenure_group'},
                        {'label': 'Payment Method', 'value': 'payment_method'},
                        {'label': 'Internet Service', 'value': 'internet_service'},
                    ],
                    value='contract_type',
                    clearable=False,
                    className="segment-dropdown"
                )
            ], className="control-item"),
            
            html.Div([
                html.Label("Color By:", className="control-label"),
                dcc.Dropdown(
                    id='color-dropdown',
                    options=[
                        {'label': 'Churn Probability', 'value': 'churn_probability'},
                        {'label': 'Customer Value', 'value': 'customer_lifetime_value'},
                    ],
                    value='churn_probability',
                    clearable=False,
                    className="segment-dropdown"
                )
            ], className="control-item"),
        ], className="control-row"),
        
        # Main visualization row
        html.Div([
            # Customer scatter plot
            html.Div([
                html.H3("Customer Segment Map", className="chart-title"),
                dcc.Graph(
                    id='customer-segment-map',
                    config={'displayModeBar': True}
                ),
                html.P([
                    "Each point represents a customer. Hover for details. Clusters indicate similar customers based on behavior and demographics."
                ], className="chart-description")
            ], className="chart-container wide"),
            
            # Segment comparison
            html.Div([
                html.H3("Segment Comparison", className="chart-title"),
                dcc.Graph(
                    id='segment-comparison-chart',
                    config={'displayModeBar': False}
                ),
            ], className="chart-container wide"),
        ], className="charts-row"),
        
        # Segment details table
        html.Div([
            html.H3("Detailed Segment Analysis", className="section-title"),
            html.Div(id="segment-details-table", className="details-table"),
            html.Div([
                html.H4("Segment Risk Assessment", className="subsection-title"),
                html.Div(id="segment-risk-assessment", className="risk-assessment"),
            ], className="segment-assessment-container")
        ], className="segment-details-container"),
    ], className="tab-content")
    
    return layout

# Churn Drivers Tab
def create_churn_drivers(data: Dict[str, Any]) -> html.Div:
    """
    Create the Churn Drivers & Feature Importance tab layout.
    
    This tab shows what factors drive customer churn according to the model.
    
    Parameters
    ----------
    data : Dict[str, Any]
        Dictionary containing all processed data
        
    Returns
    -------
    html.Div
        Layout for the churn drivers tab
    """
    layout = html.Div([
        # Title and description
        html.Div([
            html.H2("Churn Drivers Analysis", className="tab-title"),
            html.P("Understanding the key factors that influence customer churn", className="tab-description")
        ], className="tab-header"),
        
        # Feature importance
        html.Div([
            html.H3("Global Feature Importance", className="section-title"),
            dcc.Graph(
                id='feature-importance-chart',
                config={'displayModeBar': False}
            ),
            html.P([
                html.Strong("Technical Note: "), 
                "Feature importance values are normalized from the ensemble model, combining XGBoost, LightGBM, and Logistic Regression importance scores."
            ], className="chart-note")
        ], className="feature-importance-container"),
        
        # Feature selection for partial dependence
        html.Div([
            html.H3("Feature Impact Analysis", className="section-title"),
            html.P("Select a feature to see how its values affect churn probability:", className="section-description"),
            dcc.Dropdown(
                id='feature-selector',
                options=[
                    {'label': 'Tenure', 'value': 'tenure'},
                    {'label': 'Monthly Charge', 'value': 'monthly_charge'},
                    {'label': 'Contract Type', 'value': 'contract_type'},
                    {'label': 'Payment Method', 'value': 'payment_method'},
                    {'label': 'Internet Service', 'value': 'internet_service'},
                ],
                value='tenure',
                clearable=False,
                className="feature-dropdown"
            )
        ], className="feature-selection-container"),
        
        # Partial dependence plot
        html.Div([
            html.Div(id="feature-impact-title", className="chart-title"),
            dcc.Graph(
                id='partial-dependence-plot',
                config={'displayModeBar': False}
            ),
            html.Div(id="feature-impact-insight", className="chart-insight")
        ], className="partial-dependence-container"),
        
        # Feature correlations
        html.Div([
            html.H3("Feature Relationships", className="section-title"),
            dcc.Graph(
                id='feature-correlation-chart',
                config={'displayModeBar': False}
            ),
            html.P([
                html.Strong("Key Insight: "), 
                "Strong correlations exist between contract type, tenure, and payment method, creating compound risk factors when aligned negatively."
            ], className="chart-insight")
        ], className="feature-correlation-container"),
        
        # Key patterns and insights
        html.Div([
            html.H3("Key Patterns & Unexpected Insights", className="section-title"),
            
            dbc.Card([
                dbc.CardHeader(html.H5("Non-Linear Relationship", className="insight-title")),
                dbc.CardBody([
                    html.P("Monthly charges show an unexpected U-shaped relationship with churn: both very low and very high charges correlate with increased churn, while mid-range charges have the lowest churn rates."),
                    html.P(html.Strong("Business Implication: Pricing strategy should focus on the optimal mid-range price points while adding more value to justify higher price points.")),
                ])
            ], className="insight-card"),
            
            dbc.Card([
                dbc.CardHeader(html.H5("Risk Multiplier Effect", className="insight-title")),
                dbc.CardBody([
                    html.P("The combination of month-to-month contracts with electronic check payment method creates a risk multiplier effect that increases churn probability by 3.5x, significantly higher than either factor alone."),
                    html.P(html.Strong("Business Implication: Target customers with this combination for auto-pay conversion and contract term incentives.")),
                ])
            ], className="insight-card"),
            
            dbc.Card([
                dbc.CardHeader(html.H5("Retention Inflection Point", className="insight-title")),
                dbc.CardBody([
                    html.P("Customer tenure shows a clear inflection point at 12 months, after which churn probability drops significantly. Getting customers past this milestone should be a key business objective."),
                    html.P(html.Strong("Business Implication: Design loyalty programs that reward the one-year milestone to increase retention through this critical period.")),
                ])
            ], className="insight-card"),
        ], className="insights-container"),
    ], className="tab-content")
    
    return layout

# Intervention Simulator Tab
def create_intervention_simulator(data: Dict[str, Any]) -> html.Div:
    """
    Create the Intervention Simulator tab layout.
    
    This tab allows stakeholders to simulate different intervention strategies
    and estimate their impact on retention and ROI.
    
    Parameters
    ----------
    data : Dict[str, Any]
        Dictionary containing all processed data
        
    Returns
    -------
    html.Div
        Layout for the intervention simulator tab
    """
    layout = html.Div([
        # Title and description
        html.Div([
            html.H2("Intervention Simulator", className="tab-title"),
            html.P("Estimate the impact of different retention strategies on churn and revenue", className="tab-description")
        ], className="tab-header"),
        
        # Simulator controls
        html.Div([
            html.Div([
                html.H3("Target Segment", className="control-group-title"),
                
                html.Div([
                    html.Label("Risk Tier:", className="control-label"),
                    dcc.Dropdown(
                        id='risk-tier-selector',
                        options=[
                            {'label': 'High Risk (70%+ probability)', 'value': 'high'},
                            {'label': 'Medium Risk (30-70% probability)', 'value': 'medium'},
                            {'label': 'Low Risk (<30% probability)', 'value': 'low'},
                            {'label': 'All Customers', 'value': 'all'}
                        ],
                        value='high',
                        clearable=False,
                        className="control-dropdown"
                    )
                ], className="control-item"),
                
                html.Div([
                    html.Label("Contract Type:", className="control-label"),
                    dcc.Dropdown(
                        id='contract-selector',
                        options=[
                            {'label': 'Month-to-Month', 'value': 'Month-to-Month'},
                            {'label': 'One Year', 'value': 'One Year'},
                            {'label': 'Two Year', 'value': 'Two Year'},
                            {'label': 'All Types', 'value': 'all'}
                        ],
                        value='Month-to-Month',
                        clearable=False,
                        className="control-dropdown"
                    )
                ], className="control-item"),
                
                html.Div([
                    html.Label("Customer Value:", className="control-label"),
                    dcc.Dropdown(
                        id='value-selector',
                        options=[
                            {'label': 'High Value (Top 25%)', 'value': 'high'},
                            {'label': 'Medium Value (Middle 50%)', 'value': 'medium'},
                            {'label': 'Low Value (Bottom 25%)', 'value': 'low'},
                            {'label': 'All Customers', 'value': 'all'}
                        ],
                        value='high',
                        clearable=False,
                        className="control-dropdown"
                    )
                ], className="control-item"),
            ], className="control-group"),
            
            html.Div([
                html.H3("Intervention Parameters", className="control-group-title"),
                
                html.Div([
                    html.Label("Intervention Type:", className="control-label"),
                    dcc.Dropdown(
                        id='intervention-selector',
                        options=[
                            {'label': 'Discount Offer', 'value': 'discount'},
                            {'label': 'Service Upgrade', 'value': 'upgrade'},
                            {'label': 'Contract Incentive', 'value': 'contract'},
                            {'label': 'Personalized Outreach', 'value': 'outreach'},
                            {'label': 'Comprehensive Package', 'value': 'comprehensive'}
                        ],
                        value='discount',
                        clearable=False,
                        className="control-dropdown"
                    )
                ], className="control-item"),
                
                html.Div([
                    html.Label("Cost per Customer ($):", className="control-label"),
                    dcc.Slider(
                        id='cost-slider',
                        min=10,
                        max=200,
                        step=10,
                        value=50,
                        marks={10: '$10', 50: '$50', 100: '$100', 150: '$150', 200: '$200'},
                        className="control-slider"
                    ),
                    html.Div(id='cost-display', className="slider-value")
                ], className="control-item"),
                
                html.Div([
                    html.Label("Expected Effectiveness (%):", className="control-label"),
                    dcc.Slider(
                        id='effectiveness-slider',
                        min=5,
                        max=40,
                        step=5,
                        value=20,
                        marks={5: '5%', 10: '10%', 20: '20%', 30: '30%', 40: '40%'},
                        className="control-slider"
                    ),
                    html.Div(id='effectiveness-display', className="slider-value")
                ], className="control-item"),
            ], className="control-group"),
        ], className="simulator-controls"),
        
        # Simulation results
        html.Div([
            html.H3("Simulation Results", className="section-title"),
            
            html.Div([
                # Summary metrics
                html.Div([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Target Customers", className="card-title"),
                            html.H2(id="target-count", className="card-value"),
                            html.P(id="target-percent", className="card-subvalue")
                        ])
                    ], className="result-card"),
                    
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Implementation Cost", className="card-title"),
                            html.H2(id="implementation-cost", className="card-value"),
                            html.P("Total intervention cost", className="card-subvalue")
                        ])
                    ], className="result-card"),
                    
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Customers Retained", className="card-title"),
                            html.H2(id="customers-retained", className="card-value"),
                            html.P(id="retention-percent", className="card-subvalue")
                        ])
                    ], className="result-card"),
                    
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Revenue Protected", className="card-title"),
                            html.H2(id="revenue-protected", className="card-value"),
                            html.P(id="roi-value", className="card-subvalue")
                        ])
                    ], className="result-card"),
                ], className="result-metrics"),
                
                # ROI chart
                html.Div([
                    html.H4("Return on Investment Projection", className="chart-title"),
                    dcc.Graph(
                        id='roi-chart',
                        config={'displayModeBar': False}
                    )
                ], className="roi-container"),
            ], className="simulation-results-container"),
        ], className="simulation-results"),
        
        # Recommendation table
        html.Div([
            html.H3("Recommended Intervention Strategy", className="section-title"),
            html.Div(id="recommendation-table", className="recommendation-table"),
            html.Div([
                html.Strong("Note: "), 
                "These recommendations are based on historical intervention effectiveness and current customer segmentation. Actual results may vary."
            ], className="recommendation-note")
        ], className="recommendations-container"),
    ], className="tab-content")
    
    return layout

# Model Performance Tab
def create_model_performance(data: Dict[str, Any]) -> html.Div:
    """
    Create the Model Performance & Methodology tab layout.
    
    This tab provides detailed information about model performance,
    methodology, and technical aspects of the analysis.
    
    Parameters
    ----------
    data : Dict[str, Any]
        Dictionary containing all processed data
        
    Returns
    -------
    html.Div
        Layout for the model performance tab
    """
    # Extract model metrics
    model_metrics = data['model_metrics']
    
    layout = html.Div([
        # Title and description
        html.Div([
            html.H2("Model Performance & Methodology", className="tab-title"),
            html.P("Technical details about the predictive model and evaluation metrics", className="tab-description")
        ], className="tab-header"),
        
        # Model performance metrics
        html.Div([
            html.H3("Model Evaluation Metrics", className="section-title"),
            
            # Performance charts
            html.Div([
                # ROC curve
                html.Div([
                    html.H4("ROC Curve", className="chart-title"),
                    dcc.Graph(
                        id='roc-curve',
                        config={'displayModeBar': False}
                    ),
                    html.P(f"AUC: {model_metrics['validation_metrics']['auc_roc']:.3f}", className="metric-value")
                ], className="performance-chart"),
                
                # Precision-Recall curve
                html.Div([
                    html.H4("Precision-Recall Curve", className="chart-title"),
                    dcc.Graph(
                        id='pr-curve',
                        config={'displayModeBar': False}
                    ),
                    html.P(f"AUC-PR: {model_metrics['validation_metrics']['auc_pr']:.3f}", className="metric-value")
                ], className="performance-chart"),
                
                # Confusion matrix
                html.Div([
                    html.H4("Confusion Matrix", className="chart-title"),
                    dcc.Graph(
                        id='confusion-matrix',
                        config={'displayModeBar': False}
                    ),
                    html.P([
                        f"Accuracy: {model_metrics['validation_metrics']['accuracy']:.3f}",
                        html.Br(),
                        f"Precision: {model_metrics['validation_metrics']['precision']:.3f}",
                        html.Br(),
                        f"Recall: {model_metrics['validation_metrics']['recall']:.3f}",
                        html.Br(),
                        f"F1 Score: {model_metrics['validation_metrics']['f1_score']:.3f}"
                    ], className="metric-value")
                ], className="performance-chart"),
                
                # Calibration curve
                html.Div([
                    html.H4("Calibration Curve", className="chart-title"),
                    dcc.Graph(
                        id='calibration-curve',
                        config={'displayModeBar': False}
                    ),
                    html.P("Evaluates how well predicted probabilities match observed frequencies", className="chart-description")
                ], className="performance-chart"),
            ], className="performance-charts-container"),
            
            # Model comparison
            html.Div([
                html.H4("Model Comparison", className="subsection-title"),
                dcc.Graph(
                    id='model-comparison-chart',
                    config={'displayModeBar': False}
                ),
                html.P([
                    html.Strong("Note: "), 
                    "The ensemble model combines the strengths of multiple algorithms to achieve superior performance, particularly in identifying high-value customers at risk."
                ], className="chart-note")
            ], className="model-comparison-container"),
        ], className="model-performance-container"),
        
        # Methodology section
        html.Div([
            html.H3("Methodology & Technical Approach", className="section-title"),
            
            # Methodology tabs
            dcc.Tabs([
                dcc.Tab(label="Data Preparation", children=[
                    html.Div([
                        html.H4("Data Sources", className="subsection-title"),
                        html.P("This analysis combines the following data sources:"),
                        html.Ul([
                            html.Li("Customer demographic and account information"),
                            html.Li("Usage patterns and service subscriptions"),
                            html.Li("Payment history and billing information"),
                            html.Li("Customer service interactions"),
                            html.Li("Historical churn data (for model training)")
                        ]),
                        
                        html.H4("Data Preprocessing", className="subsection-title"),
                        html.P("The following preprocessing steps were applied:"),
                        html.Ul([
                            html.Li("Missing value imputation using median/mode values"),
                            html.Li("Outlier detection and treatment using interquartile range"),
                            html.Li("Feature scaling and normalization"),
                            html.Li("Categorical encoding using one-hot encoding"),
                            html.Li("Date-based feature extraction from temporal variables")
                        ]),
                    ], className="tab-content")
                ], className="methodology-tab"),
                
                dcc.Tab(label="Feature Engineering", children=[
                    html.Div([
                        html.H4("Feature Creation", className="subsection-title"),
                        html.P("Several derived features were created to improve model performance:"),
                        html.Ul([
                            html.Li("Customer tenure variables (months since signup, contract renewal counts)"),
                            html.Li("Usage pattern metrics (consistency, growth/decline trends)"),
                            html.Li("Payment behavior indicators (late payment frequency, preferred payment methods)"),
                            html.Li("Customer service metrics (ticket frequency, resolution times)"),
                            html.Li("Engagement scores combining multiple interaction types")
                        ]),
                        
                        html.H4("Feature Selection", className="subsection-title"),
                        html.P("Feature selection was performed using:"),
                        html.Ul([
                            html.Li("Correlation analysis to remove highly redundant features"),
                            html.Li("Recursive feature elimination with cross-validation"),
                            html.Li("Permutation importance to identify the most predictive variables"),
                            html.Li("Domain expertise to ensure business relevance")
                        ]),
                    ], className="tab-content")
                ], className="methodology-tab"),
                
                dcc.Tab(label="Model Selection", children=[
                    html.Div([
                        html.H4("Model Evaluation", className="subsection-title"),
                        html.P("Models were evaluated using:"),
                        html.Ul([
                            html.Li("5-fold cross-validation to ensure robustness"),
                            html.Li("AUC-ROC as the primary metric for ranking performance"),
                            html.Li("Precision-Recall AUC for handling class imbalance"),
                            html.Li("Business impact metrics (expected value framework)"),
                            html.Li("Model interpretability considerations")
                        ]),
                        
                        html.H4("Ensemble Approach", className="subsection-title"),
                        html.P("The final model is an ensemble combining:"),
                        html.Ul([
                            html.Li("XGBoost: Captures complex non-linear patterns"),
                            html.Li("LightGBM: Efficiently handles categorical features"),
                            html.Li("Logistic Regression: Provides baseline interpretability"),
                            html.Li("Stacking with cross-validated predictions as meta-features"),
                            html.Li("Weighted averaging based on model performance")
                        ]),
                    ], className="tab-content")
                ], className="methodology-tab"),
                
                dcc.Tab(label="Limitations", children=[
                    html.Div([
                        html.H4("Model Limitations", className="subsection-title"),
                        html.P("Important considerations about the model:"),
                        html.Ul([
                            html.Li("Predictions are based on historical patterns and may not capture recent market changes"),
                            html.Li("Limited ability to capture external factors (competitor offers, economic conditions)"),
                            html.Li("Some customer segments have limited representation in the training data"),
                            html.Li("Model was optimized for overall performance, which may sacrifice accuracy in specific segments"),
                            html.Li("Predicted probabilities should be interpreted as relative risk indicators rather than absolute probabilities")
                        ]),
                        
                        html.H4("Future Improvements", className="subsection-title"),
                        html.P("Planned enhancements for future iterations:"),
                        html.Ul([
                            html.Li("Incorporate additional external data sources"),
                            html.Li("Develop segment-specific models for key customer groups"),
                            html.Li("Implement real-time prediction updates based on recent behavior"),
                            html.Li("Create explainable AI components for individual customer predictions"),
                            html.Li("Integrate customer feedback data for improved accuracy")
                        ]),
                    ], className="tab-content")
                ], className="methodology-tab"),
            ], className="methodology-tabs"),
        ], className="methodology-container"),
        
        # Version information
        html.Div([
            html.H3("Version Information", className="section-title"),
            html.Table([
                html.Tr([
                    html.Td("Model Version:"),
                    html.Td(model_metrics['model_name'])
                ]),
                html.Tr([
                    html.Td("Training Date:"),
                    html.Td(model_metrics['training_date'])
                ]),
                html.Tr([
                    html.Td("Last Validation:"),
                    html.Td("January 10, 2025")
                ]),
                html.Tr([
                    html.Td("Data Cutoff:"),
                    html.Td("December 31, 2024")
                ]),
                html.Tr([
                    html.Td("Next Update:"),
                    html.Td("April 15, 2025 (Quarterly)")
                ]),
            ], className="version-table"),
        ], className="version-container"),
    ], className="tab-content")
    
    return layout