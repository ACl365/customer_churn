from dash import Input, Output, State, html, dash_table
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import pandas as pd
import numpy as np
from typing import Dict, Any, List
import dash_bootstrap_components as dbc
from visualizations import (
    create_churn_distribution_histogram,
    create_risk_tier_pie,
    create_customer_segment_map,
    create_segment_comparison_chart,
    create_feature_importance_chart,
    create_partial_dependence_plot,
    create_feature_correlation_heatmap,
    create_roi_projection,
    create_roc_curve,
    create_pr_curve,
    create_confusion_matrix,
    create_calibration_curve,
    create_model_comparison_chart
)

def register_callbacks(app, data, cache):
    """
    Register all callbacks for the dashboard application.
    
    Parameters
    ----------
    app : dash.Dash
        The Dash application instance
    data : Dict[str, Any]
        Dictionary containing all processed data
    cache : flask_caching.Cache
        Cache instance for performance optimization
    """
    # Tab content callback
    @app.callback(
        Output("tab-content", "children"),
        Input("tabs", "value")
    )
    def render_tab_content(tab):
        """Render the content of the selected tab."""
        if tab == "tab-exec-summary":
            return create_executive_summary(data)
        elif tab == "tab-segmentation":
            return create_customer_segmentation(data)
        elif tab == "tab-drivers":
            return create_churn_drivers(data)
        elif tab == "tab-simulator":
            return create_intervention_simulator(data)
        elif tab == "tab-model":
            return create_model_performance(data)
        return html.Div("No content available")
    
    # ----- Executive Summary Tab Callbacks -----
    
    @app.callback(
        Output("churn-probability-dist", "figure"),
        Input("tabs", "value")
    )
    @cache.memoize()
    def update_churn_probability_distribution(tab):
        """Create churn probability distribution histogram."""
        if tab != "tab-exec-summary":
            return {}
        
        # Create histogram with risk zones
        fig = create_churn_distribution_histogram(data['merged_data'])
        return fig
    
    @app.callback(
        Output("risk-tier-breakdown", "figure"),
        Input("tabs", "value")
    )
    @cache.memoize()
    def update_risk_tier_breakdown(tab):
        """Create risk tier breakdown pie chart."""
        if tab != "tab-exec-summary":
            return {}
        
        # Create pie chart
        fig = create_risk_tier_pie(data['derived_metrics']['risk_tiers'])
        return fig
    
    # ----- Customer Segmentation Tab Callbacks -----
    
    @app.callback(
        Output("customer-segment-map", "figure"),
        [Input("segment-dropdown", "value"),
         Input("color-dropdown", "value")]
    )
    @cache.memoize()
    def update_customer_segment_map(segment_by, color_by):
        """Create customer segment scatter plot."""
        # Create scatter plot visualization
        fig = create_customer_segment_map(
            data['merged_data'],
            segment_by,
            color_by
        )
        return fig
    
    @app.callback(
        Output("segment-comparison-chart", "figure"),
        Input("segment-dropdown", "value")
    )
    @cache.memoize()
    def update_segment_comparison(segment_by):
        """Create segment comparison chart."""
        # Create comparison chart
        fig = create_segment_comparison_chart(
            data['merged_data'],
            segment_by
        )
        return fig
    
    @app.callback(
        Output("segment-details-table", "children"),
        Input("segment-dropdown", "value")
    )
    def update_segment_details_table(segment_by):
        """Create segment details table."""
        df = data['merged_data']
        
        if segment_by not in df.columns:
            return html.Div("Segment data not available")
        
        # Group by segment and calculate metrics
        segment_df = df.groupby(segment_by).agg({
            'CustomerID': 'count',
            'churn_probability': ['mean', 'std'],
            'predicted_churn': 'mean'
        }).reset_index()
        
        # Flatten multi-level columns
        segment_df.columns = [
            '_'.join(col).strip('_') for col in segment_df.columns.values
        ]
        
        # Rename columns for clarity
        segment_df = segment_df.rename(columns={
            segment_by: 'Segment',
            'CustomerID_count': 'Customer Count',
            'churn_probability_mean': 'Avg Churn Probability',
            'churn_probability_std': 'Churn Probability Std',
            'predicted_churn_mean': 'Churn Rate'
        })
        
        # Add revenue metrics if available
        if 'annual_revenue' in df.columns:
            revenue_metrics = df.groupby(segment_by).agg({
                'annual_revenue': ['sum', 'mean']
            }).reset_index()
            
            # Flatten multi-level columns
            revenue_metrics.columns = [
                '_'.join(col).strip('_') for col in revenue_metrics.columns.values
            ]
            
            # Rename for clarity
            revenue_metrics = revenue_metrics.rename(columns={
                segment_by: 'Segment',
                'annual_revenue_sum': 'Total Revenue',
                'annual_revenue_mean': 'Avg Revenue per Customer'
            })
            
            # Merge with main segment dataframe
            segment_df = pd.merge(segment_df, revenue_metrics, on='Segment', how='left')
            
            # Calculate revenue at risk
            segment_df['Revenue at Risk'] = segment_df['Total Revenue'] * segment_df['Churn Rate']
        
        # Sort by churn rate for better insights
        segment_df = segment_df.sort_values('Churn Rate', ascending=False)
        
        # Format values for display
        segment_df['Avg Churn Probability'] = segment_df['Avg Churn Probability'].map(lambda x: f"{x:.1%}")
        segment_df['Churn Rate'] = segment_df['Churn Rate'].map(lambda x: f"{x:.1%}")
        segment_df['Customer Count'] = segment_df['Customer Count'].map(lambda x: f"{x:,}")
        
        if 'Total Revenue' in segment_df.columns:
            segment_df['Total Revenue'] = segment_df['Total Revenue'].map(lambda x: f"${x:,.0f}")
            segment_df['Avg Revenue per Customer'] = segment_df['Avg Revenue per Customer'].map(lambda x: f"${x:.2f}")
            segment_df['Revenue at Risk'] = segment_df['Revenue at Risk'].map(lambda x: f"${x:,.0f}")
        
        # Create the data table
        table = dash_table.DataTable(
            data=segment_df.to_dict('records'),
            columns=[{'name': col, 'id': col} for col in segment_df.columns],
            style_table={'overflowX': 'auto'},
            style_cell={
                'textAlign': 'left',
                'padding': '10px',
                'font-family': 'Arial, sans-serif'
            },
            style_header={
                'backgroundColor': '#f0f0f0',
                'fontWeight': 'bold',
                'border': '1px solid #ddd'
            },
            style_data={
                'border': '1px solid #ddd'
            },
            style_data_conditional=[
                {
                    'if': {
                        'filter_query': '{Churn Rate} contains "0.3"',
                    },
                    'backgroundColor': '#ffebee',
                    'color': '#c62828'
                },
            ]
        )
        
        return table
    
    @app.callback(
        Output("segment-risk-assessment", "children"),
        Input("segment-dropdown", "value")
    )
    def update_segment_risk_assessment(segment_by):
        """Create segment risk assessment text."""
        df = data['merged_data']
        
        if segment_by not in df.columns:
            return html.Div("Segment data not available")
        
        # Identify highest risk segment
        segment_risk = df.groupby(segment_by)['churn_probability'].mean()
        highest_risk_segment = segment_risk.idxmax()
        highest_risk_value = segment_risk.max()
        
        # Get count of customers in this segment
        segment_counts = df.groupby(segment_by)['CustomerID'].count()
        highest_risk_count = segment_counts[highest_risk_segment]
        total_customers = len(df)
        
        # Calculate revenue impact if available
        if 'annual_revenue' in df.columns:
            segment_revenue = df.groupby(segment_by)['annual_revenue'].sum()
            highest_risk_revenue = segment_revenue[highest_risk_segment]
            highest_risk_rev_at_risk = highest_risk_revenue * highest_risk_value
            total_revenue = df['annual_revenue'].sum()
            revenue_impact_text = f"This segment represents ${highest_risk_revenue:,.0f} in annual revenue, with approximately ${highest_risk_rev_at_risk:,.0f} at risk due to churn."
        else:
            revenue_impact_text = ""
        
        # Create assessment text
        assessment = html.Div([
            html.P([
                f"The ", 
                html.Strong(f"{highest_risk_segment}"), 
                f" segment shows the highest churn risk with an average probability of ", 
                html.Strong(f"{highest_risk_value:.1%}"), 
                f". This segment contains {highest_risk_count:,} customers ({highest_risk_count/total_customers:.1%} of total customer base)."
            ]),
            html.P(revenue_impact_text),
            html.P([
                html.Strong("Priority Assessment: "), 
                html.Span(
                    "High Priority - Immediate Intervention Recommended", 
                    style={"color": "#c62828", "fontWeight": "bold"}
                ) if highest_risk_value > 0.5 else
                html.Span(
                    "Medium Priority - Targeted Intervention Recommended", 
                    style={"color": "#ff8f00", "fontWeight": "bold"}
                ) if highest_risk_value > 0.3 else
                html.Span(
                    "Low Priority - Standard Retention Measures Sufficient", 
                    style={"color": "#2e7d32", "fontWeight": "bold"}
                )
            ]),
            html.Hr(),
            html.P([
                html.Strong("Recommended Action: "), 
                "Based on historical intervention effectiveness, we recommend ",
                html.Strong(
                    "personalized outreach with contract incentives" if highest_risk_value > 0.5 else
                    "targeted service upgrade offers" if highest_risk_value > 0.3 else
                    "standard satisfaction surveys and loyalty rewards"
                ),
                " for this segment."
            ])
        ])
        
        return assessment
    
    # ----- Churn Drivers Tab Callbacks -----
    
    @app.callback(
        Output("feature-importance-chart", "figure"),
        Input("tabs", "value")
    )
    @cache.memoize()
    def update_feature_importance(tab):
        """Create feature importance bar chart."""
        if tab != "tab-drivers":
            return {}
        
        # Get feature importance from model metrics
        feature_importance = data['model_metrics']['feature_importance']
        
        # Create the feature importance chart
        fig = create_feature_importance_chart(feature_importance)
        return fig
    
    @app.callback(
        [Output("feature-impact-title", "children"),
         Output("partial-dependence-plot", "figure"),
         Output("feature-impact-insight", "children")],
        Input("feature-selector", "value")
    )
    @cache.memoize()
    def update_partial_dependence(feature):
        """Create partial dependence plot for selected feature."""
        if not feature:
            return "Select a feature", {}, ""
        
        # Feature impact title
        title = html.H3(f"Impact of {feature.replace('_', ' ').title()} on Churn Probability", className="chart-title")
        
        # Create partial dependence plot
        fig, insight_text = create_partial_dependence_plot(
            data['merged_data'],
            feature
        )
        
        # Create insight text
        insight = html.P([
            html.Strong("Key Insight: "), 
            insight_text
        ])
        
        return title, fig, insight
    
    @app.callback(
        Output("feature-correlation-chart", "figure"),
        Input("tabs", "value")
    )
    @cache.memoize()
    def update_feature_correlation(tab):
        """Create feature correlation matrix."""
        if tab != "tab-drivers":
            return {}
        
        # Get feature importance from model metrics
        feature_importance = data['model_metrics']['feature_importance']
        
        # Select top features
        top_features = list(feature_importance.keys())[:8]  # Limit to top 8 for readability
        
        # Create correlation heatmap
        fig = create_feature_correlation_heatmap(
            data['merged_data'],
            top_features
        )
        
        return fig
    
    # ----- Intervention Simulator Tab Callbacks -----
    
    @app.callback(
        [Output("cost-display", "children"),
         Output("effectiveness-display", "children")],
        [Input("cost-slider", "value"),
         Input("effectiveness-slider", "value")]
    )
    def update_slider_displays(cost, effectiveness):
        """Update text displays for sliders."""
        return f"${cost} per customer", f"{effectiveness}% effectiveness"
    
    @app.callback(
        [Output("target-count", "children"),
         Output("target-percent", "children"),
         Output("implementation-cost", "children"),
         Output("customers-retained", "children"),
         Output("retention-percent", "children"),
         Output("revenue-protected", "children"),
         Output("roi-value", "children"),
         Output("roi-chart", "figure")],
        [Input("risk-tier-selector", "value"),
         Input("contract-selector", "value"),
         Input("value-selector", "value"),
         Input("intervention-selector", "value"),
         Input("cost-slider", "value"),
         Input("effectiveness-slider", "value")]
    )
    def update_simulation_results(risk_tier, contract_type, customer_value, intervention_type, cost_per_customer, effectiveness):
        """Update simulation results based on selected parameters."""
        df = data['merged_data']
        
        # Filter target customers
        mask = pd.Series(True, index=df.index)
        
        # Apply risk tier filter
        if risk_tier == 'high':
            mask &= df['churn_probability'] >= 0.7
        elif risk_tier == 'medium':
            mask &= (df['churn_probability'] >= 0.3) & (df['churn_probability'] < 0.7)
        elif risk_tier == 'low':
            mask &= df['churn_probability'] < 0.3
        
        # Apply contract type filter if column exists
        if 'contract_type' in df.columns and contract_type != 'all':
            mask &= df['contract_type'] == contract_type
        
        # Apply customer value filter if column exists
        if 'customer_lifetime_value' in df.columns and customer_value != 'all':
            if customer_value == 'high':
                mask &= df['customer_lifetime_value'] >= df['customer_lifetime_value'].quantile(0.75)
            elif customer_value == 'medium':
                mask &= (df['customer_lifetime_value'] >= df['customer_lifetime_value'].quantile(0.25)) & \
                      (df['customer_lifetime_value'] < df['customer_lifetime_value'].quantile(0.75))
            elif customer_value == 'low':
                mask &= df['customer_lifetime_value'] < df['customer_lifetime_value'].quantile(0.25)
        
        # Select target customers
        target_customers = df[mask]
        
        # Calculate metrics
        target_count = len(target_customers)
        target_percent = f"{target_count / len(df):.1%} of customer base"
        
        # Total implementation cost
        total_cost = target_count * cost_per_customer
        cost_display = f"${total_cost:,.0f}"
        
        # Expected customers retained
        retained_count = int(target_count * (effectiveness / 100))
        retained_percent = f"{retained_count / target_count:.1%} retention rate"
        
        # Revenue protected
        if 'annual_revenue' in df.columns:
            avg_revenue = target_customers['annual_revenue'].mean()
        else:
            avg_revenue = 500  # Placeholder value
            
        revenue_protected = retained_count * avg_revenue
        revenue_display = f"${revenue_protected:,.0f}"
        
        # ROI calculation
        roi = (revenue_protected - total_cost) / total_cost if total_cost > 0 else 0
        roi_display = f"ROI: {roi:.1%}"
        
        # Create ROI projection chart
        fig = create_roi_projection(
            total_cost=total_cost,
            monthly_revenue=revenue_protected / 12
        )
        
        return target_count, target_percent, cost_display, retained_count, retained_percent, revenue_display, roi_display, fig
    
    @app.callback(
        Output("recommendation-table", "children"),
        [Input("risk-tier-selector", "value"),
         Input("contract-selector", "value"),
         Input("value-selector", "value")]
    )
    def update_recommendation_table(risk_tier, contract_type, customer_value):
        """Create recommendation table based on selected segment."""
        # Create recommendations based on selected parameters
        recommendations = []
        
        # High-risk customers
        if risk_tier == 'high':
            if customer_value == 'high':
                recommendations.append({
                    'intervention': 'Personalized Outreach',
                    'description': 'High-touch personal call from account manager with custom retention offer',
                    'cost': '$100-150 per customer',
                    'effectiveness': '30-35%',
                    'roi': '3.5-4.2x',
                    'priority': 'Very High'
                })
                recommendations.append({
                    'intervention': 'Premium Service Upgrade',
                    'description': 'Complimentary upgrade to premium service tier for 3 months',
                    'cost': '$75-125 per customer',
                    'effectiveness': '25-30%',
                    'roi': '2.8-3.5x',
                    'priority': 'High'
                })
            else:
                recommendations.append({
                    'intervention': 'Targeted Discount',
                    'description': '15% discount offer with 12-month contract commitment',
                    'cost': '$50-80 per customer',
                    'effectiveness': '20-25%',
                    'roi': '2.1-2.8x',
                    'priority': 'High'
                })
                recommendations.append({
                    'intervention': 'Service Add-on Bundle',
                    'description': 'Free add-on services for 6 months with contract renewal',
                    'cost': '$40-60 per customer',
                    'effectiveness': '15-20%',
                    'roi': '1.8-2.5x',
                    'priority': 'Medium'
                })
        
        # Medium-risk customers
        elif risk_tier == 'medium':
            recommendations.append({
                'intervention': 'Loyalty Rewards',
                'description': 'Enhanced loyalty program benefits with milestone rewards',
                'cost': '$30-50 per customer',
                'effectiveness': '15-20%',
                'roi': '2.0-2.5x',
                'priority': 'Medium'
            })
            recommendations.append({
                'intervention': 'Educational Outreach',
                'description': 'Proactive tutorials on features that improve retention',
                'cost': '$20-35 per customer',
                'effectiveness': '10-15%',
                'roi': '1.7-2.2x',
                'priority': 'Medium'
            })
        
        # Low-risk customers
        else:
            recommendations.append({
                'intervention': 'Satisfaction Survey',
                'description': 'Customer satisfaction check-in with small incentive',
                'cost': '$10-20 per customer',
                'effectiveness': '5-10%',
                'roi': '1.2-1.8x',
                'priority': 'Low'
            })
            recommendations.append({
                'intervention': 'Digital Engagement',
                'description': 'Enhanced digital engagement campaign via email/app',
                'cost': '$5-15 per customer',
                'effectiveness': '3-8%',
                'roi': '1.0-1.5x',
                'priority': 'Low'
            })
        
        # Create the recommendations table
        table = html.Table([
            # Header
            html.Thead(
                html.Tr([
                    html.Th("Intervention Strategy", style={"width": "20%"}),
                    html.Th("Description", style={"width": "35%"}),
                    html.Th("Cost", style={"width": "10%"}),
                    html.Th("Expected Effectiveness", style={"width": "15%"}),
                    html.Th("Est. ROI", style={"width": "10%"}),
                    html.Th("Priority", style={"width": "10%"})
                ])
            ),
            # Body
            html.Tbody([
                html.Tr([
                    html.Td(rec['intervention']),
                    html.Td(rec['description']),
                    html.Td(rec['cost']),
                    html.Td(rec['effectiveness']),
                    html.Td(rec['roi']),
                    html.Td(
                        rec['priority'],
                        style={
                            "color": "#c62828" if rec['priority'] == 'Very High' else
                                    "#e53935" if rec['priority'] == 'High' else
                                    "#ff8f00" if rec['priority'] == 'Medium' else
                                    "#2e7d32",
                            "font-weight": "bold"
                        }
                    )
                ]) for rec in recommendations
            ])
        ], className="recommendation-table-element")
        
        return table
    
    # ----- Model Performance Tab Callbacks -----
    
    @app.callback(
        Output("roc-curve", "figure"),
        Input("tabs", "value")
    )
    @cache.memoize()
    def update_roc_curve(tab):
        """Create ROC curve visualization."""
        if tab != "tab-model":
            return {}
        
        # Create ROC curve with the model's AUC value
        auc_value = data['model_metrics']['validation_metrics']['auc_roc']
        fig = create_roc_curve(auc_value)
        
        return fig
    
    @app.callback(
        Output("pr-curve", "figure"),
        Input("tabs", "value")
    )
    @cache.memoize()
    def update_pr_curve(tab):
        """Create Precision-Recall curve visualization."""
        if tab != "tab-model":
            return {}
        
        # Create PR curve with model metrics
        auc_pr = data['model_metrics']['validation_metrics']['auc_pr']
        churn_rate = data['derived_metrics']['churn_rate']
        fig = create_pr_curve(auc_pr, churn_rate)
        
        return fig
    
    @app.callback(
        Output("confusion-matrix", "figure"),
        Input("tabs", "value")
    )
    @cache.memoize()
    def update_confusion_matrix(tab):
        """Create confusion matrix visualization."""
        if tab != "tab-model":
            return {}
        
        # Compute confusion matrix values based on available metrics
        metrics = data['model_metrics']['validation_metrics']
        
        # Estimate TP, TN, FP, FN from metrics
        accuracy = metrics['accuracy']
        precision = metrics['precision']
        recall = metrics['recall']
        
        # Assuming a 15% churn rate for calculations
        churn_rate = data['derived_metrics']['churn_rate']
        validation_size = 5000  # Placeholder value
        
        # Expected number of actual positive cases (churners)
        actual_positives = int(validation_size * churn_rate)
        actual_negatives = validation_size - actual_positives
        
        # Calculate TP, FP, TN, FN
        TP = int(recall * actual_positives)
        FN = actual_positives - TP
        FP = int(TP / precision - TP)
        TN = actual_negatives - FP
        
        # Create confusion matrix
        fig = create_confusion_matrix(TP, FP, TN, FN, validation_size)
        
        return fig
    
    @app.callback(
        Output("calibration-curve", "figure"),
        Input("tabs", "value")
    )
    @cache.memoize()
    def update_calibration_curve(tab):
        """Create calibration curve visualization."""
        if tab != "tab-model":
            return {}
        
        # Create calibration curve
        fig = create_calibration_curve()
        
        return fig
    
    @app.callback(
        Output("model-comparison-chart", "figure"),
        Input("tabs", "value")
    )
    @cache.memoize()
    def update_model_comparison(tab):
        """Create model comparison bar chart."""
        if tab != "tab-model":
            return {}
        
        # Get model comparison data
        model_comparison = data['model_metrics']['model_comparison']
        
        # Create model comparison chart
        fig = create_model_comparison_chart(model_comparison)
        
        return fig


# Function to create each tab's layout - these are called from the main layout callback
def create_executive_summary(data):
    """Create the executive summary tab layout."""
    from layouts import create_executive_summary
    return create_executive_summary(data)

def create_customer_segmentation(data):
    """Create the customer segmentation tab layout."""
    from layouts import create_customer_segmentation
    return create_customer_segmentation(data)

def create_churn_drivers(data):
    """Create the churn drivers tab layout."""
    from layouts import create_churn_drivers
    return create_churn_drivers(data)

def create_intervention_simulator(data):
    """Create the intervention simulator tab layout."""
    from layouts import create_intervention_simulator
    return create_intervention_simulator(data)

def create_model_performance(data):
    """Create the model performance tab layout."""
    from layouts import create_model_performance
    return create_model_performance(data)