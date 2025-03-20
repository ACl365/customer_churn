import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

# Color palette - consistent colors across all visualizations
COLORS = {
    'primary': '#55acee',      # Primary blue for main elements
    'secondary': '#884dff',    # Purple for secondary elements
    'accent': '#ff6b6b',       # Red accent for highlights/alerts
    'success': '#4CAF50',      # Green for positive indicators
    'warning': '#FFA500',      # Orange for warnings
    'text': '#444444',         # Main text color
    'light_gray': '#f0f0f0',   # Light gray background
}

###########################################
# Executive Summary Tab Visualizations
###########################################

def create_pr_curve(auc_pr_value: float = 0.84, churn_rate: float = 0.24) -> go.Figure:
    """
    Create a Precision-Recall curve visualization.
    
    In a real implementation, these values would come from model evaluation.
    Here we use placeholder data for demonstration.
    
    Parameters
    ----------
    auc_pr_value : float
        AUC-PR value to display
    churn_rate : float
        Overall churn rate (baseline precision)
        
    Returns
    -------
    go.Figure
        Plotly figure object
    """
    # Create figure
    fig = go.Figure()
    
    # Add baseline reference line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[churn_rate, churn_rate],
        mode='lines',
        line=dict(color='gray', dash='dash'),
        name='Baseline'
    ))
    
    # Add PR curve - using placeholder data
    recall = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    precision = [1.0, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, churn_rate]
    
    fig.add_trace(go.Scatter(
        x=recall,
        y=precision,
        mode='lines',
        line=dict(color=COLORS['primary'], width=2),
        name=f"Ensemble Model (AUC = {auc_pr_value})"
    ))
    
    # Update layout
    fig.update_layout(
        xaxis_title="Recall",
        yaxis_title="Precision",
        margin=dict(l=40, r=40, t=20, b=40),
        plot_bgcolor="white",
        height=300,
        legend=dict(
            yanchor="bottom",
            y=0.01,
            xanchor="right",
            x=0.99
        )
    )
    
    # Set axis ranges
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])
    
    return fig

def create_confusion_matrix(
    TP: int,
    FP: int, 
    TN: int, 
    FN: int, 
    validation_size: int
) -> go.Figure:
    """
    Create a confusion matrix visualization.
    
    Parameters
    ----------
    TP : int
        True Positives
    FP : int
        False Positives
    TN : int
        True Negatives
    FN : int
        False Negatives
    validation_size : int
        Total size of validation set
        
    Returns
    -------
    go.Figure
        Plotly figure object
    """
    # Create confusion matrix
    z = [[TN, FP], [FN, TP]]
    x = ['Predicted No Churn', 'Predicted Churn']
    y = ['Actual No Churn', 'Actual Churn']
    
    # Create annotations
    annotations = [
        [f"{TN}<br>({TN/validation_size:.1%})", f"{FP}<br>({FP/validation_size:.1%})"],
        [f"{FN}<br>({FN/validation_size:.1%})", f"{TP}<br>({TP/validation_size:.1%})"]
    ]
    
    # Create heatmap figure
    fig = ff.create_annotated_heatmap(
        z=z,
        x=x,
        y=y,
        annotation_text=annotations,
        colorscale=[[0, '#edf8e9'], [1, COLORS['primary']]],
        showscale=False
    )
    
    # Update layout
    fig.update_layout(
        margin=dict(l=40, r=40, t=30, b=40),
        plot_bgcolor="white",
        height=300,
        xaxis=dict(side='bottom'),
        yaxis=dict(side='left', autorange='reversed')
    )
    
    return fig

def create_calibration_curve() -> go.Figure:
    """
    Create a calibration curve visualization.
    
    In a real implementation, these values would come from model evaluation.
    Here we use placeholder data for demonstration.
    
    Returns
    -------
    go.Figure
        Plotly figure object
    """
    # Create figure
    fig = go.Figure()
    
    # Add perfect calibration reference line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        line=dict(color='gray', dash='dash'),
        name='Perfect Calibration'
    ))
    
    # Create placeholder calibration curve data
    prob_bins = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
    # Slightly off perfect calibration for more realism
    actual_freqs = [0.02, 0.11, 0.22, 0.38, 0.49, 0.57, 0.68, 0.78, 0.82, 0.93]
    
    fig.add_trace(go.Scatter(
        x=prob_bins,
        y=actual_freqs,
        mode='lines+markers',
        line=dict(color=COLORS['primary'], width=2),
        marker=dict(size=8, color=COLORS['primary']),
        name='Ensemble Model'
    ))
    
    # Update layout
    fig.update_layout(
        xaxis_title="Predicted Probability",
        yaxis_title="Observed Frequency",
        margin=dict(l=40, r=40, t=20, b=40),
        plot_bgcolor="white",
        height=300,
        legend=dict(
            yanchor="bottom",
            y=0.01,
            xanchor="right",
            x=0.99
        )
    )
    
    # Set axis ranges
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])
    
    return fig

def create_model_comparison_chart(model_comparison: Dict[str, float]) -> go.Figure:
    """
    Create a model comparison bar chart.
    
    Parameters
    ----------
    model_comparison : Dict[str, float]
        Dictionary mapping model names to performance values
        
    Returns
    -------
    go.Figure
        Plotly figure object
    """
    # Create dataframe for plotting
    comparison_df = pd.DataFrame({
        'Model': list(model_comparison.keys()),
        'AUC': list(model_comparison.values())
    })
    
    # Sort by performance
    comparison_df = comparison_df.sort_values('AUC', ascending=False)
    
    # Create horizontal bar chart
    fig = go.Figure()
    
    # Add bars with color highlighting for ensemble
    colors = [COLORS['accent'] if model == 'Ensemble' else COLORS['primary'] for model in comparison_df['Model']]
    
    fig.add_trace(go.Bar(
        y=comparison_df['Model'],
        x=comparison_df['AUC'],
        orientation='h',
        marker_color=colors,
        text=comparison_df['AUC'].map(lambda x: f"{x:.3f}"),
        textposition='inside'
    ))
    
    # Update layout
    fig.update_layout(
        xaxis_title="AUC-ROC Score",
        yaxis_title="",
        margin=dict(l=20, r=20, t=20, b=20),
        height=300,
        plot_bgcolor="white",
        xaxis=dict(range=[0.7, max(comparison_df['AUC']) * 1.05])
    )
    
    return fig

###########################################
# Utility Functions
###########################################

def format_currency(value: float) -> str:
    """Format a value as currency."""
    return f"${value:,.2f}"

def format_percentage(value: float) -> str:
    """Format a value as percentage."""
    return f"{value:.1%}"

def format_number(value: float) -> str:
    """Format a value as a number with thousands separator."""
    return f"{value:,.0f}"

def get_color_scale(values: List[float], colorscale_name: str = 'Blues') -> List[str]:
    """
    Generate a list of colors for values based on a colorscale.
    
    Parameters
    ----------
    values : List[float]
        List of values to map to colors
    colorscale_name : str, optional
        Name of the Plotly colorscale, by default 'Blues'
    
    Returns
    -------
    List[str]
        List of color strings
    """
    if len(values) == 0:
        return []
    
    min_val = min(values)
    max_val = max(values)
    
    if max_val == min_val:
        # If all values are the same, return middle color
        normalized = [0.5] * len(values)
    else:
        # Normalize values to [0, 1]
        normalized = [(v - min_val) / (max_val - min_val) for v in values]
    
    # Create a colorscale
    colorscale = px.colors.sequential.__getattribute__(colorscale_name)
    
    # Map normalized values to colors
    colors = []
    for norm_val in normalized:
        idx = int(norm_val * (len(colorscale) - 1))
        colors.append(colorscale[idx])
    
    return colors
def create_churn_distribution_histogram(df: pd.DataFrame) -> go.Figure:
    """
    Create a histogram of churn probabilities with risk zones.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with churn_probability column
        
    Returns
    -------
    go.Figure
        Plotly figure object
    """
    # Create histogram with custom styling
    fig = px.histogram(
        df,
        x="churn_probability",
        nbins=20,
        labels={"churn_probability": "Churn Probability"},
        title="",
        opacity=0.8
    )
    
    # Add risk zones
    fig.add_vrect(
        x0=0, x1=0.3,
        fillcolor="green", opacity=0.15,
        layer="below", line_width=0
    )
    fig.add_vrect(
        x0=0.3, x1=0.7, 
        fillcolor="orange", opacity=0.15,
        layer="below", line_width=0
    )
    fig.add_vrect(
        x0=0.7, x1=1, 
        fillcolor="red", opacity=0.15,
        layer="below", line_width=0
    )
    
    # Add risk labels
    fig.add_annotation(x=0.15, y=fig.data[0].y.max() * 0.95, text="Low Risk", showarrow=False)
    fig.add_annotation(x=0.5, y=fig.data[0].y.max() * 0.95, text="Medium Risk", showarrow=False)
    fig.add_annotation(x=0.85, y=fig.data[0].y.max() * 0.95, text="High Risk", showarrow=False)
    
    # Update layout
    fig.update_layout(
        xaxis_title="Churn Probability",
        yaxis_title="Number of Customers",
        showlegend=False,
        margin=dict(l=40, r=40, t=30, b=40),
        plot_bgcolor="white",
        height=350,
        xaxis=dict(
            tickformat=".0%",
            tickvals=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            ticktext=["0%", "10%", "20%", "30%", "40%", "50%", "60%", "70%", "80%", "90%", "100%"]
        )
    )
    
    # Update trace
    fig.update_traces(marker_color=COLORS['primary'])
    
    return fig

def create_risk_tier_pie(risk_tiers: Dict[str, int]) -> go.Figure:
    """
    Create a pie chart showing the distribution of customers across risk tiers.
    
    Parameters
    ----------
    risk_tiers : Dict[str, int]
        Dictionary with keys 'low', 'medium', 'high' and their counts
        
    Returns
    -------
    go.Figure
        Plotly figure object
    """
    # Create pie chart
    fig = go.Figure(data=[go.Pie(
        labels=["Low Risk", "Medium Risk", "High Risk"],
        values=[risk_tiers['low'], risk_tiers['medium'], risk_tiers['high']],
        hole=0.5,
        marker_colors=[COLORS['success'], COLORS['warning'], COLORS['accent']]
    )])
    
    # Update layout
    fig.update_layout(
        margin=dict(l=20, r=20, t=30, b=20),
        height=350,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5
        )
    )
    
    # Add counts as annotations
    total = sum(risk_tiers.values())
    fig.add_annotation(
        text=f"{total:,}<br>Total",
        x=0.5, y=0.5,
        font_size=14,
        showarrow=False
    )
    
    return fig

###########################################
# Customer Segmentation Tab Visualizations
###########################################

def create_customer_segment_map(
    df: pd.DataFrame, 
    segment_col: str, 
    color_col: str = "churn_probability"
) -> go.Figure:
    """
    Create a 2D visualization of customer segments.
    
    In a real implementation, t-SNE or UMAP would be used for dimensionality reduction.
    Here we use random coordinates for demonstration purposes.
    
    Parameters
    ----------
    df : pd.DataFrame
        Customer data
    segment_col : str
        Column name to use for segmentation
    color_col : str
        Column name to use for point colors
        
    Returns
    -------
    go.Figure
        Plotly figure object
    """
    # Generate placeholder coordinates for demonstration
    n = len(df)
    np.random.seed(42)  # For reproducibility
    
    # Create temporary dataframe with coordinates
    plot_df = df.copy()
    plot_df['x'] = np.random.normal(0, 1, n)
    plot_df['y'] = np.random.normal(0, 1, n)
    
    # Add some separation between groups
    if segment_col in plot_df.columns:
        segments = plot_df[segment_col].unique()
        for i, segment in enumerate(segments):
            mask = plot_df[segment_col] == segment
            plot_df.loc[mask, 'x'] += i * 2
    
    # Create scatter plot
    fig = px.scatter(
        plot_df,
        x="x",
        y="y",
        color=color_col if color_col in plot_df.columns else "churn_probability",
        color_continuous_scale=px.colors.sequential.Blues_r if color_col == "churn_probability" else px.colors.sequential.Viridis,
        opacity=0.7,
        hover_name="CustomerID",
        hover_data={
            "x": False,
            "y": False,
            segment_col: True if segment_col in plot_df.columns else False,
            "churn_probability": ":.1%",
            "predicted_churn": True,
            "annual_revenue": ":$,.2f" if "annual_revenue" in plot_df.columns else False,
            "contract_type": True if "contract_type" in plot_df.columns else False,
            "tenure": True if "tenure" in plot_df.columns else False
        },
        title=""
    )
    
    # If segment_col is available, add ellipses around segments
    if segment_col in plot_df.columns:
        segments = plot_df[segment_col].unique()
        for segment in segments:
            segment_df = plot_df[plot_df[segment_col] == segment]
            x_mean, y_mean = segment_df['x'].mean(), segment_df['y'].mean()
            x_std, y_std = segment_df['x'].std() * 2, segment_df['y'].std() * 2
            
            fig.add_shape(
                type="circle",
                xref="x", yref="y",
                x0=x_mean - x_std, y0=y_mean - y_std,
                x1=x_mean + x_std, y1=y_mean + y_std,
                line_color="gray",
                line_width=1,
                line_dash="dash",
                opacity=0.3
            )
            
            # Add segment label
            fig.add_annotation(
                x=x_mean, y=y_mean + y_std + 0.2,
                text=str(segment),
                showarrow=False,
                font=dict(size=12)
            )
    
    # Update layout
    fig.update_layout(
        xaxis_title="",
        yaxis_title="",
        xaxis_showticklabels=False,
        yaxis_showticklabels=False,
        plot_bgcolor="white",
        height=500,
        margin=dict(l=40, r=40, t=30, b=40)
    )
    
    return fig

def create_segment_comparison_chart(df: pd.DataFrame, segment_col: str) -> go.Figure:
    """
    Create a comparison chart for different segments showing churn rate, 
    customer count, and revenue at risk.
    
    Parameters
    ----------
    df : pd.DataFrame
        Customer data
    segment_col : str
        Column name to use for segmentation
        
    Returns
    -------
    go.Figure
        Plotly figure object
    """
    if segment_col not in df.columns:
        # Return empty figure if segment_col is not available
        return go.Figure().update_layout(
            title="Segment data not available",
            height=400
        )
    
    # Group by segment and calculate metrics
    segment_df = df.groupby(segment_col).agg({
        'CustomerID': 'count',
        'churn_probability': 'mean',
        'predicted_churn': 'mean'
    }).reset_index()
    
    segment_df = segment_df.rename(columns={
        'CustomerID': 'customer_count',
        'predicted_churn': 'churn_rate',
        segment_col: 'segment'
    })
    
    # Calculate revenue at risk
    if 'annual_revenue' in df.columns:
        revenue_by_segment = df.groupby(segment_col)['annual_revenue'].sum().reset_index()
        revenue_by_segment = revenue_by_segment.rename(columns={segment_col: 'segment'})
        segment_df = pd.merge(segment_df, revenue_by_segment, on='segment', how='left')
        segment_df['revenue_at_risk'] = segment_df['annual_revenue'] * segment_df['churn_rate']
    else:
        # Use placeholder values
        segment_df['annual_revenue'] = segment_df['customer_count'] * 500
        segment_df['revenue_at_risk'] = segment_df['annual_revenue'] * segment_df['churn_rate']
    
    # Sort by churn rate for better visualization
    segment_df = segment_df.sort_values('churn_rate', ascending=False)
    
    # Create figure with two y-axes
    fig = go.Figure()
    
    # Add churn rate bars
    fig.add_trace(go.Bar(
        x=segment_df['segment'],
        y=segment_df['churn_rate'],
        name='Churn Rate',
        marker_color=COLORS['accent'],
        opacity=0.8,
        yaxis='y'
    ))
    
    # Add customer count bars
    fig.add_trace(go.Bar(
        x=segment_df['segment'],
        y=segment_df['customer_count'],
        name='Customer Count',
        marker_color=COLORS['primary'],
        opacity=0.8,
        yaxis='y2'
    ))
    
    # Add markers for revenue at risk
    fig.add_trace(go.Scatter(
        x=segment_df['segment'],
        y=segment_df['revenue_at_risk'] / 1000,  # Convert to thousands for better display
        mode='markers',
        name='Revenue at Risk ($K)',
        marker=dict(
            size=segment_df['revenue_at_risk'] / segment_df['revenue_at_risk'].max() * 30 + 10,
            color=COLORS['secondary'],
            line=dict(width=1, color='#333')
        ),
        yaxis='y3'
    ))
    
    # Update layout with three y-axes
    fig.update_layout(
        yaxis=dict(
            title="Churn Rate",
            titlefont=dict(color=COLORS['accent']),
            tickfont=dict(color=COLORS['accent']),
            tickformat=".0%",
            side="left",
            range=[0, max(segment_df['churn_rate']) * 1.1]
        ),
        yaxis2=dict(
            title="Customer Count",
            titlefont=dict(color=COLORS['primary']),
            tickfont=dict(color=COLORS['primary']),
            anchor="free",
            overlaying="y",
            side="right",
            position=1.0,
            range=[0, max(segment_df['customer_count']) * 1.1]
        ),
        yaxis3=dict(
            title="Revenue at Risk ($K)",
            titlefont=dict(color=COLORS['secondary']),
            tickfont=dict(color=COLORS['secondary']),
            anchor="free",
            overlaying="y",
            side="right",
            position=0.9,
            range=[0, max(segment_df['revenue_at_risk'] / 1000) * 1.1]
        ),
        barmode='group',
        height=400,
        margin=dict(l=40, r=100, t=30, b=80),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5
        ),
        plot_bgcolor="white"
    )
    
    return fig

###########################################
# Churn Drivers Tab Visualizations
###########################################

def create_feature_importance_chart(features: Dict[str, float]) -> go.Figure:
    """
    Create a horizontal bar chart showing feature importance.
    
    Parameters
    ----------
    features : Dict[str, float]
        Dictionary mapping feature names to importance values
        
    Returns
    -------
    go.Figure
        Plotly figure object
    """
    # Create dataframe for plotting
    fi_df = pd.DataFrame({
        'Feature': list(features.keys()),
        'Importance': list(features.values())
    })
    
    # Sort by importance
    fi_df = fi_df.sort_values('Importance', ascending=True)
    
    # Create horizontal bar chart
    fig = go.Figure()
    
    # Add bars
    fig.add_trace(go.Bar(
        y=fi_df['Feature'].str.replace('_', ' ').str.title(),
        x=fi_df['Importance'],
        orientation='h',
        marker_color=COLORS['primary']
    ))
    
    # Update layout
    fig.update_layout(
        xaxis_title="Relative Importance",
        yaxis_title="",
        margin=dict(l=20, r=20, t=20, b=20),
        height=400,
        plot_bgcolor="white"
    )
    
    return fig

def create_partial_dependence_plot(
    df: pd.DataFrame, 
    feature: str
) -> Tuple[go.Figure, str]:
    """
    Create a partial dependence plot showing how a feature affects churn probability.
    
    Parameters
    ----------
    df : pd.DataFrame
        Customer data
    feature : str
        Feature name to analyze
        
    Returns
    -------
    Tuple[go.Figure, str]
        Plotly figure object and insight text
    """
    if feature not in df.columns:
        # Return empty figure if feature is not available
        fig = go.Figure()
        fig.add_annotation(
            x=0.5, y=0.5,
            text=f"Feature '{feature}' not available in dataset",
            showarrow=False,
            font=dict(size=14)
        )
        
        insight = "No data available for this feature."
        return fig, insight
    
    # Check if categorical or numerical
    is_categorical = df[feature].dtype == 'object' or len(df[feature].unique()) < 10
    
    if is_categorical:
        # Create categorical partial dependence plot
        pdp_df = df.groupby(feature).agg({
            'churn_probability': ['mean', 'std', 'count']
        }).reset_index()
        
        # Flatten multi-level columns
        pdp_df.columns = ['_'.join(col).strip('_') for col in pdp_df.columns.values]
        
        # Sort by churn probability
        pdp_df = pdp_df.sort_values('churn_probability_mean', ascending=False)
        
        # Calculate error bars
        pdp_df['error_upper'] = pdp_df['churn_probability_mean'] + pdp_df['churn_probability_std'] / np.sqrt(pdp_df['churn_probability_count'])
        pdp_df['error_lower'] = pdp_df['churn_probability_mean'] - pdp_df['churn_probability_std'] / np.sqrt(pdp_df['churn_probability_count'])
        
        # Ensure error bounds are within [0, 1]
        pdp_df['error_upper'] = pdp_df['error_upper'].clip(upper=1)
        pdp_df['error_lower'] = pdp_df['error_lower'].clip(lower=0)
        
        # Create bar chart with error bars
        fig = go.Figure()
        
        # Add bars
        fig.add_trace(go.Bar(
            x=pdp_df[feature],
            y=pdp_df['churn_probability_mean'],
            marker_color=COLORS['primary'],
            name='Average Churn Probability'
        ))
        
        # Add error bars
        fig.add_trace(go.Scatter(
            x=pdp_df[feature],
            y=pdp_df['churn_probability_mean'],
            error_y=dict(
                type='data',
                symmetric=False,
                array=pdp_df['error_upper'] - pdp_df['churn_probability_mean'],
                arrayminus=pdp_df['churn_probability_mean'] - pdp_df['error_lower']
            ),
            mode='markers',
            marker=dict(color='rgba(0,0,0,0)'),
            showlegend=False
        ))
        
        # Create insight text
        highest_category = pdp_df.iloc[0][feature]
        highest_prob = pdp_df.iloc[0]['churn_probability_mean']
        lowest_category = pdp_df.iloc[-1][feature]
        lowest_prob = pdp_df.iloc[-1]['churn_probability_mean']
        
        insight = f"Customers with {feature.replace('_', ' ')} of {highest_category} have a {highest_prob:.1%} average churn probability, which is {highest_prob/lowest_prob:.1f}x higher than those with {feature.replace('_', ' ')} of {lowest_category} ({lowest_prob:.1%})."
        
    else:
        # Create numerical partial dependence plot
        # Bin the numerical feature
        bins = 10
        df['binned'] = pd.cut(df[feature], bins=bins)
        
        # Calculate mean churn probability for each bin
        pdp_df = df.groupby('binned').agg({
            'churn_probability': ['mean', 'std', 'count'],
            feature: 'mean'
        }).reset_index()
        
        # Flatten multi-level columns
        pdp_df.columns = ['binned'] + ['_'.join(col).strip('_') for col in pdp_df.columns[1:].values]
        
        # Calculate error bars
        pdp_df['error_upper'] = pdp_df['churn_probability_mean'] + pdp_df['churn_probability_std'] / np.sqrt(pdp_df['churn_probability_count'])
        pdp_df['error_lower'] = pdp_df['churn_probability_mean'] - pdp_df['churn_probability_std'] / np.sqrt(pdp_df['churn_probability_count'])
        
        # Ensure error bounds are within [0, 1]
        pdp_df['error_upper'] = pdp_df['error_upper'].clip(upper=1)
        pdp_df['error_lower'] = pdp_df['error_lower'].clip(lower=0)
        
        # Create line chart
        fig = go.Figure()
        
        # Add line
        fig.add_trace(go.Scatter(
            x=pdp_df[f'{feature}_mean'],
            y=pdp_df['churn_probability_mean'],
            mode='lines+markers',
            marker=dict(color=COLORS['primary'], size=8),
            line=dict(color=COLORS['primary'], width=2),
            name='Average Churn Probability'
        ))
        
        # Add confidence band
        fig.add_trace(go.Scatter(
            x=pdp_df[f'{feature}_mean'],
            y=pdp_df['error_upper'],
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            name='Upper Bound'
        ))
        
        fig.add_trace(go.Scatter(
            x=pdp_df[f'{feature}_mean'],
            y=pdp_df['error_lower'],
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor=f'rgba({COLORS["primary"].replace("rgb(","").replace(")","")},.2)',
            showlegend=False,
            name='Lower Bound'
        ))
        
        # Find where churn is highest and lowest
        highest_idx = pdp_df['churn_probability_mean'].idxmax()
        lowest_idx = pdp_df['churn_probability_mean'].idxmin()
        highest_value = pdp_df.iloc[highest_idx][f'{feature}_mean']
        lowest_value = pdp_df.iloc[lowest_idx][f'{feature}_mean']
        highest_prob = pdp_df.iloc[highest_idx]['churn_probability_mean']
        lowest_prob = pdp_df.iloc[lowest_idx]['churn_probability_mean']
        
        # Determine relationship type
        first_bin_prob = pdp_df.iloc[0]['churn_probability_mean']
        last_bin_prob = pdp_df.iloc[-1]['churn_probability_mean']
        
        if abs(first_bin_prob - last_bin_prob) < 0.05:
            relationship = "non-linear or complex"
        elif first_bin_prob > last_bin_prob:
            relationship = "decreasing"
        else:
            relationship = "increasing"
        
        insight = f"There is a {relationship} relationship between {feature.replace('_', ' ')} and churn probability. Churn risk is highest ({highest_prob:.1%}) when {feature.replace('_', ' ')} is around {highest_value:.1f}, and lowest ({lowest_prob:.1%}) when {feature.replace('_', ' ')} is around {lowest_value:.1f}."
    
    # Update layout
    fig.update_layout(
        xaxis_title=feature.replace('_', ' ').title(),
        yaxis_title="Churn Probability",
        yaxis_tickformat=".0%",
        margin=dict(l=40, r=40, t=30, b=40),
        height=400,
        plot_bgcolor="white"
    )
    
    return fig, insight

def create_feature_correlation_heatmap(df: pd.DataFrame, features: List[str]) -> go.Figure:
    """
    Create a correlation heatmap for the most important features.
    
    Parameters
    ----------
    df : pd.DataFrame
        Customer data
    features : List[str]
        List of feature names to include
        
    Returns
    -------
    go.Figure
        Plotly figure object
    """
    # Get correlation matrix for these features if they exist in the data
    existing_features = [f for f in features if f in df.columns]
    
    if len(existing_features) > 1:
        # Calculate correlation matrix
        corr_matrix = df[existing_features].corr()
        
        # Create heatmap
        z = corr_matrix.values
        x = [f.replace('_', ' ').title() for f in corr_matrix.columns]
        y = [f.replace('_', ' ').title() for f in corr_matrix.index]
        
        # Custom colorscale
        colorscale = [
            [0, 'rgb(0, 0, 255)'],       # Strong negative correlation (blue)
            [0.25, 'rgb(180, 180, 255)'], # Weak negative correlation (light blue)
            [0.5, 'rgb(255, 255, 255)'],  # No correlation (white)
            [0.75, 'rgb(255, 180, 180)'], # Weak positive correlation (light red)
            [1, 'rgb(255, 0, 0)']        # Strong positive correlation (red)
        ]
        
        fig = ff.create_annotated_heatmap(
            z=z,
            x=x,
            y=y,
            annotation_text=np.around(z, decimals=2),
            colorscale=colorscale,
            hoverinfo='z',
            showscale=True
        )
        
        # Update layout
        fig.update_layout(
            margin=dict(l=40, r=40, t=30, b=40),
            height=400
        )
    else:
        # Create placeholder if not enough features
        fig = go.Figure()
        fig.add_annotation(
            x=0.5, y=0.5,
            text="Not enough features available for correlation analysis",
            showarrow=False,
            font=dict(size=14)
        )
        
        fig.update_layout(
            height=400
        )
    
    return fig

###########################################
# Intervention Simulator Tab Visualizations
###########################################

def create_roi_projection(
    total_cost: float,
    monthly_revenue: float,
    months: List[int] = None
) -> go.Figure:
    """
    Create a ROI projection chart showing cumulative costs vs. benefits over time.
    
    Parameters
    ----------
    total_cost : float
        Total implementation cost
    monthly_revenue : float
        Monthly revenue protected
    months : List[int], optional
        List of months to display, by default 1-12
        
    Returns
    -------
    go.Figure
        Plotly figure object
    """
    if months is None:
        months = list(range(1, 13))
    
    # Create figure
    fig = go.Figure()
    
    # Calculate cumulative costs and benefits over time
    cumulative_cost = [total_cost] * len(months)
    cumulative_benefit = [monthly_revenue * i for i in months]
    
    # Add cost line
    fig.add_trace(go.Scatter(
        x=months,
        y=cumulative_cost,
        mode='lines',
        name='Intervention Cost',
        line=dict(color=COLORS['accent'], width=2)
    ))
    
    # Add benefit line
    fig.add_trace(go.Scatter(
        x=months,
        y=cumulative_benefit,
        mode='lines',
        name='Cumulative Revenue Protected',
        line=dict(color=COLORS['success'], width=2)
    ))
    
    # Add breakeven point
    if max(cumulative_benefit) > total_cost:
        # Find when benefit exceeds cost
        for i, benefit in enumerate(cumulative_benefit):
            if benefit >= total_cost:
                breakeven_month = months[i]
                # Add annotation
                fig.add_annotation(
                    x=breakeven_month,
                    y=total_cost,
                    text=f"Breakeven at month {breakeven_month}",
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor="#333",
                    arrowsize=1,
                    arrowwidth=1
                )
                break
    
    # Update layout
    fig.update_layout(
        xaxis_title="Months",
        yaxis_title="Amount ($)",
        margin=dict(l=40, r=40, t=20, b=40),
        height=300,
        plot_bgcolor="white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5
        )
    )
    
    return fig

###########################################
# Model Performance Tab Visualizations
###########################################

def create_roc_curve(auc_value: float = 0.91) -> go.Figure:
    """
    Create a ROC curve visualization.
    
    In a real implementation, these values would come from model evaluation.
    Here we use placeholder data for demonstration.
    
    Parameters
    ----------
    auc_value : float
        AUC-ROC value to display
        
    Returns
    -------
    go.Figure
        Plotly figure object
    """
    # Create figure
    fig = go.Figure()
    
    # Add diagonal reference line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        line=dict(color='gray', dash='dash'),
        name='Random'
    ))
    
    # Add ROC curve - using placeholder data
    fpr = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    tpr = [0, 0.4, 0.55, 0.68, 0.75, 0.8, 0.85, 0.9, 0.94, 0.98, 1.0]
    
    fig.add_trace(go.Scatter(
        x=fpr,
        y=tpr,
        mode='lines',
        line=dict(color=COLORS['primary'], width=2),
        name=f"Ensemble Model (AUC = {auc_value})"
    ))
    
    # Update layout
    fig.update_layout(
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        margin=dict(l=40, r=40, t=20, b=40),
        plot_bgcolor="white",
        height=300,
        legend=dict(
            yanchor="bottom",
            y=0.01,
            xanchor="right",
            x=0.99
        )
    )
    
    # Set axis ranges
    fig.update_xaxes(range=[0, 1], constrain="domain")
    fig.update_yaxes(range=[0, 1], scaleanchor="x", scaleratio=1)
    
    return fig

def create_pr_curve(auc_pr_value: float = 0.84, churn_rate: float = 0.24) -> go.Figure:
    """
    Create a Precision-Recall curve visualization.
    
    In a real implementation, these values would come from model evaluation.
    Here we use placeholder data for demonstration.
    
    Parameters
    ----------
    auc_pr_value : float
        AUC-PR value to display
    churn_rate : float
        Overall churn rate (baseline precision)
        
    Returns
    -------
    go.Figure
        Plotly figure object
    """
    # Create figure
    fig = go.Figure()
    
    # Add baseline reference line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[churn_rate, churn_rate],
        mode='lines',
        line=dict(color='gray', dash='dash'),
        name='Baseline'
    ))
    
    # Add PR curve - using placeholder data
    recall = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    precision = [1.0, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, churn_rate]
    
    fig.add_trace(go.Scatter(
        x=recall,
        y=precision,
        mode='lines',
        line=dict(color=COLORS['primary'], width=2),
        name=f"Ensemble Model (AUC = {auc_pr_value})"
    ))
    
    # Update layout
    fig.update_layout(
        xaxis_title="Recall",
        yaxis_title="Precision",
        margin=dict(l=40, r=40, t=20, b=40),
        plot_bgcolor="white",
        height=300,
        legend=dict(
            yanchor="bottom",
            y=0.01,
            xanchor="right",
            x=0.99
        )
    )
    
    # Set axis ranges
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])
    
    return fig

def create_confusion_matrix(
    TP: int,
    FP: int, 
    TN: int, 
    FN: int, 
    validation_size: int
) -> go.Figure:
    """
    Create a confusion matrix visualization.
    
    Parameters
    ----------
    TP : int
        True Positives
    FP : int
        False Positives
    TN : int
        True Negatives
    FN : int
        False Negatives
    validation_size : int
        Total size of validation set
        
    Returns
    -------
    go.Figure
        Plotly figure object
    """
    # Create confusion matrix
    z = [[TN, FP], [FN, TP]]
    x = ['Predicted No Churn', 'Predicted Churn']
    y = ['Actual No Churn', 'Actual Churn']
    
    # Create annotations
    annotations = [
        [f"{TN}<br>({TN/validation_size:.1%})", f"{FP}<br>({FP/validation_size:.1%})"],
        [f"{FN}<br>({FN/validation_size:.1%})", f"{TP}<br>({TP/validation_size:.1%})"]
    ]
    
    # Create heatmap figure
    fig = ff.create_annotated_heatmap(
        z=z,
        x=x,
        y=y,
        annotation_text=annotations,
        colorscale=[[0, '#edf8e9'], [1, COLORS['primary']]],
        showscale=False
    )
    
    # Update layout
    fig.update_layout(
        margin=dict(l=40, r=40, t=30, b=40),
        plot_bgcolor="white",
        height=300,
        xaxis=dict(side='bottom'),
        yaxis=dict(side='left', autorange='reversed')
    )
    
    return fig

def create_calibration_curve() -> go.Figure:
    """
    Create a calibration curve visualization.
    
    In a real implementation, these values would come from model evaluation.
    Here we use placeholder data for demonstration.
    
    Returns
    -------
    go.Figure
        Plotly figure object
    """
    # Create figure
    fig = go.Figure()
    
    # Add perfect calibration reference line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        line=dict(color='gray', dash='dash'),
        name='Perfect Calibration'
    ))
    
    # Create placeholder calibration curve data
    prob_bins = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
    # Slightly off perfect calibration for more realism
    actual_freqs = [0.02, 0.11, 0.22, 0.38, 0.49, 0.57, 0.68, 0.78, 0.82, 0.93]
    
    fig.add_trace(go.Scatter(
        x=prob_bins,
        y=actual_freqs,
        mode='lines+markers',
        line=dict(color=COLORS['primary'], width=2),
        marker=dict(size=8, color=COLORS['primary']),
        name='Ensemble Model'
    ))
    
    # Update layout
    fig.update_layout(
        xaxis_title="Predicted Probability",
        yaxis_title="Observed Frequency",
        margin=dict(l=40, r=40, t=20, b=40),
        plot_bgcolor="white",
        height=300,
        legend=dict(
            yanchor="bottom",
            y=0.01,
            xanchor="right",
            x=0.99
        )
    )
    
    # Set axis ranges
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])
    
    return fig

def create_model_comparison_chart(model_comparison: Dict[str, float]) -> go.Figure:
    """
    Create a model comparison bar chart.
    
    Parameters
    ----------
    model_comparison : Dict[str, float]
        Dictionary mapping model names to performance values
        
    Returns
    -------
    go.Figure
        Plotly figure object
    """
    # Create dataframe for plotting
    comparison_df = pd.DataFrame({
        'Model': list(model_comparison.keys()),
        'AUC': list(model_comparison.values())
    })
    
    # Sort by performance
    comparison_df = comparison_df.sort_values('AUC', ascending=False)
    
    # Create horizontal bar chart
    fig = go.Figure()
    
    # Add bars with color highlighting for ensemble
    colors = [COLORS['accent'] if model == 'Ensemble' else COLORS['primary'] for model in comparison_df['Model']]
    
    fig.add_trace(go.Bar(
        y=comparison_df['Model'],
        x=comparison_df['AUC'],
        orientation='h',
        marker_color=colors,
        text=comparison_df['AUC'].map(lambda x: f"{x:.3f}"),
        textposition='inside'
    ))
    
    # Update layout
    fig.update_layout(
        xaxis_title="AUC-ROC Score",
        yaxis_title="",
        margin=dict(l=20, r=20, t=20, b=20),
        height=300,
        plot_bgcolor="white",
        xaxis=dict(range=[0.7, max(comparison_df['AUC']) * 1.05])
    )
    
    return fig

###########################################
# Utility Functions
###########################################

def format_currency(value: float) -> str:
    """Format a value as currency."""
    return f"${value:,.2f}"

def format_percentage(value: float) -> str:
    """Format a value as percentage."""
    return f"{value:.1%}"

def format_number(value: float) -> str:
    """Format a value as a number with thousands separator."""
    return f"{value:,.0f}"

def get_color_scale(values: List[float], colorscale_name: str = 'Blues') -> List[str]:
    """
    Generate a list of colors for values based on a colorscale.
    
    Parameters
    ----------
    values : List[float]
        List of values to map to colors
    colorscale_name : str, optional
        Name of the Plotly colorscale, by default 'Blues'
    
    Returns
    -------
    List[str]
        List of color strings
    """
    if len(values) == 0:
        return []
    
    min_val = min(values)
    max_val = max(values)
    
    if max_val == min_val:
        # If all values are the same, return middle color
        normalized = [0.5] * len(values)
    else:
        # Normalize values to [0, 1]
        normalized = [(v - min_val) / (max_val - min_val) for v in values]
    
    # Create a colorscale
    colorscale = px.colors.sequential.__getattribute__(colorscale_name)
    
    # Map normalized values to colors
    colors = []
    for norm_val in normalized:
        idx = int(norm_val * (len(colorscale) - 1))
        colors.append(colorscale[idx])
    
    return colors