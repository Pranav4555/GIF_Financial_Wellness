import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")


class DashboardComponents:
    """Custom dashboard components for the gig worker financial wellness app"""

    def __init__(self):
        self.colors = {
            "primary": "#1f77b4",
            "secondary": "#ff7f0e",
            "success": "#2ca02c",
            "warning": "#d62728",
            "info": "#9467bd",
            "background": "#f0f2f6",
        }

    def create_wellness_gauge(self, score, title="Financial Wellness Score"):
        """Create an animated gauge chart for wellness score"""

        # Determine color based on score
        if score >= 80:
            color = self.colors["success"]
            status = "Excellent"
        elif score >= 60:
            color = self.colors["primary"]
            status = "Good"
        elif score >= 40:
            color = self.colors["secondary"]
            status = "Fair"
        else:
            color = self.colors["warning"]
            status = "Needs Improvement"

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number+delta",
                value=score,
                domain={"x": [0, 1], "y": [0, 1]},
                title={
                    "text": f"{title}<br><span style='font-size:0.8em;color:gray'>{status}</span>"
                },
                delta={"reference": 50, "position": "top"},
                gauge={
                    "axis": {
                        "range": [None, 100],
                        "tickwidth": 1,
                        "tickcolor": "darkblue",
                    },
                    "bar": {"color": color, "thickness": 0.3},
                    "bgcolor": "white",
                    "borderwidth": 2,
                    "bordercolor": "gray",
                    "steps": [
                        {"range": [0, 40], "color": "#ffcccc"},
                        {"range": [40, 60], "color": "#fff3cd"},
                        {"range": [60, 80], "color": "#d4edda"},
                        {"range": [80, 100], "color": "#d1ecf1"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": 90,
                    },
                },
            )
        )

        fig.update_layout(
            height=350,
            font={"color": "darkblue", "family": "Arial"},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )

        return fig

    def create_income_trend_chart(self, user_earnings):
        """Create an enhanced income trend visualization"""
        if len(user_earnings) == 0:
            return self._empty_chart("No earnings data available")

        user_earnings["date"] = pd.to_datetime(user_earnings["date"])

        # Daily earnings aggregation
        daily_earnings = (
            user_earnings.groupby("date")
            .agg({"earnings": "sum", "hours_worked": "sum"})
            .reset_index()
        )

        # Calculate 7-day moving average
        daily_earnings["earnings_ma7"] = (
            daily_earnings["earnings"].rolling(window=7, min_periods=1).mean()
        )

        # Create subplot with secondary y-axis
        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=("Daily Earnings & Trend", "Hours Worked"),
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3],
        )

        # Daily earnings bars
        fig.add_trace(
            go.Bar(
                x=daily_earnings["date"],
                y=daily_earnings["earnings"],
                name="Daily Earnings",
                marker_color=self.colors["primary"],
                opacity=0.7,
            ),
            row=1,
            col=1,
        )

        # Moving average line
        fig.add_trace(
            go.Scatter(
                x=daily_earnings["date"],
                y=daily_earnings["earnings_ma7"],
                mode="lines",
                name="7-Day Average",
                line=dict(color=self.colors["warning"], width=3),
            ),
            row=1,
            col=1,
        )

        # Hours worked
        fig.add_trace(
            go.Scatter(
                x=daily_earnings["date"],
                y=daily_earnings["hours_worked"],
                mode="lines+markers",
                name="Hours Worked",
                line=dict(color=self.colors["success"], width=2),
                marker=dict(size=4),
            ),
            row=2,
            col=1,
        )

        fig.update_layout(
            height=500,
            title_text="Income and Activity Trends",
            showlegend=True,
            hovermode="x unified",
        )

        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Earnings ($)", row=1, col=1)
        fig.update_yaxes(title_text="Hours", row=2, col=1)

        return fig

    def create_platform_breakdown(self, user_earnings):
        """Create platform earnings breakdown with enhanced visuals"""
        if len(user_earnings) == 0:
            return self._empty_chart("No platform data available")

        platform_data = (
            user_earnings.groupby("platform")
            .agg(
                {
                    "earnings": ["sum", "count", "mean"],
                    "hours_worked": "sum",
                    "rating": "mean",
                }
            )
            .round(2)
        )

        platform_data.columns = [
            "total_earnings",
            "jobs_count",
            "avg_earnings",
            "total_hours",
            "avg_rating",
        ]
        platform_data = platform_data.reset_index()

        # Calculate hourly rate
        platform_data["hourly_rate"] = (
            platform_data["total_earnings"] / platform_data["total_hours"]
        ).round(2)

        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Earnings by Platform",
                "Jobs Completed",
                "Average Rating",
                "Hourly Rate",
            ),
            specs=[
                [{"type": "pie"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "bar"}],
            ],
        )

        # Pie chart for earnings
        fig.add_trace(
            go.Pie(
                labels=platform_data["platform"],
                values=platform_data["total_earnings"],
                hole=0.3,
                textinfo="label+percent",
                textposition="outside",
            ),
            row=1,
            col=1,
        )

        # Bar chart for jobs count
        fig.add_trace(
            go.Bar(
                x=platform_data["platform"],
                y=platform_data["jobs_count"],
                name="Jobs",
                marker_color=self.colors["secondary"],
            ),
            row=1,
            col=2,
        )

        # Bar chart for ratings
        fig.add_trace(
            go.Bar(
                x=platform_data["platform"],
                y=platform_data["avg_rating"],
                name="Rating",
                marker_color=self.colors["success"],
            ),
            row=2,
            col=1,
        )

        # Bar chart for hourly rate
        fig.add_trace(
            go.Bar(
                x=platform_data["platform"],
                y=platform_data["hourly_rate"],
                name="Hourly Rate",
                marker_color=self.colors["info"],
            ),
            row=2,
            col=2,
        )

        fig.update_layout(
            height=600, showlegend=False, title_text="Platform Performance Analysis"
        )

        # Update y-axis labels
        fig.update_yaxes(title_text="Jobs", row=1, col=2)
        fig.update_yaxes(title_text="Rating (1-5)", row=2, col=1)
        fig.update_yaxes(title_text="$ per Hour", row=2, col=2)

        return fig

    def create_expense_analysis(self, user_transactions):
        """Create expense analysis visualization"""
        if len(user_transactions) == 0:
            return self._empty_chart("No transaction data available")

        # Filter expenses (negative amounts)
        expenses = user_transactions[user_transactions["amount"] < 0].copy()
        expenses["amount"] = abs(expenses["amount"])

        # Group by category
        expense_summary = (
            expenses.groupby("category")
            .agg({"amount": ["sum", "count", "mean"]})
            .round(2)
        )

        expense_summary.columns = [
            "total_spent",
            "transaction_count",
            "avg_transaction",
        ]
        expense_summary = expense_summary.reset_index()
        expense_summary = expense_summary.sort_values("total_spent", ascending=False)

        # Create subplots
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Spending by Category", "Average Transaction Size"),
            specs=[[{"type": "bar"}, {"type": "bar"}]],
        )

        # Total spending by category
        fig.add_trace(
            go.Bar(
                x=expense_summary["category"],
                y=expense_summary["total_spent"],
                name="Total Spent",
                marker_color=self.colors["warning"],
                text=expense_summary["total_spent"],
                texttemplate="$%{text}",
                textposition="outside",
            ),
            row=1,
            col=1,
        )

        # Average transaction size
        fig.add_trace(
            go.Bar(
                x=expense_summary["category"],
                y=expense_summary["avg_transaction"],
                name="Avg Transaction",
                marker_color=self.colors["secondary"],
                text=expense_summary["avg_transaction"],
                texttemplate="$%{text}",
                textposition="outside",
            ),
            row=1,
            col=2,
        )

        fig.update_layout(height=400, title_text="Spending Analysis", showlegend=False)

        fig.update_xaxes(tickangle=45)
        fig.update_yaxes(title_text="Amount ($)", row=1, col=1)
        fig.update_yaxes(title_text="Avg Amount ($)", row=1, col=2)

        return fig

    def create_financial_health_radar(self, metrics):
        """Create radar chart for financial health dimensions"""

        categories = [
            "Income Stability",
            "Expense Management",
            "Emergency Fund",
            "Credit Health",
            "Investment Readiness",
            "Platform Diversity",
        ]

        # Normalize metrics to 0-5 scale for radar chart
        values = [
            min(5, metrics.get("income_stability", 2) * 5),
            min(5, metrics.get("expense_management", 2) * 5),
            min(5, metrics.get("emergency_fund", 2) * 5),
            min(5, metrics.get("credit_health", 2) * 5),
            min(5, metrics.get("investment_readiness", 2) * 5),
            min(5, metrics.get("platform_diversity", 2) * 5),
        ]

        fig = go.Figure()

        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=categories,
                fill="toself",
                fillcolor="rgba(31, 119, 180, 0.2)",
                line=dict(color=self.colors["primary"], width=2),
                name="Current Score",
            )
        )

        # Add target/ideal scores
        target_values = [4.5] * 6  # Target score for all categories
        fig.add_trace(
            go.Scatterpolar(
                r=target_values,
                theta=categories,
                fill="toself",
                fillcolor="rgba(46, 160, 44, 0.1)",
                line=dict(color=self.colors["success"], width=1, dash="dash"),
                name="Target Score",
            )
        )

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
            showlegend=True,
            title="Financial Health Radar",
            height=400,
        )

        return fig

    def create_prediction_confidence_chart(self, predictions, confidence_intervals):
        """Create chart showing predictions with confidence intervals"""

        dates = pd.date_range(start=datetime.now(), periods=len(predictions), freq="M")

        fig = go.Figure()

        # Add confidence interval as filled area
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=confidence_intervals["upper"],
                fill=None,
                mode="lines",
                line_color="rgba(0,0,0,0)",
                showlegend=False,
            )
        )

        fig.add_trace(
            go.Scatter(
                x=dates,
                y=confidence_intervals["lower"],
                fill="tonexty",
                mode="lines",
                line_color="rgba(0,0,0,0)",
                fillcolor="rgba(31, 119, 180, 0.2)",
                name="95% Confidence Interval",
            )
        )

        # Add prediction line
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=predictions,
                mode="lines+markers",
                line=dict(color=self.colors["primary"], width=3),
                marker=dict(size=6),
                name="Predicted Income",
            )
        )

        fig.update_layout(
            title="Income Prediction with Confidence Intervals",
            xaxis_title="Date",
            yaxis_title="Predicted Income ($)",
            height=400,
            hovermode="x unified",
        )

        return fig

    def create_metric_cards(self, metrics):
        """Create metric cards for dashboard"""

        cards_html = """
        <style>
        .metric-container {
            display: flex;
            gap: 20px;
            margin: 20px 0;
        }
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            flex: 1;
            text-align: center;
        }
        .metric-value {
            font-size: 2.5em;
            font-weight: bold;
            margin: 10px 0;
        }
        .metric-label {
            font-size: 0.9em;
            opacity: 0.9;
        }
        .metric-change {
            font-size: 0.8em;
            margin-top: 5px;
        }
        .positive { color: #4CAF50; }
        .negative { color: #f44336; }
        </style>
        """

        return cards_html

    def create_savings_goal_progress(
        self, current_savings, savings_goal, monthly_contribution
    ):
        """Create savings goal progress visualization"""

        progress_percentage = (
            (current_savings / savings_goal) * 100 if savings_goal > 0 else 0
        )
        months_to_goal = (
            max(0, (savings_goal - current_savings) / monthly_contribution)
            if monthly_contribution > 0
            else float("inf")
        )

        fig = go.Figure()

        # Progress bar
        fig.add_trace(
            go.Bar(
                x=[progress_percentage],
                y=["Savings Progress"],
                orientation="h",
                marker=dict(
                    color=(
                        self.colors["success"]
                        if progress_percentage >= 100
                        else self.colors["primary"]
                    )
                ),
                text=f"{progress_percentage:.1f}%",
                textposition="inside",
            )
        )

        fig.update_layout(
            title=f"Emergency Fund Goal: ${current_savings:,.0f} / ${savings_goal:,.0f}",
            xaxis=dict(range=[0, 100], title="Progress (%)"),
            height=150,
            showlegend=False,
        )

        if months_to_goal != float("inf"):
            fig.add_annotation(
                text=f"Est. completion: {months_to_goal:.1f} months",
                xref="paper",
                yref="paper",
                x=0.5,
                y=-0.3,
                showarrow=False,
            )

        return fig

    def _empty_chart(self, message):
        """Create empty chart with message"""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16, color="gray"),
        )
        fig.update_layout(
            height=300, xaxis={"visible": False}, yaxis={"visible": False}
        )
        return fig

    def display_feature_importance(
        self, feature_importance_dict, title="Model Feature Importance"
    ):
        """Display feature importance as horizontal bar chart"""

        # Sort features by importance
        sorted_features = sorted(
            feature_importance_dict.items(), key=lambda x: x[1], reverse=True
        )
        features, importance = zip(*sorted_features[:10])  # Top 10 features

        fig = go.Figure(
            go.Bar(
                x=importance,
                y=features,
                orientation="h",
                marker_color=self.colors["info"],
                text=[f"{imp:.3f}" for imp in importance],
                textposition="outside",
            )
        )

        fig.update_layout(
            title=title,
            xaxis_title="Importance Score",
            height=400,
            yaxis=dict(autorange="reversed"),  # Top feature at top
        )

        return fig

    def create_comparison_chart(self, user_metrics, benchmark_metrics):
        """Create comparison chart between user and benchmark"""

        categories = list(user_metrics.keys())
        user_values = list(user_metrics.values())
        benchmark_values = list(benchmark_metrics.values())

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                name="Your Score",
                x=categories,
                y=user_values,
                marker_color=self.colors["primary"],
            )
        )

        fig.add_trace(
            go.Bar(
                name="Benchmark",
                x=categories,
                y=benchmark_values,
                marker_color=self.colors["secondary"],
                opacity=0.7,
            )
        )

        fig.update_layout(
            title="Performance vs. Benchmark",
            xaxis_title="Metrics",
            yaxis_title="Score",
            barmode="group",
            height=400,
        )

        return fig


# Example usage and testing
if __name__ == "__main__":
    # Create sample data for testing
    dashboard = DashboardComponents()

    # Test wellness gauge
    test_score = 75
    gauge_fig = dashboard.create_wellness_gauge(test_score)
    print(f"Created wellness gauge for score: {test_score}")

    # Test with sample earnings data
    sample_earnings = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=30, freq="D"),
            "earnings": np.random.uniform(50, 200, 30),
            "hours_worked": np.random.uniform(2, 10, 30),
            "platform": np.random.choice(["uber", "doordash", "upwork"], 30),
            "rating": np.random.uniform(4.0, 5.0, 30),
        }
    )

    income_fig = dashboard.create_income_trend_chart(sample_earnings)
    platform_fig = dashboard.create_platform_breakdown(sample_earnings)

    print("Dashboard components created successfully!")
    print("Ready for integration with Streamlit app.")
