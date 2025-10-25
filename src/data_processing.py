import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")


class DataProcessor:
    def __init__(self):
        self.earnings_df = None
        self.transactions_df = None
        self.users_df = None
        self.financial_products_df = None

    def load_datasets(self, data_dir="data"):
        """Load all CSV datasets"""
        try:
            self.earnings_df = pd.read_csv(f"{data_dir}/sample_gig_earnings.csv")
            self.transactions_df = pd.read_csv(f"{data_dir}/sample_transactions.csv")
            self.users_df = pd.read_csv(f"{data_dir}/sample_user_profiles.csv")
            self.financial_products_df = pd.read_csv(
                f"{data_dir}/financial_products.csv"
            )

            # Convert date columns
            self.earnings_df["date"] = pd.to_datetime(self.earnings_df["date"])
            self.transactions_df["date"] = pd.to_datetime(self.transactions_df["date"])

            print("✅ All datasets loaded successfully!")
            return True

        except FileNotFoundError as e:
            print(f"❌ Error loading datasets: {e}")
            print("🔧 Please run: python generate_datasets.py")
            return False

    def get_user_summary(self, user_id):
        """Get comprehensive summary for a specific user"""
        if self.earnings_df is None:
            print("❌ Datasets not loaded. Call load_datasets() first.")
            return None

        # User profile
        user_profile = self.users_df[self.users_df["user_id"] == user_id].iloc[0]

        # Earnings summary
        user_earnings = self.earnings_df[self.earnings_df["user_id"] == user_id]
        earnings_summary = {
            "total_earnings": user_earnings["earnings"].sum(),
            "avg_daily_earnings": user_earnings["earnings"].mean(),
            "total_hours": user_earnings["hours_worked"].sum(),
            "avg_rating": user_earnings["rating"].mean(),
            "active_platforms": user_earnings["platform"].nunique(),
            "working_days": len(user_earnings),
        }

        # Transaction summary
        user_transactions = self.transactions_df[
            self.transactions_df["user_id"] == user_id
        ]
        transaction_summary = {
            "total_expenses": abs(user_transactions["amount"].sum()),
            "avg_daily_expenses": abs(user_transactions["amount"].mean()),
            "current_balance": user_transactions["account_balance"].iloc[-1],
            "min_balance": user_transactions["account_balance"].min(),
            "transaction_count": len(user_transactions),
        }

        return {
            "profile": user_profile.to_dict(),
            "earnings": earnings_summary,
            "transactions": transaction_summary,
        }

    def calculate_income_volatility(self, user_id):
        """Calculate income volatility metrics"""
        user_earnings = self.earnings_df[self.earnings_df["user_id"] == user_id]

        if len(user_earnings) == 0:
            return 0

        daily_earnings = user_earnings.groupby("date")["earnings"].sum()

        # Coefficient of variation
        mean_earnings = daily_earnings.mean()
        std_earnings = daily_earnings.std()

        if mean_earnings == 0:
            return 1  # High volatility if no earnings

        cv = std_earnings / mean_earnings
        return min(cv, 2)  # Cap at 2 for extreme cases

    def get_spending_patterns(self, user_id):
        """Analyze spending patterns by category"""
        user_transactions = self.transactions_df[
            self.transactions_df["user_id"] == user_id
        ]

        # Group by category (only expenses - negative amounts)
        expenses = user_transactions[user_transactions["amount"] < 0].copy()
        expenses["amount"] = abs(expenses["amount"])  # Make positive for analysis

        category_spending = (
            expenses.groupby("category")["amount"]
            .agg(["sum", "mean", "count"])
            .round(2)
        )

        return category_spending.to_dict("index")

    def detect_financial_stress_indicators(self, user_id):
        """Detect early warning signs of financial stress"""
        user_transactions = self.transactions_df[
            self.transactions_df["user_id"] == user_id
        ]

        indicators = {}

        # Low balance frequency
        low_balance_days = (user_transactions["account_balance"] < 100).sum()
        indicators["low_balance_frequency"] = low_balance_days / len(user_transactions)

        # Negative balance occurrences
        negative_balance_days = (user_transactions["account_balance"] < 0).sum()
        indicators["negative_balance_frequency"] = negative_balance_days / len(
            user_transactions
        )

        # Large expense volatility
        expenses = user_transactions[user_transactions["amount"] < 0]["amount"]
        if len(expenses) > 0:
            indicators["expense_volatility"] = abs(expenses.std() / expenses.mean())
        else:
            indicators["expense_volatility"] = 0

        # Recent balance trend (last 30 days)
        recent_transactions = user_transactions.tail(30)
        if len(recent_transactions) > 1:
            balance_trend = (
                recent_transactions["account_balance"].iloc[-1]
                - recent_transactions["account_balance"].iloc[0]
            )
            indicators["recent_balance_trend"] = balance_trend
        else:
            indicators["recent_balance_trend"] = 0

        return indicators

    def get_platform_performance(self, user_id):
        """Analyze performance across different gig platforms"""
        user_earnings = self.earnings_df[self.earnings_df["user_id"] == user_id]

        platform_stats = (
            user_earnings.groupby("platform")
            .agg(
                {
                    "earnings": ["sum", "mean", "count"],
                    "hours_worked": ["sum", "mean"],
                    "rating": "mean",
                }
            )
            .round(2)
        )

        # Flatten column names
        platform_stats.columns = [
            "_".join(col).strip() for col in platform_stats.columns
        ]

        # Calculate hourly rate for each platform
        platform_stats["hourly_rate"] = (
            platform_stats["earnings_sum"] / platform_stats["hours_worked_sum"]
        ).round(2)

        return platform_stats.to_dict("index")

    def export_processed_data(self, output_dir="processed_data"):
        """Export processed data for ML models"""
        import os

        os.makedirs(output_dir, exist_ok=True)

        # Create aggregated features for all users
        all_users = self.users_df["user_id"].unique()
        processed_data = []

        for user_id in all_users:
            try:
                user_summary = self.get_user_summary(user_id)
                volatility = self.calculate_income_volatility(user_id)
                stress_indicators = self.detect_financial_stress_indicators(user_id)

                # Combine all features
                features = {
                    "user_id": user_id,
                    "income_volatility": volatility,
                    **user_summary["profile"],
                    **user_summary["earnings"],
                    **user_summary["transactions"],
                    **stress_indicators,
                }

                processed_data.append(features)

            except Exception as e:
                print(f"⚠️ Error processing user {user_id}: {e}")
                continue

        # Save processed features
        processed_df = pd.DataFrame(processed_data)
        processed_df.to_csv(f"{output_dir}/processed_features.csv", index=False)

        print(f"✅ Processed data exported to {output_dir}/processed_features.csv")
        return processed_df


if __name__ == "__main__":
    # Test data processing
    processor = DataProcessor()

    if processor.load_datasets():
        print("\n📊 Dataset Statistics:")
        print(f"Users: {len(processor.users_df)}")
        print(f"Earnings records: {len(processor.earnings_df)}")
        print(f"Transaction records: {len(processor.transactions_df)}")

        # Test with first user
        first_user = processor.users_df["user_id"].iloc[0]
        print(f"\n🔍 Sample analysis for {first_user}:")

        summary = processor.get_user_summary(first_user)
        print(f"Total earnings: ${summary['earnings']['total_earnings']:.2f}")
        print(
            f"Income volatility: {processor.calculate_income_volatility(first_user):.3f}"
        )

        # Export processed data
        processed_df = processor.export_processed_data()
        print(f"\n✅ Processed {len(processed_df)} user profiles successfully!")
