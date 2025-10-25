#!/usr/bin/env python3
"""
Quick fix script to resolve common errors in the Gig Worker Financial Wellness project
"""

import pandas as pd
import numpy as np
import sys
import os


def fix_data_issues():
    """Fix common data-related issues"""
    print("🔧 Checking and fixing data issues...")

    # Check if data files exist
    required_files = [
        "data/sample_gig_earnings.csv",
        "data/sample_transactions.csv",
        "data/sample_user_profiles.csv",
        "data/financial_products.csv",
    ]

    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)

    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        print("💡 Run: python generate_datasets.py")
        return False

    # Check data quality
    try:
        earnings = pd.read_csv("data/sample_gig_earnings.csv")
        transactions = pd.read_csv("data/sample_transactions.csv")
        users = pd.read_csv("data/sample_user_profiles.csv")

        print(f"✅ Data loaded successfully:")
        print(f"   - Earnings records: {len(earnings)}")
        print(f"   - Transaction records: {len(transactions)}")
        print(f"   - User profiles: {len(users)}")

        # Check for consistent user IDs
        earnings_users = set(earnings["user_id"].unique())
        transaction_users = set(transactions["user_id"].unique())
        profile_users = set(users["user_id"].unique())

        if not (earnings_users == transaction_users == profile_users):
            print("⚠️  Warning: Inconsistent user IDs across datasets")
        else:
            print("✅ User IDs consistent across all datasets")

        return True

    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False


def fix_ml_model_issues():
    """Fix machine learning model issues"""
    print("\n🤖 Checking ML model configuration...")

    try:
        sys.path.append("src")
        from ml_models import GigWorkerMLModels

        # Test model initialization
        ml_models = GigWorkerMLModels()
        print("✅ ML models class initialized successfully")

        # Test feature preparation
        earnings = pd.read_csv("data/sample_gig_earnings.csv")
        transactions = pd.read_csv("data/sample_transactions.csv")
        users = pd.read_csv("data/sample_user_profiles.csv")

        features_df = ml_models.prepare_features(earnings, transactions, users)
        print(f"✅ Features prepared: {features_df.shape}")

        # Test model training
        income_results = ml_models.train_income_prediction_model(features_df)
        stress_results = ml_models.train_financial_stress_model(features_df)

        print(f"✅ Income model trained - R²: {income_results['r2_score']:.3f}")
        print(f"✅ Stress model trained - Accuracy: {stress_results['accuracy']:.3f}")

        # Test predictions on first user
        first_user_features = features_df.head(1)

        try:
            wellness_score = ml_models.calculate_financial_wellness_score(
                first_user_features
            )
            stress_prob = ml_models.predict_financial_stress(first_user_features)
            income_pred = ml_models.predict_income(first_user_features)

            print(f"✅ Test predictions successful:")
            print(f"   - Wellness Score: {wellness_score}")
            print(f"   - Stress Probability: {stress_prob}")
            print(f"   - Income Prediction: ${income_pred}")

        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return False

        return True

    except Exception as e:
        print(f"❌ ML model error: {e}")
        return False


def fix_api_configuration():
    """Check API configuration"""
    print("\n🔑 Checking API configuration...")

    try:
        sys.path.append("config")
        from api_keys import OPENAI_API_KEY, USE_MOCK_AI

        if OPENAI_API_KEY and not USE_MOCK_AI:
            print("✅ OpenAI API key configured")

            # Test OpenAI integration
            try:
                sys.path.append("src")
                from openai_integration import FinancialAdvisorAI

                advisor = FinancialAdvisorAI()
                test_advice = advisor.generate_personalized_advice(
                    {"age": 30, "location": "Test", "primary_platform": "uber"}, 75, 0.3
                )

                if len(test_advice) > 50:  # Basic check for response
                    print("✅ OpenAI integration working")
                else:
                    print("⚠️  OpenAI response seems short, check API key")

            except Exception as e:
                print(f"⚠️  OpenAI integration error: {e}")
                print("💡 Will fall back to mock responses")

        elif USE_MOCK_AI:
            print("✅ Using mock AI responses (no API key needed)")

        else:
            print("⚠️  No API key found, using mock responses")

        return True

    except Exception as e:
        print(f"❌ API configuration error: {e}")
        return False


def run_comprehensive_test():
    """Run a comprehensive test of the entire system"""
    print("\n🧪 Running comprehensive system test...")

    try:
        # Test data loading and processing
        if not fix_data_issues():
            return False

        # Test ML models
        if not fix_ml_model_issues():
            return False

        # Test API configuration
        fix_api_configuration()

        print("\n🎉 System test completed successfully!")
        print("\n🚀 Ready to run: streamlit run main.py")
        return True

    except Exception as e:
        print(f"❌ Comprehensive test failed: {e}")
        return False


def main():
    """Main function to run all fixes"""
    print("🔧 Gig Worker Financial Wellness - System Fix Script")
    print("=" * 60)

    # Check Python version
    if sys.version_info < (3, 8):
        print("⚠️  Warning: Python 3.8+ recommended")

    # Check current directory
    if not os.path.exists("main.py"):
        print("❌ Please run this script from the project root directory")
        return

    # Run fixes
    success = run_comprehensive_test()

    if success:
        print("\n✅ All systems operational!")
        print("\n🎯 Next steps:")
        print("   1. Run: streamlit run main.py")
        print("   2. Open: http://localhost:8501")
        print("   3. Select a user and explore the dashboard")
    else:
        print("\n❌ Some issues found. Please check the errors above.")
        print("\n💡 Common solutions:")
        print("   - Run: python generate_datasets.py")
        print("   - Check your internet connection for API calls")
        print("   - Verify all required packages are installed")


if __name__ == "__main__":
    main()
