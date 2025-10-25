import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)


def generate_gig_earnings():
    """Generate synthetic gig earnings data"""
    users = [f"USER{str(i).zfill(3)}" for i in range(1, 101)]  # 100 users
    platforms = [
        "uber",
        "lyft",
        "doordash",
        "ubereats",
        "instacart",
        "upwork",
        "fiverr",
        "taskrabbit",
    ]
    job_types = ["rideshare", "delivery", "freelance", "services"]

    data = []
    start_date = datetime(2024, 1, 1)

    for user in users:
        # Each user works on 1-3 platforms
        user_platforms = random.sample(platforms, random.randint(1, 3))

        for platform in user_platforms:
            # Generate 90 days of data
            for day in range(90):
                current_date = start_date + timedelta(days=day)

                # Not everyone works every day (60% chance of working)
                if random.random() < 0.6:
                    # Generate earnings based on platform type
                    if platform in ["uber", "lyft"]:
                        base_hourly = random.uniform(15, 25)
                        hours = random.uniform(2, 12)
                        job_type = "rideshare"
                    elif platform in ["doordash", "ubereats", "instacart"]:
                        base_hourly = random.uniform(12, 20)
                        hours = random.uniform(1, 8)
                        job_type = "delivery"
                    elif platform in ["upwork", "fiverr"]:
                        base_hourly = random.uniform(20, 80)
                        hours = random.uniform(1, 10)
                        job_type = "freelance"
                    else:
                        base_hourly = random.uniform(18, 35)
                        hours = random.uniform(1, 6)
                        job_type = "services"

                    earnings = round(base_hourly * hours * random.uniform(0.8, 1.2), 2)
                    rating = round(random.uniform(4.0, 5.0), 1)

                    data.append(
                        {
                            "user_id": user,
                            "platform": platform,
                            "date": current_date.strftime("%Y-%m-%d"),
                            "earnings": earnings,
                            "hours_worked": round(hours, 1),
                            "rating": rating,
                            "job_type": job_type,
                        }
                    )

    return pd.DataFrame(data)


def generate_transactions():
    """Generate synthetic transaction data"""
    users = [f"USER{str(i).zfill(3)}" for i in range(1, 101)]
    categories = [
        "food",
        "gas",
        "rent",
        "utilities",
        "entertainment",
        "shopping",
        "healthcare",
        "transport",
    ]

    data = []
    start_date = datetime(2024, 1, 1)

    for user in users:
        balance = random.uniform(500, 5000)  # Starting balance

        for day in range(90):
            current_date = start_date + timedelta(days=day)

            # Generate 1-5 transactions per day
            num_transactions = random.randint(1, 5)

            for _ in range(num_transactions):
                category = random.choice(categories)

                # Transaction amounts based on category
                if category == "rent":
                    amount = -random.uniform(800, 2000)
                elif category == "food":
                    amount = -random.uniform(5, 50)
                elif category == "gas":
                    amount = -random.uniform(20, 80)
                elif category == "utilities":
                    amount = -random.uniform(50, 200)
                else:
                    amount = -random.uniform(10, 150)

                balance += amount

                descriptions = {
                    "food": ["McDonald's", "Starbucks", "Grocery Store", "Pizza Hut"],
                    "gas": ["Shell Station", "Chevron", "BP", "Exxon"],
                    "rent": ["Monthly Rent", "Apartment Rent"],
                    "utilities": ["Electric Bill", "Phone Bill", "Internet"],
                    "entertainment": ["Netflix", "Spotify", "Movie Theater"],
                    "shopping": ["Amazon", "Target", "Walmart"],
                    "healthcare": ["Doctor Visit", "Pharmacy"],
                    "transport": ["Metro Card", "Parking", "Car Maintenance"],
                }

                data.append(
                    {
                        "user_id": user,
                        "date": current_date.strftime("%Y-%m-%d"),
                        "amount": round(amount, 2),
                        "category": category,
                        "description": random.choice(descriptions[category]),
                        "account_balance": round(balance, 2),
                    }
                )

    return pd.DataFrame(data)


def generate_user_profiles():
    """Generate synthetic user profiles"""
    users = [f"USER{str(i).zfill(3)}" for i in range(1, 101)]
    locations = [
        "New York",
        "San Francisco",
        "Austin",
        "Chicago",
        "Los Angeles",
        "Seattle",
        "Boston",
        "Miami",
    ]
    platforms = [
        "uber",
        "lyft",
        "doordash",
        "upwork",
        "instacart",
        "fiverr",
        "taskrabbit",
    ]
    education_levels = ["high_school", "some_college", "bachelors", "masters", "phd"]

    data = []
    for user in users:
        data.append(
            {
                "user_id": user,
                "age": random.randint(18, 65),
                "location": random.choice(locations),
                "primary_platform": random.choice(platforms),
                "months_active": random.randint(3, 48),
                "education": random.choice(education_levels),
                "dependents": random.randint(0, 4),
            }
        )

    return pd.DataFrame(data)


def generate_financial_products():
    """Generate financial products data"""
    products = [
        {
            "product_id": "LOAN001",
            "product_name": "Gig Worker Loan",
            "provider": "FinTech Bank",
            "type": "personal_loan",
            "min_credit_score": 600,
            "max_amount": 5000,
            "interest_rate": 12.5,
        },
        {
            "product_id": "LOAN002",
            "product_name": "Quick Cash Advance",
            "provider": "PayDay Plus",
            "type": "cash_advance",
            "min_credit_score": 500,
            "max_amount": 1000,
            "interest_rate": 24.9,
        },
        {
            "product_id": "SAVE001",
            "product_name": "Emergency Fund",
            "provider": "Credit Union",
            "type": "savings",
            "min_credit_score": 500,
            "max_amount": 999999,
            "interest_rate": 2.1,
        },
        {
            "product_id": "INVEST001",
            "product_name": "Micro Investment",
            "provider": "Robo Advisor",
            "type": "investment",
            "min_credit_score": 550,
            "max_amount": 999999,
            "interest_rate": 0.25,
        },
        {
            "product_id": "CREDIT001",
            "product_name": "Gig Worker Credit Card",
            "provider": "Neo Bank",
            "type": "credit_card",
            "min_credit_score": 650,
            "max_amount": 3000,
            "interest_rate": 18.99,
        },
        {
            "product_id": "INSURE001",
            "product_name": "Gig Worker Insurance",
            "provider": "InsureTech",
            "type": "insurance",
            "min_credit_score": 500,
            "max_amount": 999999,
            "interest_rate": 0.05,
        },
    ]

    return pd.DataFrame(products)


if __name__ == "__main__":
    # Create data directory if it doesn't exist
    import os

    os.makedirs("data", exist_ok=True)

    # Generate all datasets
    print("Generating gig earnings data...")
    gig_earnings = generate_gig_earnings()
    gig_earnings.to_csv("data/sample_gig_earnings.csv", index=False)

    print("Generating transaction data...")
    transactions = generate_transactions()
    transactions.to_csv("data/sample_transactions.csv", index=False)

    print("Generating user profiles...")
    user_profiles = generate_user_profiles()
    user_profiles.to_csv("data/sample_user_profiles.csv", index=False)

    print("Generating financial products...")
    financial_products = generate_financial_products()
    financial_products.to_csv("data/financial_products.csv", index=False)

    print("All datasets generated successfully!")
    print(f"Gig earnings: {len(gig_earnings)} records")
    print(f"Transactions: {len(transactions)} records")
    print(f"User profiles: {len(user_profiles)} records")
    print(f"Financial products: {len(financial_products)} records")
