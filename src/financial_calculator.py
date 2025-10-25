import numpy as np
import pandas as pd
from datetime import datetime, timedelta


class FinancialCalculator:
    """Financial calculations and metrics for gig workers"""

    def __init__(self):
        pass

    def calculate_emergency_fund_target(
        self, monthly_expenses, dependents=0, risk_level="medium"
    ):
        """Calculate recommended emergency fund target"""
        base_months = {
            "low": 3,  # Stable income, no dependents
            "medium": 4,  # Average gig worker
            "high": 6,  # High income volatility or dependents
        }

        months = base_months.get(risk_level, 4)

        # Adjust for dependents
        months += dependents * 0.5

        target = monthly_expenses * months
        return round(target, 2)

    def calculate_debt_to_income_ratio(self, monthly_debt_payments, monthly_income):
        """Calculate debt-to-income ratio"""
        if monthly_income <= 0:
            return float("inf")

        ratio = (monthly_debt_payments / monthly_income) * 100
        return round(ratio, 2)

    def assess_credit_worthiness(
        self, income_stability, debt_ratio, payment_history_score=0.8
    ):
        """Assess creditworthiness for gig workers (0-1 scale)"""

        # Income stability score (0-1)
        if income_stability < 0.2:
            stability_score = 0.9
        elif income_stability < 0.5:
            stability_score = 0.7
        elif income_stability < 1.0:
            stability_score = 0.5
        else:
            stability_score = 0.3

        # Debt ratio score (0-1)
        if debt_ratio < 10:
            debt_score = 1.0
        elif debt_ratio < 20:
            debt_score = 0.8
        elif debt_ratio < 30:
            debt_score = 0.6
        elif debt_ratio < 40:
            debt_score = 0.4
        else:
            debt_score = 0.2

        # Weighted credit score
        credit_score = (
            stability_score * 0.4 + debt_score * 0.3 + payment_history_score * 0.3
        )

        return round(credit_score, 3)

    def calculate_optimal_savings_rate(
        self, monthly_income, monthly_expenses, financial_goals=None
    ):
        """Calculate optimal savings rate for gig workers"""

        if monthly_income <= monthly_expenses:
            return 0  # Can't save if expenses exceed income

        disposable_income = monthly_income - monthly_expenses

        # Base savings rate: 20% of disposable income
        base_savings = disposable_income * 0.20

        # Adjust for financial goals
        if financial_goals:
            goal_savings = sum(
                [goal.get("monthly_target", 0) for goal in financial_goals]
            )
            recommended_savings = max(base_savings, goal_savings)
        else:
            recommended_savings = base_savings

        # Cap at 50% of disposable income for practicality
        max_savings = disposable_income * 0.5
        optimal_savings = min(recommended_savings, max_savings)

        savings_rate = (optimal_savings / monthly_income) * 100
        return round(savings_rate, 2)

    def project_retirement_savings(
        self,
        current_age,
        target_retirement_age,
        monthly_contribution,
        annual_return_rate=0.07,
    ):
        """Project retirement savings for gig workers"""

        years_to_retirement = target_retirement_age - current_age
        months_to_retirement = years_to_retirement * 12
        monthly_return_rate = annual_return_rate / 12

        if monthly_return_rate == 0:
            # Simple case without returns
            total_savings = monthly_contribution * months_to_retirement
        else:
            # Future value of annuity formula
            total_savings = monthly_contribution * (
                ((1 + monthly_return_rate) ** months_to_retirement - 1)
                / monthly_return_rate
            )

        return round(total_savings, 2)

    def calculate_insurance_needs(self, annual_income, dependents, existing_coverage=0):
        """Calculate life insurance needs for gig workers"""

        # Basic rule: 10x annual income + additional for dependents
        base_coverage = annual_income * 10

        # Additional coverage for dependents (college costs, etc.)
        dependent_coverage = dependents * 250000  # $250k per dependent

        total_needed = base_coverage + dependent_coverage - existing_coverage
        return max(0, round(total_needed, 2))

    def analyze_cash_flow(self, income_data, expense_data, period_days=30):
        """Analyze cash flow patterns for gig workers"""

        # Convert to pandas series if needed
        if not isinstance(income_data, pd.Series):
            income_data = pd.Series(income_data)
        if not isinstance(expense_data, pd.Series):
            expense_data = pd.Series(expense_data)

        # Daily averages
        avg_daily_income = income_data.mean()
        avg_daily_expenses = abs(expense_data.mean())

        # Volatility measures
        income_volatility = (
            income_data.std() / avg_daily_income if avg_daily_income > 0 else 0
        )
        expense_volatility = (
            abs(expense_data.std()) / avg_daily_expenses
            if avg_daily_expenses > 0
            else 0
        )

        # Net cash flow
        net_daily_flow = avg_daily_income - avg_daily_expenses
        projected_monthly_flow = net_daily_flow * 30

        # Cash flow stability score (0-1, higher is better)
        stability_score = 1 / (1 + income_volatility + expense_volatility)

        return {
            "avg_daily_income": round(avg_daily_income, 2),
            "avg_daily_expenses": round(avg_daily_expenses, 2),
            "net_daily_flow": round(net_daily_flow, 2),
            "projected_monthly_flow": round(projected_monthly_flow, 2),
            "income_volatility": round(income_volatility, 3),
            "expense_volatility": round(expense_volatility, 3),
            "stability_score": round(stability_score, 3),
        }

    def recommend_investment_allocation(
        self, age, risk_tolerance="medium", gig_worker_specific=True
    ):
        """Recommend investment portfolio allocation for gig workers"""

        # Base allocation by age (stocks vs bonds)
        stock_percentage = max(20, 100 - age)
        bond_percentage = min(80, age)

        # Adjust for risk tolerance
        risk_adjustments = {"low": -20, "medium": 0, "high": +20}

        adjustment = risk_adjustments.get(risk_tolerance, 0)
        stock_percentage = max(10, min(90, stock_percentage + adjustment))
        bond_percentage = 100 - stock_percentage

        # Gig worker specific adjustments
        if gig_worker_specific:
            # Higher emergency fund allocation
            emergency_fund_percentage = max(10, 100 - age * 1.5)

            # Reduce stock/bond percentages proportionally
            reduction_factor = (100 - emergency_fund_percentage) / 100
            stock_percentage = round(stock_percentage * reduction_factor)
            bond_percentage = round(bond_percentage * reduction_factor)
        else:
            emergency_fund_percentage = 0

        return {
            "stocks": stock_percentage,
            "bonds": bond_percentage,
            "emergency_fund": round(emergency_fund_percentage),
            "total": stock_percentage + bond_percentage + emergency_fund_percentage,
        }

    def calculate_tax_optimization_savings(
        self, annual_gig_income, business_expenses=0, retirement_contribution=0
    ):
        """Calculate potential tax savings for gig workers"""

        # Simplified tax calculation (self-employment + income tax)
        se_tax_rate = 0.153  # Self-employment tax (~15.3%)

        # Progressive income tax (simplified)
        if annual_gig_income <= 40000:
            income_tax_rate = 0.12
        elif annual_gig_income <= 85000:
            income_tax_rate = 0.22
        else:
            income_tax_rate = 0.24

        # Calculate taxes without optimization
        gross_income = annual_gig_income
        se_tax = gross_income * se_tax_rate
        income_tax = gross_income * income_tax_rate
        total_tax_before = se_tax + income_tax

        # Calculate taxes with optimization
        net_business_income = gross_income - business_expenses
        adjusted_gross_income = net_business_income - retirement_contribution

        optimized_se_tax = net_business_income * se_tax_rate
        optimized_income_tax = adjusted_gross_income * income_tax_rate
        total_tax_after = optimized_se_tax + optimized_income_tax

        savings = total_tax_before - total_tax_after

        return {
            "tax_before_optimization": round(total_tax_before, 2),
            "tax_after_optimization": round(total_tax_after, 2),
            "total_savings": round(savings, 2),
            "savings_percentage": (
                round((savings / total_tax_before) * 100, 2)
                if total_tax_before > 0
                else 0
            ),
        }


# Example usage and testing
if __name__ == "__main__":
    calculator = FinancialCalculator()

    # Test emergency fund calculation
    emergency_target = calculator.calculate_emergency_fund_target(
        3000, dependents=1, risk_level="high"
    )
    print(f"Emergency fund target: ${emergency_target}")

    # Test investment allocation
    allocation = calculator.recommend_investment_allocation(
        age=30, risk_tolerance="medium"
    )
    print(f"Investment allocation: {allocation}")

    # Test cash flow analysis
    income_data = [120, 80, 150, 90, 110, 130, 95]
    expense_data = [-50, -60, -45, -70, -55, -65, -48]

    cash_flow = calculator.analyze_cash_flow(income_data, expense_data)
    print(f"Cash flow analysis: {cash_flow}")

    # Test tax optimization
    tax_savings = calculator.calculate_tax_optimization_savings(
        annual_gig_income=60000, business_expenses=8000, retirement_contribution=6000
    )
    print(f"Tax savings: {tax_savings}")
