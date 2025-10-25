import os
import json
from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd

load_dotenv()


class FinancialAdvisorAI:
    def __init__(self):
        """Initialize Financial Advisor AI class with API setup and mock mode toggle"""
        from config.api_keys import OPENAI_API_KEY, USE_MOCK_AI

        self.use_mock = USE_MOCK_AI

        if not self.use_mock and OPENAI_API_KEY:
            try:
                # Initialize OpenAI client with explicit parameters to avoid proxy conflicts
                self.client = OpenAI(api_key=OPENAI_API_KEY)
                print("✅ OpenAI client initialized successfully")
            except TypeError as e:
                if "unexpected keyword argument" in str(e):
                    print(f"⚠️ OpenAI client parameter conflict: {e}")
                    print("🔄 Falling back to mock mode")
                    self.use_mock = True
                else:
                    raise
            except Exception as e:
                print(f"❌ OpenAI client initialization failed: {e}")
                print("🔄 Falling back to mock mode")
                self.use_mock = True
        else:
            self.client = None
            if not self.use_mock:
                print("⚠️ OpenAI API key not found, using mock mode")
                self.use_mock = True

    # -------------------------------------------------
    # 1. Personalized Financial Advice
    # -------------------------------------------------
    def generate_personalized_advice(self, user_data, wellness_score, stress_level):
        """Generate personalized financial advice using AI or mock data"""
        if self.use_mock:
            return self._mock_financial_advice(user_data, wellness_score, stress_level)

        prompt = f"""
        You are a financial advisor specializing in gig economy workers. 
        Analyze this user profile and provide personalized advice.

        **User Profile**
        - Age: {user_data.get('age', 'Unknown')}
        - Location: {user_data.get('location', 'Unknown')}
        - Primary Platform: {user_data.get('primary_platform', 'Unknown')}
        - Months Active: {user_data.get('months_active', 'Unknown')}
        - Dependents: {user_data.get('dependents', 'Unknown')}
        - Financial Wellness Score: {wellness_score}/100
        - Financial Stress Level: {stress_level * 100:.1f}%

        Provide specific, actionable advice in these areas:
        1. Income optimization strategies
        2. Emergency fund recommendations
        3. Investment suggestions
        4. Risk management tips
        5. Platform-specific advice

        Keep advice practical, empathetic, and tailored to gig workers.
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a supportive financial advisor for gig economy workers.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=600,
                temperature=0.7,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"[Error] OpenAI API failed: {e}")
            return self._mock_financial_advice(user_data, wellness_score, stress_level)

    # -------------------------------------------------
    # 2. Mock Advice Generation
    # -------------------------------------------------
    def _mock_financial_advice(self, user_data, wellness_score, stress_level):
        """Fallback mock advice for demo/testing"""
        age = user_data.get("age", 30)
        platform = user_data.get("primary_platform", "uber")
        location = user_data.get("location", "Unknown")
        dependents = user_data.get("dependents", 0)

        advice = []

        # Income optimization
        if platform in ["uber", "lyft"]:
            advice.append(
                "🚗 Drive during peak hours (7–9 AM, 5–7 PM) for surge pricing."
            )
        elif platform in ["doordash", "ubereats"]:
            advice.append(
                "🍕 Focus on dinner hours/weekends and use multiple apps to reduce idle time."
            )
        elif platform in ["upwork", "fiverr"]:
            advice.append(
                "💻 Specialize in high-demand skills, build long-term clients, and raise rates gradually."
            )

        # Emergency fund
        if stress_level > 0.7:
            advice.append(
                "🚨 Start with a $200–500 emergency fund. Save $5–10 daily from your earnings."
            )
        elif stress_level > 0.4:
            advice.append(
                "💰 Aim for $1,000, then 3 months of expenses. Automate transfers on high-earning days."
            )
        else:
            advice.append(
                "✅ Maintain 6 months of expenses. Great job staying financially stable!"
            )

        # Investment
        if age < 35 and wellness_score > 60:
            advice.append(
                "📈 Invest in low-cost index funds. Start small ($25–50/month)."
            )
        elif wellness_score > 40:
            advice.append(
                "📊 Build your emergency fund first, then move into conservative investments."
            )

        # Risk management
        if platform in ["uber", "lyft", "doordash"]:
            advice.append(
                "🛡️ Get rideshare/delivery insurance and track expenses for tax deductions."
            )

        # Family planning
        if dependents > 0:
            advice.append(
                f"👨‍👩‍👧‍👦 With {dependents} dependent(s), prioritize life insurance and education savings."
            )

        # Location-based
        if location in ["New York", "San Francisco"]:
            advice.append(
                "🏙️ High cost of living detected. Focus on maximizing earnings and reducing housing costs."
            )

        return "\n\n".join(advice)

    # -------------------------------------------------
    # 3. Wellness Score Explanation
    # -------------------------------------------------
    def explain_wellness_score(self, wellness_score, user_data):
        """Explain user's financial wellness score"""
        if self.use_mock:
            return self._mock_score_explanation(wellness_score, user_data)
        return self._mock_score_explanation(wellness_score, user_data)

    def _mock_score_explanation(self, wellness_score, user_data):
        """Fallback explanation for demo"""
        if wellness_score >= 80:
            level, emoji, message = (
                "Excellent",
                "🌟",
                "Your financial health is outstanding!",
            )
        elif wellness_score >= 60:
            level, emoji, message = (
                "Good",
                "👍",
                "You're financially stable with room to grow.",
            )
        elif wellness_score >= 40:
            level, emoji, message = (
                "Fair",
                "⚠️",
                "You need to stabilize income and budget better.",
            )
        else:
            level, emoji, message = (
                "Needs Improvement",
                "🚨",
                "Focus on increasing income consistency.",
            )

        breakdown = f"""
{emoji} **Financial Wellness Score: {wellness_score}/100 ({level})**

{message}

**Score Breakdown:**
• Income Stability: {min(40, wellness_score * 0.4):.0f}/40  
• Financial Stress: {min(30, (100 - wellness_score) * 0.3):.0f}/30  
• Platform Performance: {min(20, wellness_score * 0.2):.0f}/20  
• Growth Potential: {min(10, wellness_score * 0.1):.0f}/10  

**Next Steps:**
"""

        if wellness_score < 40:
            breakdown += "1. Stabilize income\n2. Build small emergency fund\n3. Track expenses daily"
        elif wellness_score < 60:
            breakdown += "1. Diversify income\n2. Build 3-month buffer\n3. Optimize gig platform strategy"
        elif wellness_score < 80:
            breakdown += (
                "1. Grow emergency fund\n2. Begin investing\n3. Plan long-term goals"
            )
        else:
            breakdown += "1. Increase investments\n2. Explore property/business\n3. Plan retirement"

        return breakdown

    # -------------------------------------------------
    # 4. Financial Product Recommendations
    # -------------------------------------------------
    def generate_product_recommendations(
        self, user_data, financial_products_df, wellness_score
    ):
        """Generate top 3 product recommendations"""
        if not isinstance(financial_products_df, pd.DataFrame):
            raise ValueError("financial_products_df must be a pandas DataFrame")

        estimated_credit_score = int(500 + (wellness_score * 2))  # 500–700 range

        suitable = financial_products_df[
            financial_products_df["min_credit_score"] <= estimated_credit_score
        ]

        if wellness_score < 40:
            priority_types = ["savings", "cash_advance"]
        elif wellness_score < 60:
            priority_types = ["savings", "personal_loan", "insurance"]
        else:
            priority_types = ["investment", "credit_card", "personal_loan"]

        recommendations = []
        for ptype in priority_types:
            match = suitable[suitable["type"] == ptype]
            if not match.empty:
                best = match.iloc[0]

                reasons = {
                    "savings": "Build your emergency fund with a high-yield account",
                    "personal_loan": "Consolidate debt or invest in your gig setup",
                    "investment": "Start long-term wealth building with low fees",
                    "credit_card": "Build credit and earn gig-related rewards",
                    "insurance": "Protect your income and assets",
                    "cash_advance": "Get quick access to emergency funds",
                }

                recommendations.append(
                    {
                        "product_name": best["product_name"],
                        "provider": best["provider"],
                        "type": best["type"],
                        "interest_rate": best["interest_rate"],
                        "reason": reasons.get(
                            ptype, "Useful financial support for gig workers"
                        ),
                        "max_amount": best["max_amount"],
                    }
                )

        return recommendations[:3]
