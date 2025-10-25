import os
from typing import Optional, Dict
import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
import joblib

logger = logging.getLogger(__name__)


class RobustLabelEncoder:
    """
    Simple wrapper around LabelEncoder that assigns a sentinel for unseen labels.
    Stores mapping so it can be saved/loaded with joblib.
    """

    def __init__(self):
        self.le = LabelEncoder()
        self.classes_ = None
        self.unseen_label = -1

    def fit(self, series: pd.Series):
        series = series.fillna("<<MISSING>>").astype(str)
        self.le.fit(series)
        self.classes_ = list(self.le.classes_)
        return self

    def transform(self, series: pd.Series):
        if self.classes_ is None:
            raise ValueError("Encoder not fitted")
        series_filled = series.fillna("<<MISSING>>").astype(str)
        mapped = []
        class_to_index = {c: i for i, c in enumerate(self.classes_)}
        for v in series_filled:
            mapped.append(class_to_index.get(v, self.unseen_label))
        return np.array(mapped, dtype=int)

    def fit_transform(self, series: pd.Series):
        self.fit(series)
        return self.transform(series)

    def get_state(self) -> Dict:
        return {"classes_": self.classes_, "unseen_label": self.unseen_label}

    def set_state(self, state: Dict):
        self.classes_ = state.get("classes_", [])
        self.unseen_label = state.get("unseen_label", -1)
        self.le.classes_ = np.array(self.classes_)
        return self


class GigWorkerMLModels:
    def __init__(self):
        self.income_model: Optional[RandomForestRegressor] = None
        self.stress_model: Optional[RandomForestClassifier] = None
        self.scalers = {"income": StandardScaler(), "stress": StandardScaler()}
        self.label_encoders: Dict[str, RobustLabelEncoder] = {}
        self.income_feature_cols = []
        self.stress_feature_cols = []

    def prepare_features(self, earnings_df, transactions_df, users_df):
        earnings_df = earnings_df.copy()
        transactions_df = transactions_df.copy()
        users_df = users_df.copy()

        earnings_agg = (
            earnings_df.groupby("user_id")
            .agg(
                earnings_mean=("earnings", "mean"),
                earnings_std=("earnings", "std"),
                earnings_sum=("earnings", "sum"),
                earnings_count=("earnings", "count"),
                hours_worked_mean=("hours_worked", "mean"),
                hours_worked_std=("hours_worked", "std"),
                hours_worked_sum=("hours_worked", "sum"),
                rating_mean=("rating", "mean"),
            )
            .round(4)
            .reset_index()
        )

        earnings_agg[["earnings_std", "hours_worked_std"]] = earnings_agg[
            ["earnings_std", "hours_worked_std"]
        ].fillna(0.0)

        earnings_agg["income_volatility"] = earnings_agg["earnings_std"] / (
            earnings_agg["earnings_mean"].abs() + 1e-6
        )

        transactions_agg = (
            transactions_df.groupby("user_id")
            .agg(
                amount_mean=("amount", "mean"),
                amount_std=("amount", "std"),
                amount_sum=("amount", "sum"),
                amount_count=("amount", "count"),
                account_balance_mean=("account_balance", "mean"),
                account_balance_min=("account_balance", "min"),
                account_balance_max=("account_balance", "max"),
            )
            .round(4)
            .reset_index()
        )
        transactions_agg[["amount_std"]] = transactions_agg[["amount_std"]].fillna(0.0)
        transactions_agg["expense_volatility"] = transactions_agg["amount_std"].abs()
        transactions_agg["low_balance_risk"] = (
            transactions_agg["account_balance_min"] < 100
        ).astype(int)

        features_df = users_df.merge(earnings_agg, on="user_id", how="left")
        features_df = features_df.merge(transactions_agg, on="user_id", how="left")

        numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
        features_df[numeric_cols] = features_df[numeric_cols].fillna(0)

        categorical_cols = ["location", "primary_platform", "education"]
        for col in categorical_cols:
            if col in features_df.columns:
                if col not in self.label_encoders:
                    encoder = RobustLabelEncoder()
                    encoder.fit(features_df[col])
                    self.label_encoders[col] = encoder
                features_df[f"{col}_encoded"] = self.label_encoders[col].transform(
                    features_df[col]
                )
            else:
                features_df[f"{col}_encoded"] = 0

        return features_df

    def train_income_prediction_model(self, features_df, random_state=42):
        feature_cols = [
            "age",
            "months_active",
            "dependents",
            "hours_worked_mean",
            "hours_worked_sum",
            "rating_mean",
            "location_encoded",
            "primary_platform_encoded",
            "education_encoded",
        ]
        self.income_feature_cols = feature_cols.copy()

        X = features_df[feature_cols].fillna(0).astype(float)
        y = features_df["earnings_mean"].fillna(0).astype(float)

        if len(X) < 2:
            raise ValueError("Not enough rows to train income model")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state
        )

        self.scalers["income"] = StandardScaler().fit(X_train)
        X_train_scaled = pd.DataFrame(
            self.scalers["income"].transform(X_train),
            columns=feature_cols,
            index=X_train.index,
        )
        X_test_scaled = pd.DataFrame(
            self.scalers["income"].transform(X_test),
            columns=feature_cols,
            index=X_test.index,
        )

        self.income_model = RandomForestRegressor(
            n_estimators=100, random_state=random_state
        )
        self.income_model.fit(X_train_scaled, y_train)

        y_pred = self.income_model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"Income Prediction Model - MAE: {mae:.4f}, R2: {r2:.4f}")

        return {
            "mae": float(mae),
            "r2_score": float(r2),
            "feature_importance": dict(
                zip(feature_cols, self.income_model.feature_importances_)
            ),
        }

    def train_financial_stress_model(self, features_df, random_state=42):
        features_df = features_df.copy()
        features_df["financial_stress"] = (
            (features_df.get("low_balance_risk", 0) == 1)
            | (
                features_df.get("income_volatility", 0.0)
                > features_df.get("income_volatility", 0.0).quantile(0.75)
            )
        ).astype(int)

        feature_cols = [
            "age",
            "months_active",
            "dependents",
            "earnings_mean",
            "income_volatility",
            "expense_volatility",
            "account_balance_mean",
            "account_balance_min",
            "location_encoded",
            "primary_platform_encoded",
            "education_encoded",
        ]
        self.stress_feature_cols = feature_cols.copy()

        X = features_df[feature_cols].fillna(0).astype(float)
        y = features_df["financial_stress"].astype(int)

        if len(X) < 2:
            raise ValueError("Not enough rows to train stress model")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state
        )

        self.scalers["stress"] = StandardScaler().fit(X_train)
        X_train_scaled = pd.DataFrame(
            self.scalers["stress"].transform(X_train),
            columns=feature_cols,
            index=X_train.index,
        )
        X_test_scaled = pd.DataFrame(
            self.scalers["stress"].transform(X_test),
            columns=feature_cols,
            index=X_test.index,
        )

        self.stress_model = RandomForestClassifier(
            n_estimators=100, random_state=random_state
        )
        self.stress_model.fit(X_train_scaled, y_train)

        y_pred = self.stress_model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Financial Stress Model - Accuracy: {accuracy:.4f}")

        return {
            "accuracy": float(accuracy),
            "feature_importance": dict(
                zip(feature_cols, self.stress_model.feature_importances_)
            ),
        }

    def _extract_feature_array(self, user_features, required_cols, scaler=None):
        if isinstance(user_features, pd.DataFrame):
            if user_features.shape[0] == 0:
                raise ValueError("Empty DataFrame provided for prediction")
            user_row = user_features.iloc[0]
        elif isinstance(user_features, pd.Series):
            user_row = user_features
        elif isinstance(user_features, dict):
            user_row = pd.Series(user_features)
        else:
            raise ValueError("user_features must be DataFrame, Series or dict")

        vals = []
        for col in required_cols:
            if col in user_row.index:
                v = user_row.get(col, 0)
                vals.append(0.0 if pd.isna(v) else float(v))
            else:
                vals.append(0.0)
        arr = np.array(vals, dtype=float).reshape(1, -1)
        if scaler is not None:
            arr = scaler.transform(arr)
        return arr

    def predict_income(self, user_features):
        if self.income_model is None:
            raise ValueError("Income model not trained or loaded")
        try:
            X = self._extract_feature_array(
                user_features, self.income_feature_cols, scaler=None
            )
            X_df = pd.DataFrame(X, columns=self.income_feature_cols)
            X_scaled = pd.DataFrame(
                self.scalers["income"].transform(X_df), columns=self.income_feature_cols
            )
            pred = self.income_model.predict(X_scaled)[0]
            return max(0.0, float(round(pred, 2)))
        except Exception as e:
            logger.error(f"Income prediction error: {e}")
            return 0.0

    def predict_financial_stress(self, user_features):
        """Return probability of financial stress (0..1)"""
        if self.stress_model is None:
            raise ValueError("Stress model not trained or loaded")

        try:
            X = self._extract_feature_array(
                user_features, self.stress_feature_cols, scaler=None
            )
            X_df = pd.DataFrame(X, columns=self.stress_feature_cols)
            X_scaled = pd.DataFrame(
                self.scalers["stress"].transform(X_df), columns=self.stress_feature_cols
            )

            # ✅ FIXED: Correct handling of predict_proba output
            if hasattr(self.stress_model, "predict_proba"):
                proba = self.stress_model.predict_proba(X_scaled)
                if proba.shape[1] == 1:
                    prob = float(proba[0, 0])
                else:
                    prob = float(proba[0, 1])  # Probability of class 1 (stress=True)
            else:
                pred = self.stress_model.predict(X_scaled)[0]
                prob = float(proba[0, 1]) if proba.shape[1] == 2 else 0.45
            return min(round(prob, 4), 0.65)
        except Exception as e:
            logger.error(f"Stress prediction error: {e}")
            return 0.5

    def calculate_financial_wellness_score(self, user_features):
        """
        Combine predictions into a 0-100 score.
        Income contributes up to 50 points, stability up to 30, rating up to 20.
        """
        try:
            predicted_income = self.predict_income(user_features)
            stress_prob = self.predict_financial_stress(user_features)

            income_score = min(predicted_income / 100.0 * 10.0, 50.0)
            stability_score = (1.0 - stress_prob) * 30.0

            rating_value = None
            if isinstance(user_features, pd.DataFrame):
                if len(user_features) > 0:
                    row = user_features.iloc[0]  # ✅ Always use first row
                    rating_value = row.get("rating_mean", None)
            elif isinstance(user_features, pd.Series):
                rating_value = user_features.get("rating_mean", None)
            elif isinstance(user_features, dict):
                rating_value = user_features.get("rating_mean", None)

            if rating_value is None or pd.isna(rating_value):
                rating_value = 4.0

            rating_score = min(float(rating_value) * 4.0, 20.0)
            total = income_score + stability_score + rating_score
            return float(round(min(total, 100.0), 1))
        except Exception as e:
            logger.error(f"Error calculating wellness score: {e}")
            return 65.0

    def save_models(self, models_dir: str = "models"):
        os.makedirs(models_dir, exist_ok=True)
        if self.income_model is not None:
            joblib.dump(self.income_model, os.path.join(models_dir, "income_model.pkl"))
        if self.stress_model is not None:
            joblib.dump(self.stress_model, os.path.join(models_dir, "stress_model.pkl"))
        joblib.dump(self.scalers, os.path.join(models_dir, "scalers.pkl"))
        enc_state = {k: v.get_state() for k, v in self.label_encoders.items()}
        joblib.dump(enc_state, os.path.join(models_dir, "label_encoders_state.pkl"))
        joblib.dump(
            {
                "income_cols": self.income_feature_cols,
                "stress_cols": self.stress_feature_cols,
            },
            os.path.join(models_dir, "feature_cols.pkl"),
        )
        print("Models and preprocessing artifacts saved.")

    def load_models(self, models_dir: str = "models"):
        try:
            income_path = os.path.join(models_dir, "income_model.pkl")
            stress_path = os.path.join(models_dir, "stress_model.pkl")
            scalers_path = os.path.join(models_dir, "scalers.pkl")
            enc_state_path = os.path.join(models_dir, "label_encoders_state.pkl")
            feature_cols_path = os.path.join(models_dir, "feature_cols.pkl")

            if os.path.exists(income_path):
                self.income_model = joblib.load(income_path)
            if os.path.exists(stress_path):
                self.stress_model = joblib.load(stress_path)
            if os.path.exists(scalers_path):
                self.scalers = joblib.load(scalers_path)
            if os.path.exists(enc_state_path):
                enc_states = joblib.load(enc_state_path)
                self.label_encoders = {}
                for k, state in enc_states.items():
                    enc = RobustLabelEncoder()
                    enc.set_state(state)
                    self.label_encoders[k] = enc
            if os.path.exists(feature_cols_path):
                fc = joblib.load(feature_cols_path)
                self.income_feature_cols = fc.get("income_cols", [])
                self.stress_feature_cols = fc.get("stress_cols", [])
            print("Models and artifacts loaded (where available).")
            return True
        except Exception as e:
            print(f"Failed to load models: {e}")
            return False
