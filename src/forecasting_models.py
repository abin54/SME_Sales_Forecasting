"""
Sales Forecasting Models
Multiple approaches: Prophet, ARIMA, XGBoost
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit
import joblib
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def setup_matplotlib():
    """Setup matplotlib for plotting"""
    plt.style.use("seaborn-v0_8")
    sns.set_palette("husl")
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

class SalesForecastingPipeline:
    """End-to-end sales forecasting pipeline"""

    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None

    def prepare_data(self, df, target_col='revenue', freq='D'):
        """Prepare time series data"""
        # Aggregate to daily level
        daily_df = df.groupby('date').agg({
            target_col: 'sum',
            'units_sold': 'sum',
            'is_weekend': 'first',
            'is_festival': 'max',
            'festival_multiplier': 'max'
        }).reset_index()

        daily_df['date'] = pd.to_datetime(daily_df['date'])
        daily_df = daily_df.sort_values('date')

        return daily_df

    def create_features(self, df, date_col='date'):
        """Create time-series features for ML models"""
        df = df.copy()
        df['dayofweek'] = df[date_col].dt.dayofweek
        df['month'] = df[date_col].dt.month
        df['quarter'] = df[date_col].dt.quarter
        df['dayofmonth'] = df[date_col].dt.day
        df['weekofyear'] = df[date_col].dt.isocalendar().week.astype(int)
        df['is_month_start'] = (df[date_col].dt.day <= 5).astype(int)
        df['is_month_end'] = (df[date_col].dt.day >= 28).astype(int)

        # Lag features
        for lag in [1, 7, 14, 30]:
            df[f'lag_{lag}'] = df['revenue'].shift(lag)

        # Rolling features
        for window in [7, 14, 30]:
            df[f'rolling_mean_{window}'] = df['revenue'].rolling(window).mean()
            df[f'rolling_std_{window}'] = df['revenue'].rolling(window).std()

        return df

    def train_prophet(self, df, forecast_days=30):
        """Train Facebook Prophet model"""
        print("Training Prophet model...")

        # Prepare data for Prophet
        prophet_df = df[['date', 'revenue']].copy()
        prophet_df.columns = ['ds', 'y']

        # Add Indian holidays as custom seasonality
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode='multiplicative'
        )

        # Add custom regressors for festivals
        if 'is_festival' in df.columns:
            prophet_df['is_festival'] = df['is_festival'].values
            model.add_regressor('is_festival')

        # Fit model
        model.fit(prophet_df)

        # Make future dataframe
        future = model.make_future_dataframe(periods=forecast_days)
        if 'is_festival' in prophet_df.columns:
            future['is_festival'] = 0  # Default to no festival

        forecast = model.predict(future)

        self.models['prophet'] = {
            'model': model,
            'forecast': forecast
        }

        return forecast

    def train_arima(self, df, order=(5, 1, 2), forecast_days=30):
        """Train ARIMA model"""
        print("Training ARIMA model...")

        y = df['revenue'].values

        # Fit ARIMA
        model = ARIMA(y, order=order)
        fitted = model.fit()

        # Forecast
        forecast = fitted.forecast(steps=forecast_days)

        self.models['arima'] = {
            'model': fitted,
            'forecast': forecast,
            'aic': fitted.aic
        }

        return forecast

    def train_xgboost(self, df, forecast_days=30):
        """Train XGBoost model with time features"""
        print("Training XGBoost model...")

        # Create features
        df_features = self.create_features(df)
        df_features = df_features.dropna()

        feature_cols = ['dayofweek', 'month', 'quarter', 'dayofmonth', 'weekofyear',
                       'is_month_start', 'is_month_end', 'is_weekend', 'is_festival',
                       'lag_1', 'lag_7', 'lag_14', 'lag_30',
                       'rolling_mean_7', 'rolling_mean_14', 'rolling_mean_30']

        X = df_features[feature_cols]
        y = df_features['revenue']

        # Train-test split (last 30 days as test)
        train_size = len(X) - forecast_days
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Train model
        model = XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        model.fit(X_train, y_train)

        # Predictions
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        self.models['xgboost'] = {
            'model': model,
            'feature_cols': feature_cols,
            'train_pred': train_pred,
            'test_pred': test_pred,
            'y_test': y_test.values
        }

        return test_pred

    def evaluate_models(self, df):
        """Evaluate all models on test data"""
        results = {}

        # Split data
        test_size = 30
        train_df = df[:-test_size]
        test_df = df[-test_size:]
        y_true = test_df['revenue'].values

        # Prophet evaluation
        if 'prophet' in self.models:
            prophet_pred = self.models['prophet']['forecast']['yhat'].values[-test_size:]
            results['Prophet'] = {
                'MAE': mean_absolute_error(y_true, prophet_pred),
                'RMSE': np.sqrt(mean_squared_error(y_true, prophet_pred)),
                'MAPE': mean_absolute_percentage_error(y_true, prophet_pred) * 100
            }

        # ARIMA evaluation
        if 'arima' in self.models:
            arima_pred = self.models['arima']['forecast'][:test_size]
            results['ARIMA'] = {
                'MAE': mean_absolute_error(y_true, arima_pred),
                'RMSE': np.sqrt(mean_squared_error(y_true, arima_pred)),
                'MAPE': mean_absolute_percentage_error(y_true, arima_pred) * 100
            }

        # XGBoost evaluation
        if 'xgboost' in self.models:
            xgb_pred = self.models['xgboost']['test_pred']
            xgb_true = self.models['xgboost']['y_test']
            results['XGBoost'] = {
                'MAE': mean_absolute_error(xgb_true, xgb_pred),
                'RMSE': np.sqrt(mean_squared_error(xgb_true, xgb_pred)),
                'MAPE': mean_absolute_percentage_error(xgb_true, xgb_pred) * 100
            }

        # Find best model
        best_mape = float('inf')
        for name, metrics in results.items():
            if metrics['MAPE'] < best_mape:
                best_mape = metrics['MAPE']
                self.best_model_name = name

        print("\nModel Comparison:")
        print("-" * 60)
        for name, metrics in results.items():
            print(f"{name:15} | MAE: {metrics['MAE']:,.0f} | RMSE: {metrics['RMSE']:,.0f} | MAPE: {metrics['MAPE']:.2f}%")
        print(f"\nBest Model: {self.best_model_name}")

        return results

    def save_models(self, output_dir='../models'):
        """Save trained models"""
        os.makedirs(output_dir, exist_ok=True)

        if 'xgboost' in self.models:
            joblib.dump(self.models['xgboost']['model'], f'{output_dir}/xgboost_model.joblib')
            joblib.dump(self.models['xgboost']['feature_cols'], f'{output_dir}/feature_cols.joblib')

        print(f"Models saved to {output_dir}/")


class InventoryOptimizer:
    """Inventory optimization based on forecasts"""

    def __init__(self, safety_stock_days=3, lead_time_days=2):
        self.safety_stock_days = safety_stock_days
        self.lead_time_days = lead_time_days

    def calculate_optimal_stock(self, forecast, avg_daily_sales, category_info=None):
        """Calculate optimal stock level"""
        # Average forecasted daily demand
        avg_forecast = np.mean(forecast)

        # Safety stock
        safety_stock = avg_forecast * self.safety_stock_days

        # Reorder point
        reorder_point = (avg_forecast * self.lead_time_days) + safety_stock

        # Weekly order quantity
        weekly_demand = avg_forecast * 7
        order_quantity = weekly_demand + safety_stock

        return {
            'avg_daily_demand': round(avg_forecast, 0),
            'safety_stock': round(safety_stock, 0),
            'reorder_point': round(reorder_point, 0),
            'weekly_order_qty': round(order_quantity, 0)
        }

    def what_if_analysis(self, base_forecast, scenario):
        """What-if scenario analysis"""
        adjusted_forecast = base_forecast.copy()

        if scenario == 'festival_boost':
            adjusted_forecast = base_forecast * 1.5
        elif scenario == 'economic_slowdown':
            adjusted_forecast = base_forecast * 0.8
        elif scenario == 'competitor_entry':
            adjusted_forecast = base_forecast * 0.9
        elif scenario == 'price_increase':
            adjusted_forecast = base_forecast * 0.85

        return adjusted_forecast


def main():
    """Main forecasting pipeline"""
    print("="*60)
    print("INDIAN SME SALES FORECASTING")
    print("="*60)

    # Load data
    df = pd.read_csv('../data/retail_sales_data.csv')
    df['date'] = pd.to_datetime(df['date'])

    # Initialize pipeline
    pipeline = SalesForecastingPipeline()

    # Prepare data
    daily_df = pipeline.prepare_data(df)
    print(f"\nData range: {daily_df['date'].min()} to {daily_df['date'].max()}")
    print(f"Total days: {len(daily_df)}")

    # Train models
    pipeline.train_prophet(daily_df, forecast_days=30)
    pipeline.train_arima(daily_df, order=(5, 1, 2), forecast_days=30)
    pipeline.train_xgboost(daily_df, forecast_days=30)

    # Evaluate
    results = pipeline.evaluate_models(daily_df)

    # Save models
    pipeline.save_models()

    # Inventory optimization example
    print("\n" + "="*60)
    print("INVENTORY OPTIMIZATION")
    print("="*60)

    optimizer = InventoryOptimizer()
    avg_daily = daily_df['revenue'].mean()

    if 'xgboost' in pipeline.models:
        forecast = pipeline.models['xgboost']['test_pred']
        stock_levels = optimizer.calculate_optimal_stock(forecast, avg_daily)

        print("\nOptimal Stock Levels:")
        for key, value in stock_levels.items():
            print(f"  {key}: {value:,.0f}")

    print("\nForecasting pipeline complete!")

if __name__ == "__main__":
    main()
