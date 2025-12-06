"""
Indian Retail Sales Data Generator
Generates realistic sales data with India-specific patterns
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

np.random.seed(42)

# Indian festival dates (approximate) for 2023-2024
INDIAN_FESTIVALS = {
    '2023-01-26': ('Republic Day', 1.3),
    '2023-03-08': ('Holi', 1.8),
    '2023-04-14': ('Baisakhi', 1.4),
    '2023-08-15': ('Independence Day', 1.3),
    '2023-08-30': ('Raksha Bandhan', 1.6),
    '2023-09-07': ('Janmashtami', 1.4),
    '2023-10-24': ('Dussehra', 1.7),
    '2023-11-12': ('Diwali', 2.5),  # Biggest shopping season
    '2023-11-13': ('Diwali', 2.5),
    '2023-11-14': ('Diwali', 2.3),
    '2023-12-25': ('Christmas', 1.5),
    '2024-01-26': ('Republic Day', 1.3),
    '2024-03-25': ('Holi', 1.8),
    '2024-08-15': ('Independence Day', 1.3),
    '2024-10-12': ('Dussehra', 1.7),
    '2024-11-01': ('Diwali', 2.5),
    '2024-11-02': ('Diwali', 2.5),
}

# Product categories with base sales and margins
PRODUCT_CATEGORIES = {
    'grocery': {'base_sales': 500, 'margin': 0.08, 'perishable': True},
    'snacks': {'base_sales': 300, 'margin': 0.15, 'perishable': True},
    'beverages': {'base_sales': 250, 'margin': 0.12, 'perishable': False},
    'personal_care': {'base_sales': 150, 'margin': 0.20, 'perishable': False},
    'household': {'base_sales': 100, 'margin': 0.18, 'perishable': False},
    'dairy': {'base_sales': 400, 'margin': 0.10, 'perishable': True},
    'fruits_vegetables': {'base_sales': 350, 'margin': 0.25, 'perishable': True},
    'packaged_foods': {'base_sales': 200, 'margin': 0.14, 'perishable': False},
    'confectionery': {'base_sales': 180, 'margin': 0.22, 'perishable': False},
    'stationery': {'base_sales': 80, 'margin': 0.30, 'perishable': False},
}

# Indian states with regional weights
STATES = {
    'Maharashtra': 1.2,
    'Delhi': 1.15,
    'Karnataka': 1.1,
    'Tamil Nadu': 1.08,
    'Gujarat': 1.05,
    'West Bengal': 1.0,
    'Rajasthan': 0.95,
    'Uttar Pradesh': 0.9,
    'Kerala': 1.05,
    'Telangana': 1.08,
}

def get_festival_multiplier(date):
    """Get sales multiplier for festival dates"""
    date_str = date.strftime('%Y-%m-%d')

    # Check exact festival date
    if date_str in INDIAN_FESTIVALS:
        return INDIAN_FESTIVALS[date_str][1]

    # Check pre-festival shopping (7 days before Diwali)
    for festival_date, (name, mult) in INDIAN_FESTIVALS.items():
        if 'Diwali' in name:
            festival_dt = datetime.strptime(festival_date, '%Y-%m-%d')
            days_before = (festival_dt - date).days
            if 1 <= days_before <= 7:
                return 1 + (mult - 1) * (1 - days_before / 10)

    return 1.0

def get_payday_multiplier(date):
    """Month-end and month-start payday effect"""
    day = date.day
    if day <= 5 or day >= 28:
        return 1.25
    return 1.0

def get_weekend_multiplier(date):
    """Weekend shopping boost"""
    if date.weekday() >= 5:  # Saturday, Sunday
        return 1.15
    return 1.0

def get_seasonal_multiplier(date, category):
    """Seasonal effects based on category"""
    month = date.month

    # Summer boost for beverages
    if category == 'beverages' and month in [4, 5, 6]:
        return 1.4

    # Monsoon effect
    if month in [7, 8, 9]:
        if category in ['grocery', 'packaged_foods']:
            return 1.1
        if category == 'fruits_vegetables':
            return 0.9

    # Winter boost for dairy
    if category == 'dairy' and month in [11, 12, 1, 2]:
        return 1.15

    # School season for stationery
    if category == 'stationery' and month in [6, 7]:
        return 1.8

    return 1.0

def generate_daily_sales(start_date, end_date, store_id, state, category):
    """Generate daily sales for a store-category combination"""
    records = []
    current_date = start_date
    category_info = PRODUCT_CATEGORIES[category]
    base_sales = category_info['base_sales']
    state_multiplier = STATES[state]

    # Add some store-specific randomness
    store_factor = np.random.uniform(0.8, 1.2)

    while current_date <= end_date:
        # Calculate all multipliers
        festival_mult = get_festival_multiplier(current_date)
        payday_mult = get_payday_multiplier(current_date)
        weekend_mult = get_weekend_multiplier(current_date)
        seasonal_mult = get_seasonal_multiplier(current_date, category)

        # Combine multipliers
        total_mult = festival_mult * payday_mult * weekend_mult * seasonal_mult * state_multiplier * store_factor

        # Add noise
        noise = np.random.normal(1, 0.15)

        # Calculate daily sales
        daily_sales = max(0, base_sales * total_mult * noise)

        # Calculate units and revenue
        avg_price = np.random.uniform(50, 500)
        units_sold = int(daily_sales / avg_price * 10)
        revenue = units_sold * avg_price

        records.append({
            'date': current_date,
            'store_id': store_id,
            'state': state,
            'category': category,
            'units_sold': units_sold,
            'revenue': round(revenue, 2),
            'avg_price': round(avg_price, 2),
            'day_of_week': current_date.weekday(),
            'is_weekend': current_date.weekday() >= 5,
            'month': current_date.month,
            'is_month_end': current_date.day >= 28,
            'is_month_start': current_date.day <= 5,
            'festival_multiplier': festival_mult,
            'is_festival': festival_mult > 1.0,
        })

        current_date += timedelta(days=1)

    return records

def generate_dataset(n_stores=20, start_date='2023-01-01', end_date='2024-11-30'):
    """Generate complete sales dataset"""
    print("Generating Indian retail sales data...")

    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')

    all_records = []

    # Generate data for each store
    for i in range(n_stores):
        store_id = f"STORE_{i+1:03d}"
        state = np.random.choice(list(STATES.keys()))

        print(f"Generating data for {store_id} in {state}...")

        # Generate for each category
        for category in PRODUCT_CATEGORIES.keys():
            records = generate_daily_sales(start, end, store_id, state, category)
            all_records.extend(records)

    df = pd.DataFrame(all_records)

    print(f"\nDataset generated successfully!")
    print(f"Total records: {len(df):,}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Unique stores: {df['store_id'].nunique()}")
    print(f"Categories: {df['category'].nunique()}")

    return df

def save_dataset(df, output_dir='../data'):
    """Save generated dataset"""
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(f'{output_dir}/retail_sales_data.csv', index=False)
    print(f"Saved to {output_dir}/retail_sales_data.csv")

    # Also save aggregated data
    weekly = df.groupby([pd.Grouper(key='date', freq='W'), 'category']).agg({
        'units_sold': 'sum',
        'revenue': 'sum'
    }).reset_index()
    weekly.to_csv(f'{output_dir}/weekly_sales.csv', index=False)
    print(f"Saved weekly aggregation to {output_dir}/weekly_sales.csv")

if __name__ == "__main__":
    df = generate_dataset(n_stores=20)
    save_dataset(df)
    print("\nSample data:")
    print(df.head(10))
