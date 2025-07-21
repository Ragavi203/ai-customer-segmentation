import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_customer_data(n_customers=5000):
    """Generate realistic e-commerce customer data for segmentation"""
    
    np.random.seed(42)
    random.seed(42)
    
    # Customer demographics
    customer_ids = [f"CUST_{i:05d}" for i in range(1, n_customers + 1)]
    
    # Age distribution (realistic e-commerce demographics)
    ages = np.random.choice(
        [18, 25, 30, 35, 40, 45, 50, 55, 60, 65], 
        size=n_customers, 
        p=[0.05, 0.15, 0.20, 0.18, 0.15, 0.12, 0.08, 0.04, 0.02, 0.01]
    )
    
    # Gender
    genders = np.random.choice(['M', 'F', 'Other'], size=n_customers, p=[0.45, 0.52, 0.03])
    
    # Location (major cities)
    locations = np.random.choice(
        ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia', 
         'San Antonio', 'San Diego', 'Dallas', 'San Jose', 'Austin', 'Jacksonville'],
        size=n_customers
    )
    
    # Customer acquisition date (last 3 years)
    start_date = datetime.now() - timedelta(days=3*365)
    acquisition_dates = [
        start_date + timedelta(days=random.randint(0, 3*365)) 
        for _ in range(n_customers)
    ]
    
    # Purchase behavior (correlated with demographics)
    data = []
    
    for i in range(n_customers):
        # Base purchase frequency based on age and demographics
        if ages[i] < 30:
            base_frequency = np.random.poisson(8)  # Young customers - more frequent
            avg_order_value = np.random.normal(75, 25)
        elif ages[i] < 50:
            base_frequency = np.random.poisson(12)  # Middle-aged - highest frequency
            avg_order_value = np.random.normal(120, 40)
        else:
            base_frequency = np.random.poisson(6)  # Older customers - less frequent but higher value
            avg_order_value = np.random.normal(150, 50)
        
        # Ensure positive values
        total_orders = max(1, base_frequency)
        avg_order_value = max(20, avg_order_value)
        
        # Calculate other metrics
        total_spent = total_orders * avg_order_value + np.random.normal(0, avg_order_value * 0.2)
        total_spent = max(total_spent, avg_order_value)
        
        # Days since last purchase (recency)
        days_since_last_purchase = max(1, int(np.random.exponential(30)))
        
        # Product categories (simulate preferences)
        categories = ['Electronics', 'Clothing', 'Home', 'Books', 'Sports', 'Beauty']
        preferred_categories = random.sample(categories, random.randint(1, 3))
        
        # Customer lifetime value (CLV) calculation
        days_as_customer = (datetime.now() - acquisition_dates[i]).days
        if days_as_customer > 0:
            clv = total_spent + (total_spent / days_as_customer) * 365 * 0.5  # Projected annual value
        else:
            clv = total_spent
        
        # Support tickets (customer service interactions)
        support_tickets = np.random.poisson(2) if total_orders > 5 else np.random.poisson(1)
        
        # Email engagement
        email_open_rate = np.random.beta(2, 3)  # Skewed towards lower open rates
        email_click_rate = email_open_rate * np.random.beta(2, 5)  # Click rate lower than open rate
        
        # Mobile vs desktop usage
        mobile_usage_pct = np.random.beta(3, 2)  # Skewed towards higher mobile usage
        
        # Return rate
        return_rate = np.random.beta(1, 9)  # Most customers have low return rates
        
        # Seasonal purchasing (higher in Q4)
        q4_purchase_ratio = np.random.beta(3, 2) + 0.2  # Boost for holiday season
        
        data.append({
            'customer_id': customer_ids[i],
            'age': ages[i],
            'gender': genders[i],
            'location': locations[i],
            'acquisition_date': acquisition_dates[i],
            'total_orders': total_orders,
            'total_spent': round(total_spent, 2),
            'avg_order_value': round(avg_order_value, 2),
            'days_since_last_purchase': days_since_last_purchase,
            'preferred_categories': ', '.join(preferred_categories),
            'customer_lifetime_value': round(clv, 2),
            'support_tickets': support_tickets,
            'email_open_rate': round(email_open_rate, 3),
            'email_click_rate': round(email_click_rate, 3),
            'mobile_usage_pct': round(mobile_usage_pct, 3),
            'return_rate': round(return_rate, 3),
            'q4_purchase_ratio': round(q4_purchase_ratio, 3),
            'days_as_customer': days_as_customer
        })
    
    df = pd.DataFrame(data)
    return df

def save_sample_data():
    """Generate and save sample data to CSV"""
    print("Generating sample customer data...")
    df = generate_customer_data(5000)
    df.to_csv('data/customer_data.csv', index=False)
    print(f"✅ Generated {len(df)} customer records")
    print(f"📁 Saved to: data/customer_data.csv")
    print("\nData preview:")
    print(df.head())
    print(f"\nData shape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    return df

if __name__ == "__main__":
    # Create data directory if it doesn't exist
    import os
    os.makedirs('data', exist_ok=True)
    
    # Generate and save data
    df = save_sample_data()