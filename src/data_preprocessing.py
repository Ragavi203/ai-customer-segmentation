import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

class CustomerDataPreprocessor:
    """Preprocess customer data for segmentation analysis"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        self.categorical_columns = []
        self.numerical_columns = []
        
    def load_data(self, file_path):
        """Load customer data from CSV"""
        try:
            df = pd.read_csv(file_path)
            print(f"✅ Loaded {len(df)} customer records")
            return df
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
    
    def create_rfm_features(self, df):
        """Create RFM (Recency, Frequency, Monetary) features"""
        # Recency: Days since last purchase (already in data)
        df['recency'] = df['days_since_last_purchase']
        
        # Frequency: Total number of orders (already in data)
        df['frequency'] = df['total_orders']
        
        # Monetary: Total amount spent (already in data)
        df['monetary'] = df['total_spent']
        
        # RFM Score calculation (1-5 scale for each component)
        df['recency_score'] = pd.qcut(df['recency'], 5, labels=[5,4,3,2,1])  # Lower recency = higher score
        df['frequency_score'] = pd.qcut(df['frequency'].rank(method='first'), 5, labels=[1,2,3,4,5])
        df['monetary_score'] = pd.qcut(df['monetary'], 5, labels=[1,2,3,4,5])
        
        # Combined RFM score
        df['rfm_score'] = (
            df['recency_score'].astype(int) * 100 + 
            df['frequency_score'].astype(int) * 10 + 
            df['monetary_score'].astype(int)
        )
        
        return df
    
    def create_behavioral_features(self, df):
        """Create additional behavioral features"""
        # Purchase patterns
        df['orders_per_day'] = df['total_orders'] / (df['days_as_customer'] + 1)
        df['spend_per_day'] = df['total_spent'] / (df['days_as_customer'] + 1)
        
        # Engagement metrics
        df['email_engagement'] = (df['email_open_rate'] + df['email_click_rate']) / 2
        df['high_value_customer'] = (df['customer_lifetime_value'] > df['customer_lifetime_value'].quantile(0.8)).astype(int)
        
        # Customer maturity
        df['customer_maturity'] = pd.cut(
            df['days_as_customer'], 
            bins=[0, 90, 365, 730, float('inf')], 
            labels=['New', 'Regular', 'Loyal', 'VIP']
        )
        
        # Purchase intensity
        df['purchase_intensity'] = df['total_orders'] / df['days_as_customer'].replace(0, 1)
        
        # Value per order efficiency
        df['value_efficiency'] = df['total_spent'] / df['total_orders']
        
        return df
    
    def encode_categorical_features(self, df):
        """Encode categorical variables"""
        categorical_cols = ['gender', 'location', 'customer_maturity']
        
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
                self.categorical_columns.append(f'{col}_encoded')
        
        return df
    
    def select_features_for_clustering(self, df):
        """Select and prepare features for clustering"""
        # Core features for segmentation
        feature_cols = [
            # RFM features
            'recency', 'frequency', 'monetary',
            'recency_score', 'frequency_score', 'monetary_score',
            
            # Behavioral features
            'avg_order_value', 'orders_per_day', 'spend_per_day',
            'email_engagement', 'mobile_usage_pct', 'return_rate',
            'q4_purchase_ratio', 'purchase_intensity', 'value_efficiency',
            
            # Demographic features (encoded)
            'age', 'gender_encoded', 'location_encoded', 'customer_maturity_encoded',
            
            # Customer lifecycle
            'days_as_customer', 'customer_lifetime_value',
            'support_tickets'
        ]
        
        # Filter to existing columns
        available_cols = [col for col in feature_cols if col in df.columns]
        self.feature_columns = available_cols
        
        # Reset categorical columns list to only include those that exist
        self.categorical_columns = [col for col in self.categorical_columns if col in df.columns]
        self.numerical_columns = [col for col in available_cols if col not in self.categorical_columns]
        
        return df[available_cols].copy()
    
    def handle_missing_values(self, df):
        """Handle missing values"""
        # Use median for numerical columns
        if self.numerical_columns:
            num_imputer = SimpleImputer(strategy='median')
            df[self.numerical_columns] = num_imputer.fit_transform(df[self.numerical_columns])
        
        # Use most frequent for categorical columns
        if self.categorical_columns:
            # Filter categorical columns to only those that exist in the dataframe
            existing_cat_cols = [col for col in self.categorical_columns if col in df.columns]
            if existing_cat_cols:
                cat_imputer = SimpleImputer(strategy='most_frequent')
                df[existing_cat_cols] = cat_imputer.fit_transform(df[existing_cat_cols])
        
        return df
    
    def scale_features(self, df):
        """Scale numerical features"""
        df_scaled = df.copy()
        df_scaled[self.numerical_columns] = self.scaler.fit_transform(df[self.numerical_columns])
        return df_scaled
    
    def preprocess(self, file_path):
        """Main preprocessing pipeline"""
        print("🔄 Starting data preprocessing...")
        
        # Reset lists
        self.categorical_columns = []
        self.numerical_columns = []
        self.feature_columns = []
        
        # Load data
        df = self.load_data(file_path)
        if df is None:
            return None, None
        
        print(f"📊 Original data shape: {df.shape}")
        
        # Create features
        print("🎯 Creating RFM features...")
        df = self.create_rfm_features(df)
        
        print("🎯 Creating behavioral features...")
        df = self.create_behavioral_features(df)
        
        print("🎯 Encoding categorical features...")
        df = self.encode_categorical_features(df)
        
        # Select features for clustering
        print("🎯 Selecting features for clustering...")
        features_df = self.select_features_for_clustering(df)
        
        print(f"📊 Selected {len(features_df.columns)} features: {list(features_df.columns)}")
        
        # Handle missing values
        print("🎯 Handling missing values...")
        features_df = self.handle_missing_values(features_df)
        
        # Scale features
        print("🎯 Scaling features...")
        features_scaled = self.scale_features(features_df)
        
        print("✅ Preprocessing completed!")
        print(f"📊 Final feature matrix shape: {features_scaled.shape}")
        
        return df, features_scaled
    
    def get_feature_importance_summary(self):
        """Get summary of features used"""
        summary = {
            'total_features': len(self.feature_columns),
            'numerical_features': len(self.numerical_columns),
            'categorical_features': len(self.categorical_columns),
            'feature_list': self.feature_columns
        }
        return summary

# Example usage and testing
if __name__ == "__main__":
    # Test the preprocessor
    preprocessor = CustomerDataPreprocessor()
    
    # Check if data exists
    import os
    if not os.path.exists('data/customer_data.csv'):
        print("❌ Customer data not found. Run sample_data.py first!")
        print("💡 Execute: python data/sample_data.py")
    else:
        # Preprocess data
        original_df, features_df = preprocessor.preprocess('data/customer_data.csv')
        
        if original_df is not None:
            print("\n📋 Feature Summary:")
            summary = preprocessor.get_feature_importance_summary()
            for key, value in summary.items():
                if key != 'feature_list':
                    print(f"  {key}: {value}")
            
            print("\n🎯 Feature List:")
            for i, feature in enumerate(summary['feature_list'], 1):
                print(f"  {i:2d}. {feature}")
            
            print(f"\n📊 Sample of processed features:")
            print(features_df.head())