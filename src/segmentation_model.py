import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class CustomerSegmentationModel:
    """Advanced customer segmentation using multiple clustering algorithms"""
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.pca = None
        self.cluster_profiles = {}
        self.evaluation_results = {}
        
    def find_optimal_clusters(self, X, max_clusters=10):
        """Find optimal number of clusters using elbow method and silhouette analysis"""
        sse = []
        silhouette_scores = []
        k_range = range(2, min(max_clusters + 1, len(X)))
        
        print("🔍 Finding optimal number of clusters...")
        
        for k in k_range:
            # KMeans clustering
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X)
            
            # Calculate metrics
            sse.append(kmeans.inertia_)
            sil_score = silhouette_score(X, cluster_labels)
            silhouette_scores.append(sil_score)
            
            print(f"  K={k}: SSE={kmeans.inertia_:.2f}, Silhouette={sil_score:.3f}")
        
        # Find elbow point (simplified)
        # Calculate the rate of change in SSE
        sse_diff = np.diff(sse)
        sse_diff2 = np.diff(sse_diff)
        
        # Find the point where the second derivative is maximum (elbow)
        if len(sse_diff2) > 0:
            elbow_k = k_range[np.argmax(sse_diff2) + 2]
        else:
            elbow_k = k_range[len(k_range)//2]
        
        # Find best silhouette score
        best_sil_k = k_range[np.argmax(silhouette_scores)]
        
        # Choose final k (balance between elbow and silhouette)
        optimal_k = best_sil_k  # Prioritize silhouette score
        
        print(f"📊 Elbow method suggests: {elbow_k} clusters")
        print(f"📊 Best silhouette score at: {best_sil_k} clusters")
        print(f"✅ Selected optimal clusters: {optimal_k}")
        
        return optimal_k, sse, silhouette_scores
    
    def train_multiple_models(self, X, n_clusters=None):
        """Train multiple clustering models and compare performance"""
        if n_clusters is None:
            n_clusters, _, _ = self.find_optimal_clusters(X)
        
        print(f"\n🤖 Training clustering models with {n_clusters} clusters...")
        
        # Initialize models
        models_config = {
            'KMeans': KMeans(n_clusters=n_clusters, random_state=42, n_init=10),
            'GaussianMixture': GaussianMixture(n_components=n_clusters, random_state=42),
            'AgglomerativeClustering': AgglomerativeClustering(n_clusters=n_clusters),
            'DBSCAN': DBSCAN(eps=0.5, min_samples=5)
        }
        
        results = {}
        
        for name, model in models_config.items():
            try:
                print(f"  Training {name}...")
                
                # Fit model
                if name == 'GaussianMixture':
                    model.fit(X)
                    labels = model.predict(X)
                else:
                    labels = model.fit_predict(X)
                
                # Store model
                self.models[name] = model
                
                # Evaluate if we have more than 1 cluster
                unique_labels = len(np.unique(labels))
                if unique_labels > 1 and unique_labels < len(X):
                    silhouette = silhouette_score(X, labels)
                    calinski = calinski_harabasz_score(X, labels)
                    
                    results[name] = {
                        'model': model,
                        'labels': labels,
                        'n_clusters': unique_labels,
                        'silhouette_score': silhouette,
                        'calinski_harabasz_score': calinski,
                        'score': silhouette * 0.7 + (calinski / 1000) * 0.3  # Combined score
                    }
                    
                    print(f"    ✅ Clusters: {unique_labels}, Silhouette: {silhouette:.3f}, Calinski-Harabasz: {calinski:.2f}")
                else:
                    print(f"    ❌ Invalid clustering result (clusters: {unique_labels})")
                    
            except Exception as e:
                print(f"    ❌ Error training {name}: {e}")
        
        # Select best model
        if results:
            best_model_name = max(results.keys(), key=lambda k: results[k]['score'])
            self.best_model_name = best_model_name
            self.best_model = results[best_model_name]['model']
            self.evaluation_results = results
            
            print(f"\n🏆 Best model: {best_model_name}")
            print(f"📊 Score: {results[best_model_name]['score']:.3f}")
            
            return results[best_model_name]['labels']
        else:
            print("❌ No valid clustering results!")
            return None
    
    def create_cluster_profiles(self, original_df, features_df, cluster_labels):
        """Create detailed profiles for each cluster"""
        print("\n📊 Creating cluster profiles...")
        
        # Add cluster labels to dataframes
        df_with_clusters = original_df.copy()
        df_with_clusters['cluster'] = cluster_labels
        
        profiles = {}
        n_clusters = len(np.unique(cluster_labels))
        
        for cluster_id in range(n_clusters):
            cluster_data = df_with_clusters[df_with_clusters['cluster'] == cluster_id]
            cluster_size = len(cluster_data)
            
            if cluster_size == 0:
                continue
            
            # Basic statistics
            profile = {
                'cluster_id': cluster_id,
                'size': cluster_size,
                'percentage': (cluster_size / len(df_with_clusters)) * 100,
                
                # Demographics
                'avg_age': cluster_data['age'].mean(),
                'gender_distribution': cluster_data['gender'].value_counts(normalize=True).to_dict(),
                'top_locations': cluster_data['location'].value_counts().head(3).to_dict(),
                
                # RFM Analysis
                'avg_recency': cluster_data['recency'].mean(),
                'avg_frequency': cluster_data['frequency'].mean(),
                'avg_monetary': cluster_data['monetary'].mean(),
                'avg_rfm_score': cluster_data['rfm_score'].mean(),
                
                # Purchase Behavior
                'avg_order_value': cluster_data['avg_order_value'].mean(),
                'avg_total_spent': cluster_data['total_spent'].mean(),
                'avg_clv': cluster_data['customer_lifetime_value'].mean(),
                
                # Engagement
                'avg_email_engagement': cluster_data['email_engagement'].mean(),
                'avg_mobile_usage': cluster_data['mobile_usage_pct'].mean(),
                'avg_return_rate': cluster_data['return_rate'].mean(),
                
                # Customer Lifecycle
                'avg_days_as_customer': cluster_data['days_as_customer'].mean(),
                'maturity_distribution': cluster_data['customer_maturity'].value_counts(normalize=True).to_dict(),
                
                # Support
                'avg_support_tickets': cluster_data['support_tickets'].mean(),
            }
            
            # Cluster characteristics (compared to overall mean)
            overall_means = df_with_clusters.select_dtypes(include=[np.number]).mean()
            cluster_means = cluster_data.select_dtypes(include=[np.number]).mean()
            
            profile['characteristics'] = {}
            key_metrics = ['total_spent', 'total_orders', 'avg_order_value', 'customer_lifetime_value', 
                          'recency', 'email_engagement', 'mobile_usage_pct']
            
            for metric in key_metrics:
                if metric in cluster_means and metric in overall_means:
                    ratio = cluster_means[metric] / overall_means[metric] if overall_means[metric] != 0 else 1
                    if ratio > 1.2:
                        profile['characteristics'][metric] = 'High'
                    elif ratio < 0.8:
                        profile['characteristics'][metric] = 'Low'
                    else:
                        profile['characteristics'][metric] = 'Average'
            
            profiles[cluster_id] = profile
            
            print(f"  Cluster {cluster_id}: {cluster_size} customers ({profile['percentage']:.1f}%)")
        
        self.cluster_profiles = profiles
        return profiles
    
    def generate_cluster_names(self, profiles):
        """Generate meaningful names for clusters based on their characteristics"""
        cluster_names = {}
        
        for cluster_id, profile in profiles.items():
            chars = profile['characteristics']
            
            # Analyze key characteristics to generate names
            if chars.get('customer_lifetime_value') == 'High' and chars.get('total_spent') == 'High':
                name = "VIP Champions"
            elif chars.get('recency') == 'Low' and chars.get('total_orders') == 'High':
                name = "Loyal Customers"
            elif chars.get('recency') == 'High' and chars.get('total_spent') == 'Low':
                name = "At-Risk Customers"
            elif chars.get('total_orders') == 'Low' and profile['avg_days_as_customer'] < 90:
                name = "New Customers"
            elif chars.get('avg_order_value') == 'High' and chars.get('total_orders') == 'Low':
                name = "Big Spenders"
            elif chars.get('email_engagement') == 'High':
                name = "Engaged Customers"
            elif chars.get('mobile_usage_pct') == 'High':
                name = "Mobile-First Users"
            else:
                name = f"Segment {cluster_id}"
            
            cluster_names[cluster_id] = name
        
        return cluster_names
    
    def perform_pca_analysis(self, X, n_components=2):
        """Perform PCA for visualization"""
        print(f"\n🔍 Performing PCA analysis with {n_components} components...")
        
        self.pca = PCA(n_components=n_components, random_state=42)
        X_pca = self.pca.fit_transform(X)
        
        # Explained variance
        explained_variance = self.pca.explained_variance_ratio_
        total_variance = sum(explained_variance)
        
        print(f"📊 PCA Results:")
        for i, var in enumerate(explained_variance):
            print(f"  Component {i+1}: {var:.3f} ({var*100:.1f}% variance)")
        print(f"  Total variance explained: {total_variance:.3f} ({total_variance*100:.1f}%)")
        
        return X_pca
    
    def segment_customers(self, original_df, features_df, n_clusters=None):
        """Main segmentation pipeline"""
        print("🎯 Starting customer segmentation...")
        
        # Train models and get best clustering
        cluster_labels = self.train_multiple_models(features_df.values, n_clusters)
        
        if cluster_labels is None:
            print("❌ Segmentation failed!")
            return None, None, None
        
        # Create cluster profiles
        profiles = self.create_cluster_profiles(original_df, features_df, cluster_labels)
        
        # Generate meaningful names
        cluster_names = self.generate_cluster_names(profiles)
        
        # Add names to profiles
        for cluster_id, name in cluster_names.items():
            if cluster_id in profiles:
                profiles[cluster_id]['name'] = name
        
        # PCA for visualization
        X_pca = self.perform_pca_analysis(features_df.values)
        
        print("\n✅ Customer segmentation completed!")
        print(f"📊 Created {len(profiles)} customer segments")
        
        # Print segment summary
        print("\n🏷️ Customer Segments:")
        for cluster_id, profile in profiles.items():
            print(f"  {profile['name']}: {profile['size']} customers ({profile['percentage']:.1f}%)")
        
        return cluster_labels, profiles, X_pca
    
    def get_customer_segment(self, customer_features):
        """Predict segment for a new customer"""
        if self.best_model is None:
            return None
        
        try:
            if hasattr(self.best_model, 'predict'):
                return self.best_model.predict([customer_features])[0]
            else:
                # For models without predict method (like DBSCAN)
                return self.best_model.fit_predict([customer_features])[0]
        except:
            return None
    
    def get_model_summary(self):
        """Get summary of model performance"""
        if not self.evaluation_results:
            return None
        
        summary = {
            'best_model': self.best_model_name,
            'models_tested': len(self.evaluation_results),
            'performance_metrics': {}
        }
        
        for name, results in self.evaluation_results.items():
            summary['performance_metrics'][name] = {
                'silhouette_score': results['silhouette_score'],
                'calinski_harabasz_score': results['calinski_harabasz_score'],
                'n_clusters': results['n_clusters']
            }
        
        return summary

# Example usage and testing
if __name__ == "__main__":
    from data_preprocessing import CustomerDataPreprocessor
    import os
    
    # Check if preprocessed data exists
    if not os.path.exists('data/customer_data.csv'):
        print("❌ Customer data not found. Run sample_data.py first!")
        print("💡 Execute: python data/sample_data.py")
    else:
        print("🚀 Testing Customer Segmentation Model...")
        
        # Load and preprocess data
        preprocessor = CustomerDataPreprocessor()
        original_df, features_df = preprocessor.preprocess('data/customer_data.csv')
        
        if original_df is not None and features_df is not None:
            # Create segmentation model
            segmentation_model = CustomerSegmentationModel()
            
            # Perform segmentation
            cluster_labels, profiles, X_pca = segmentation_model.segment_customers(
                original_df, features_df, n_clusters=5
            )
            
            if cluster_labels is not None:
                # Display detailed results
                print("\n" + "="*60)
                print("📊 DETAILED CLUSTER ANALYSIS")
                print("="*60)
                
                for cluster_id, profile in profiles.items():
                    print(f"\n🏷️ {profile['name']} (Cluster {cluster_id})")
                    print(f"   Size: {profile['size']} customers ({profile['percentage']:.1f}%)")
                    print(f"   Avg Age: {profile['avg_age']:.1f} years")
                    print(f"   Avg CLV: ${profile['avg_clv']:,.2f}")
                    print(f"   Avg Order Value: ${profile['avg_order_value']:.2f}")
                    print(f"   Avg Total Spent: ${profile['avg_total_spent']:,.2f}")
                    print(f"   Recency: {profile['avg_recency']:.1f} days")
                    print(f"   Frequency: {profile['avg_frequency']:.1f} orders")
                    print(f"   Email Engagement: {profile['avg_email_engagement']:.3f}")
                    print(f"   Key Characteristics: {', '.join([f'{k}={v}' for k, v in profile['characteristics'].items()])}")
                
                # Model performance summary
                print(f"\n📈 MODEL PERFORMANCE:")
                summary = segmentation_model.get_model_summary()
                if summary:
                    print(f"   Best Model: {summary['best_model']}")
                    print(f"   Models Tested: {summary['models_tested']}")
                    
                    best_metrics = summary['performance_metrics'][summary['best_model']]
                    print(f"   Silhouette Score: {best_metrics['silhouette_score']:.3f}")
                    print(f"   Calinski-Harabasz Score: {best_metrics['calinski_harabasz_score']:.2f}")
                
                print("\n✅ Segmentation model testing completed!")
            else:
                print("❌ Segmentation failed!")
        else:
            print("❌ Data preprocessing failed!")