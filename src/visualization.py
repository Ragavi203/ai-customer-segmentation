import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class CustomerSegmentationVisualizer:
    """Create interactive visualizations for customer segmentation analysis"""
    
    def __init__(self):
        self.color_palette = px.colors.qualitative.Set3
        
    def plot_cluster_scatter(self, X_pca, cluster_labels, cluster_names=None, title="Customer Segments (PCA Visualization)"):
        """Create 2D scatter plot of clusters using PCA"""
        
        # Create DataFrame for plotting
        df_plot = pd.DataFrame({
            'PC1': X_pca[:, 0],
            'PC2': X_pca[:, 1],
            'Cluster': cluster_labels
        })
        
        # Add cluster names if provided
        if cluster_names:
            df_plot['Segment'] = df_plot['Cluster'].map(cluster_names)
            color_col = 'Segment'
        else:
            df_plot['Segment'] = df_plot['Cluster'].astype(str)
            color_col = 'Segment'
        
        fig = px.scatter(
            df_plot, 
            x='PC1', 
            y='PC2', 
            color=color_col,
            title=title,
            labels={
                'PC1': 'First Principal Component',
                'PC2': 'Second Principal Component'
            },
            color_discrete_sequence=self.color_palette
        )
        
        fig.update_layout(
            width=800,
            height=600,
            title_x=0.5
        )
        
        return fig
    
    def plot_segment_sizes(self, cluster_profiles, title="Customer Segment Distribution"):
        """Create pie chart showing segment sizes"""
        
        segments = []
        sizes = []
        percentages = []
        
        for cluster_id, profile in cluster_profiles.items():
            name = profile.get('name', f'Cluster {cluster_id}')
            segments.append(name)
            sizes.append(profile['size'])
            percentages.append(profile['percentage'])
        
        fig = go.Figure(data=[go.Pie(
            labels=segments,
            values=sizes,
            hole=0.3,
            textinfo='label+percent',
            textposition='outside',
            marker_colors=self.color_palette[:len(segments)]
        )])
        
        fig.update_layout(
            title=title,
            title_x=0.5,
            width=600,
            height=500
        )
        
        return fig
    
    def plot_rfm_heatmap(self, cluster_profiles, title="RFM Analysis by Segment"):
        """Create heatmap showing RFM scores by segment"""
        
        segments = []
        recency_scores = []
        frequency_scores = []
        monetary_scores = []
        
        for cluster_id, profile in cluster_profiles.items():
            name = profile.get('name', f'Cluster {cluster_id}')
            segments.append(name)
            
            # Normalize scores to 1-5 scale for better visualization
            recency_scores.append(5 - min(4, profile['avg_recency'] // 30))  # Lower recency = higher score
            frequency_scores.append(min(5, max(1, profile['avg_frequency'])))
            monetary_scores.append(min(5, max(1, profile['avg_monetary'] // 200)))
        
        # Create heatmap data
        heatmap_data = np.array([recency_scores, frequency_scores, monetary_scores]).T
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            x=['Recency', 'Frequency', 'Monetary'],
            y=segments,
            colorscale='RdYlGn',
            text=heatmap_data,
            texttemplate="%{text}",
            textfont={"size": 12},
            colorbar=dict(title="Score (1-5)")
        ))
        
        fig.update_layout(
            title=title,
            title_x=0.5,
            width=500,
            height=400
        )
        
        return fig
    
    def plot_segment_metrics_radar(self, cluster_profiles, metrics=None):
        """Create radar chart comparing segments across key metrics"""
        
        if metrics is None:
            metrics = ['avg_clv', 'avg_order_value', 'avg_email_engagement', 
                      'avg_mobile_usage', 'avg_frequency', 'avg_total_spent']
        
        fig = go.Figure()
        
        for cluster_id, profile in cluster_profiles.items():
            name = profile.get('name', f'Cluster {cluster_id}')
            
            # Normalize metrics to 0-1 scale for radar chart
            values = []
            for metric in metrics:
                if metric in profile:
                    val = profile[metric]
                    # Normalize based on metric type
                    if 'clv' in metric or 'spent' in metric:
                        normalized = min(1.0, val / 3000)  # Normalize CLV/spending
                    elif 'order_value' in metric:
                        normalized = min(1.0, val / 200)  # Normalize AOV
                    elif 'engagement' in metric or 'usage' in metric:
                        normalized = val  # Already 0-1 scale
                    elif 'frequency' in metric:
                        normalized = min(1.0, val / 20)  # Normalize frequency
                    else:
                        normalized = val
                    values.append(normalized)
                else:
                    values.append(0)
            
            # Close the radar chart
            values.append(values[0])
            metrics_labels = metrics + [metrics[0]]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics_labels,
                fill='toself',
                name=name,
                line_color=self.color_palette[cluster_id % len(self.color_palette)]
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Segment Comparison - Key Metrics",
            title_x=0.5,
            width=700,
            height=500
        )
        
        return fig
    
    def plot_clv_distribution(self, original_df, cluster_labels, cluster_names=None):
        """Plot Customer Lifetime Value distribution by segment"""
        
        df_plot = original_df.copy()
        df_plot['cluster'] = cluster_labels
        
        if cluster_names:
            df_plot['segment'] = df_plot['cluster'].map(cluster_names)
        else:
            df_plot['segment'] = df_plot['cluster'].astype(str)
        
        fig = px.box(
            df_plot,
            x='segment',
            y='customer_lifetime_value',
            color='segment',
            title="Customer Lifetime Value Distribution by Segment",
            color_discrete_sequence=self.color_palette
        )
        
        fig.update_layout(
            xaxis_title="Customer Segment",
            yaxis_title="Customer Lifetime Value ($)",
            title_x=0.5,
            width=800,
            height=500
        )
        
        return fig
    
    def plot_purchase_behavior_comparison(self, cluster_profiles):
        """Create comparison charts for purchase behavior metrics"""
        
        segments = []
        avg_orders = []
        avg_spending = []
        avg_frequency = []
        
        for cluster_id, profile in cluster_profiles.items():
            name = profile.get('name', f'Cluster {cluster_id}')
            segments.append(name)
            avg_orders.append(profile['avg_order_value'])
            avg_spending.append(profile['avg_total_spent'])
            avg_frequency.append(profile['avg_frequency'])
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Average Order Value', 'Total Spending', 'Purchase Frequency', 'Segment Sizes'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "pie"}]]
        )
        
        # Average Order Value
        fig.add_trace(
            go.Bar(x=segments, y=avg_orders, name="AOV", marker_color=self.color_palette[0]),
            row=1, col=1
        )
        
        # Total Spending
        fig.add_trace(
            go.Bar(x=segments, y=avg_spending, name="Total Spending", marker_color=self.color_palette[1]),
            row=1, col=2
        )
        
        # Purchase Frequency
        fig.add_trace(
            go.Bar(x=segments, y=avg_frequency, name="Frequency", marker_color=self.color_palette[2]),
            row=2, col=1
        )
        
        # Segment Sizes (Pie)
        sizes = [profile['size'] for profile in cluster_profiles.values()]
        fig.add_trace(
            go.Pie(labels=segments, values=sizes, name="Segment Size"),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="Purchase Behavior Analysis by Segment",
            title_x=0.5,
            width=1000,
            height=700,
            showlegend=False
        )
        
        return fig
    
    def plot_engagement_metrics(self, cluster_profiles):
        """Plot engagement metrics comparison"""
        
        segments = []
        email_engagement = []
        mobile_usage = []
        return_rates = []
        
        for cluster_id, profile in cluster_profiles.items():
            name = profile.get('name', f'Cluster {cluster_id}')
            segments.append(name)
            email_engagement.append(profile['avg_email_engagement'])
            mobile_usage.append(profile['avg_mobile_usage'])
            return_rates.append(profile['avg_return_rate'])
        
        fig = go.Figure()
        
        # Add traces for each metric
        fig.add_trace(go.Bar(
            name='Email Engagement',
            x=segments,
            y=email_engagement,
            marker_color=self.color_palette[0]
        ))
        
        fig.add_trace(go.Bar(
            name='Mobile Usage',
            x=segments,
            y=mobile_usage,
            marker_color=self.color_palette[1]
        ))
        
        fig.add_trace(go.Bar(
            name='Return Rate',
            x=segments,
            y=return_rates,
            marker_color=self.color_palette[2]
        ))
        
        fig.update_layout(
            title='Customer Engagement Metrics by Segment',
            xaxis_title='Customer Segment',
            yaxis_title='Engagement Rate',
            barmode='group',
            title_x=0.5,
            width=800,
            height=500
        )
        
        return fig
    
    def create_dashboard_layout(self, original_df, features_df, cluster_labels, cluster_profiles, X_pca, cluster_names=None):
        """Create a comprehensive dashboard with all visualizations"""
        
        dashboard_plots = {}
        
        print("📊 Creating visualization dashboard...")
        
        # 1. PCA Scatter Plot
        dashboard_plots['pca_scatter'] = self.plot_cluster_scatter(X_pca, cluster_labels, cluster_names)
        
        # 2. Segment Distribution
        dashboard_plots['segment_sizes'] = self.plot_segment_sizes(cluster_profiles)
        
        # 3. RFM Heatmap
        dashboard_plots['rfm_heatmap'] = self.plot_rfm_heatmap(cluster_profiles)
        
        # 4. Radar Chart
        dashboard_plots['radar_chart'] = self.plot_segment_metrics_radar(cluster_profiles)
        
        # 5. CLV Distribution
        dashboard_plots['clv_distribution'] = self.plot_clv_distribution(original_df, cluster_labels, cluster_names)
        
        # 6. Purchase Behavior
        dashboard_plots['purchase_behavior'] = self.plot_purchase_behavior_comparison(cluster_profiles)
        
        # 7. Engagement Metrics
        dashboard_plots['engagement_metrics'] = self.plot_engagement_metrics(cluster_profiles)
        
        print("✅ Dashboard visualizations created!")
        return dashboard_plots
    
    def save_plots_as_html(self, dashboard_plots, output_dir="visualizations"):
        """Save all plots as HTML files"""
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        for plot_name, fig in dashboard_plots.items():
            filename = f"{output_dir}/{plot_name}.html"
            fig.write_html(filename)
            print(f"💾 Saved {plot_name} to {filename}")
        
        # Create an index HTML file linking all plots
        index_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Customer Segmentation Dashboard</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .plot-link { 
                    display: inline-block; 
                    margin: 10px; 
                    padding: 15px 25px; 
                    background-color: #4CAF50; 
                    color: white; 
                    text-decoration: none; 
                    border-radius: 5px; 
                }
                .plot-link:hover { background-color: #45a049; }
                h1 { color: #333; }
            </style>
        </head>
        <body>
            <h1>🎯 Customer Segmentation Dashboard</h1>
            <p>Click on any visualization below to view the interactive plot:</p>
        """
        
        for plot_name in dashboard_plots.keys():
            display_name = plot_name.replace('_', ' ').title()
            index_html += f'<a href="{plot_name}.html" class="plot-link">{display_name}</a>\n'
        
        index_html += """
        </body>
        </html>
        """
        
        with open(f"{output_dir}/index.html", "w") as f:
            f.write(index_html)
        
        print(f"📋 Created dashboard index at {output_dir}/index.html")
        
        return f"{output_dir}/index.html"

# Example usage and testing
if __name__ == "__main__":
    from data_preprocessing import CustomerDataPreprocessor
    from segmentation_model import CustomerSegmentationModel
    import os
    
    if not os.path.exists('data/customer_data.csv'):
        print("❌ Customer data not found. Run sample_data.py first!")
    else:
        print("🎨 Testing Customer Segmentation Visualizations...")
        
        # Load and preprocess data
        preprocessor = CustomerDataPreprocessor()
        original_df, features_df = preprocessor.preprocess('data/customer_data.csv')
        
        if original_df is not None and features_df is not None:
            # Create segmentation model
            segmentation_model = CustomerSegmentationModel()
            cluster_labels, profiles, X_pca = segmentation_model.segment_customers(
                original_df, features_df, n_clusters=5
            )
            
            if cluster_labels is not None:
                # Create cluster names mapping
                cluster_names = {}
                for cluster_id, profile in profiles.items():
                    if 'name' in profile:
                        cluster_names[cluster_id] = profile['name']
                
                # Create visualizer
                visualizer = CustomerSegmentationVisualizer()
                
                # Generate all visualizations
                dashboard_plots = visualizer.create_dashboard_layout(
                    original_df, features_df, cluster_labels, profiles, X_pca, cluster_names
                )
                
                # Save visualizations
                index_path = visualizer.save_plots_as_html(dashboard_plots)
                
                print(f"\n✅ Visualization testing completed!")
                print(f"🌐 Open {index_path} in your browser to view the dashboard")
                
                # Show a sample plot
                print("\n📊 Displaying sample visualization...")
                dashboard_plots['pca_scatter'].show()
            else:
                print("❌ Segmentation failed!")
        else:
            print("❌ Data preprocessing failed!")