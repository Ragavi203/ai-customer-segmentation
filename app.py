import streamlit as st
import pandas as pd
import os
import json
from datetime import datetime
import plotly.express as px

# Import our custom modules
from src.data_preprocessing import CustomerDataPreprocessor
from src.segmentation_model import CustomerSegmentationModel
from src.claude_analyzer import ClaudeMarketingAnalyzer
from src.visualization import CustomerSegmentationVisualizer
from data.sample_data import generate_customer_data

# Page configuration
st.set_page_config(
    page_title="AI Customer Segmentation",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .segment-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
    .warning-box {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #ffeaa7;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'segmentation_complete' not in st.session_state:
    st.session_state.segmentation_complete = False
if 'claude_analysis_complete' not in st.session_state:
    st.session_state.claude_analysis_complete = False

# Main header
st.markdown('<h1 class="main-header">🎯 AI-Driven Customer Segmentation</h1>', unsafe_allow_html=True)
st.markdown("**Automate customer segmentation for personalized marketing in e-commerce using Claude API**")

# Sidebar
st.sidebar.header("🛠️ Configuration")

# Check for Claude API key
claude_api_key = os.getenv('ANTHROPIC_API_KEY')
if claude_api_key:
    st.sidebar.success("✅ Claude API key configured")
else:
    st.sidebar.error("❌ Claude API key not found")
    st.sidebar.info("Add ANTHROPIC_API_KEY to your .env file")

# Main tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Data", "🤖 Segmentation", "🧠 AI Analysis", "📈 Visualizations", "📋 Reports"])

# Tab 1: Data Management
with tab1:
    st.header("📊 Data Management")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Data Source")
        data_option = st.radio(
            "Choose data source:",
            ["Generate Sample Data", "Upload CSV File", "Use Existing Data"]
        )
        
        if data_option == "Generate Sample Data":
            st.info("Generate realistic e-commerce customer data for demonstration")
            
            n_customers = st.slider("Number of customers", 1000, 10000, 5000, 500)
            
            if st.button("🎲 Generate Sample Data", type="primary"):
                with st.spinner("Generating customer data..."):
                    try:
                        # Create data directory
                        os.makedirs('data', exist_ok=True)
                        
                        # Generate data
                        df = generate_customer_data(n_customers)
                        df.to_csv('data/customer_data.csv', index=False)
                        
                        st.session_state.data_loaded = True
                        st.session_state.customer_data = df
                        
                        st.success(f"✅ Generated {len(df)} customer records!")
                        
                    except Exception as e:
                        st.error(f"❌ Error generating data: {e}")
        
        elif data_option == "Upload CSV File":
            uploaded_file = st.file_uploader("Upload customer data CSV", type="csv")
            
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.session_state.customer_data = df
                    st.session_state.data_loaded = True
                    st.success(f"✅ Loaded {len(df)} records from uploaded file")
                except Exception as e:
                    st.error(f"❌ Error loading file: {e}")
        
        elif data_option == "Use Existing Data":
            if os.path.exists('data/customer_data.csv'):
                try:
                    df = pd.read_csv('data/customer_data.csv')
                    st.session_state.customer_data = df
                    st.session_state.data_loaded = True
                    st.success(f"✅ Loaded existing data: {len(df)} records")
                except Exception as e:
                    st.error(f"❌ Error loading existing data: {e}")
            else:
                st.warning("⚠️ No existing data found. Generate sample data first.")
    
    with col2:
        if st.session_state.data_loaded:
            st.subheader("📈 Data Overview")
            df = st.session_state.customer_data
            
            st.metric("Total Customers", f"{len(df):,}")
            st.metric("Features", len(df.columns))
            st.metric("Data Quality", f"{((1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100):.1f}%")
    
    # Data preview
    if st.session_state.data_loaded:
        st.subheader("🔍 Data Preview")
        df = st.session_state.customer_data
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Sample Records:**")
            st.dataframe(df.head(10), use_container_width=True)
        
        with col2:
            st.write("**Data Summary:**")
            st.dataframe(df.describe(), use_container_width=True)

# Tab 2: Customer Segmentation
with tab2:
    st.header("🤖 Customer Segmentation")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data first in the Data tab")
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Segmentation Configuration")
            
            n_clusters = st.selectbox(
                "Number of clusters (leave empty for auto-detection)",
                options=[None, 3, 4, 5, 6, 7, 8],
                format_func=lambda x: "Auto-detect optimal" if x is None else str(x)
            )
            
            algorithm_info = st.expander("ℹ️ Algorithm Information")
            with algorithm_info:
                st.write("""
                **Algorithms tested:**
                - K-Means: Traditional centroid-based clustering
                - Gaussian Mixture: Probabilistic clustering with overlapping clusters
                - Agglomerative: Hierarchical clustering
                - DBSCAN: Density-based clustering for irregular shapes
                
                The system automatically selects the best performing algorithm based on silhouette score and Calinski-Harabasz index.
                """)
            
            if st.button("🚀 Run Segmentation Analysis", type="primary"):
                with st.spinner("Running customer segmentation..."):
                    try:
                        # Preprocess data
                        preprocessor = CustomerDataPreprocessor()
                        original_df, features_df = preprocessor.preprocess('data/customer_data.csv')
                        
                        if original_df is not None:
                            # Run segmentation
                            segmentation_model = CustomerSegmentationModel()
                            cluster_labels, profiles, X_pca = segmentation_model.segment_customers(
                                original_df, features_df, n_clusters
                            )
                            
                            if cluster_labels is not None:
                                # Store results in session state
                                st.session_state.segmentation_complete = True
                                st.session_state.cluster_labels = cluster_labels
                                st.session_state.cluster_profiles = profiles
                                st.session_state.X_pca = X_pca
                                st.session_state.original_df = original_df
                                st.session_state.features_df = features_df
                                st.session_state.segmentation_model = segmentation_model
                                
                                st.success("✅ Segmentation completed successfully!")
                            else:
                                st.error("❌ Segmentation failed!")
                        else:
                            st.error("❌ Data preprocessing failed!")
                            
                    except Exception as e:
                        st.error(f"❌ Error during segmentation: {e}")
        
        with col2:
            if st.session_state.segmentation_complete:
                st.subheader("📊 Segmentation Results")
                
                profiles = st.session_state.cluster_profiles
                
                st.metric("Segments Created", len(profiles))
                
                model_summary = st.session_state.segmentation_model.get_model_summary()
                if model_summary:
                    st.metric("Best Algorithm", model_summary['best_model'])
                    best_metrics = model_summary['performance_metrics'][model_summary['best_model']]
                    st.metric("Silhouette Score", f"{best_metrics['silhouette_score']:.3f}")
        
        # Segment overview
        if st.session_state.segmentation_complete:
            st.subheader("🏷️ Customer Segments Overview")
            
            profiles = st.session_state.cluster_profiles
            
            for cluster_id, profile in profiles.items():
                with st.container():
                    st.markdown(f"""
                    <div class="segment-card">
                        <h4>🎯 {profile.get('name', f'Segment {cluster_id}')}</h4>
                        <div style="display: flex; gap: 2rem; margin: 1rem 0;">
                            <div><strong>Size:</strong> {profile['size']} customers ({profile['percentage']:.1f}%)</div>
                            <div><strong>Avg Age:</strong> {profile['avg_age']:.1f} years</div>
                            <div><strong>Avg CLV:</strong> ${profile['avg_clv']:,.2f}</div>
                            <div><strong>Avg AOV:</strong> ${profile['avg_order_value']:.2f}</div>
                        </div>
                        <div><strong>Key Characteristics:</strong> {', '.join([f'{k}={v}' for k, v in profile['characteristics'].items()])}</div>
                    </div>
                    """, unsafe_allow_html=True)

# Tab 3: AI Analysis with Claude
with tab3:
    st.header("🧠 AI Marketing Analysis with Claude")
    
    if not st.session_state.segmentation_complete:
        st.warning("⚠️ Please complete customer segmentation first")
    else:
        if not claude_api_key:
            st.error("❌ Claude API key not configured. Please add ANTHROPIC_API_KEY to your .env file")
        else:
            st.subheader("Marketing Strategy Generation")
            
            analysis_type = st.selectbox(
                "Select analysis type:",
                [
                    "Comprehensive Marketing Report",
                    "Marketing Strategies by Segment", 
                    "Risk & Opportunity Analysis",
                    "Personalization Rules",
                    "Campaign Ideas for Specific Segment"
                ]
            )
            
            if analysis_type == "Campaign Ideas for Specific Segment":
                profiles = st.session_state.cluster_profiles
                segment_names = [prof.get('name', f'Segment {cid}') for cid, prof in profiles.items()]
                selected_segment = st.selectbox("Select segment:", segment_names)
            
            if st.button("🧠 Generate AI Analysis", type="primary"):
                with st.spinner("Generating AI analysis with Claude..."):
                    try:
                        analyzer = ClaudeMarketingAnalyzer()
                        
                        if analysis_type == "Comprehensive Marketing Report":
                            model_summary = st.session_state.segmentation_model.get_model_summary()
                            result = analyzer.generate_comprehensive_report(
                                st.session_state.cluster_profiles, 
                                model_summary
                            )
                            
                            if result['status'] == 'success':
                                st.session_state.claude_analysis_complete = True
                                st.session_state.claude_report = result['report']
                                st.success("✅ Comprehensive report generated!")
                            else:
                                st.error(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")
                        
                        elif analysis_type == "Marketing Strategies by Segment":
                            result = analyzer.generate_marketing_strategies(st.session_state.cluster_profiles)
                            
                            if result['status'] == 'success':
                                st.success("✅ Marketing strategies generated!")
                                st.markdown("### 📋 Marketing Strategies")
                                st.markdown(result['strategies'])
                            else:
                                st.error(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")
                        
                        elif analysis_type == "Risk & Opportunity Analysis":
                            result = analyzer.analyze_segment_risks(st.session_state.cluster_profiles)
                            
                            if result['status'] == 'success':
                                st.success("✅ Risk analysis completed!")
                                st.markdown("### ⚠️ Risk & Opportunity Analysis")
                                st.markdown(result['risk_analysis'])
                            else:
                                st.error(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")
                        
                        elif analysis_type == "Personalization Rules":
                            result = analyzer.generate_personalization_rules(st.session_state.cluster_profiles)
                            
                            if result['status'] == 'success':
                                st.success("✅ Personalization rules generated!")
                                st.markdown("### 🎨 Personalization Rules")
                                st.markdown(result['personalization_rules'])
                            else:
                                st.error(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")
                        
                        elif analysis_type == "Campaign Ideas for Specific Segment":
                            # Find selected segment profile
                            selected_profile = None
                            for cid, prof in st.session_state.cluster_profiles.items():
                                if prof.get('name', f'Segment {cid}') == selected_segment:
                                    selected_profile = prof
                                    break
                            
                            if selected_profile:
                                result = analyzer.generate_campaign_ideas(selected_segment, selected_profile)
                                
                                if result['status'] == 'success':
                                    st.success(f"✅ Campaign ideas generated for {selected_segment}!")
                                    st.markdown(f"### 🎯 Campaign Ideas for {selected_segment}")
                                    st.markdown(result['campaigns'])
                                else:
                                    st.error(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")
                            else:
                                st.error("❌ Selected segment not found")
                    
                    except Exception as e:
                        st.error(f"❌ Error during AI analysis: {e}")
            
            # Display comprehensive report if available
            if st.session_state.claude_analysis_complete and 'claude_report' in st.session_state:
                st.subheader("📊 Comprehensive Marketing Report")
                
                report = st.session_state.claude_report
                
                if 'executive_summary' in report:
                    with st.expander("📋 Executive Summary", expanded=True):
                        st.markdown(report['executive_summary'])
                
                if 'marketing_strategies' in report:
                    with st.expander("🎯 Marketing Strategies"):
                        st.markdown(report['marketing_strategies'])
                
                if 'risk_analysis' in report:
                    with st.expander("⚠️ Risk Analysis"):
                        st.markdown(report['risk_analysis'])
                
                if 'personalization_rules' in report:
                    with st.expander("🎨 Personalization Rules"):
                        st.markdown(report['personalization_rules'])

# Tab 4: Visualizations
with tab4:
    st.header("📈 Interactive Visualizations")
    
    if not st.session_state.segmentation_complete:
        st.warning("⚠️ Please complete customer segmentation first")
    else:
        # Create visualizations
        visualizer = CustomerSegmentationVisualizer()
        
        # Get cluster names mapping
        cluster_names = {}
        for cluster_id, profile in st.session_state.cluster_profiles.items():
            if 'name' in profile:
                cluster_names[cluster_id] = profile['name']
        
        # Visualization options
        viz_option = st.selectbox(
            "Select visualization:",
            [
                "Customer Segments (PCA)",
                "Segment Distribution", 
                "RFM Analysis Heatmap",
                "Segment Metrics Comparison",
                "Customer Lifetime Value",
                "Purchase Behavior",
                "Engagement Metrics"
            ]
        )
        
        try:
            if viz_option == "Customer Segments (PCA)":
                fig = visualizer.plot_cluster_scatter(
                    st.session_state.X_pca, 
                    st.session_state.cluster_labels, 
                    cluster_names
                )
                st.plotly_chart(fig, use_container_width=True)
            
            elif viz_option == "Segment Distribution":
                fig = visualizer.plot_segment_sizes(st.session_state.cluster_profiles)
                st.plotly_chart(fig, use_container_width=True)
            
            elif viz_option == "RFM Analysis Heatmap":
                fig = visualizer.plot_rfm_heatmap(st.session_state.cluster_profiles)
                st.plotly_chart(fig, use_container_width=True)
            
            elif viz_option == "Segment Metrics Comparison":
                fig = visualizer.plot_segment_metrics_radar(st.session_state.cluster_profiles)
                st.plotly_chart(fig, use_container_width=True)
            
            elif viz_option == "Customer Lifetime Value":
                fig = visualizer.plot_clv_distribution(
                    st.session_state.original_df, 
                    st.session_state.cluster_labels, 
                    cluster_names
                )
                st.plotly_chart(fig, use_container_width=True)
            
            elif viz_option == "Purchase Behavior":
                fig = visualizer.plot_purchase_behavior_comparison(st.session_state.cluster_profiles)
                st.plotly_chart(fig, use_container_width=True)
            
            elif viz_option == "Engagement Metrics":
                fig = visualizer.plot_engagement_metrics(st.session_state.cluster_profiles)
                st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"❌ Error creating visualization: {e}")

# Tab 5: Reports
with tab5:
    st.header("📋 Export Reports")
    
    if not st.session_state.segmentation_complete:
        st.warning("⚠️ Please complete customer segmentation first")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Data Export")
            
            if st.button("💾 Export Segmented Customer Data"):
                try:
                    # Add cluster labels to original data
                    df_export = st.session_state.original_df.copy()
                    df_export['cluster_id'] = st.session_state.cluster_labels
                    
                    # Add cluster names
                    cluster_names_map = {}
                    for cluster_id, profile in st.session_state.cluster_profiles.items():
                        cluster_names_map[cluster_id] = profile.get('name', f'Segment {cluster_id}')
                    
                    df_export['segment_name'] = df_export['cluster_id'].map(cluster_names_map)
                    
                    # Convert to CSV
                    csv = df_export.to_csv(index=False)
                    
                    st.download_button(
                        label="⬇️ Download Customer Segments CSV",
                        data=csv,
                        file_name=f"customer_segments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    
                    st.success("✅ Customer data prepared for download!")
                    
                except Exception as e:
                    st.error(f"❌ Error preparing export: {e}")
        
        with col2:
            st.subheader("📈 Visualization Export")
            
            if st.button("🎨 Generate All Visualizations"):
                try:
                    with st.spinner("Creating all visualizations..."):
                        visualizer = CustomerSegmentationVisualizer()
                        
                        cluster_names = {}
                        for cluster_id, profile in st.session_state.cluster_profiles.items():
                            if 'name' in profile:
                                cluster_names[cluster_id] = profile['name']
                        
                        dashboard_plots = visualizer.create_dashboard_layout(
                            st.session_state.original_df,
                            st.session_state.features_df,
                            st.session_state.cluster_labels,
                            st.session_state.cluster_profiles,
                            st.session_state.X_pca,
                            cluster_names
                        )
                        
                        # Save visualizations
                        index_path = visualizer.save_plots_as_html(dashboard_plots)
                        
                        st.success("✅ All visualizations generated!")
                        st.info(f"📁 Saved to: {index_path}")
                        
                except Exception as e:
                    st.error(f"❌ Error generating visualizations: {e}")
        
        # Summary statistics
        if st.session_state.segmentation_complete:
            st.subheader("📈 Analysis Summary")
            
            profiles = st.session_state.cluster_profiles
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Customers", f"{sum(prof['size'] for prof in profiles.values()):,}")
            
            with col2:
                st.metric("Customer Segments", len(profiles))
            
            with col3:
                avg_clv = sum(prof['avg_clv'] * prof['size'] for prof in profiles.values()) / sum(prof['size'] for prof in profiles.values())
                st.metric("Average CLV", f"${avg_clv:,.2f}")
            
            with col4:
                if st.session_state.segmentation_model:
                    model_summary = st.session_state.segmentation_model.get_model_summary()
                    if model_summary:
                        best_score = model_summary['performance_metrics'][model_summary['best_model']]['silhouette_score']
                        st.metric("Model Quality", f"{best_score:.3f}")

# Footer
st.markdown("---")
st.markdown("**🎯 AI Customer Segmentation Dashboard** | Built with Streamlit & Claude API")