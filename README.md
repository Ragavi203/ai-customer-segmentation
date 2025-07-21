# 🎯 AI-Driven Customer Segmentation

> Automate customer segmentation for personalized marketing in e-commerce using advanced machine learning and AI-powered insights.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-v1.29+-red.svg)
![Claude API](https://img.shields.io/badge/Claude%20API-Haiku-green.svg)
![Machine Learning](https://img.shields.io/badge/ML-scikit--learn-orange.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 🌟 Project Overview

This project revolutionizes e-commerce customer segmentation by combining traditional machine learning clustering algorithms with AI-powered marketing analysis. Built with Streamlit, it provides an intuitive dashboard for businesses to understand their customers, identify distinct segments, and generate actionable marketing strategies using Claude API.

### 🎯 Key Use Cases

- **E-commerce Personalization**: Segment customers based on purchase behavior, demographics, and engagement patterns
- **Marketing Strategy Optimization**: Generate AI-powered marketing strategies tailored to each customer segment
- **Customer Lifetime Value Analysis**: Predict and optimize CLV for different customer groups
- **Churn Prevention**: Identify at-risk customers and develop retention strategies
- **Product Recommendations**: Create targeted product recommendation systems
- **Campaign Optimization**: Design personalized marketing campaigns for maximum ROI

## 🚀 Features

### 🤖 Advanced Machine Learning
- **Multiple Clustering Algorithms**: K-Means, Gaussian Mixture Models, Agglomerative Clustering, DBSCAN
- **Automatic Model Selection**: System automatically chooses the best-performing algorithm
- **RFM Analysis**: Comprehensive Recency, Frequency, Monetary value analysis
- **Feature Engineering**: 20+ behavioral and demographic features
- **PCA Visualization**: Dimensionality reduction for cluster visualization

### 🧠 AI-Powered Marketing Intelligence
- **Claude API Integration**: Leverages Anthropic's Claude for marketing analysis
- **Personalized Strategies**: Custom marketing strategies for each customer segment
- **Campaign Generation**: AI-generated campaign ideas with specific targeting
- **Risk Assessment**: Automated churn risk and opportunity analysis
- **Personalization Rules**: Website and app personalization recommendations

### 📊 Interactive Dashboard
- **Real-time Analytics**: Live customer segmentation and analysis
- **Rich Visualizations**: Interactive plots with Plotly (scatter plots, heatmaps, radar charts)
- **Export Capabilities**: CSV data export and HTML dashboard generation
- **Responsive Design**: Works on desktop and mobile devices

## 🛠️ Tech Stack

### Backend & ML
- **Python 3.8+** - Core programming language
- **scikit-learn** - Machine learning algorithms and preprocessing
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **PyCaret** - Low-code machine learning library

### AI & NLP
- **Anthropic Claude API** - AI-powered marketing analysis and strategy generation
- **SHAP** - Model explainability and feature importance

### Frontend & Visualization
- **Streamlit** - Web application framework
- **Plotly** - Interactive visualizations and dashboards
- **Seaborn & Matplotlib** - Statistical data visualization

### Data Processing
- **Pandas** - Data manipulation and analysis
- **Python-dotenv** - Environment variable management

## 📊 Dataset & Features

### Input Data Schema
| Feature | Description | Type | Example |
|---------|-------------|------|---------|
| `customer_id` | Unique customer identifier | String | CUST_00001 |
| `age` | Customer age | Integer | 34 |
| `gender` | Customer gender | String | F, M, Other |
| `location` | Customer location | String | New York |
| `total_orders` | Number of orders placed | Integer | 12 |
| `total_spent` | Total money spent | Float | $2,450.00 |
| `avg_order_value` | Average order value | Float | $125.50 |
| `days_since_last_purchase` | Recency metric | Integer | 15 |
| `customer_lifetime_value` | Predicted CLV | Float | $3,200.00 |
| `email_open_rate` | Email engagement rate | Float | 0.65 |
| `mobile_usage_pct` | Mobile usage percentage | Float | 0.78 |

### Generated Features
- **RFM Scores**: Recency, Frequency, Monetary scores (1-5 scale)
- **Behavioral Metrics**: Purchase intensity, value efficiency, engagement scores
- **Customer Lifecycle**: Customer maturity, days as customer
- **Channel Preferences**: Mobile vs desktop usage patterns

## 🎯 Business Impact

### For E-commerce Businesses
- **Increase Revenue**: Targeted marketing strategies can improve conversion rates by 15-25%
- **Reduce Churn**: Early identification of at-risk customers reduces churn by 10-20%
- **Optimize Marketing Spend**: Personalized campaigns improve ROI by 20-30%
- **Enhance Customer Experience**: Tailored experiences increase customer satisfaction

### For Marketing Teams
- **Data-Driven Decisions**: Replace intuition with AI-powered insights
- **Campaign Automation**: Generate campaign ideas and strategies automatically
- **Performance Tracking**: Monitor segment performance with detailed analytics
- **Scalable Personalization**: Implement personalization at scale

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/Ragavi203/ai-customer-segmentation.git
cd ai-customer-segmentation
```

### 2. Set Up Environment
```bash
# Create virtual environment
python -m venv customer_seg_env

# Activate environment
# Windows:
customer_seg_env\Scripts\activate
# Mac/Linux:
source customer_seg_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment
```bash
# Copy environment template
cp .env.template .env

# Edit .env file and add your Claude API key
# Get your API key from: https://console.anthropic.com/
```

### 4. Run the Application
```bash
# Generate sample data (optional)
python data/sample_data.py

# Launch Streamlit app
streamlit run app.py
```

Visit `http://localhost:8501` to access the dashboard.

## 💡 Usage Guide

### Step 1: Data Loading
- **Generate Sample Data**: Create realistic e-commerce customer data for testing
- **Upload CSV**: Use your own customer data
- **Data Validation**: System automatically validates and processes your data

### Step 2: Customer Segmentation
- **Algorithm Selection**: Choose specific algorithm or let the system auto-select
- **Feature Engineering**: System creates 20+ behavioral and demographic features
- **Model Training**: Multiple algorithms tested with performance comparison
- **Segment Profiling**: Detailed analysis of each customer segment

### Step 3: AI-Powered Analysis
- **Marketing Strategies**: Generate comprehensive marketing strategies for each segment
- **Campaign Ideas**: Create specific campaign concepts with targeting details
- **Risk Analysis**: Identify churn risks and growth opportunities
- **Personalization Rules**: Get specific rules for website/app personalization

### Step 4: Visualization & Export
- **Interactive Dashboards**: Explore segments with interactive visualizations
- **Export Data**: Download segmented customer data as CSV
- **Generate Reports**: Create comprehensive HTML dashboard reports

## 📈 Sample Results

### Customer Segments Identified
1. **VIP Champions** (8%): High-value, loyal customers with excellent engagement
2. **Loyal Customers** (23%): Regular purchasers with strong brand loyalty
3. **New Customers** (15%): Recent acquisitions with growth potential
4. **At-Risk Customers** (12%): Previously active customers showing decline
5. **Price-Sensitive Shoppers** (42%): Value-conscious customers needing incentives

### AI-Generated Insights
- **Personalized Email Campaigns**: Segment-specific subject lines and content
- **Dynamic Pricing Strategies**: Tailored offers based on price sensitivity
- **Product Recommendations**: Category preferences by segment
- **Channel Optimization**: Preferred communication channels and timing

## 🔧 Customization

### Adding Custom Features
```python
# Edit src/data_preprocessing.py
def create_custom_features(self, df):
    # Add custom business logic
    df['custom_metric'] = df['feature1'] / df['feature2']
    df['seasonal_score'] = self.calculate_seasonality(df)
    return df
```

### Custom AI Prompts
```python
# Edit src/claude_analyzer.py
def custom_analysis(self, segment_data):
    prompt = f"""
    Analyze this {segment_data['name']} segment for retail industry.
    Focus on: {self.business_context}
    """
    return self.generate_analysis(prompt)
```

## 📁 Project Structure

```
ai-customer-segmentation/
├── 📄 README.md                 # Project documentation
├── 📄 requirements.txt          # Python dependencies
├── 📄 .env.template             # Environment template
├── 📄 .gitignore               # Git ignore rules
├── 🚀 app.py                   # Main Streamlit application
├── 📊 data/
│   ├── 🐍 sample_data.py       # Sample data generator
│   └── 📋 .gitkeep            # Preserve directory structure
├── 🔧 src/
│   ├── 📝 __init__.py
│   ├── 🔄 data_preprocessing.py  # Data preprocessing & feature engineering
│   ├── 🤖 segmentation_model.py  # ML clustering algorithms
│   ├── 🧠 claude_analyzer.py     # Claude API integration
│   └── 📈 visualization.py       # Interactive visualizations
└── 📊 visualizations/          # Generated HTML dashboards (git-ignored)
```

## 🔍 Performance & Scalability

### Recommended Specifications
- **Dataset Size**: Optimal for 1K-50K customers
- **Memory Requirements**: 4GB+ RAM recommended
- **Processing Time**: ~30 seconds for 5K customers
- **Storage**: ~100MB for sample dataset

### Optimization Tips
- **Feature Selection**: Remove irrelevant features for better performance
- **Batch Processing**: Process large datasets in chunks
- **Caching**: Enable Streamlit caching for faster repeated analysis

## 🤝 Contributing

1. **Fork the repository**
2. **Create feature branch** (`git checkout -b feature/AmazingFeature`)
3. **Commit changes** (`git commit -m 'Add AmazingFeature'`)
4. **Push to branch** (`git push origin feature/AmazingFeature`)
5. **Open Pull Request**

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Code formatting
black src/ app.py
flake8 src/ app.py
```

## 🛡️ Security & Privacy

- **API Key Protection**: Environment variables and gitignore protection
- **Data Privacy**: No customer data stored permanently
- **Secure Communication**: HTTPS for all API communications
- **Access Control**: Environment-based configuration management

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **[Anthropic](https://anthropic.com/)** - For the powerful Claude API
- **[Streamlit](https://streamlit.io/)** - For the amazing web app framework
- **[Plotly](https://plotly.com/)** - For interactive visualization capabilities
- **[scikit-learn](https://scikit-learn.org/)** - For robust machine learning algorithms

---

<div align="center">

**⭐ Star this repository if you found it helpful!**

**🔄 Fork it to customize for your business needs!**

**🤝 Contribute to make it even better!**

[⬆ Back to Top](#-ai-driven-customer-segmentation)

</div>
