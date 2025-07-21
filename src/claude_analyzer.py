import json
import os
from anthropic import Anthropic
from dotenv import load_dotenv
import pandas as pd

# Load environment variables
load_dotenv()

class ClaudeMarketingAnalyzer:
    """Use Claude API to generate marketing insights and strategies"""
    
    def __init__(self):
        self.client = None
        self.initialize_client()
    
    def initialize_client(self):
        """Initialize Claude API client"""
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            print("⚠️ ANTHROPIC_API_KEY not found in environment variables!")
            print("💡 Please add your Claude API key to the .env file")
            return False
        
        try:
            self.client = Anthropic(api_key=api_key)
            print("✅ Claude API client initialized successfully!")
            return True
        except Exception as e:
            print(f"❌ Error initializing Claude API: {e}")
            return False
    
    def format_cluster_data_for_claude(self, profiles):
        """Format cluster profiles for Claude analysis"""
        formatted_data = {}
        
        for cluster_id, profile in profiles.items():
            cluster_name = profile.get('name', f"Cluster {cluster_id}")
            
            formatted_data[cluster_name] = {
                'size': f"{profile['size']} customers ({profile['percentage']:.1f}%)",
                'demographics': {
                    'average_age': f"{profile['avg_age']:.1f} years",
                    'gender_distribution': profile['gender_distribution'],
                    'top_locations': profile['top_locations']
                },
                'purchase_behavior': {
                    'average_order_value': f"${profile['avg_order_value']:.2f}",
                    'total_spent_avg': f"${profile['avg_total_spent']:,.2f}",
                    'customer_lifetime_value': f"${profile['avg_clv']:,.2f}",
                    'purchase_frequency': f"{profile['avg_frequency']:.1f} orders",
                    'days_since_last_purchase': f"{profile['avg_recency']:.1f} days"
                },
                'engagement_metrics': {
                    'email_engagement': f"{profile['avg_email_engagement']:.3f}",
                    'mobile_usage': f"{profile['avg_mobile_usage']:.1%}",
                    'return_rate': f"{profile['avg_return_rate']:.1%}",
                    'support_tickets': f"{profile['avg_support_tickets']:.1f}"
                },
                'key_characteristics': profile['characteristics'],
                'customer_maturity': profile['maturity_distribution']
            }
        
        return formatted_data
    
    def generate_marketing_strategies(self, cluster_profiles):
        """Generate personalized marketing strategies for each segment"""
        if not self.client:
            return {"error": "Claude API client not initialized"}
        
        formatted_data = self.format_cluster_data_for_claude(cluster_profiles)
        
        prompt = f"""
As an expert marketing strategist, analyze these customer segments and provide detailed, actionable marketing strategies for each segment.

Customer Segments Data:
{json.dumps(formatted_data, indent=2)}

For each customer segment, provide:

1. **Segment Overview**: Brief description of the segment's key characteristics
2. **Marketing Strategy**: Specific tactics and approaches
3. **Communication Style**: Tone, messaging, and channel preferences
4. **Product Recommendations**: What products/services to promote
5. **Retention Strategy**: How to keep these customers engaged
6. **Growth Opportunities**: How to increase value from this segment
7. **Risk Factors**: Potential challenges and how to address them
8. **Success Metrics**: KPIs to track for this segment

Please provide practical, data-driven recommendations that can be implemented immediately. Focus on ROI and customer experience optimization.

Format your response in clear sections for each customer segment.
"""

        try:
            response = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=4000,
                temperature=0.7,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            
            return {
                "status": "success",
                "strategies": response.content[0].text,
                "segments_analyzed": len(formatted_data)
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "segments_analyzed": 0
            }
    
    def generate_campaign_ideas(self, segment_name, segment_profile):
        """Generate specific campaign ideas for a segment"""
        if not self.client:
            return {"error": "Claude API client not initialized"}
        
        prompt = f"""
Create 3 specific marketing campaign ideas for the customer segment: {segment_name}

Segment Profile:
- Size: {segment_profile['size']} customers ({segment_profile['percentage']:.1f}%)
- Average CLV: ${segment_profile['avg_clv']:,.2f}
- Average Order Value: ${segment_profile['avg_order_value']:.2f}
- Purchase Frequency: {segment_profile['avg_frequency']:.1f} orders
- Email Engagement: {segment_profile['avg_email_engagement']:.3f}
- Key Characteristics: {segment_profile['characteristics']}

For each campaign, provide:
1. Campaign Name
2. Objective
3. Target Channels (email, SMS, social, etc.)
4. Key Messages
5. Offer/Incentive
6. Timeline
7. Expected ROI
8. Success Metrics

Make campaigns creative, practical, and tailored to this specific segment's behavior and preferences.
"""

        try:
            response = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=2000,
                temperature=0.8,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            
            return {
                "status": "success",
                "campaigns": response.content[0].text,
                "segment": segment_name
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "segment": segment_name
            }
    
    def analyze_segment_risks(self, cluster_profiles):
        """Analyze risks and opportunities for each segment"""
        if not self.client:
            return {"error": "Claude API client not initialized"}
        
        formatted_data = self.format_cluster_data_for_claude(cluster_profiles)
        
        prompt = f"""
As a customer analytics expert, analyze these customer segments for business risks and opportunities.

Customer Segments:
{json.dumps(formatted_data, indent=2)}

For each segment, identify:

1. **Risk Assessment**:
   - Churn risk level (High/Medium/Low)
   - Revenue risk factors
   - Competitive threats

2. **Opportunity Analysis**:
   - Upsell/cross-sell potential
   - Lifetime value growth opportunities
   - Market expansion possibilities

3. **Action Priorities**:
   - Immediate actions needed (next 30 days)
   - Medium-term strategies (3-6 months)
   - Long-term investments (6+ months)

4. **Resource Allocation**:
   - Recommended marketing budget allocation (%)
   - Sales team focus areas
   - Customer success priorities

Provide specific, quantifiable recommendations with clear reasoning based on the data.
"""

        try:
            response = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=3500,
                temperature=0.6,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            
            return {
                "status": "success",
                "risk_analysis": response.content[0].text,
                "segments_analyzed": len(formatted_data)
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "segments_analyzed": 0
            }
    
    def generate_personalization_rules(self, cluster_profiles):
        """Generate personalization rules for website/app experience"""
        if not self.client:
            return {"error": "Claude API client not initialized"}
        
        formatted_data = self.format_cluster_data_for_claude(cluster_profiles)
        
        prompt = f"""
Create personalization rules for an e-commerce website based on these customer segments.

Customer Segments:
{json.dumps(formatted_data, indent=2)}

For each segment, define:

1. **Homepage Personalization**:
   - Hero banner messaging
   - Featured products/categories
   - Layout preferences

2. **Product Recommendations**:
   - Recommendation algorithms to use
   - Product categories to emphasize
   - Price range focus

3. **Content Strategy**:
   - Blog topics and content themes
   - Educational vs promotional content mix
   - Content formats (video, text, infographics)

4. **Navigation & UX**:
   - Menu structure preferences
   - Search behavior optimization
   - Mobile vs desktop experience

5. **Promotional Strategy**:
   - Discount types and thresholds
   - Urgency messaging effectiveness
   - Seasonal campaign timing

6. **Email Personalization**:
   - Subject line strategies
   - Content length preferences
   - Send time optimization

Provide implementable rules that can be coded into a personalization engine.
"""

        try:
            response = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=4000,
                temperature=0.7,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            
            return {
                "status": "success",
                "personalization_rules": response.content[0].text,
                "segments_covered": len(formatted_data)
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "segments_covered": 0
            }

    def generate_comprehensive_report(self, cluster_profiles, model_summary=None):
        """Generate a comprehensive marketing analysis report"""
        if not self.client:
            return {"error": "Claude API client not initialized"}
        
        # Get all analyses
        print("🧠 Generating comprehensive marketing analysis with Claude...")
        
        strategies = self.generate_marketing_strategies(cluster_profiles)
        risks = self.analyze_segment_risks(cluster_profiles)
        personalization = self.generate_personalization_rules(cluster_profiles)
        
        # Combine all insights
        report_data = {
            "segmentation_overview": {
                "total_segments": len(cluster_profiles),
                "model_used": model_summary.get('best_model', 'Unknown') if model_summary else 'Unknown',
                "segments": {name: prof['percentage'] for name, prof in cluster_profiles.items() if 'name' in prof}
            },
            "marketing_strategies": strategies.get('strategies', 'Analysis failed'),
            "risk_analysis": risks.get('risk_analysis', 'Analysis failed'),
            "personalization_rules": personalization.get('personalization_rules', 'Analysis failed')
        }
        
        # Generate executive summary
        summary_prompt = f"""
Create an executive summary for this customer segmentation analysis.

Analysis Results:
{json.dumps(report_data, indent=2)}

Provide:
1. **Key Findings** (3-4 bullet points)
2. **Strategic Recommendations** (top 3 priorities)
3. **Expected Business Impact** (revenue, retention, efficiency gains)
4. **Implementation Timeline** (quick wins vs long-term projects)
5. **Resource Requirements** (budget, team, technology)

Keep it concise and executive-friendly (1-2 pages max).
"""

        try:
            summary_response = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=2000,
                temperature=0.6,
                messages=[{
                    "role": "user",
                    "content": summary_prompt
                }]
            )
            
            report_data["executive_summary"] = summary_response.content[0].text
            
            return {
                "status": "success",
                "report": report_data,
                "timestamp": pd.Timestamp.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "partial_report": report_data
            }

# Example usage and testing
if __name__ == "__main__":
    # Test Claude analyzer
    analyzer = ClaudeMarketingAnalyzer()
    
    if analyzer.client:
        print("✅ Claude API connection successful!")
        
        # Test with sample data
        sample_profile = {
            0: {
                'name': 'VIP Champions',
                'size': 500,
                'percentage': 10.0,
                'avg_age': 45.2,
                'avg_clv': 2500.00,
                'avg_order_value': 150.00,
                'avg_frequency': 12.5,
                'avg_email_engagement': 0.75,
                'characteristics': {'customer_lifetime_value': 'High', 'total_spent': 'High'}
            }
        }
        
        print("\n🧠 Testing Claude analysis...")
        result = analyzer.generate_campaign_ideas('VIP Champions', sample_profile[0])
        
        if result['status'] == 'success':
            print("✅ Claude analysis test successful!")
            print(f"📝 Sample output: {result['campaigns'][:200]}...")
        else:
            print(f"❌ Claude analysis test failed: {result.get('error', 'Unknown error')}")
    else:
        print("❌ Claude API connection failed!")
        print("💡 Make sure to add your ANTHROPIC_API_KEY to the .env file")