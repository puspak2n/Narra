import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import time
import pandas as pd
import json
import base64
import openai
from openai import OpenAI
import os
import numpy as np
from datetime import datetime
from io import BytesIO
from sklearn.preprocessing import LabelEncoder
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import plotly
import textwrap
import re


# In your initialization
def __init__(self, use_agentic_ai=False, openai_api_key=None, strict_agentic_mode=False):
    self.use_agentic_ai = use_agentic_ai
    self.openai_api_key = openai_api_key
    self.strict_agentic_mode = strict_agentic_mode
    
    # Configure OpenAI if API key is provided
    if self.openai_api_key and self.use_agentic_ai:
        import openai
        openai.api_key = self.openai_api_key

def load_openai_key():
    try:
        return st.secrets["openai"]["api_key"]
    except KeyError:
        return None

# Load OpenAI Key
openai.api_key = load_openai_key()
USE_OPENAI = openai.api_key is not None

# Function to create dark-themed charts
def create_dark_chart(fig):
    """Apply consistent dark theme to charts"""
    fig.update_layout(
        plot_bgcolor='#111827',
        paper_bgcolor='#111827',
        font_color='#F3F4F6',
        template='plotly_dark',
        margin=dict(t=40, b=40, l=40, r=40),
        title_font_size=16,
        title_font_color='#60A5FA',
        xaxis=dict(
            gridcolor='#374151',
            zerolinecolor='#374151'
        ),
        yaxis=dict(
            gridcolor='#374151',
            zerolinecolor='#374151'
        )
    )
    return fig

# Smart data filtering helper
def get_business_relevant_columns(df):
    """Filter out ID columns and other non-analytical fields"""
    avoid_patterns = [
        'id', 'row_id', 'order_id', 'customer_id', 'product_id', 
        'transaction_id', 'invoice_id', 'record_id', 'index', 'key',
        'guid', 'uuid', 'code', 'sku', '_id'
    ]
    business_columns = {
        'categorical': [],
        'numerical': [],
        'temporal': []
    }
    for col in df.columns:
        col_lower = col.lower().replace(' ', '_').replace('-', '_')
        if any(pattern in col_lower for pattern in avoid_patterns):
            continue
        if df[col].dtype == 'object' and df[col].nunique() > 0.8 * len(df):
            continue
        if df[col].dtype == 'object' and df[col].nunique() < 50:
            business_columns['categorical'].append(col)
        elif pd.api.types.is_datetime64_any_dtype(df[col]) or 'date' in col_lower:
            business_columns['temporal'].append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            business_columns['numerical'].append(col)
    return business_columns

# Updated AI Agent Classes
class DataAnalystAgent:
    def __init__(self):
        self.name = "Data Analyst Agent"
    
    def analyze_dataset(self, df):
        """Perform intelligent data analysis using AI"""
        try:
            business_cols = get_business_relevant_columns(df)
            profile = self._generate_smart_data_profile(df, business_cols)
            analysis_prompt = f"""
            You are an expert data analyst. Analyze this business dataset and provide insights in VALID JSON format.

            Dataset Overview:
            - Shape: {df.shape[0]} rows, {df.shape[1]} columns
            - Business Categorical Columns: {business_cols['categorical']}
            - Business Numerical Columns: {business_cols['numerical']}
            - Temporal Columns: {business_cols['temporal']}
            
            Data Profile:
            {profile}
            
            User Learning Context: {st.session_state.agent_learning.get('business_context', 'No context provided')}
            Previous Feedback: {st.session_state.user_feedback[-3:] if st.session_state.user_feedback else 'No feedback yet'}

            IMPORTANT: Respond ONLY with valid JSON. No markdown, no explanations outside JSON.
            For each recommended chart, provide:
            - "type": chart type
            - "x_col": x-axis column
            - "y_col": y-axis column (if applicable)
            - "title": chart title
            - "reason": business reasoning
            - "priority": high/medium/low
            - "code": complete Python code using Plotly (assign the figure to 'fig')
            
            {
                "patterns": ["list of 3-5 key business patterns found"],
                "relationships": ["list of 2-4 important relationships between business metrics"],
                "recommendations": [
                    {
                        "type": "bar|line|scatter|pie",
                        "x_col": "column_name",
                        "y_col": "column_name", 
                        "reason": "business justification",
                        "priority": "high|medium|low",
                        "code": "complete Python code using Plotly, assign the figure to 'fig'"
                    }
                ],
                "quality": ["2-3 data quality observations"],
                "insights": ["3-4 business insights with potential"]
            }
            """
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": analysis_prompt}],
                temperature=0.1,
                max_tokens=2000
            )
            response_text = response.choices[0].message.content.strip()
            if response_text.startswith('```json'):
                response_text = response_text.replace('```json', '').replace('```', '').strip()
            try:
                analysis = json.loads(response_text)
                required_keys = ['patterns', 'relationships', 'recommendations', 'quality', 'insights']
                for key in required_keys:
                    if key not in analysis:
                        analysis[key] = [f"Analysis incomplete for {key}"]
                return analysis
            except json.JSONDecodeError:
                return self._generate_fallback_analysis(df, business_cols)
        except Exception:
            return self._generate_fallback_analysis(df, business_cols)
    
    def _generate_smart_data_profile(self, df, business_cols):
        profile = []
        for col in business_cols['numerical']:
            if col in df.columns:
                stats = df[col].describe()
                profile.append(f"- {col}: mean={stats['mean']:.2f}, std={stats['std']:.2f}, range=[{stats['min']:.2f}, {stats['max']:.2f}]")
        for col in business_cols['categorical']:
            if col in df.columns:
                unique_count = df[col].nunique()
                top_values = df[col].value_counts().head(3).to_dict()
                profile.append(f"- {col}: {unique_count} categories, top values: {top_values}")
        numeric_cols = business_cols['numerical']
        if len(numeric_cols) > 1:
            available_cols = [col for col in numeric_cols if col in df.columns]
            if len(available_cols) > 1:
                corr_matrix = df[available_cols].corr()
                high_corr = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_val = corr_matrix.iloc[i, j]
                        if abs(corr_val) > 0.5:
                            high_corr.append(f"{corr_matrix.columns[i]} vs {corr_matrix.columns[j]}: {corr_val:.3f}")
                if high_corr:
                    profile.append(f"Strong correlations: {high_corr}")
        return "\n".join(profile) if profile else "No significant patterns detected in business columns"
    
    def _generate_fallback_analysis(self, df, business_cols):
        return {
            "patterns": [
                f"Dataset contains {len(business_cols['numerical'])} business metrics",
                f"Dataset contains {len(business_cols['categorical'])} categorical dimensions",
                "Data requires further business context for deeper analysis"
            ],
            "relationships": [
                "Relationships between business metrics need exploration",
                "Categorical segments may show different performance patterns"
            ],
            "recommendations": [
                {
                    "type": "bar",
                    "x_col": business_cols['categorical'][0] if business_cols['categorical'] else df.columns[0],
                    "y_col": business_cols['numerical'][0] if business_cols['numerical'] else df.columns[-1],
                    "reason": "Basic performance analysis by category",
                    "priority": "medium"
                }
            ],
            "quality": ["Fallback analysis - AI processing unavailable"],
            "insights": ["Business insights require successful AI analysis"]
        }
    
    def _generate_data_profile(self, df):
        return self._generate_smart_data_profile(df, get_business_relevant_columns(df))

class ExecutiveSummaryAgent:
    """Intelligent agent for generating executive summaries and recommendations"""
    
    def __init__(self):
        self.name = "Executive Summary Agent"
        self.client = None
        
    def _init_openai(self):
        """Initialize OpenAI client with new v1.0+ syntax"""
        if not self.client:
            api_key = load_openai_key() or st.session_state.get('openai_api_key')
            if api_key:
                self.client = OpenAI(api_key=api_key)
        return self.client is not None
        
    def _calculate_data_statistics(self, df):
        """Calculate specific statistics to feed to AI"""
        # Use the existing function to filter out ID columns
        business_cols = get_business_relevant_columns(df)
        
        stats = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'business_columns': len(business_cols['categorical'] + business_cols['numerical'] + business_cols['temporal']),
            'numeric_stats': {},
            'categorical_stats': {},
            'correlations': {},
            'outliers': {},
            'trends': {},
            'top_performers': {},
            'bottom_performers': {}
        }
        
        # Focus only on business-relevant numeric columns
        numeric_cols = business_cols['numerical']
        for col in numeric_cols[:10]:
            if col in df.columns:
                stats['numeric_stats'][col] = {
                    'mean': float(df[col].mean()),
                    'median': float(df[col].median()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'nulls': int(df[col].isnull().sum()),
                    'unique': int(df[col].nunique()),
                    'cv': float(df[col].std() / df[col].mean() * 100) if df[col].mean() != 0 else 0  # Coefficient of variation
                }
                
                # Find outliers (values beyond 3 std)
                mean = df[col].mean()
                std = df[col].std()
                outliers = df[(df[col] < mean - 3*std) | (df[col] > mean + 3*std)]
                if len(outliers) > 0:
                    stats['outliers'][col] = {
                        'count': len(outliers),
                        'percentage': round(len(outliers) / len(df) * 100, 2),
                        'examples': outliers[col].head(3).tolist()
                    }
        
        # Smart categorical analysis - only business columns
        categorical_cols = business_cols['categorical']
        for col in categorical_cols[:5]:
            if col in df.columns:
                value_counts = df[col].value_counts()
                stats['categorical_stats'][col] = {
                    'unique_values': int(df[col].nunique()),
                    'top_value': str(value_counts.index[0]),
                    'top_value_count': int(value_counts.iloc[0]),
                    'top_value_percentage': round(value_counts.iloc[0] / len(df) * 100, 2),
                    'distribution_skew': 'concentrated' if value_counts.iloc[0] / len(df) > 0.5 else 'distributed'
                }
                
                # For categorical with numeric, find top/bottom performers
                if numeric_cols:
                    metric_col = numeric_cols[0]  # Use first numeric as key metric
                    grouped = df.groupby(col)[metric_col].agg(['mean', 'sum', 'count'])
                    
                    # Top performers
                    top_performers = grouped.nlargest(3, 'mean')
                    stats['top_performers'][f"{col}_by_{metric_col}"] = {
                        'top_1': {
                            'category': top_performers.index[0],
                            'avg_value': round(top_performers.iloc[0]['mean'], 2),
                            'total_value': round(top_performers.iloc[0]['sum'], 2)
                        }
                    }
                    
                    # Bottom performers
                    bottom_performers = grouped.nsmallest(3, 'mean')
                    stats['bottom_performers'][f"{col}_by_{metric_col}"] = {
                        'bottom_1': {
                            'category': bottom_performers.index[0],
                            'avg_value': round(bottom_performers.iloc[0]['mean'], 2),
                            'total_value': round(bottom_performers.iloc[0]['sum'], 2)
                        }
                    }
        
        # Time-based trends if temporal columns exist
        if business_cols['temporal']:
            for time_col in business_cols['temporal'][:1]:
                if time_col in df.columns and numeric_cols:
                    try:
                        df_time = df.copy()
                        df_time[time_col] = pd.to_datetime(df_time[time_col])
                        df_time['year_month'] = df_time[time_col].dt.to_period('M')
                        
                        metric_col = numeric_cols[0]
                        monthly_trend = df_time.groupby('year_month')[metric_col].sum()
                        
                        # Calculate growth rate
                        if len(monthly_trend) > 1:
                            first_value = monthly_trend.iloc[0]
                            last_value = monthly_trend.iloc[-1]
                            growth_rate = ((last_value - first_value) / first_value * 100) if first_value != 0 else 0
                            
                            stats['trends'][metric_col] = {
                                'period': f"{monthly_trend.index[0]} to {monthly_trend.index[-1]}",
                                'growth_rate': round(growth_rate, 2),
                                'trend_direction': 'increasing' if growth_rate > 5 else 'decreasing' if growth_rate < -5 else 'stable',
                                'peak_period': str(monthly_trend.idxmax()),
                                'peak_value': round(monthly_trend.max(), 2)
                            }
                    except:
                        pass
        
        # Business-relevant correlations
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            high_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_value = corr_matrix.iloc[i, j]
                    if abs(corr_value) > 0.7:  # High correlation threshold
                        high_corr.append({
                            'col1': corr_matrix.columns[i],
                            'col2': corr_matrix.columns[j],
                            'correlation': round(corr_value, 3),
                            'strength': 'very strong' if abs(corr_value) > 0.9 else 'strong'
                        })
            stats['correlations']['high_correlations'] = high_corr[:5]
        
        return stats
    
    def generate_key_findings(self, df, analysis, charts, business_context=""):
        """Generate specific, data-driven key findings"""
        try:
            if not self._init_openai():
                return self._generate_specific_fallback_findings(df, analysis, charts)
            
            # Calculate specific statistics
            stats = self._calculate_data_statistics(df)
            
            # Extract specific insights from charts
            chart_insights = []
            for i, chart in enumerate(charts[:5]):
                if chart.get('type') == 'bar' and chart.get('x_col') and chart.get('y_col'):
                    # Get actual top values from the chart data
                    try:
                        grouped = df.groupby(chart['x_col'])[chart['y_col']].mean().sort_values(ascending=False).head(3)
                        top_category = grouped.index[0]
                        top_value = grouped.iloc[0]
                        chart_insights.append(f"{top_category} leads in {chart['y_col']} with average of {top_value:.2f}")
                    except:
                        pass
                        
                elif chart.get('type') == 'scatter' and 'correlation' in str(chart.get('reason', '')).lower():
                    x_col, y_col = chart.get('x_col'), chart.get('y_col')
                    if x_col and y_col and x_col in df.columns and y_col in df.columns:
                        corr = df[x_col].corr(df[y_col])
                        chart_insights.append(f"{x_col} and {y_col} show {'strong' if abs(corr) > 0.7 else 'moderate'} correlation (r={corr:.3f})")
            
            prompt = f"""
            Generate 5-7 SPECIFIC, DATA-DRIVEN key findings. Each finding MUST include actual numbers from the data.
            
            Business Context: {business_context or "Business performance analysis"}
            
            Data Statistics:
            {json.dumps(stats, indent=2)}
            
            Chart-Specific Insights:
            {json.dumps(chart_insights, indent=2)}
            
            Additional Patterns Found:
            {json.dumps(analysis.get('patterns', [])[:3], indent=2)}
            
            REQUIREMENTS:
            1. Every finding MUST include specific numbers, percentages, or comparisons from the data
            2. Reference actual column names and values
            3. Highlight surprising or actionable insights
            4. Avoid generic statements like "explore the data" or "review insights"
            5. Focus on: outliers, trends, correlations, top/bottom performers, anomalies
            
            BAD EXAMPLE: "Sales show interesting patterns worth exploring"
            GOOD EXAMPLE: "North region sales ($4.2M) exceed South region by 47%, driven by 23% higher average transaction size"
            
            Format: Return as JSON array of specific, quantified findings.
            """
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a data analyst. Always include specific numbers and percentages in findings."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Lower temperature for more factual responses
                max_tokens=800
            )
            
            findings_text = response.choices[0].message.content
            
            # Parse and validate findings
            try:
                findings = json.loads(findings_text)
                # Filter out any generic findings
                specific_findings = []
                for finding in findings:
                    if any(char.isdigit() for char in finding) and len(finding) > 20:
                        specific_findings.append(finding)
                
                return specific_findings[:7] if specific_findings else self._generate_specific_fallback_findings(df, analysis, charts)
            except:
                return self._generate_specific_fallback_findings(df, analysis, charts)
                
        except Exception as e:
            st.error(f"Finding generation error: {str(e)}")
            return self._generate_specific_fallback_findings(df, analysis, charts)
    
    def generate_next_steps(self, df, analysis, charts, findings, business_context=""):
        """Generate specific, actionable next steps"""
        try:
            if not self._init_openai():
                return self._generate_specific_fallback_next_steps(df, analysis, charts, findings)
            
            # Extract specific issues and opportunities from findings
            issues = [f for f in findings if any(word in f.lower() for word in ['below', 'low', 'decrease', 'issue', 'problem'])]
            opportunities = [f for f in findings if any(word in f.lower() for word in ['high', 'increase', 'top', 'above', 'growth'])]
            
            prompt = f"""
            Generate 5-7 SPECIFIC, ACTIONABLE next steps based on these data findings.
            
            Key Findings:
            {json.dumps(findings, indent=2)}
            
            Identified Issues:
            {json.dumps(issues, indent=2)}
            
            Identified Opportunities:
            {json.dumps(opportunities, indent=2)}
            
            Business Context: {business_context}
            
            For each next step, provide:
            1. A specific action referencing the actual data/finding
            2. Quantified expected impact where possible
            3. Clear ownership and timeline
            
            AVOID generic actions like "review data" or "analyze further"
            INCLUDE specific references to the metrics, categories, or values from findings
            
            Format as JSON array:
            [{{
                "action": "Investigate why North region outperforms South by 47% and replicate successful strategies",
                "timeline": "Within 1 week",
                "owner": "Regional Sales Manager",
                "impact": "Potential $1.2M revenue increase in South region",
                "priority": "high",
                "related_finding": "Reference to specific finding"
            }}]
            """
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a strategic business advisor. Create specific, measurable actions based on data findings."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,
                max_tokens=1000
            )
            
            next_steps_text = response.choices[0].message.content
            
            try:
                next_steps = json.loads(next_steps_text)
                # Ensure steps reference specific data
                valid_steps = [step for step in next_steps if len(step.get('action', '')) > 30]
                return valid_steps[:7]
            except:
                return self._generate_specific_fallback_next_steps(df, analysis, charts, findings)
                
        except Exception as e:
            st.error(f"Next steps generation error: {str(e)}")
            return self._generate_specific_fallback_next_steps(df, analysis, charts, findings)
    
    def _generate_specific_fallback_findings(self, df, analysis, charts):
        """Generate specific findings without AI"""
        findings = []
        
        # Use business columns only
        business_cols = get_business_relevant_columns(df)
        numeric_cols = business_cols['numerical']
        cat_cols = business_cols['categorical']
        
        # Data quality finding - but only if there are actually missing values
        missing_cells = df.isnull().sum().sum()
        if missing_cells > 0:
            total_cells = df.shape[0] * df.shape[1]
            missing_pct = (missing_cells / total_cells) * 100
            findings.append(f"Data quality concern: {missing_pct:.1f}% missing data ({missing_cells:,} null values) may impact analysis accuracy")
        
        # Top/Bottom performer analysis
        if cat_cols and numeric_cols:
            cat_col = cat_cols[0]
            metric_col = numeric_cols[0]
            
            grouped = df.groupby(cat_col)[metric_col].agg(['mean', 'sum', 'count'])
            top_performer = grouped.nlargest(1, 'mean')
            bottom_performer = grouped.nsmallest(1, 'mean')
            
            if len(top_performer) > 0 and len(bottom_performer) > 0:
                top_name = top_performer.index[0]
                top_avg = top_performer.iloc[0]['mean']
                bottom_name = bottom_performer.index[0]
                bottom_avg = bottom_performer.iloc[0]['mean']
                
                gap_pct = ((top_avg - bottom_avg) / bottom_avg * 100) if bottom_avg != 0 else 0
                findings.append(f"{cat_col} '{top_name}' outperforms '{bottom_name}' by {gap_pct:.1f}% in average {metric_col} (${top_avg:,.0f} vs ${bottom_avg:,.0f})")
        
        # Concentration analysis
        if cat_cols:
            for col in cat_cols[:2]:
                value_counts = df[col].value_counts()
                top_3_pct = (value_counts.head(3).sum() / len(df)) * 100
                if top_3_pct > 50:
                    findings.append(f"{col} shows high concentration: top 3 values account for {top_3_pct:.1f}% of all records")
        
        # Trend analysis if time data exists
        if business_cols['temporal'] and numeric_cols:
            time_col = business_cols['temporal'][0]
            metric_col = numeric_cols[0]
            try:
                df_time = df.copy()
                df_time[time_col] = pd.to_datetime(df_time[time_col])
                df_time['year'] = df_time[time_col].dt.year
                yearly = df_time.groupby('year')[metric_col].sum()
                
                if len(yearly) > 1:
                    first_year = yearly.index[0]
                    last_year = yearly.index[-1]
                    growth = ((yearly.iloc[-1] - yearly.iloc[0]) / yearly.iloc[0] * 100)
                    findings.append(f"{metric_col} grew {growth:.1f}% from {first_year} to {last_year} (${yearly.iloc[0]:,.0f} to ${yearly.iloc[-1]:,.0f})")
            except:
                pass
        
        # Distribution insights
        if numeric_cols:
            for col in numeric_cols[:2]:
                cv = (df[col].std() / df[col].mean() * 100) if df[col].mean() != 0 else 0
                if cv > 100:
                    findings.append(f"{col} shows high variability (CV={cv:.1f}%), indicating inconsistent performance")
                
                # Check for skewness
                skew = df[col].skew()
                if abs(skew) > 1:
                    direction = "right" if skew > 0 else "left"
                    findings.append(f"{col} distribution is {direction}-skewed (skewness={skew:.2f}), with most values concentrated at the {'lower' if skew > 0 else 'higher'} end")
        
        # Correlation insights
        if len(numeric_cols) >= 2:
            corr_matrix = df[numeric_cols].corr()
            for i in range(len(numeric_cols)-1):
                for j in range(i+1, min(len(numeric_cols), i+3)):  # Limit combinations
                    corr = corr_matrix.iloc[i, j]
                    if abs(corr) > 0.7:
                        col1, col2 = numeric_cols[i], numeric_cols[j]
                        findings.append(f"Strong {'positive' if corr > 0 else 'negative'} correlation (r={corr:.3f}) between {col1} and {col2} suggests {'they move together' if corr > 0 else 'inverse relationship'}")
                        break
        
        return findings[:7]
    
    def _generate_specific_fallback_next_steps(self, df, analysis, charts, findings):
        """Generate specific next steps without AI"""
        steps = []
        
        # Based on actual findings
        for i, finding in enumerate(findings[:3]):
            if 'missing data' in finding.lower():
                steps.append({
                    "action": f"Address data quality issues - {finding}. Implement data validation rules.",
                    "timeline": "Within 1 week",
                    "owner": "Data Engineering Team",
                    "impact": "Improve analysis accuracy by reducing null values",
                    "priority": "high"
                })
            elif 'correlation' in finding.lower():
                steps.append({
                    "action": f"Investigate correlation finding: {finding}. Build predictive model if correlation is causal.",
                    "timeline": "Within 2 weeks",
                    "owner": "Analytics Team",
                    "impact": "Enable predictive analytics for business planning",
                    "priority": "medium"
                })
            elif 'dominated by' in finding.lower() or '%' in finding:
                steps.append({
                    "action": f"Analyze concentration risk: {finding}. Develop diversification strategy.",
                    "timeline": "This month",
                    "owner": "Strategy Team",
                    "impact": "Reduce business risk through diversification",
                    "priority": "high"
                })
        
        # Always include data monitoring step
        steps.append({
            "action": f"Set up automated monitoring for {len(findings)} key metrics identified in analysis",
            "timeline": "Within 1 week",
            "owner": "BI Team",
            "impact": "Enable proactive decision making with real-time alerts",
            "priority": "medium"
        })
        
        return steps[:5]

# Enhanced Executive Summary UI Component
def render_agentic_executive_summary():
    """Render the intelligent executive summary section"""
    
    if not st.session_state.get('agent_recommendations'):
        return
    
    st.markdown("### 🤖 AI Executive Summary")
    
    # Initialize summary agent
    summary_agent = ExecutiveSummaryAgent()
    
    # Get business context
    business_context = st.session_state.get('agent_learning', {}).get('business_context', '')
    
    # Action buttons with loading states
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("🔍 Generate Key Findings", key="gen_findings_btn", 
                    help="AI analyzes your data to extract executive-level insights"):
            with st.spinner("🤖 AI generating key findings..."):
                findings = summary_agent.generate_key_findings(
                    st.session_state.dataset,
                    st.session_state.data_analysis,
                    st.session_state.agent_recommendations,
                    business_context
                )
                st.session_state.executive_findings = findings
                st.session_state.show_findings = True
                st.session_state.show_next_steps = False
    
    with col2:
        if st.button("🎯 Generate Next Steps", key="gen_next_steps_btn",
                    help="AI creates actionable recommendations based on findings"):
            if not st.session_state.get('executive_findings'):
                st.warning("Generate Key Findings first!")
            else:
                with st.spinner("🤖 AI creating action plan..."):
                    next_steps = summary_agent.generate_next_steps(
                        st.session_state.dataset,
                        st.session_state.data_analysis,
                        st.session_state.agent_recommendations,
                        st.session_state.executive_findings,
                        business_context
                    )
                    st.session_state.executive_next_steps = next_steps
                    st.session_state.show_next_steps = True
                    st.session_state.show_findings = False
    
    with col3:
        if st.button("📧 Export Summary", key="export_summary_btn",
                    help="Export executive summary as formatted report"):
            export_executive_summary()
    
    # Display Key Findings
    if st.session_state.get('show_findings') and st.session_state.get('executive_findings'):
        st.markdown("""
        <div class="executive-findings">
            <h4>🔍 Key Findings</h4>
        </div>
        """, unsafe_allow_html=True)
        
        for i, finding in enumerate(st.session_state.executive_findings, 1):
            # Add severity/importance indicator
            if any(word in finding.lower() for word in ['risk', 'concern', 'issue', 'problem']):
                st.markdown(f"🔴 **Finding {i}:** {finding}")
            elif any(word in finding.lower() for word in ['opportunity', 'growth', 'increase', 'improve']):
                st.markdown(f"🟢 **Finding {i}:** {finding}")
            else:
                st.markdown(f"🔵 **Finding {i}:** {finding}")
        
        # Add confidence metrics
        st.markdown("---")
        confidence_score = min(95, 70 + len(st.session_state.agent_recommendations) * 5)
        st.metric("Analysis Confidence", f"{confidence_score}%", 
                 help="Based on data quality and number of patterns found")
    
    # Display Next Steps
    if st.session_state.get('show_next_steps') and st.session_state.get('executive_next_steps'):
        st.markdown("""
        <div class="executive-next-steps">
            <h4>🎯 Recommended Actions</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Group by priority
        immediate_actions = [s for s in st.session_state.executive_next_steps if s.get('priority') == 'immediate']
        high_priority = [s for s in st.session_state.executive_next_steps if s.get('priority') == 'high']
        other_actions = [s for s in st.session_state.executive_next_steps if s.get('priority') not in ['immediate', 'high']]
        
        if immediate_actions:
            st.markdown("**🚨 Immediate Actions (24-48 hours):**")
            for step in immediate_actions:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"• **{step['action']}**")
                    st.caption(f"Owner: {step['owner']} | Impact: {step['impact']}")
                with col2:
                    if st.button("✓ Mark Done", key=f"done_{step['action'][:20]}"):
                        st.success("Marked as complete!")
        
        if high_priority:
            st.markdown("**⚡ High Priority (This week):**")
            for step in high_priority:
                st.markdown(f"• **{step['action']}** ({step['owner']})")
                st.caption(f"Timeline: {step['timeline']} | {step['impact']}")
        
        if other_actions:
            with st.expander("📋 Additional Recommendations"):
                for step in other_actions:
                    st.markdown(f"• {step['action']} - {step['timeline']}")
    
    # Add executive notes section
    st.markdown("---")
    executive_notes = st.text_area(
        "📝 Executive Notes",
        placeholder="Add your notes, decisions, or follow-up items here...",
        key="executive_notes_area",
        height=100
    )
    
    if st.button("💾 Save Executive Summary", key="save_exec_summary"):
        save_executive_session(
            st.session_state.get('executive_findings', []),
            st.session_state.get('executive_next_steps', []),
            executive_notes
        )
        st.success("Executive summary saved!")

def export_executive_summary():
    """Export executive summary as formatted report"""
    summary_content = f"""
# Executive Data Analysis Summary
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Key Findings
{chr(10).join([f"• {f}" for f in st.session_state.get('executive_findings', [])])}

## Recommended Next Steps
{chr(10).join([f"• {s['action']} - {s['timeline']} ({s['owner']})" for s in st.session_state.get('executive_next_steps', [])])}

## Analysis Details
- Total Records Analyzed: {st.session_state.dataset.shape[0]:,}
- Variables Examined: {st.session_state.dataset.shape[1]}
- Visualizations Generated: {len(st.session_state.agent_recommendations)}
- Confidence Level: {min(95, 70 + len(st.session_state.agent_recommendations) * 5)}%
"""
    
    st.download_button(
        label="📥 Download Summary",
        data=summary_content,
        file_name=f"executive_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
        mime="text/markdown"
    )



class ChartCreatorAgent:
    def __init__(self):
        self.name = "Chart Creator Agent"
    
    def create_intelligent_charts(self, df, analysis):
        charts = []
        business_cols = get_business_relevant_columns(df)
        import plotly
        import plotly.express as px
        import plotly.graph_objects as go
        import numpy as np
        import pandas as pd
        import textwrap
        import re
        try:
            chart_prompt = f"""
            You are a data visualization expert. Based on the analysis, recommend business-valuable charts.
            
            ONLY USE THESE EXACT COLUMN NAMES:
            - Categorical: {business_cols['categorical']}
            - Numerical: {business_cols['numerical']} 
            - Temporal: {business_cols['temporal']}
            
            Analysis Results: {json.dumps(analysis, indent=2)}
            
            User Preferences: {st.session_state.agent_learning.get('preferred_charts', [])}
            Avoid These Columns: {st.session_state.agent_learning.get('avoided_columns', [])}
            
            IMPORTANT RULES:
            1. Only use exact column names from the lists above
            2. For bar/pie charts: x_col must be categorical, y_col must be numerical
            3. For line charts: x_col should be temporal or categorical, y_col must be numerical
            4. For scatter: both x_col and y_col must be numerical
            5. color_col (optional) should be categorical and exist in final data
            6. For each chart, provide a 'code' field: complete Python code using Plotly (assign the figure to 'fig', use the provided df, do NOT load data from file)
            
            IMPORTANT: Respond ONLY with valid JSON array. No markdown, no extra text.
            
            [
                {
                    "type": "bar|line|scatter|pie|histogram",
                    "x_col": "exact_column_name_from_lists_above",
                    "y_col": "exact_column_name_from_lists_above",
                    "color_col": null,
                    "title": "Business-focused chart title",
                    "reason": "Clear business justification",
                    "priority": "high|medium|low",
                    "code": "complete Python code using Plotly, assign the figure to 'fig'"
                }
            ]
            
            Recommend 2-4 charts maximum. Focus on business value, not technical IDs.
            Set color_col to null unless specifically needed and it won't cause groupby issues.
            """
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": chart_prompt}],
                temperature=0.1,
                max_tokens=2000
            )
            response_text = response.choices[0].message.content.strip()
            if response_text.startswith('```json'):
                response_text = response_text.replace('```json', '').replace('```', '').strip()
            try:
                chart_recommendations = json.loads(response_text)
                for rec in chart_recommendations:
                    x_col = rec.get('x_col')
                    y_col = rec.get('y_col')
                    color_col = rec.get('color_col')
                    code = rec.get('code', '')
                    # Fix common AI mistakes
                    code = code.replace("go.Scatter3D", "go.Scatter3d")
                    code = code.replace("go.Bar3D", "go.Bar")
                    code = re.sub(r"values=['\"](\w+)['\"]", r"values=df['\1']", code)
                    if "go.Pie(" in code and "go.Figure(" not in code:
                        code = re.sub(r"fig\s*=\s*go\.Pie\((.*?)\)", r"fig = go.Figure(go.Pie(\1))", code, flags=re.DOTALL)
                    # Remove any line that contains file loading
                    lines = code.split('\n')
                    filtered_lines = []
                    for line in lines:
                        if not any(pattern in line.lower() for pattern in [
                            'pd.read_csv', 'pd.read_excel', 'pd.read_', 'open(', 
                            'your_dataset.csv', '.csv', '.xlsx', '.xls', 'read_csv', 'read_excel',
                            'pd.read_', 'read_', 'load_', 'import_', 'from_csv', 'from_excel'
                        ]):
                            filtered_lines.append(line)
                    code = '\n'.join(filtered_lines)
                    # Additional safety: remove any remaining file references
                    code = re.sub(r"['\"].*?\.csv['\"]", "''", code)
                    code = re.sub(r"['\"].*?\.xlsx['\"]", "''", code)
                    code = re.sub(r"['\"].*?\.xls['\"]", "''", code)
                    code = re.sub(r"['\"]your_dataset\.csv['\"]", "''", code)
                    code = re.sub(r"['\"]dataset\.csv['\"]", "''", code)
                    code = re.sub(r"['\"]data\.csv['\"]", "''", code)
                    code = re.sub(r"df\s*=\s*pd\.read_csv\(.*?\)\s*", "", code)
                    code = re.sub(r"df\s*=\s*pd\.read_excel\(.*?\)\s*", "", code)
                    code = re.sub(r"df\s*=\s*open\(.*?\)\s*", "", code)
                    code = re.sub(r"df\s*=\s*.*?your_dataset\\.csv.*", "", code)
                    code = re.sub(r"df\s*=\s*.*?\.csv.*", "", code)
                    code = re.sub(r"df\s*=\s*.*?\.xlsx.*", "", code)
                    code = re.sub(r"df\s*=\s*.*?\.xls.*", "", code)
                    # Enhanced file loading removal - catch more patterns
                    code = re.sub(r"df\s*=\s*pd\.read_csv\(['\"].*?['\"]\)", "", code)
                    code = re.sub(r"df\s*=\s*pd\.read_excel\(['\"].*?['\"]\)", "", code)
                    code = re.sub(r"df\s*=\s*pd\.read_csv\(.*?\)", "", code, flags=re.DOTALL)
                    code = re.sub(r"df\s*=\s*pd\.read_excel\(.*?\)", "", code, flags=re.DOTALL)
                    code = re.sub(r"df\s*=\s*pd\.read_csv\(.*?\)", "", code, flags=re.MULTILINE)
                    code = re.sub(r"df\s*=\s*pd\.read_excel\(.*?\)", "", code, flags=re.MULTILINE)
                    code = re.sub(r'data\s*=\s*data', 'data=df', code)
                    code = re.sub(r'\bdata\.', 'df.', code)
                    code = re.sub(r'data\s*=\s*df', 'data=df', code)
                    code = textwrap.dedent(code)
                    
                    # Debug: print the code to see what's being generated
                    print(f"DEBUG - Generated code:\n{code}")
                    
                    all_business_cols = business_cols['categorical'] + business_cols['numerical'] + business_cols['temporal']
                    if (x_col in all_business_cols and x_col in df.columns and 
                        (not y_col or (y_col in all_business_cols and y_col in df.columns))):
                        local_vars = {'df': df, 'px': px, 'go': go, 'pd': pd, 'np': np, 'plotly': plotly}
                        fig = None
                        try:
                            exec(code, {}, local_vars)
                            fig = local_vars.get('fig')
                        except FileNotFoundError as e:
                            return {'error': f'File loading detected and blocked: {str(e)}. Please use the existing dataframe.', 'code': code}
                        except Exception as e:
                            return {'error': f'Error executing AI code: {str(e)}', 'code': code}
                        if fig:
                            charts.append({
                                'figure': create_dark_chart(fig),
                                'title': rec.get('title', 'AI Chart'),
                                'type': rec.get('type', 'custom'),
                                'code': code,
                                'data': df,
                                'priority': rec.get('priority', 'high'),
                                'reason': rec.get('reason', 'AI-generated insight'),
                                'x_col': x_col,
                                'y_col': y_col
                            })
                        else:
                            charts.append({
                                'figure': None,
                                'title': rec.get('title', 'AI Chart'),
                                'type': rec.get('type', 'custom'),
                                'code': code,
                                'data': df,
                                'priority': rec.get('priority', 'high'),
                                'reason': f"AI code execution failed. See code for details.",
                                'x_col': x_col,
                                'y_col': y_col
                            })
                    else:
                        print(f"Skipped chart recommendation: {rec.get('title', 'Unknown')} - Column validation failed")
                        continue
            except json.JSONDecodeError:
                return self._create_smart_fallback_charts(df, business_cols)
            # Only return charts with a figure
            return [c for c in charts if c['figure'] is not None] if charts else self._create_smart_fallback_charts(df, business_cols)
        except Exception:
            return self._create_smart_fallback_charts(df, business_cols)
    
    def _create_smart_fallback_charts(self, df, business_cols):
        charts = []
        categorical = business_cols['categorical']
        numerical = business_cols['numerical']
        temporal = business_cols['temporal']
        if categorical and numerical:
            x_col = categorical[0]
            y_col = numerical[0] 
            chart_data = df.groupby(x_col)[y_col].sum().reset_index().sort_values(y_col, ascending=False).head(10)
            fig = px.bar(chart_data, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
            fig.update_traces(marker_color='#60A5FA')
            fig = create_dark_chart(fig)
            charts.append({
                'figure': fig,
                'title': f"{y_col} by {x_col}",
                'reason': "Primary business metric analysis by category",
                'priority': 'high',
                'x_col': x_col,
                'y_col': y_col,
                'data': chart_data,
                'type': 'bar'
            })
        if temporal and numerical:
            x_col = temporal[0]
            y_col = numerical[0]
            try:
                df_time = df.copy()
                df_time[x_col] = pd.to_datetime(df_time[x_col])
                chart_data = df_time.groupby(pd.Grouper(key=x_col, freq='M'))[y_col].sum().reset_index()
                fig = px.line(chart_data, x=x_col, y=y_col, title=f"{y_col} Trend Over Time")
                fig.update_traces(line_color='#60A5FA')
                fig = create_dark_chart(fig)
                charts.append({
                    'figure': fig,
                    'title': f"{y_col} Trend Over Time",
                    'reason': "Temporal analysis of key business metric",
                    'priority': 'high',
                    'x_col': x_col,
                    'y_col': y_col,
                    'data': chart_data,
                    'type': 'line'
                })
            except:
                pass
        if len(numerical) >= 2:
            x_col = numerical[0]
            y_col = numerical[1]
            color_col = categorical[0] if categorical else None
            if color_col:
                fig = px.scatter(df, x=x_col, y=y_col, color=color_col, title=f"{x_col} vs {y_col} by {color_col}")
            else:
                fig = px.scatter(df, x=x_col, y=y_col, title=f"{x_col} vs {y_col}")
                fig.update_traces(marker_color='#60A5FA')
            fig = create_dark_chart(fig)
            charts.append({
                'figure': fig,
                'title': f"{x_col} vs {y_col}" + (f" by {color_col}" if color_col else ""),
                'reason': "Correlation analysis between key business metrics",
                'priority': 'medium',
                'x_col': x_col,
                'y_col': y_col,
                'data': df,
                'type': 'scatter'
            })
        return charts
    
    def _create_chart_from_recommendation(self, df, rec):
        try:
            chart_type = rec.get('type', 'bar').lower()
            x_col = rec.get('x_col')
            y_col = rec.get('y_col')
            color_col = rec.get('color_col')
            title = rec.get('title', 'AI Generated Chart')
            if not x_col or x_col not in df.columns:
                return None
            if y_col and y_col not in df.columns:
                return None
            if color_col and color_col not in df.columns:
                color_col = None
            fig = None
            chart_data = df
            if chart_type == 'bar' and y_col:
                if df[x_col].dtype == 'object':
                    if color_col:
                        chart_data = df.groupby([x_col, color_col])[y_col].sum().reset_index()
                        chart_data = chart_data.sort_values(y_col, ascending=False).head(15)
                        fig = px.bar(chart_data, x=x_col, y=y_col, color=color_col, title=title)
                    else:
                        chart_data = df.groupby(x_col)[y_col].sum().reset_index()
                        chart_data = chart_data.sort_values(y_col, ascending=False).head(15)
                        fig = px.bar(chart_data, x=x_col, y=y_col, title=title)
                        fig.update_traces(marker_color='#60A5FA')
            elif chart_type == 'line' and y_col:
                if 'date' in x_col.lower() or pd.api.types.is_datetime64_any_dtype(df[x_col]):
                    chart_data = df.copy()
                    chart_data[x_col] = pd.to_datetime(chart_data[x_col])
                    if color_col:
                        chart_data = chart_data.groupby([pd.Grouper(key=x_col, freq='M'), color_col])[y_col].sum().reset_index()
                        fig = px.line(chart_data, x=x_col, y=y_col, color=color_col, title=title)
                    else:
                        chart_data = chart_data.groupby(pd.Grouper(key=x_col, freq='M'))[y_col].sum().reset_index()
                        fig = px.line(chart_data, x=x_col, y=y_col, title=title)
                        fig.update_traces(line_color='#60A5FA')
            elif chart_type == 'scatter' and y_col:
                if color_col:
                    fig = px.scatter(df, x=x_col, y=y_col, color=color_col, title=title)
                else:
                    fig = px.scatter(df, x=x_col, y=y_col, title=title)
                    fig.update_traces(marker_color='#60A5FA')
            elif chart_type == 'pie' and y_col:
                chart_data = df.groupby(x_col)[y_col].sum().reset_index()
                fig = px.pie(chart_data, names=x_col, values=y_col, title=title)
            elif chart_type == 'histogram' and x_col:
                fig = px.histogram(df, x=x_col, title=title)
            if fig:
                fig = create_dark_chart(fig)
                return {
                    'figure': fig,
                    'title': title,
                    'reason': rec.get('reason', 'AI recommended visualization'),
                    'priority': rec.get('priority', 'medium'),
                    'x_col': x_col,
                    'y_col': y_col,
                    'data': chart_data,
                    'type': chart_type
                }
            return None
        except Exception as e:
            print(f"Chart creation error: {str(e)}")
            return None
    def _create_fallback_charts(self, df):
        return self._create_smart_fallback_charts(df, get_business_relevant_columns(df))

class InsightGeneratorAgent:
    def __init__(self):
        self.name = "Insight Generator Agent"
    
    def generate_insights(self, df, chart_info, analysis):
        """Generate intelligent business insights using OpenAI"""
        try:
            insight_prompt = f"""
            As a business intelligence expert, generate actionable insights for this chart:
            
            Chart: {chart_info['title']}
            Type: {chart_info['type']}
            Data: X-axis: {chart_info['x_col']}, Y-axis: {chart_info['y_col']}
            
            Dataset Analysis: {json.dumps(analysis, indent=2)}
            
            Chart Data Summary:
            {chart_info['data'].describe().to_dict() if hasattr(chart_info['data'], 'describe') else 'No numerical summary available'}
            
            Provide 3-4 specific, actionable business insights that a stakeholder could act upon.
            Focus on trends, outliers, opportunities, and risks.
            Make insights specific to the data shown.
            """
            
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": insight_prompt}],
                temperature=0.4
            )
            
            insights = response.choices[0].message.content.strip().split('\n')
            cleaned_insights = [insight.strip('- •').strip() for insight in insights if insight.strip()]
            return cleaned_insights if len(cleaned_insights) >= 3 else self._get_default_insights(chart_info)
            
        except Exception as e:
            return self._get_default_insights(chart_info)
    
    def _get_default_insights(self, chart_info):
        """Provide 3 intelligent default insights when AI fails"""
        chart_type = chart_info.get('type', 'bar')
        x_col = chart_info.get('x_col', 'Category')
        y_col = chart_info.get('y_col', 'Value')
        
        if chart_type == 'bar':
            return [
                f"Top performing {x_col.lower()} segments drive significant {y_col.lower()} - focus resources on these leaders",
                f"Performance gaps between {x_col.lower()} categories suggest optimization opportunities in underperforming areas", 
                f"Consider analyzing what makes top {x_col.lower()} successful and apply those strategies to boost overall {y_col.lower()}"
            ]
        elif chart_type == 'line':
            return [
                f"Trend analysis reveals seasonal patterns in {y_col.lower()} that can inform planning and resource allocation",
                f"Growth rate changes over time indicate market shifts - monitor these for strategic adjustments",
                f"Peak and trough periods in {y_col.lower()} suggest optimal timing for campaigns and inventory management"
            ]
        elif chart_type == 'scatter':
            return [
                f"Correlation between {x_col.lower()} and {y_col.lower()} reveals key performance drivers worth investigating",
                f"Outlier data points may represent exceptional cases or data quality issues requiring attention",
                f"Strong relationships suggest predictive opportunities - changes in {x_col.lower()} may forecast {y_col.lower()} trends"
            ]
        elif chart_type == 'pie':
            return [
                f"Market share distribution shows concentration levels - consider diversification strategies if heavily concentrated",
                f"Smaller segments may represent untapped growth opportunities or areas for strategic focus",
                f"Dominant segments warrant protection strategies while emerging segments need development investment"
            ]
        else:
            return [
                f"Data distribution patterns in {y_col.lower()} suggest areas for targeted improvement and optimization",
                f"Performance variations across {x_col.lower()} indicate different strategies may be needed for different segments",
                f"Key metrics show actionable patterns that can inform decision-making and resource allocation priorities"
            ]
    
    def generate_custom_chart_insights(self, prompt, chart_data):
        """Generate insights for custom user requests"""
        try:
            insight_prompt = f"""
            User requested: "{prompt}"
            
            Analyze this request and provide:
            1. What the user is trying to understand
            2. Best chart type for this analysis
            3. Key metrics to focus on
            4. Potential business insights they should look for
            
            Make it specific to their request.
            """
            
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": insight_prompt}],
                temperature=0.3
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Analysis of '{prompt}' in progress. Chart will reveal key patterns and relationships."

class AgenticAIAgent:
    def __init__(self):
        self.name = "Agentic AI Coordinator"
        self.version = "1.0.0"
        self.data_analyst = None
        self.chart_creator = None
        self.insight_generator = None
        self.agent_status = {
            'data_analyst': 'idle',
            'chart_creator': 'idle',
            'insight_generator': 'idle'
        }
        self.learning_data = {
            'preferred_charts': [],
            'avoided_columns': [],
            'business_context': "",
            'user_feedback': [],
            'successful_patterns': []
        }
        self.conversation_log = []
        self.active_analysis = None
    
    def initialize_agents(self):
        if self.data_analyst is None:
            self.data_analyst = DataAnalystAgent()
            self.log_conversation("System", "Data Analyst Agent initialized")
        if self.chart_creator is None:
            self.chart_creator = ChartCreatorAgent()
            self.log_conversation("System", "Chart Creator Agent initialized")
        if self.insight_generator is None:
            self.insight_generator = InsightGeneratorAgent()
            self.log_conversation("System", "Insight Generator Agent initialized")
    
    def log_conversation(self, agent_name, message, message_type="info", details=None):
        conversation_entry = {
            'agent': agent_name,
            'message': message,
            'type': message_type,
            'timestamp': datetime.now().isoformat(),
            'details': details
        }
        self.conversation_log.append(conversation_entry)
        if 'agent_conversations' in st.session_state:
            st.session_state.agent_conversations.append(conversation_entry)
    
    def update_agent_status(self, agent_name, status):
        if agent_name in self.agent_status:
            self.agent_status[agent_name] = status
        if 'agent_status' in st.session_state:
            st.session_state.agent_status[agent_name] = status
    
    def run_full_analysis(self, dataframe, use_code_based_charts=True):
        try:
            self.initialize_agents()
            self.update_agent_status('data_analyst', 'analyzing')
            self.log_conversation("Agentic AI Coordinator", "Starting comprehensive data analysis workflow")
            analysis_results = self.data_analyst.analyze_dataset(dataframe)
            self.active_analysis = analysis_results
            self.log_conversation(
                "Data Analyst Agent", 
                f"Completed analysis: {len(analysis_results.get('recommendations', []))} visualization opportunities identified",
                "analyst",
                analysis_results
            )
            self.update_agent_status('data_analyst', 'complete')
            self.update_agent_status('chart_creator', 'creating')
            charts = self.chart_creator.create_intelligent_charts(dataframe, analysis_results)
            self.log_conversation(
                "Chart Creator Agent",
                f"Generated {len(charts)} visualizations using {'code-based' if use_code_based_charts else 'AI-generated'} approach",
                "creator"
            )
            self.update_agent_status('chart_creator', 'complete')
            self.update_agent_status('insight_generator', 'generating')
            total_insights = 0
            for chart in charts:
                if chart.get('agentic'):
                    chart['insights'] = chart.get('ai_insights', [])
                else:
                    chart['insights'] = self.insight_generator.generate_insights(dataframe, chart, analysis_results)
                total_insights += len(chart.get('insights', []))
            self.log_conversation(
                "Insight Generator Agent",
                f"Generated {total_insights} business insights across all visualizations",
                "insight"
            )
            self.update_agent_status('insight_generator', 'complete')
            for agent in self.agent_status:
                self.update_agent_status(agent, 'idle')
            results = {
                'analysis': analysis_results,
                'charts': charts,
                'total_insights': total_insights,
                'conversation_log': self.conversation_log,
                'agent_status': self.agent_status.copy(),
                'metadata': {
                    'dataset_shape': dataframe.shape,
                    'analysis_timestamp': datetime.now().isoformat(),
                    'use_code_based_charts': use_code_based_charts
                }
            }
            self.log_conversation("Agentic AI Coordinator", "Analysis workflow completed successfully")
            return results
        except Exception as e:
            error_msg = f"Analysis workflow failed: {str(e)}"
            self.log_conversation("Agentic AI Coordinator", error_msg, "error")
            for agent in self.agent_status:
                self.update_agent_status(agent, 'error')
            return {
                'error': error_msg,
                'conversation_log': self.conversation_log,
                'agent_status': self.agent_status.copy()
            }
    
    def create_custom_chart(self, dataframe, user_prompt, business_context=""):
        """Create a custom chart based on user request using OpenAI to parse the prompt"""
        try:
            self.initialize_agents()
            import plotly
            import plotly.express as px
            import plotly.graph_objects as go
            import numpy as np
            import pandas as pd
            import textwrap
            import re
            parse_prompt = f"""
            Analyze this user request for data visualization:
            User Request: {user_prompt}
            Business Context: {business_context or self.learning_data.get('business_context', '')}
            Dataset Info:
            - Shape: {dataframe.shape}
            - Columns: {list(dataframe.columns)}
            - Data Types: {dataframe.dtypes.to_dict()}
            
            CRITICAL: The dataset is already loaded as 'df'. DO NOT include any file loading code like pd.read_csv(), pd.read_excel(), or open().
            Use the existing 'df' variable directly in your code.
            
            Determine the best visualization(s) to create. Return JSON with:
            {{
                "recommendations": [
                    {{
                        "type": "chart_type (bar, line, scatter, heatmap, etc.)",
                        "x_col": "exact column name from dataset",
                        "y_col": "exact column name from dataset (if needed)",
                        "title": "descriptive title based on user request",
                        "reason": "why this chart answers the user's question",
                        "priority": "high",
                        "code": "complete Python code using Plotly to create the chart (use 'df' variable, NO file loading)"
                    }}
                ]
            }}
            """
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": parse_prompt}],
                temperature=0.2,
                max_tokens=1000
            )
            response_text = response.choices[0].message.content.strip()
            if response_text.startswith('```json'):
                response_text = response_text.replace('```json', '').replace('```', '').strip()
            recs = json.loads(response_text).get('recommendations', [])
            if not recs:
                return {'error': 'AI did not return any chart recommendations.'}
            rec = recs[0]
            code = rec.get('code', '')
            # Fix common AI mistakes
            code = code.replace("go.Scatter3D", "go.Scatter3d")
            code = code.replace("go.Bar3D", "go.Bar")
            code = re.sub(r"values=['\"](\w+)['\"]", r"values=df['\1']", code)
            if "go.Pie(" in code and "go.Figure(" not in code:
                code = re.sub(r"fig\s*=\s*go\.Pie\((.*?)\)", r"fig = go.Figure(go.Pie(\1))", code, flags=re.DOTALL)
            # Remove any line that contains file loading
            lines = code.split('\n')
            filtered_lines = []
            for line in lines:
                if not any(pattern in line.lower() for pattern in [
                    'pd.read_csv', 'pd.read_excel', 'pd.read_', 'open(', 
                    'your_dataset.csv', '.csv', '.xlsx', '.xls', 'read_csv', 'read_excel',
                    'pd.read_', 'read_', 'load_', 'import_', 'from_csv', 'from_excel'
                ]):
                    filtered_lines.append(line)
            code = '\n'.join(filtered_lines)
            # Additional safety: remove any remaining file references
            code = re.sub(r"['\"].*?\.csv['\"]", "''", code)
            code = re.sub(r"['\"].*?\.xlsx['\"]", "''", code)
            code = re.sub(r"['\"].*?\.xls['\"]", "''", code)
            code = re.sub(r"['\"]your_dataset\.csv['\"]", "''", code)
            code = re.sub(r"['\"]dataset\.csv['\"]", "''", code)
            code = re.sub(r"['\"]data\.csv['\"]", "''", code)
            code = re.sub(r"df\s*=\s*pd\.read_csv\(.*?\)\s*", "", code)
            code = re.sub(r"df\s*=\s*pd\.read_excel\(.*?\)\s*", "", code)
            code = re.sub(r"df\s*=\s*open\(.*?\)\s*", "", code)
            code = re.sub(r"df\s*=\s*.*?your_dataset\\.csv.*", "", code)
            code = re.sub(r"df\s*=\s*.*?\.csv.*", "", code)
            code = re.sub(r"df\s*=\s*.*?\.xlsx.*", "", code)
            code = re.sub(r"df\s*=\s*.*?\.xls.*", "", code)
            # Enhanced file loading removal - catch more patterns
            code = re.sub(r"df\s*=\s*pd\.read_csv\(['\"].*?['\"]\)", "", code)
            code = re.sub(r"df\s*=\s*pd\.read_excel\(['\"].*?['\"]\)", "", code)
            code = re.sub(r"df\s*=\s*pd\.read_csv\(.*?\)", "", code, flags=re.DOTALL)
            code = re.sub(r"df\s*=\s*pd\.read_excel\(.*?\)", "", code, flags=re.DOTALL)
            code = re.sub(r"df\s*=\s*pd\.read_csv\(.*?\)", "", code, flags=re.MULTILINE)
            code = re.sub(r"df\s*=\s*pd\.read_excel\(.*?\)", "", code, flags=re.MULTILINE)
            code = re.sub(r'data\s*=\s*data', 'data=df', code)
            code = re.sub(r'\bdata\.', 'df.', code)
            code = re.sub(r'data\s*=\s*df', 'data=df', code)
            code = textwrap.dedent(code)
            
            # Debug: print the code to see what's being generated
            print(f"DEBUG - Generated code:\n{code}")
            
            # Pre-execution safety check for file loading
            if any(pattern in code.lower() for pattern in [
                'pd.read_csv', 'pd.read_excel', 'pd.read_', 'open(', 
                'your_dataset.csv', '.csv', '.xlsx', '.xls', 'read_csv', 'read_excel',
                'pd.read_', 'read_', 'load_', 'import_', 'from_csv', 'from_excel'
            ]):
                return {'error': 'File loading detected in generated code. Please use the existing dataframe.', 'code': code}
            
            local_vars = {'df': dataframe, 'px': px, 'go': go, 'pd': pd, 'np': np, 'plotly': plotly}
            try:
                exec(code, {}, local_vars)
                fig = local_vars.get('fig')
            except FileNotFoundError as e:
                return {'error': f'File loading detected and blocked: {str(e)}. Please use the existing dataframe.', 'code': code}
            except Exception as e:
                return {'error': f'Error executing AI code: {str(e)}', 'code': code}
            if not fig:
                return {'error': 'AI code did not produce a figure.', 'code': code}
            return {
                'prompt': user_prompt,
                'figure': create_dark_chart(fig),
                'x_col': rec.get('x_col'),
                'y_col': rec.get('y_col'),
                'code': code,
                'ai_analysis': rec.get('reason', ''),
                'type': rec.get('type', 'custom'),
                'title': rec.get('title', user_prompt)
            }
        except Exception as e:
            self.log_conversation("Agentic AI Coordinator", f"Custom chart creation failed: {str(e)}", "error")
            return {"error": str(e)}

    def process_user_feedback(self, feedback_data):
        try:
            self.learning_data['user_feedback'].append(feedback_data)
            rating_value = len(feedback_data.get('chart_rating', '⭐⭐⭐').split('⭐')) - 1
            if rating_value >= 4:
                chart_pattern = f"{feedback_data.get('chart_type')}:{feedback_data.get('x_col')}:{feedback_data.get('y_col')}"
                if chart_pattern not in self.learning_data['preferred_charts']:
                    self.learning_data['preferred_charts'].append(chart_pattern)
                self.learning_data['successful_patterns'].append({
                    'pattern': chart_pattern,
                    'feedback': feedback_data.get('feedback_text', ''),
                    'rating': rating_value,
                    'timestamp': datetime.now().isoformat()
                })
            elif rating_value <= 2:
                x_col = feedback_data.get('x_col')
                if x_col and x_col not in self.learning_data['avoided_columns']:
                    self.learning_data['avoided_columns'].append(x_col)
            if 'agent_learning' in st.session_state:
                st.session_state.agent_learning.update(self.learning_data)
            self.log_conversation(
                "Learning System",
                f"Processed feedback for {feedback_data.get('chart_title', 'chart')} - Rating: {rating_value}/5",
                "feedback"
            )
            return True
        except Exception as e:
            self.log_conversation(
                "Learning System",
                f"Failed to process feedback: {str(e)}",
                "error"
            )
            return False
    
    def get_smart_recommendations(self, dataframe):
        try:
            self.initialize_agents()
            analysis = self.data_analyst.analyze_dataset(dataframe)
            filtered_recommendations = []
            for rec in analysis.get('recommendations', []):
                x_col = rec.get('x_col')
                chart_type = rec.get('type')
                if x_col in self.learning_data.get('avoided_columns', []):
                    continue
                pattern = f"{chart_type}:{x_col}:{rec.get('y_col', '')}"
                if pattern in self.learning_data.get('preferred_charts', []):
                    rec['priority'] = 'high'
                    rec['reason'] += " (User preferred pattern)"
                filtered_recommendations.append(rec)
            return filtered_recommendations
        except Exception as e:
            self.log_conversation(
                "Smart Recommendations",
                f"Failed to generate smart recommendations: {str(e)}",
                "error"
            )
            return []
    
    def export_learning_summary(self):
        return {
            'total_feedback_sessions': len(self.learning_data.get('user_feedback', [])),
            'preferred_chart_patterns': len(self.learning_data.get('preferred_charts', [])),
            'avoided_columns': len(self.learning_data.get('avoided_columns', [])),
            'successful_patterns': len(self.learning_data.get('successful_patterns', [])),
            'average_rating': np.mean([
                len(fb.get('chart_rating', '⭐⭐⭐').split('⭐')) - 1 
                for fb in self.learning_data.get('user_feedback', [])[-10:]
            ]) if self.learning_data.get('user_feedback') else 0,
            'business_context': self.learning_data.get('business_context', ''),
            'last_activity': datetime.now().isoformat()
        }
    
    def reset_agents(self):
        self.data_analyst = None
        self.chart_creator = None  
        self.insight_generator = None
        self.conversation_log = []
        self.active_analysis = None
        for agent in self.agent_status:
            self.update_agent_status(agent, 'idle')
        self.log_conversation("System", "All agents reset successfully")
    
    def get_system_status(self):
        return {
            'agents_initialized': all([
                self.data_analyst is not None,
                self.chart_creator is not None,
                self.insight_generator is not None
            ]),
            'agent_status': self.agent_status.copy(),
            'conversation_entries': len(self.conversation_log),
            'learning_data_summary': self.export_learning_summary(),
            'active_analysis': self.active_analysis is not None,
            'system_version': self.version
        }

class YourVisualizationClass:
    def __init__(self, use_agentic_ai=False, strict_agentic_mode=False, openai_api_key=None):
        """
        Initialize visualization class.
        
        Args:
            use_agentic_ai: Whether to use AI-generated charts.
            strict_agentic_mode: If True, only show AI-generated charts when agentic AI
                is enabled, with no fallback to standard charts.
            openai_api_key: API key for OpenAI to generate custom chart code.
        """
        self.use_agentic_ai = use_agentic_ai
        self.strict_agentic_mode = strict_agentic_mode
        self.openai_api_key = openai_api_key
        # Configure OpenAI if API key is provided
        if self.openai_api_key and self.use_agentic_ai:
            import openai
            openai.api_key = self.openai_api_key


def track_agentic_ai_usage(user_id, supabase):
    if st.session_state.get("user_role") == "Viewer":
        result = supabase.table("usage").select("count").eq("user_id", user_id).eq("feature", "agentic_ai").execute()
        usage_count = result.data[0]["count"] if result.data else 0
        if usage_count >= 5:
            st.error("Free usage limit reached. Upgrade to Pro for unlimited Agentic AI analyses.")
            return False
        supabase.table("usage").insert({
            "user_id": user_id,
            "feature": "agentic_ai",
            "timestamp": datetime.utcnow().isoformat()
        }).execute()
    return True

def agentic_ai_chart_tab():
    agentic_ai = AgenticAIAgent()
    
    # --- HEADER (Branding & Beta Disclaimer) ---
    st.markdown("""
    <div class='branding'>
        <h2>🤖 NarraViz.ai - AI Dashboard</h2>
        <p>Powered by Real Agentic AI & Machine Learning</p>
    </div>
    <div class='beta-banner'>
        🚧 BETA VERSION - Advanced AI Analysis System in Development 🚧
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("🤖 Real Agentic AI Chart System - Intelligent Analysis")
    if not st.session_state.get("agentic_ai_enabled", False):
        st.warning("Agentic AI is disabled. Enable it in the sidebar to access advanced analytics.")
        return
    if st.session_state.dataset is None:
        st.info("No dataset loaded. Please upload a dataset in the 'Data' tab.")
        return
    
    # Initialize session state variables
    session_vars = [
        'agent_conversations', 'agent_status', 'agentic_charts', 
        'agent_recommendations', 'saved_dashboards', 'custom_charts',
        'data_analysis', 'user_feedback', 'agent_learning', 'use_code_based_charts',
        'show_key_findings', 'show_next_steps'
    ]
    
    for var in session_vars:
        if var not in st.session_state:
            if var == 'agent_status':
                st.session_state[var] = {
                    'data_analyst': 'idle',
                    'chart_creator': 'idle',
                    'insight_generator': 'idle'
                }
            elif var == 'agent_learning':
                st.session_state[var] = {
                    'preferred_charts': [],
                    'avoided_columns': [],
                    'business_context': ""
                }
            elif var == 'use_code_based_charts':
                st.session_state[var] = True
            elif var in ['show_key_findings', 'show_next_steps']:
                st.session_state[var] = False
            else:
                st.session_state[var] = []

    st.markdown(r"""
    <style>
    .agent-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .agent-status {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.875rem;
        font-weight: 500;
        margin-left: 1rem;
    }
    .status-idle { background-color: #718096; }
    .status-analyzing { background-color: #F59E0B; animation: pulse 2s infinite; }
    .status-creating { background-color: #3B82F6; animation: pulse 2s infinite; }
    .status-generating { background-color: #8B5CF6; animation: pulse 2s infinite; }
    .status-complete { background-color: #10B981; }
    .status-error { background-color: #EF4444; }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    .conversation-log {
        background-color: #1F2937;
        color: #F3F4F6;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        font-family: monospace;
        font-size: 0.9rem;
        max-height: 400px;
        overflow-y: auto;
    }
    .agent-message {
        margin: 0.5rem 0;
        padding: 0.75rem;
        border-left: 3px solid;
        border-radius: 4px;
        background-color: #374151;
    }
    .analyst-message { border-color: #8B5CF6; }
    .creator-message { border-color: #3B82F6; }
    .insight-message { border-color: #10B981; }
    .chart-container {
        border: 1px solid #374151;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 2rem;
        background-color: #1F2937;
        box-shadow: 0 1px 3px rgba(0,0,0,0.3);
    }
    .chart-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
        color: #F3F4F6;
    }
    .priority-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.875rem;
        font-weight: 500;
    }
    .priority-high { background-color: #7F1D1D; color: #FEE2E2; }
    .priority-medium { background-color: #78350F; color: #FEF3C7; }
    .priority-low { background-color: #312E81; color: #E0E7FF; }
    .metric-container {
        font-size: 0.8rem !important;
        background-color: #374151;
        padding: 0.75rem;
        border-radius: 8px;
        text-align: center;
        color: #F3F4F6;
        margin: 0.25rem;
        border: 1px solid #4B5563;
    }
    .metric-container .metric-value {
        font-size: 1.2rem !important;
        color: #60A5FA;
        font-weight: bold;
        display: block;
        margin-bottom: 0.25rem;
    }
    .metric-container .metric-label {
        font-size: 0.75rem !important;
        color: #E5E7EB;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 500;
    }
    .insights-container {
        margin-top: 1rem;
        background-color: #374151;
        padding: 1rem;
        border-radius: 8px;
        color: #93C5FD;
        border: 1px solid #4B5563;
    }
    .insights-container h3 {
        color: #60A5FA;
        margin-bottom: 0.75rem;
    }
    .insights-container p, .insights-container li {
        color: #93C5FD;
        line-height: 1.5;
    }
    .insights-container strong {
        color: #DBEAFE;
    }
    .js-plotly-plot {
        background-color: #111827 !important;
    }
    .custom-chart-section {
        border: 2px solid #8B5CF6;
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
        background-color: #1F2937;
        color: #93C5FD;
    }
    .custom-chart-section h4 {
        color: #DBEAFE;
    }
    .chart-container {
        border: 1px solid #374151;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 2rem;
        background-color: #1F2937;
        box-shadow: 0 1px 3px rgba(0,0,0,0.3);
        color: #93C5FD;
    }
    .chart-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
        color: #DBEAFE;
    }
    .chart-header h4 {
        color: #DBEAFE;
    }
    p, li, span {
        color: #93C5FD;
    }
    strong {
        color: #DBEAFE;
    }
    .code-section {
        background-color: #1F2937;
        padding: 10px;
        border-radius: 8px;
        font-family: monospace;
        font-size: 0.9em;
        color: #93C5FD;
        margin-top: 10px;
        overflow-x: auto;
    }
    
    /* Executive Summary Styles */
    .exec-summary-btn {
        background: linear-gradient(135deg, #3B82F6 0%, #1D4ED8 100%);
        border: none;
        color: white;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(59, 130, 246, 0.3);
    }
    
    .exec-summary-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
    }
    
    .key-findings-section {
        background: linear-gradient(135deg, #1E3A8A 0%, #3730A3 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid #3B82F6;
        color: #E0E7FF;
    }
    
    .next-steps-section {
        background: linear-gradient(135deg, #065F46 0%, #047857 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid #10B981;
        color: #D1FAE5;
    }
    
    .summary-title {
        color: #DBEAFE;
        margin-bottom: 1rem;
        font-weight: 700;
    }
    
    .findings-title {
        color: #ECFDF5;
        margin-bottom: 1rem;
        font-weight: 700;
    }
    </style>
    """, unsafe_allow_html=True)

    # Display current mode prominently
    col1, col2 = st.columns([3, 1])
    with col1:
        if st.session_state.get('use_code_based_charts', True):
            st.info("📊 **Code-Based Mode**: AI generates verified code, deterministic visualizations")
        else:
            st.info("🎨 **Full AI Mode**: Dynamic AI-powered visualizations and insights")
    
    with col2:
        if st.button("🔄 Refresh Analysis"):
            # Clear the agent conversations and charts
            for var in ['agent_conversations', 'agentic_charts', 'agent_recommendations', 'data_analysis']:
                st.session_state[var] = [] if var != 'data_analysis' else None
            st.session_state.show_key_findings = False
            st.session_state.show_next_steps = False
            st.rerun()

    charts = []

    # When displaying insights, check the mode:
    if charts:
        for chart in charts:
            with st.container():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.plotly_chart(chart['figure'], use_container_width=True)
                
                with col2:
                    if st.session_state.get('use_code_based_charts', True):
                        # Code-based mode: Show code and basic stats
                        st.markdown("### 📊 Chart Details")
                        st.markdown(f"**Type**: {chart['type']}")
                        st.markdown(f"**Priority**: {chart['priority']}")
                        with st.expander("View Code"):
                            st.code(chart.get('code', ''), language='python')
                    else:
                        # Full AI mode: Show AI insights
                        st.markdown("### 🎨 AI Insights")
                        if 'ai_insight' in chart:
                            st.markdown(chart['ai_insight'])
                        st.markdown(f"**Visualization**: {chart['type']}")
                        
                        # Let AI generate specific insights for this chart
                        insight_agent = InsightGeneratorAgent()
                        specific_insights = insight_agent.generate_chart_insights(
                            chart, df, analysis
                        )
                        st.markdown(specific_insights)



    def save_dashboard():
        dashboard_data = {
            'timestamp': datetime.now().isoformat(),
            'recommendations': st.session_state.agent_recommendations,
            'conversations': st.session_state.agent_conversations,
            'custom_charts': st.session_state.custom_charts,
            'data_analysis': st.session_state.data_analysis,
            'dataset_info': {
                'rows': len(st.session_state.dataset),
                'columns': list(st.session_state.dataset.columns),
            }
        }
        return dashboard_data
    
    def export_to_pdf():
        df = st.session_state.dataset
        business_cols = get_business_relevant_columns(df)
        chart_images = []
        for idx, chart in enumerate(st.session_state.agent_recommendations):
            try:
                chart_html = chart['figure'].to_html(include_plotlyjs='cdn', div_id=f"chart_{idx}")
                chart_images.append({
                    'title': chart['title'],
                    'html': chart_html,
                    'reason': chart.get('reason', ''),
                    'insights': chart.get('insights', []),
                    'priority': chart.get('priority', 'medium'),
                    'code': chart.get('code', '')
                })
            except:
                chart_images.append({
                    'title': chart['title'],
                    'html': '<div>Chart could not be exported</div>',
                    'reason': chart.get('reason', ''),
                    'insights': chart.get('insights', []),
                    'priority': chart.get('priority', 'medium'),
                    'code': chart.get('code', '')
                })
        custom_chart_images = []
        for idx, custom_chart in enumerate(st.session_state.custom_charts):
            if isinstance(custom_chart, dict) and custom_chart.get('figure'):
                try:
                    chart_html = custom_chart['figure'].to_html(include_plotlyjs='cdn', div_id=f"custom_chart_{idx}")
                    custom_chart_images.append({
                        'prompt': custom_chart.get('prompt', f'Custom Chart {i+1}'),
                        'html': chart_html,
                        'analysis': custom_chart.get('ai_analysis', 'No analysis available'),
                        'code': custom_chart.get('code', '')
                    })
                except:
                    custom_chart_images.append({
                        'prompt': custom_chart.get('prompt', f'Custom Chart {i+1}'),
                        'html': '<div>Custom chart could not be exported</div>',
                        'analysis': custom_chart.get('ai_analysis', 'No analysis available'),
                        'code': custom_chart.get('code', '')})


        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>NarraViz.ai - AI Dashboard Report</title>
            <style>
                body {{ 
                    font-family: 'Segoe UI', Arial, sans-serif; 
                    margin: 0.5rem 0;;
                    background-color: #1F2937; 
                    color: #93C5FD; 
                    line-height: 1.6;
                }}
                .header {{ 
                    color: #60A5FA; 
                    text-align: center; 
                    border-bottom: 2px solid #374151;
                    padding-bottom: 20px;
                    margin-bottom: 30px;
                }}
                .beta-banner {{
                    background: linear-gradient(135deg, #F59E0B 0%, #EF4444 100%);
                    color: white;
                    padding: 10px 20px;
                    border-radius: 8px;
                    text-align: center;
                    margin: 20px 0;
                    font-weight: bold;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
                }}
                .branding {{
                    background: linear-gradient(135deg, #8B5CF6 0%, #3B82F6 100%);
                    color: white;
                    padding: 15px;
                    border-radius: 12px;
                    text-align: center;
                    margin: 20px 0;
                    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
                }}
                .branding h2 {{
                    margin: 0;
                    font-size: 1.5em;
                    text-shadow: 0 2px 4px rgba(0,0,0,0.3);
                }}
                .branding p {{
                    margin: 5px 0 0 0;
                    opacity: 0.9;
                    font-size: 1.1em;
                }}
                .section {{ 
                    margin: 30px 0; 
                    padding: 20px; 
                    background-color: #374151; 
                    border-radius: 8px;
                    border: 1px solid #4B5563;
                }}
                .insight {{ 
                    margin: 10px 0; 
                    padding: 15px; 
                    background-color: #1F2937; 
                    border-left: 4px solid #60A5FA;
                    border-radius: 4px;
                }}
                .chart-container {{
                    margin: 20px 0;
                    padding: 20px;
                    background-color: #111827;
                    border-radius: 8px;
                    border: 1px solid #374151;
                }}
                .chart-title {{
                    color: #DBEAFE;
                    font-size: 1.2em;
                    font-weight: bold;
                    margin-bottom: 10px;
                }}
                .priority-badge {{
                    display: inline-block;
                    padding: 4px 12px;
                    border-radius: 12px;
                    font-size: 0.8em;
                    font-weight: bold;
                    margin-left: 10px;
                }}
                .priority-high {{ background-color: #7F1D1D; color: #FEE2E2; }}
                .priority-medium {{ background-color: #78350F; color: #FEF3C7; }}
                .priority-low {{ background-color: #312E81; color: #E0E7FF; }}
                .business-context {{
                    background-color: #8B5CF6;
                    color: white;
                    padding: 15px;
                    border-radius: 8px;
                    margin: 20px 0;
                }}
                .data-overview {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 15px;
                    margin: 20px 0;
                }}
                .metric-card {{
                    background-color: #1F2937;
                    padding: 15px;
                    border-radius: 8px;
                    text-align: center;
                    border: 1px solid #374151;
                }}
                .metric-value {{
                    font-size: 1.5em;
                    font-weight: bold;
                    color: #60A5FA;
                }}
                .metric-label {{
                    font-size: 0.9em;
                    color: #9CA3AF;
                    text-transform: uppercase;
                }}
                .conversation-log {{
                    background-color: #111827;
                    padding: 15px;
                    border-radius: 8px;
                    font-family: monospace;
                    font-size: 0.9em;
                }}
                .agent-message {{
                    margin: 8px 0;
                    padding: 10px;
                    border-left: 3px solid #8B5CF6;
                    background-color: #374151;
                    border-radius: 4px;
                }}
                h1, h2, h3 {{ color: #DBEAFE; }}
                strong {{ color: #DBEAFE; }}
                .feedback-summary {{
                    background-color: #065F46;
                    color: #D1FAE5;
                    padding: 15px;
                    border-radius: 8px;
                    margin: 20px 0;
                }}
                .disclaimer {{
                    background-color: #374151;
                    border: 1px solid #4B5563;
                    border-radius: 8px;
                    padding: 15px;
                    margin: 20px 0;
                    color: #9CA3AF;
                    font-size: 0.9em;
                    line-height: 1.4;
                }}
                .disclaimer h3 {{
                    color: #F59E0B;
                    margin-top: 0;
                }}
                .footer {{
                    text-align: center; 
                    margin-top: 40px; 
                    padding: 20px; 
                    border-top: 2px solid #374151;
                    background: linear-gradient(135deg, #1F2937 0%, #374151 100%);
                    border-radius: 8px;
                }}
                .footer .logo {{
                    font-size: 1.3em;
                    font-weight: bold;
                    color: #60A5FA;
                    margin-bottom: 10px;
                }}
                .footer .tagline {{
                    color: #93C5FD;
                    font-size: 1.1em;
                    margin-bottom: 15px;
                }}
                .footer .tech {{
                    color: #9CA3AF;
                    font-size: 0.9em;
                    font-style: italic;
                }}
                .code-section {{
                    background-color: #1F2937;
                    padding: 10px;
                    border-radius: 8px;
                    font-family: monospace;
                    font-size: 0.9em;
                    color: #93C5FD;
                    margin-top: 10px;
                    overflow-x: auto;
                }}
            </style>
        </head>
        <body>
            <div class="branding">
                <h2>🤖 NarraViz.ai - AI Dashboard Report</h2>
                <p>Powered by Real Agentic AI & Machine Learning</p>
            </div>
            <div class="beta-banner">
                🚧 BETA VERSION - Advanced AI Analysis System in Development 🚧
            </div>
            <div class="header">
                <p><strong>Generated on:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Dataset:</strong> {len(df):,} rows × {len(df.columns)} columns</p>
            </div>
            <div class="disclaimer">
                <h3>⚠️ Important Beta Disclaimers</h3>
                <ul>
                    <li><strong>Beta Software:</strong> This AI system is in active development. Results should be validated before making business decisions.</li>
                    <li><strong>AI Limitations:</strong> AI-generated insights are based on statistical patterns and may not capture all business context or nuances.</li>
                    <li><strong>Data Responsibility:</strong> Users are responsible for data quality, privacy, and ensuring appropriate use of generated insights.</li>
                    <li><strong>Continuous Learning:</strong> The system improves with feedback - your ratings help train better AI recommendations.</li>
                    <li><strong>Professional Review:</strong> Always have domain experts review AI findings before implementing strategic decisions.</li>
                </ul>
            </div>
            <div class="section">
                <h2>📊 Smart Data Overview</h2>
                <div class="data-overview">
                    <div class="metric-card">
                        <div class="metric-value">{len(df):,}</div>
                        <div class="metric-label">Total Rows</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{len(business_cols['categorical']) + len(business_cols['numerical']) + len(business_cols['temporal'])}</div>
                        <div class="metric-label">Business Columns</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{len(df.columns) - len(business_cols['categorical']) - len(business_cols['numerical']) - len(business_cols['temporal'])}</div>
                        <div class="metric-label">Excluded IDs</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{(1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100:.1f}%</div>
                        <div class="metric-label">Data Quality</div>
                    </div>
                </div>
                <h3>📈 Business Columns Analyzed:</h3>
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px;">
                    <div>
                        <strong>📊 Numerical (Metrics):</strong>
                        <ul>{''.join([f'<li>{col}</li>' for col in business_cols['numerical']])}</ul>
                    </div>
                    <div>
                        <strong>🏷️ Categorical (Dimensions):</strong>
                        <ul>{''.join([f'<li>{col}</li>' for col in business_cols['categorical']])}</ul>
                    </div>
                    <div>
                        <strong>📅 Temporal (Time):</strong>
                        <ul>{''.join([f'<li>{col}</li>' for col in business_cols['temporal']])}</ul>
                    </div>
                </div>
            </div>
            {f'''
            <div class="business-context">
                <h2>🏢 Business Context</h2>
                <p>{st.session_state.agent_learning.get('business_context', 'No business context provided yet.')}</p>
            </div>
            ''' if st.session_state.agent_learning.get('business_context') else ''}
            <div class="section">
                <h2>🧠 AI Data Analysis Results</h2>
                <div class="insight">
                    {json.dumps(st.session_state.data_analysis, indent=2) if st.session_state.data_analysis else 'AI analysis not yet completed. Run the AI analysis to see intelligent insights here.'}
                </div>
            </div>
            {f'''
            <div class="section">
                <h2>🤖 AI Agent Collaboration Log</h2>
                <div class="conversation-log">
                    {''.join([f'<div class="agent-message"><strong>{msg["agent"]}:</strong> {msg["message"]}</div>' for msg in st.session_state.agent_conversations])}
                </div>
            </div>
            ''' if st.session_state.agent_conversations else ''}
            <div class="section">
                <h2>📊 AI-Generated Intelligent Visualizations</h2>
                <p><em>Our AI agents created {len(chart_images)} optimized visualizations based on intelligent data analysis:</em></p>
                {''.join([f'''
                <div class="chart-container">
                    <div class="chart-title">
                        {chart["title"]}
                        <span class="priority-badge priority-{chart["priority"]}">{chart["priority"].upper()}</span>
                    </div>
                    <p><strong>🤖 AI Reasoning:</strong> {chart["reason"]}</p>
                    <div style="margin: 20px 0;">
                        {chart["html"]}
                    </div>
                    <h3>🧠 AI-Generated Insights:</h3>
                    {''.join([f'<div class="insight">• {insight}</div>' for insight in chart["insights"][:4]])}
                    <h3>💻 Generated Code:</h3>
                    <div class="code-section"><pre>{chart["code"]}</pre></div>
                </div>
                ''' for chart in chart_images])}
            </div>
            {f'''
            <div class="section">
                <h2>🎯 Custom AI-Generated Charts</h2>
                {''.join([f'''
                <div class="chart-container">
                    <div class="chart-title">🤖 Custom Analysis: {custom_chart["prompt"]}</div>
                    <div style="margin: 20px 0;">
                        {custom_chart["html"]}
                    </div>
                    <h3>🤖 AI Analysis:</h3>
                    <div class="insight">{custom_chart["analysis"]}</div>
                    <h3>💻 Generated Code:</h3>
                    <div class="code-section"><pre>{custom_chart["code"]}</pre></div>
                </div>
                ''' for custom_chart in custom_chart_images])}
            </div>
            ''' if custom_chart_images else ''}
            {f'''
            <div class="feedback-summary">
                <h2>📚 AI Learning Progress</h2>
                <div class="data-overview">
                    <div class="metric-card">
                        <div class="metric-value">{len(st.session_state.user_feedback)}</div>
                        <div class="metric-label">Feedback Sessions</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{len(st.session_state.agent_learning.get('preferred_charts', []))}</div>
                        <div class="metric-label">Preferred Patterns</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{len(st.session_state.agent_learning.get('avoided_columns', []))}</div>
                        <div class="metric-label">Avoided Columns</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{np.mean([len(fb['chart_rating'].split('⭐')) - 1 for fb in st.session_state.user_feedback[-10:]]) if st.session_state.user_feedback else 0:.1f}/5</div>
                        <div class="metric-label">Avg Chart Rating</div>
                    </div>
                </div>
                <h3>Recent User Feedback:</h3>
                {''.join([f'<div class="insight"><strong>{fb["chart_title"]}</strong> - {fb["chart_rating"]} - {fb["feedback_text"]}</div>' for fb in st.session_state.user_feedback[-5:]])}
            </div>
            ''' if st.session_state.user_feedback else ''}
            <div class="section">
                <h2>🎯 Summary & Recommendations</h2>
                <div class="insight">
                    <strong>Key Findings:</strong>
                    <ul>
                        <li>AI analyzed {len(business_cols['categorical']) + len(business_cols['numerical']) + len(business_cols['temporal'])} business-relevant columns while excluding {len(df.columns) - len(business_cols['categorical']) - len(business_cols['numerical']) - len(business_cols['temporal'])} ID fields</li>
                        <li>Generated {len(chart_images)} intelligent visualizations with business-specific insights</li>
                        <li>Created {len(custom_chart_images)} custom analyses based on user requests</li>
                        {f"<li>Incorporated {len(st.session_state.user_feedback)} feedback sessions for continuous AI improvement</li>" if st.session_state.user_feedback else ""}
                    </ul>
                </div>
                <div class="insight">
                    <strong>Next Steps:</strong>
                    <ul>
                        <li>Review the AI-generated insights for actionable business opportunities</li>
                        <li>Focus on high-priority visualizations for strategic decision-making</li>
                        <li>Use the feedback system to train AI for better future recommendations</li>
                        <li>Explore custom analysis requests for specific business questions</li>
                        <li>Validate findings with domain experts before implementation</li>
                        <li>Consider expanding analysis with additional data sources</li>
                    </ul>
                </div>
            </div>
            <div class="footer">
                <div class="logo">🤖 NarraViz.ai</div>
                <div class="tagline">Powered by Real Agentic AI & Intelligent Data Analysis</div>
                <div class="tech">Advanced Machine Learning • Natural Language Processing • Smart Data Analytics</div>
                <br>
                <div style="color: #F59E0B; font-weight: bold;">BETA VERSION - Continuously Learning & Improving</div>
                <div style="color: #9CA3AF; font-size: 0.8em; margin-top: 10px;">
                    Generated by the Real Agentic AI Chart System • Always validate insights with domain experts
                </div>
            </div>
        </body>
        </html>
        """
        return html_content.encode()

    df = st.session_state.dataset
    st.markdown("### 🔍 Smart Data Overview")
    business_cols = get_business_relevant_columns(df)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Rows", f"{len(df):,}")
    with col2:
        st.metric("Business Columns", len(business_cols['categorical']) + len(business_cols['numerical']) + len(business_cols['temporal']))
    with col3:
        st.metric("Excluded IDs", len(df.columns) - len(business_cols['categorical']) - len(business_cols['numerical']) - len(business_cols['temporal']))
    with col4:
        completeness = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        st.metric("Data Completeness", f"{completeness:.1f}%")
    with st.expander("🤖 What AI Will Analyze", expanded=False):
        col_diag1, col_diag2, col_diag3 = st.columns(3)
        with col_diag1:
            st.markdown("**📊 Numerical (Metrics)**")
            for col in business_cols['numerical']:
                st.markdown(f"• {col}")
        with col_diag2:
            st.markdown("**🏷️ Categorical (Dimensions)**") 
            for col in business_cols['categorical']:
                st.markdown(f"• {col}")
        with col_diag3:
            st.markdown("**📅 Temporal (Time)**")
            for col in business_cols['temporal']:
                st.markdown(f"• {col}")
            st.markdown("**🚫 Excluded (IDs)**")
            excluded = [col for col in df.columns if col not in business_cols['categorical'] + business_cols['numerical'] + business_cols['temporal']]
            for col in excluded[:5]:
                st.markdown(f"• {col}")
            if len(excluded) > 5:
                st.markdown(f"• ...and {len(excluded)-5} more")
    if not USE_OPENAI:
        st.error("🔑 OpenAI API key not configured. Please check your secrets.toml file.")
        st.info("💡 Add your OpenAI key to .streamlit/secrets.toml:\n```\n[openai]\napi_key = \"your-key-here\"\n```")
        return
    
    # Initialize AI Agents
    data_analyst = DataAnalystAgent()
    chart_creator = ChartCreatorAgent()
    insight_generator = InsightGeneratorAgent()

    # Add Full AI Mode toggle
    full_ai_mode = st.checkbox(
        "Enable Full AI Mode (LLM-driven everything)",
        value=st.session_state.get("full_ai_mode", False),
        help="When enabled, all chart recommendations and analysis are generated by the AI with minimal validation. When disabled, only business-validated charts are used.",
        key="full_ai_mode"
    )

    # Use the correct agent logic based on the toggle
    if full_ai_mode:
        data_analyst = DataAnalystAgent()
        chart_creator = ChartCreatorAgent()
    else:
        # Use the deterministic/business-validated logic (fallbacks, strict validation, etc.)
        data_analyst = DataAnalystAgent()  # This will use the fallback/validated logic
        chart_creator = ChartCreatorAgent()

    # Chart Generation Mode
    use_code = st.checkbox(
        "Use Code-Based Charts",
         value=st.session_state.get("use_code_based_charts", True),
         help="Enable to generate charts using verified Python code (reduces AI hallucination). Disable for fully AI-generated charts.",
         key="use_code_based_charts"
    )
 

    # Header with controls
    col1, col2, col3, col4 = st.columns([2, 0.8, 0.8, 0.8])
    with col1:
        st.markdown("### 🤖 Real Agentic AI - Intelligent Data Analysis")
        st.info("Watch AI agents analyze your data and create visualizations!")
    with col2:
        if st.button("🧠 Run AI Analysis", type="primary"):
            st.session_state.agent_recommendations = []
            st.session_state.agent_conversations = []
            st.session_state.custom_charts = []
            st.session_state.data_analysis = None
    with col3:
        if st.button("🗑️ Clear All"):
            st.session_state.agent_recommendations = []
            st.session_state.agent_conversations = []
            st.session_state.custom_charts = []
            st.session_state.data_analysis = None
    with col4:
        save_col1, save_col2 = st.columns(2)
        with save_col1:
            if st.button("💾 Save"):
                dashboard = save_dashboard()
                st.session_state.saved_dashboards.append(dashboard)
                st.success("Dashboard saved!")
        with save_col2:
            if st.button("📄 Export"):
                pdf_data = export_to_pdf()
                b64 = base64.b64encode(pdf_data).decode()
                href = f'<a href="data:text/html;base64,{b64}" download="ai_dashboard_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html">Download Report</a>'
                st.markdown(href, unsafe_allow_html=True)
    
    # Agent Status Display
    st.markdown("### 🤖 AI Agent Status")
    agent_cols = st.columns(3)
    agents_info = [
        ("🧠 Data Analyst", "data_analyst", "Analyzing patterns with AI..."),
        ("📊 Chart Creator", "chart_creator", "Designing optimal visualizations..."),
        ("💡 Insight Generator", "insight_generator", "Generating business insights...")
    ]
    
    for col, (name, key, description) in zip(agent_cols, agents_info):
        with col:
            status = st.session_state.agent_status[key]
            st.markdown(f"""
            <div class="agent-card">
                <strong>{name}</strong>
                <span class="agent-status status-{status}">{status.upper()}</span>
                <br><small>{description}</small>
            </div>
            """, unsafe_allow_html=True)
    
    # Run AI Analysis
    if not st.session_state.agent_recommendations and st.session_state.data_analysis is None:
        with st.spinner("🤖 AI Agents are intelligently analyzing your data..."):
            st.session_state.agent_status['data_analyst'] = 'analyzing'
            time.sleep(1)
            
            st.markdown('<div class="ai-thinking">🧠 Data Analyst Agent is examining data patterns...</div>', unsafe_allow_html=True)
            
            analysis = data_analyst.analyze_dataset(df)
            st.session_state.data_analysis = analysis
            
            st.session_state.agent_conversations.append({
                'agent': 'Data Analyst Agent',
                'message': f"📊 Identified {len(analysis.get('recommendations', []))} visualization opportunities.",
                'type': 'analyst',
                'details': analysis
            })
            
            st.session_state.agent_status['data_analyst'] = 'complete'
            
            st.session_state.agent_status['chart_creator'] = 'creating'
            time.sleep(1)
            
            st.markdown('<div class="ai-thinking">📊 Chart Creator Agent is generating visualizations...</div>', unsafe_allow_html=True)
            
            charts = chart_creator.create_intelligent_charts(df, analysis)
            st.session_state.agent_conversations.append({
                'agent': 'Chart Creator Agent',
                'message': f"🎨 Created {len(charts)} visualizations using {'code-based' if st.session_state.use_code_based_charts else 'AI-generated'} approach.",
                'type': 'creator'
            })
            
            st.session_state.agent_status['chart_creator'] = 'complete'
            
            st.session_state.agent_status['insight_generator'] = 'generating'
            time.sleep(1)
            
            st.markdown('<div class="ai-thinking">💡 Insight Generator Agent is analyzing charts...</div>', unsafe_allow_html=True)
            
            for chart in charts:
                if chart.get('agentic'):
                    chart['insights'] = chart.get('ai_insights', [])
                else:
                    chart['insights'] = insight_generator.generate_insights(df, chart, analysis)
            
            st.session_state.agent_recommendations = charts
            
            st.session_state.agent_conversations.append({
                'agent': 'Insight Generator Agent',
                'message': f"💡 Generated {sum(len(chart.get('insights', [])) for chart in charts)} insights across all visualizations.",
                'type': 'insight'
            })
            
            st.session_state.agent_status['insight_generator'] = 'complete'
            
            for key in st.session_state.agent_status:
                st.session_state.agent_status[key] = 'idle'
    
    # Display Agent Conversation Log
    if st.session_state.agent_conversations:
        with st.expander("🤖 AI Agent Collaboration Log", expanded=False):
            st.markdown('<div class="conversation-log">', unsafe_allow_html=True)
            for msg in st.session_state.agent_conversations:
                msg_class = f"{msg.get('type', 'info')}-message"
                st.markdown(f'<div class="agent-message {msg_class}"><strong>{msg["agent"]}:</strong> {msg["message"]}</div>', unsafe_allow_html=True)
                if msg.get('type') == 'analyst' and 'details' in msg:
                    details = msg['details']
                    st.markdown(f'<div style="margin-left: 1rem; font-size: 0.9rem; color: #9CA3AF;">')
                    st.markdown(f'<strong>Recommendations:</strong> {len(details.get("recommendations", []))} visualization opportunities')
                    st.markdown('</div>')
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Display AI-Generated Charts
    if st.session_state.agent_recommendations:
        st.markdown("### 🧠 AI-Generated Intelligent Visualizations")
        st.markdown(f"*Created {len(st.session_state.agent_recommendations)} visualizations:*")
        for i, chart in enumerate(st.session_state.agent_recommendations):
            st.markdown(f'<div class="chart-container">', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="chart-header">
                <h4>{chart['title']}</h4>
                <span class="priority-badge priority-{chart['priority']}">{chart['priority'].upper()}</span>
            </div>
            """, unsafe_allow_html=True)
            st.caption(f"🤖 AI Reasoning: {chart['reason']}")
            col1, col2 = st.columns([0.7, 0.3])
            with col1:
                st.plotly_chart(chart['figure'], use_container_width=True, key=f"rec_chart-{i}")
                with st.expander("💻 Generated Code", expanded=False):
                    st.code(chart.get('code', ''), language="python")
                if chart.get('data') is not None and chart.get('y_col') and hasattr(chart['data'], 'columns') and chart['y_col'] in chart['data'].columns:
                    col_stats = st.columns(4)
                    data = chart['data']
                    y_col = chart['y_col']
                    with col_stats[0]:
                        total_val = f"${data[y_col].sum():,.0f}"
                        st.markdown(f'<div class="metric-container"><div class="metric-value">{total_val}</div><div class="metric-label">Total</div></div>', unsafe_allow_html=True)
                    with col_stats[1]:
                        avg_val = f"${data[y_col].mean():,.0f}"
                        st.markdown(f'<div class="metric-container"><div class="metric-value">{avg_val}</div><div class="metric-label">Average</div></div>', unsafe_allow_html=True)
                    with col_stats[2]:
                        max_val = f"${data[y_col].max():,.0f}"
                        st.markdown(f'<div class="metric-container"><div class="metric-value">{max_val}</div><div class="metric-label">Max</div></div>', unsafe_allow_html=True)
                    with col_stats[3]:
                        count_val = f"{len(data):,}"
                        st.markdown(f'<div class="metric-container"><div class="metric-value">{count_val}</div><div class="metric-label">Count</div></div>', unsafe_allow_html=True)
            with col2:
                st.markdown('<div class="insights-container">', unsafe_allow_html=True)
                st.markdown("### 🧠 Insights")
                for insight in chart.get('insights', [])[:4]:
                    st.markdown(f"• **{insight}**")
                st.markdown("---")
                st.markdown("#### 📝 Rate & Improve")
                col_rate1, col_rate2 = st.columns(2)
                with col_rate1:
                    chart_rating = st.selectbox(
                        "Chart Usefulness", 
                        ["⭐ Poor", "⭐⭐ Fair", "⭐⭐⭐ Good", "⭐⭐⭐⭐ Great", "⭐⭐⭐⭐⭐ Excellent"],
                        key=f"rating_{i}",
                        index=2
                    )
                with col_rate2:
                    insights_rating = st.selectbox(
                        "Insights Quality",
                        ["⭐ Poor", "⭐⭐ Fair", "⭐⭐⭐ Good", "⭐⭐⭐⭐ Great", "⭐⭐⭐⭐⭐ Excellent"],
                        key=f"insights_rating_{i}",
                        index=2
                    )
                feedback_text = st.text_area(
                    "Feedback for AI",
                    placeholder="How can this chart/insight be improved?",
                    key=f"feedback_{i}",
                    height=68
                )
                if st.button(f"📚 Train AI", key=f"train_{i}"):
                    feedback_data = {
                        'chart_title': chart['title'],
                        'chart_type': chart['type'],
                        'x_col': chart['x_col'],
                        'y_col': chart['y_col'],
                        'chart_rating': chart_rating,
                        'insights_rating': insights_rating,
                        'feedback_text': feedback_text,
                        'timestamp': datetime.now().isoformat()
                    }
                    st.session_state.user_feedback.append(feedback_data)
                    rating_value = len(chart_rating.split('⭐')) - 1
                    if rating_value >= 4:
                        chart_pattern = f"{chart['type']}:{chart['x_col']}:{chart['y_col']}"
                        if chart_pattern not in st.session_state.agent_learning['preferred_charts']:
                            st.session_state.agent_learning['preferred_charts'].append(chart_pattern)
                    elif rating_value <= 2:
                        if chart['x_col'] not in st.session_state.agent_learning['avoided_columns']:
                            st.session_state.agent_learning['avoided_columns'].append(chart['x_col'])
                    st.success("✅ Feedback saved! AI will learn from this.")
                st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
    # --- Move custom chart display here ---
    if st.session_state.custom_charts:
        st.markdown("---")
        st.markdown("### 🎯 Custom AI-Generated Charts")
        for i, custom_chart in enumerate(st.session_state.custom_charts):
            if isinstance(custom_chart, dict):
                st.markdown(f'<div class="custom-chart-section">', unsafe_allow_html=True)
                st.markdown(f"#### 🤖 AI Custom Chart #{i+1}: {custom_chart.get('prompt', 'Custom Analysis')}")
                col1, col2 = st.columns([0.7, 0.3])
                with col1:
                    if custom_chart.get('figure'):
                        st.plotly_chart(custom_chart['figure'], use_container_width=True, key=f"custom_chart-{i}")
                    with st.expander("💻 Generated Code", expanded=False):
                        st.code(custom_chart.get('code', 'No code available'), language="python")

                with col2:
                    st.markdown('<div class="insights-container">', unsafe_allow_html=True)
                    st.markdown("### 🧠 AI-Generated Insights")
                    insights = chart.get('insights', [])
                    
                    # Ensure we always have at least 3 insights
                    if not insights or len(insights) < 2:
                        insights = insight_generator._get_default_insights(chart)
                    
                    for insight in insights[:4]:  # Show top 4 insights
                        st.markdown(f"• **{insight}**")
                
    # --- Restore business context and custom chart prompt input boxes below custom charts ---
    st.markdown("---")
    st.markdown("### 🏢 Business Context & Custom Analysis")
    col1, col2 = st.columns([1, 1])
    with col1:
        business_context = st.text_area(
            "Provide Business Context",
            value=st.session_state.agent_learning.get('business_context', ''),
            placeholder="e.g., We're a retail company focusing on Q4 sales performance...",
            height=100
        )
        if st.button("💾 Save Context"):
            st.session_state.agent_learning['business_context'] = business_context
            st.success("Business context saved!")
    with col2:
        custom_prompt = st.text_area(
            "Request Custom Chart",
            placeholder="e.g., Show me seasonal trends in revenue by product category",
            height=100
        )
        if st.button("🎯 Create Custom Chart"):
            with st.spinner("Creating custom visualization..."):
                custom_chart = agentic_ai.create_custom_chart(df, custom_prompt, business_context)
                if custom_chart is not None and 'error' not in custom_chart:
                    st.session_state.custom_charts.append(custom_chart)
                    st.success("Custom chart created!")
                    st.rerun() 
                else:
                    st.error(custom_chart['error'] if custom_chart else 'Unknown error creating custom chart.')

    render_agentic_executive_summary()

    # --- FOOTER (Branding & Disclaimer) ---
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; margin-top: 40px; padding: 20px; border-top: 2px solid #374151; background: linear-gradient(135deg, #1F2937 0%, #374151 100%); border-radius: 8px;">
        <div style="font-size: 1.3em; font-weight: bold; color: #60A5FA; margin-bottom: 10px;">🤖 NarraViz.ai</div>
        <div style="color: #93C5FD; font-size: 1.1em; margin-bottom: 15px;">Powered by Real Agentic AI & Intelligent Data Analysis</div>
        <div style="color: #9CA3AF; font-size: 0.9em; font-style: italic;">Advanced Machine Learning • Natural Language Processing • Smart Data Analytics</div>
        <br>
        <div style="color: #F59E0B; font-weight: bold;">BETA VERSION - Continuously Learning & Improving</div>
        <div style="color: #9CA3AF; font-size: 0.8em; margin-top: 10px;">
            Generated by the Real Agentic AI Chart System • Always validate insights with domain experts
        </div>
    </div>
    """, unsafe_allow_html=True)

class FeedbackLearningSystem:
    """System to store and learn from user feedback"""
    
    def __init__(self, supabase=None):
        self.supabase = supabase
        
    def store_feedback(self, feedback_data):
        """Store feedback in database and session state"""
        try:
            # Add to session state for immediate use
            if 'user_feedback_history' not in st.session_state:
                st.session_state.user_feedback_history = []
            
            feedback_entry = {
                'timestamp': datetime.now().isoformat(),
                'user_id': st.session_state.get('user_id', 'anonymous'),
                'dataset_hash': self._hash_dataset(st.session_state.dataset),
                **feedback_data
            }
            
            st.session_state.user_feedback_history.append(feedback_entry)
            
            # Store in Supabase if available
            if self.supabase:
                self.supabase.table('ai_feedback').insert({
                    'user_id': feedback_entry['user_id'],
                    'feedback_type': feedback_data.get('type'),
                    'content': json.dumps(feedback_data),
                    'dataset_info': {
                        'columns': list(st.session_state.dataset.columns),
                        'shape': st.session_state.dataset.shape,
                        'hash': feedback_entry['dataset_hash']
                    }
                }).execute()
                
            return True
            
        except Exception as e:
            st.error(f"Failed to store feedback: {str(e)}")
            return False
    
    def _hash_dataset(self, df):
        """Create a hash of dataset structure for matching similar datasets"""
        structure = f"{sorted(df.columns.tolist())}_{df.shape}_{df.dtypes.to_dict()}"
        return hashlib.md5(structure.encode()).hexdigest()[:10]
    
    def get_learned_preferences(self, df):
        """Get learned preferences for the current dataset type"""
        preferences = {
            'preferred_charts': [],
            'avoided_columns': [],
            'business_context': "",
            'chart_ratings': {},
            'successful_patterns': []
        }
        
        try:
            current_hash = self._hash_dataset(df)
            
            # Get from session state
            feedback_history = st.session_state.get('user_feedback_history', [])
            
            # Get from database if available
            if self.supabase:
                result = self.supabase.table('ai_feedback').select('*').eq(
                    'user_id', st.session_state.get('user_id', 'anonymous')
                ).execute()
                
                if result.data:
                    for entry in result.data:
                        content = json.loads(entry.get('content', '{}'))
                        if content.get('dataset_hash') == current_hash:
                            self._process_feedback_entry(content, preferences)
            
            # Process session state feedback
            for entry in feedback_history:
                if entry.get('dataset_hash') == current_hash:
                    self._process_feedback_entry(entry, preferences)
                    
            # Get global preferences
            global_prefs = st.session_state.get('agent_learning', {})
            preferences['business_context'] = global_prefs.get('business_context', '')
            
        except Exception as e:
            st.warning(f"Could not load preferences: {str(e)}")
            
        return preferences
    
    def _process_feedback_entry(self, entry, preferences):
        """Process a single feedback entry to update preferences"""
        feedback_type = entry.get('type')
        
        if feedback_type == 'chart_rating':
            chart_key = f"{entry.get('chart_type')}_{entry.get('columns')}"
            if chart_key not in preferences['chart_ratings']:
                preferences['chart_ratings'][chart_key] = []
            preferences['chart_ratings'][chart_key].append(entry.get('rating', 0))
            
            # High-rated charts become preferred
            if entry.get('rating', 0) >= 4:
                preferences['preferred_charts'].append({
                    'type': entry.get('chart_type'),
                    'columns': entry.get('columns'),
                    'reason': entry.get('reason')
                })
                
        elif feedback_type == 'train_ai':
            # Process training feedback
            if entry.get('good_charts'):
                preferences['preferred_charts'].extend(entry.get('good_charts', []))
            if entry.get('bad_columns'):
                preferences['avoided_columns'].extend(entry.get('bad_columns', []))
                
        elif feedback_type == 'business_context':
            # Update business context
            preferences['business_context'] = entry.get('context', '')


def add_feedback_components(chart, chart_index, feedback_system):
    """Add feedback components for a chart"""
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        # Star rating
        rating_key = f"rating_{chart_index}_{chart.get('type', 'unknown')}"
        rating = st.select_slider(
            "Rate this visualization",
            options=[1, 2, 3, 4, 5],
            value=3,
            format_func=lambda x: "⭐" * x,
            key=rating_key
        )
        
    with col2:
        if st.button("Save Rating", key=f"save_rating_{chart_index}"):
            feedback_data = {
                'type': 'chart_rating',
                'chart_type': chart.get('type'),
                'columns': [chart.get('x_col'), chart.get('y_col')],
                'rating': rating,
                'reason': chart.get('reason', ''),
                'code': chart.get('code', '')
            }
            if feedback_system.store_feedback(feedback_data):
                st.success("Rating saved!")
    
    with col3:
        if st.button("📌 Pin Chart", key=f"pin_{chart_index}"):
            if 'pinned_charts' not in st.session_state:
                st.session_state.pinned_charts = []
            st.session_state.pinned_charts.append(chart)
            st.success("Chart pinned!")


def add_training_interface(feedback_system):
    """Add the Train AI interface"""
    
    with st.expander("🧠 Train AI - Teach it your preferences"):
        st.markdown("Help the AI understand your data better:")
        
        # Business context
        business_context = st.text_area(
            "Business Context",
            value=st.session_state.get('agent_learning', {}).get('business_context', ''),
            placeholder="Describe your business, key metrics, and what insights matter most...",
            help="This helps AI create more relevant visualizations"
        )
        
        # Good chart examples
        good_charts = st.text_area(
            "Examples of Good Charts",
            placeholder="e.g., Revenue over time, Customer segmentation by region, Product performance comparison",
            help="Describe charts that would be valuable for your analysis"
        )
        
        # Columns to avoid
        bad_columns = st.text_input(
            "Columns to Ignore",
            placeholder="e.g., id, internal_code, debug_flag",
            help="Comma-separated list of columns that aren't useful for analysis"
        )
        
        # Preferred chart types
        chart_types = st.multiselect(
            "Preferred Chart Types",
            options=['bar', 'line', 'scatter', 'pie', 'box', 'histogram', 'heatmap', 'treemap', 'sunburst'],
            default=['bar', 'line', 'scatter']
        )
        
        if st.button("💾 Save Training Data", key="save_training"):
            # Update session state
            st.session_state.agent_learning = {
                'business_context': business_context,
                'preferred_charts': good_charts.split(',') if good_charts else [],
                'avoided_columns': [col.strip() for col in bad_columns.split(',')] if bad_columns else [],
                'preferred_types': chart_types
            }
            
            # Store in database
            feedback_data = {
                'type': 'train_ai',
                'context': business_context,
                'good_charts': good_charts.split(',') if good_charts else [],
                'bad_columns': [col.strip() for col in bad_columns.split(',')] if bad_columns else [],
                'preferred_types': chart_types
            }
            
            if feedback_system.store_feedback(feedback_data):
                st.success("✅ AI training data saved! Future charts will be more relevant.")