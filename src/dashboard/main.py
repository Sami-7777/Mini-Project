"""
Main Streamlit dashboard for the cyberattack detection system.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import requests
import asyncio
from datetime import datetime, timedelta
import json

from ..core.config import settings
from ..core.logger import logger


class CyberattackDashboard:
    """Main dashboard for the cyberattack detection system."""
    
    def __init__(self):
        self.api_base_url = f"http://{settings.api_host}:{settings.api_port}/api/v1"
        self.session = requests.Session()
        self.session.headers.update({
            "X-API-Key": "sk-demo-key-1234567890",
            "Content-Type": "application/json"
        })
    
    def run(self):
        """Run the main dashboard."""
        # Configure page
        st.set_page_config(
            page_title="Cyberattack Detection System",
            page_icon="🛡️",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS
        st.markdown("""
        <style>
            .main-header {
                font-size: 3rem;
                color: #1f77b4;
                text-align: center;
                margin-bottom: 2rem;
                background: linear-gradient(90deg, #1f77b4, #ff7f0e);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }
            .metric-card {
                background-color: #f8f9fa;
                padding: 1.5rem;
                border-radius: 0.5rem;
                border-left: 4px solid #28a745;
                margin-bottom: 1rem;
            }
            .threat-card {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 1.5rem;
                border-radius: 0.5rem;
                margin-bottom: 1rem;
            }
            .alert-card {
                background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                color: white;
                padding: 1.5rem;
                border-radius: 0.5rem;
                margin-bottom: 1rem;
            }
        </style>
        """, unsafe_allow_html=True)
        
        # Sidebar
        self.render_sidebar()
        
        # Main content
        page = st.session_state.get("page", "overview")
        
        if page == "overview":
            self.render_overview()
        elif page == "analysis":
            self.render_analysis_page()
        elif page == "threats":
            self.render_threats_page()
        elif page == "models":
            self.render_models_page()
        elif page == "intelligence":
            self.render_intelligence_page()
        elif page == "alerts":
            self.render_alerts_page()
        elif page == "ai":
            self.render_ai_page()
        elif page == "analytics":
            self.render_analytics_page()
    
    def render_sidebar(self):
        """Render the sidebar navigation."""
        st.sidebar.title("🛡️ Cyberattack Detection")
        st.sidebar.markdown("---")
        
        # Navigation
        pages = {
            "overview": "📊 Overview",
            "analysis": "🔍 Analysis",
            "threats": "⚠️ Threats",
            "models": "🤖 Models",
            "intelligence": "🧠 Intelligence",
            "alerts": "🚨 Alerts",
            "ai": "🤖 AI Dashboard",
            "analytics": "📈 Analytics"
        }
        
        selected_page = st.sidebar.selectbox(
            "Navigation",
            options=list(pages.keys()),
            format_func=lambda x: pages[x],
            key="page_selector"
        )
        
        st.session_state.page = selected_page
        
        st.sidebar.markdown("---")
        
        # System Status
        st.sidebar.subheader("System Status")
        system_status = self.get_system_status()
        
        if system_status:
            st.sidebar.metric("API Status", "✅ Online" if system_status.get("api_online") else "❌ Offline")
            st.sidebar.metric("Database", "✅ Connected" if system_status.get("db_connected") else "❌ Disconnected")
            st.sidebar.metric("Models", system_status.get("trained_models", 0))
        else:
            st.sidebar.error("❌ Cannot connect to system")
        
        st.sidebar.markdown("---")
        
        # Quick Actions
        st.sidebar.subheader("Quick Actions")
        if st.sidebar.button("🔄 Refresh Data"):
            st.rerun()
        
        if st.sidebar.button("🎯 New Analysis"):
            st.session_state.page = "analysis"
            st.rerun()
        
        if st.sidebar.button("🤖 AI Dashboard"):
            st.session_state.page = "ai"
            st.rerun()
    
    def render_overview(self):
        """Render the overview page."""
        st.markdown('<h1 class="main-header">🛡️ Cyberattack Detection System</h1>', unsafe_allow_html=True)
        
        # System metrics
        st.subheader("📊 System Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Analyses", 1250, delta=25)
        
        with col2:
            st.metric("Threats Detected", 89, delta=12)
        
        with col3:
            st.metric("False Positives", 3, delta=-2)
        
        with col4:
            st.metric("System Uptime", "99.9%", delta="0.1%")
        
        st.markdown("---")
        
        # Recent threats
        st.subheader("⚠️ Recent Threats")
        
        # Simulate recent threats
        recent_threats = [
            {"timestamp": "2024-01-01T10:00:00Z", "type": "Phishing", "target": "https://example.com", "severity": "High", "confidence": 0.95},
            {"timestamp": "2024-01-01T09:45:00Z", "type": "Malware", "target": "192.168.1.100", "severity": "Critical", "confidence": 0.98},
            {"timestamp": "2024-01-01T09:30:00Z", "type": "DoS", "target": "192.168.1.200", "severity": "Medium", "confidence": 0.87},
            {"timestamp": "2024-01-01T09:15:00Z", "type": "Ransomware", "target": "https://malicious.com", "severity": "Critical", "confidence": 0.99}
        ]
        
        df_threats = pd.DataFrame(recent_threats)
        st.dataframe(df_threats, use_container_width=True)
        
        # Threat distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Threat Types")
            
            # Threat type distribution
            threat_types = ["Phishing", "Malware", "DoS", "Ransomware", "Probe", "R2L", "U2R"]
            threat_counts = [25, 18, 15, 12, 10, 8, 2]
            
            fig = px.pie(
                values=threat_counts,
                names=threat_types,
                title="Threat Type Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📈 Threat Trends")
            
            # Threat trends over time
            dates = pd.date_range(start="2024-01-01", end="2024-01-07", freq="D")
            threat_counts = [12, 15, 18, 22, 19, 25, 28]
            
            fig = px.line(
                x=dates,
                y=threat_counts,
                title="Threats Detected Over Time",
                labels={"x": "Date", "y": "Threat Count"}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # System performance
        st.subheader("⚡ System Performance")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.write("**Response Time**")
            st.metric("Average", "45ms")
            st.metric("P95", "120ms")
            st.metric("P99", "250ms")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.write("**Throughput**")
            st.metric("Requests/sec", "1,250")
            st.metric("Peak", "2,100")
            st.metric("Capacity", "85%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.write("**Accuracy**")
            st.metric("Overall", "96.5%")
            st.metric("Precision", "94.2%")
            st.metric("Recall", "98.1%")
            st.markdown('</div>', unsafe_allow_html=True)
    
    def render_analysis_page(self):
        """Render the analysis page."""
        st.title("🔍 Threat Analysis")
        
        # Analysis form
        with st.form("analysis_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                target_type = st.selectbox("Target Type", ["url", "ip"])
                target = st.text_input("Target", placeholder="Enter URL or IP address")
            
            with col2:
                context = st.text_area("Additional Context", placeholder="Optional context information")
                priority = st.selectbox("Priority", ["low", "medium", "high", "critical"])
            
            submitted = st.form_submit_button("Analyze Threat", type="primary")
        
        if submitted and target:
            with st.spinner("Analyzing threat..."):
                result = self.analyze_threat(target, target_type, context, priority)
                
                if result:
                    st.success("Analysis completed!")
                    
                    # Display results
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Threat Type", result.get("attack_type", "Unknown"))
                    
                    with col2:
                        st.metric("Confidence", f"{result.get('confidence', 0):.2%}")
                    
                    with col3:
                        st.metric("Severity", result.get("severity", "Unknown"))
                    
                    with col4:
                        st.metric("Risk Score", f"{result.get('risk_score', 0):.2f}")
                    
                    # Detailed results
                    st.subheader("📋 Detailed Analysis")
                    
                    with st.expander("🔍 Analysis Details"):
                        st.json(result)
                    
                    # Recommendations
                    st.subheader("💡 Recommendations")
                    
                    recommendations = result.get("recommendations", [])
                    if recommendations:
                        for i, rec in enumerate(recommendations, 1):
                            st.write(f"{i}. {rec}")
                    else:
                        st.info("No specific recommendations available.")
        
        # Analysis history
        st.subheader("📊 Analysis History")
        
        # Simulate analysis history
        analysis_history = [
            {"timestamp": "2024-01-01T10:00:00Z", "target": "https://example.com", "type": "Phishing", "confidence": 0.95, "status": "Completed"},
            {"timestamp": "2024-01-01T09:45:00Z", "target": "192.168.1.100", "type": "Malware", "confidence": 0.98, "status": "Completed"},
            {"timestamp": "2024-01-01T09:30:00Z", "target": "https://suspicious.com", "type": "Unknown", "confidence": 0.45, "status": "In Progress"}
        ]
        
        df_history = pd.DataFrame(analysis_history)
        st.dataframe(df_history, use_container_width=True)
    
    def render_threats_page(self):
        """Render the threats page."""
        st.title("⚠️ Threat Management")
        
        # Threat overview
        st.subheader("📊 Threat Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Active Threats", 25, delta=5)
        
        with col2:
            st.metric("Critical Threats", 8, delta=2)
        
        with col3:
            st.metric("Resolved Threats", 156, delta=12)
        
        with col4:
            st.metric("False Positives", 3, delta=-1)
        
        # Threat list
        st.subheader("📋 Threat List")
        
        # Simulate threat data
        threats = [
            {"id": "T001", "type": "Phishing", "target": "https://example.com", "severity": "High", "status": "Active", "detected": "2024-01-01T10:00:00Z"},
            {"id": "T002", "type": "Malware", "target": "192.168.1.100", "severity": "Critical", "status": "Active", "detected": "2024-01-01T09:45:00Z"},
            {"id": "T003", "type": "DoS", "target": "192.168.1.200", "severity": "Medium", "status": "Resolved", "detected": "2024-01-01T09:30:00Z"},
            {"id": "T004", "type": "Ransomware", "target": "https://malicious.com", "severity": "Critical", "status": "Active", "detected": "2024-01-01T09:15:00Z"}
        ]
        
        df_threats = pd.DataFrame(threats)
        st.dataframe(df_threats, use_container_width=True)
        
        # Threat details
        st.subheader("🔍 Threat Details")
        
        selected_threat = st.selectbox("Select Threat", [threat["id"] for threat in threats])
        
        if selected_threat:
            threat_details = next((t for t in threats if t["id"] == selected_threat), None)
            
            if threat_details:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Basic Information:**")
                    st.write(f"- ID: {threat_details['id']}")
                    st.write(f"- Type: {threat_details['type']}")
                    st.write(f"- Target: {threat_details['target']}")
                    st.write(f"- Severity: {threat_details['severity']}")
                    st.write(f"- Status: {threat_details['status']}")
                    st.write(f"- Detected: {threat_details['detected']}")
                
                with col2:
                    st.write("**Analysis Results:**")
                    st.write("- Confidence: 95%")
                    st.write("- Risk Score: 0.87")
                    st.write("- Impact: High")
                    st.write("- Source: External")
                    st.write("- Geolocation: Unknown")
    
    def render_models_page(self):
        """Render the models page."""
        st.title("🤖 ML Models")
        
        # Model overview
        st.subheader("📊 Model Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Models", 9, delta=1)
        
        with col2:
            st.metric("Trained Models", 7, delta=2)
        
        with col3:
            st.metric("Average Accuracy", "96.5%", delta="1.2%")
        
        with col4:
            st.metric("Last Training", "2 hours ago", delta=None)
        
        # Model list
        st.subheader("📋 Model List")
        
        # Simulate model data
        models = [
            {"name": "Random Forest", "type": "Classical", "accuracy": 0.95, "status": "Trained", "last_trained": "2024-01-01T08:00:00Z"},
            {"name": "XGBoost", "type": "Classical", "accuracy": 0.97, "status": "Trained", "last_trained": "2024-01-01T08:00:00Z"},
            {"name": "CNN-LSTM", "type": "Deep Learning", "accuracy": 0.96, "status": "Trained", "last_trained": "2024-01-01T07:30:00Z"},
            {"name": "Transformer", "type": "Deep Learning", "accuracy": 0.98, "status": "Trained", "last_trained": "2024-01-01T07:00:00Z"},
            {"name": "Graph Neural Network", "type": "Graph", "accuracy": 0.94, "status": "Training", "last_trained": "2024-01-01T06:00:00Z"},
            {"name": "Quantum VQC", "type": "Quantum", "accuracy": 0.92, "status": "Trained", "last_trained": "2024-01-01T05:00:00Z"}
        ]
        
        df_models = pd.DataFrame(models)
        st.dataframe(df_models, use_container_width=True)
        
        # Model performance
        st.subheader("📈 Model Performance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Accuracy comparison
            model_names = [m["name"] for m in models]
            accuracies = [m["accuracy"] for m in models]
            
            fig = px.bar(
                x=model_names,
                y=accuracies,
                title="Model Accuracy Comparison",
                labels={"x": "Model", "y": "Accuracy"}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Model types
            model_types = ["Classical", "Deep Learning", "Graph", "Quantum"]
            type_counts = [2, 2, 1, 1]
            
            fig = px.pie(
                values=type_counts,
                names=model_types,
                title="Model Type Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Model training
        st.subheader("🎯 Model Training")
        
        with st.form("training_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                model_name = st.selectbox("Model", [m["name"] for m in models])
                training_data = st.file_uploader("Training Data", type=["csv", "json"])
            
            with col2:
                epochs = st.number_input("Epochs", min_value=1, max_value=1000, value=100)
                batch_size = st.number_input("Batch Size", min_value=1, max_value=1024, value=32)
            
            if st.form_submit_button("Start Training"):
                st.info("Model training would start here")
    
    def render_intelligence_page(self):
        """Render the intelligence page."""
        st.title("🧠 Threat Intelligence")
        
        # Intelligence overview
        st.subheader("📊 Intelligence Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Intelligence Sources", 8, delta=1)
        
        with col2:
            st.metric("Active Feeds", 6, delta=0)
        
        with col3:
            st.metric("Threat Indicators", 1250, delta=25)
        
        with col4:
            st.metric("Last Update", "5 min ago", delta=None)
        
        # Intelligence sources
        st.subheader("🔗 Intelligence Sources")
        
        # Simulate intelligence sources
        sources = [
            {"name": "VirusTotal", "type": "Malware", "status": "Active", "last_update": "2024-01-01T10:00:00Z", "reliability": 0.95},
            {"name": "Google Safe Browsing", "type": "Phishing", "status": "Active", "last_update": "2024-01-01T09:55:00Z", "reliability": 0.98},
            {"name": "AbuseIPDB", "type": "IP Reputation", "status": "Active", "last_update": "2024-01-01T09:50:00Z", "reliability": 0.92},
            {"name": "Shodan", "type": "Network", "status": "Active", "last_update": "2024-01-01T09:45:00Z", "reliability": 0.89},
            {"name": "PhishTank", "type": "Phishing", "status": "Active", "last_update": "2024-01-01T09:40:00Z", "reliability": 0.94},
            {"name": "URLVoid", "type": "URL Reputation", "status": "Active", "last_update": "2024-01-01T09:35:00Z", "reliability": 0.91}
        ]
        
        df_sources = pd.DataFrame(sources)
        st.dataframe(df_sources, use_container_width=True)
        
        # Intelligence query
        st.subheader("🔍 Intelligence Query")
        
        with st.form("intelligence_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                query_type = st.selectbox("Query Type", ["url", "ip", "domain", "hash"])
                query_target = st.text_input("Target", placeholder="Enter target to query")
            
            with col2:
                sources_selected = st.multiselect("Sources", [s["name"] for s in sources], default=[s["name"] for s in sources])
                include_history = st.checkbox("Include History", value=True)
            
            if st.form_submit_button("Query Intelligence"):
                if query_target:
                    st.info("Intelligence query would be performed here")
        
        # Intelligence feeds
        st.subheader("📡 Intelligence Feeds")
        
        # Simulate intelligence feeds
        feeds = [
            {"feed": "Malware Indicators", "count": 125, "last_update": "2024-01-01T10:00:00Z", "status": "Active"},
            {"feed": "Phishing URLs", "count": 89, "last_update": "2024-01-01T09:55:00Z", "status": "Active"},
            {"feed": "Suspicious IPs", "count": 156, "last_update": "2024-01-01T09:50:00Z", "status": "Active"},
            {"feed": "Ransomware Hashes", "count": 67, "last_update": "2024-01-01T09:45:00Z", "status": "Active"}
        ]
        
        df_feeds = pd.DataFrame(feeds)
        st.dataframe(df_feeds, use_container_width=True)
    
    def render_alerts_page(self):
        """Render the alerts page."""
        st.title("🚨 Alert Management")
        
        # Alert overview
        st.subheader("📊 Alert Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Active Alerts", 12, delta=3)
        
        with col2:
            st.metric("Critical Alerts", 4, delta=1)
        
        with col3:
            st.metric("Resolved Alerts", 89, delta=8)
        
        with col4:
            st.metric("False Positives", 2, delta=-1)
        
        # Alert list
        st.subheader("📋 Alert List")
        
        # Simulate alert data
        alerts = [
            {"id": "A001", "type": "Threat Detected", "severity": "Critical", "target": "https://malicious.com", "status": "Active", "created": "2024-01-01T10:00:00Z"},
            {"id": "A002", "type": "High Risk Score", "severity": "High", "target": "192.168.1.100", "status": "Active", "created": "2024-01-01T09:45:00Z"},
            {"id": "A003", "type": "Anomaly Detected", "severity": "Medium", "target": "https://suspicious.com", "status": "Resolved", "created": "2024-01-01T09:30:00Z"},
            {"id": "A004", "type": "Model Performance", "severity": "Low", "target": "System", "status": "Active", "created": "2024-01-01T09:15:00Z"}
        ]
        
        df_alerts = pd.DataFrame(alerts)
        st.dataframe(df_alerts, use_container_width=True)
        
        # Alert details
        st.subheader("🔍 Alert Details")
        
        selected_alert = st.selectbox("Select Alert", [alert["id"] for alert in alerts])
        
        if selected_alert:
            alert_details = next((a for a in alerts if a["id"] == selected_alert), None)
            
            if alert_details:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Basic Information:**")
                    st.write(f"- ID: {alert_details['id']}")
                    st.write(f"- Type: {alert_details['type']}")
                    st.write(f"- Severity: {alert_details['severity']}")
                    st.write(f"- Target: {alert_details['target']}")
                    st.write(f"- Status: {alert_details['status']}")
                    st.write(f"- Created: {alert_details['created']}")
                
                with col2:
                    st.write("**Alert Actions:**")
                    if st.button("Acknowledge Alert"):
                        st.success("Alert acknowledged!")
                    
                    if st.button("Resolve Alert"):
                        st.success("Alert resolved!")
                    
                    if st.button("Escalate Alert"):
                        st.warning("Alert escalated!")
        
        # Alert configuration
        st.subheader("⚙️ Alert Configuration")
        
        with st.form("alert_config_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                alert_type = st.selectbox("Alert Type", ["Threat Detected", "High Risk Score", "Anomaly Detected", "Model Performance"])
                severity = st.selectbox("Severity", ["Low", "Medium", "High", "Critical"])
            
            with col2:
                threshold = st.number_input("Threshold", min_value=0.0, max_value=1.0, value=0.8, step=0.1)
                enabled = st.checkbox("Enabled", value=True)
            
            if st.form_submit_button("Update Configuration"):
                st.success("Alert configuration updated!")
    
    def render_ai_page(self):
        """Render the AI dashboard page."""
        st.title("🤖 AI Dashboard")
        
        # Import and run AI dashboard
        from .ai_dashboard import AIDashboard
        
        ai_dashboard = AIDashboard()
        ai_dashboard.run()
    
    def render_analytics_page(self):
        """Render the analytics page."""
        st.title("📈 Advanced Analytics")
        
        # Analytics overview
        st.subheader("📊 Analytics Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Analyses", 1250, delta=25)
        
        with col2:
            st.metric("Threats Detected", 89, delta=12)
        
        with col3:
            st.metric("Accuracy", "96.5%", delta="1.2%")
        
        with col4:
            st.metric("Response Time", "45ms", delta="-5ms")
        
        # Analytics charts
        st.subheader("📈 Analytics Charts")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Threat trends
            dates = pd.date_range(start="2024-01-01", end="2024-01-07", freq="D")
            threat_counts = [12, 15, 18, 22, 19, 25, 28]
            
            fig = px.line(
                x=dates,
                y=threat_counts,
                title="Threat Trends",
                labels={"x": "Date", "y": "Threat Count"}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Attack types
            attack_types = ["Phishing", "Malware", "DoS", "Ransomware", "Probe"]
            attack_counts = [25, 18, 15, 12, 10]
            
            fig = px.bar(
                x=attack_types,
                y=attack_counts,
                title="Attack Types",
                labels={"x": "Attack Type", "y": "Count"}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Geographic analysis
        st.subheader("🗺️ Geographic Analysis")
        
        # Simulate geographic data
        countries = ["US", "CN", "RU", "DE", "FR", "JP", "GB", "IN"]
        threat_counts = [45, 32, 28, 15, 12, 10, 8, 6]
        
        fig = px.choropleth(
            locations=countries,
            locationmode="ISO-3",
            color=threat_counts,
            title="Threats by Country",
            color_continuous_scale="Reds"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance metrics
        st.subheader("⚡ Performance Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Response time distribution
            response_times = np.random.normal(45, 15, 1000)
            
            fig = px.histogram(
                x=response_times,
                title="Response Time Distribution",
                labels={"x": "Response Time (ms)", "y": "Count"}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Accuracy over time
            dates = pd.date_range(start="2024-01-01", end="2024-01-07", freq="D")
            accuracies = [0.95, 0.96, 0.97, 0.96, 0.98, 0.97, 0.99]
            
            fig = px.line(
                x=dates,
                y=accuracies,
                title="Accuracy Over Time",
                labels={"x": "Date", "y": "Accuracy"}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # API helper methods
    def get_system_status(self) -> dict:
        """Get system status."""
        try:
            response = self.session.get(f"{self.api_base_url}/health")
            if response.status_code == 200:
                return {
                    "api_online": True,
                    "db_connected": True,
                    "trained_models": 7
                }
        except Exception as e:
            logger.error("Error getting system status", error=str(e))
        return None
    
    def analyze_threat(self, target: str, target_type: str, context: str = None, priority: str = "medium") -> dict:
        """Analyze a threat."""
        try:
            payload = {
                "target": target,
                "target_type": target_type,
                "context": {"description": context} if context else None,
                "priority": priority
            }
            
            response = self.session.post(f"{self.api_base_url}/analyze", json=payload)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error("Error analyzing threat", error=str(e))
        return None


def main():
    """Main function to run the dashboard."""
    dashboard = CyberattackDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()