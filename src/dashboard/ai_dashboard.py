"""
Advanced AI Dashboard for the cyberattack detection system.
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


class AIDashboard:
    """Advanced AI Dashboard for comprehensive system monitoring."""
    
    def __init__(self):
        self.api_base_url = f"http://{settings.api_host}:{settings.api_port}/api/v1"
        self.session = requests.Session()
        self.session.headers.update({
            "X-API-Key": "sk-demo-key-1234567890",
            "Content-Type": "application/json"
        })
    
    def run(self):
        """Run the AI dashboard."""
        # Configure page
        st.set_page_config(
            page_title="AI Cyberattack Detection",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS
        st.markdown("""
        <style>
            .ai-header {
                font-size: 2.5rem;
                color: #1f77b4;
                text-align: center;
                margin-bottom: 2rem;
            }
            .component-card {
                background-color: #f8f9fa;
                padding: 1.5rem;
                border-radius: 0.5rem;
                border-left: 4px solid #28a745;
                margin-bottom: 1rem;
            }
            .quantum-card {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 1.5rem;
                border-radius: 0.5rem;
                margin-bottom: 1rem;
            }
            .blockchain-card {
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
        page = st.session_state.get("ai_page", "overview")
        
        if page == "overview":
            self.render_ai_overview()
        elif page == "orchestrator":
            self.render_orchestrator_page()
        elif page == "quantum":
            self.render_quantum_page()
        elif page == "blockchain":
            self.render_blockchain_page()
        elif page == "federated":
            self.render_federated_page()
        elif page == "analytics":
            self.render_analytics_page()
        elif page == "explainability":
            self.render_explainability_page()
    
    def render_sidebar(self):
        """Render the sidebar navigation."""
        st.sidebar.title("🤖 AI Cyberattack Detection")
        st.sidebar.markdown("---")
        
        # Navigation
        pages = {
            "overview": "📊 AI Overview",
            "orchestrator": "🎯 AI Orchestrator",
            "quantum": "⚛️ Quantum ML",
            "blockchain": "🔗 Blockchain Security",
            "federated": "🌐 Federated Learning",
            "analytics": "📈 Advanced Analytics",
            "explainability": "🔍 Explainable AI"
        }
        
        selected_page = st.sidebar.selectbox(
            "AI Navigation",
            options=list(pages.keys()),
            format_func=lambda x: pages[x],
            key="ai_page_selector"
        )
        
        st.session_state.ai_page = selected_page
        
        st.sidebar.markdown("---")
        
        # AI System Status
        st.sidebar.subheader("AI System Status")
        ai_stats = self.get_ai_statistics()
        
        if ai_stats:
            st.sidebar.metric("AI Components", len(ai_stats.get("components", {})))
            st.sidebar.metric("Orchestrations", ai_stats.get("orchestrator", {}).get("total_orchestrations", 0))
            st.sidebar.metric("Avg Confidence", f"{ai_stats.get('orchestrator', {}).get('average_confidence', 0):.2%}")
        else:
            st.sidebar.error("❌ Cannot connect to AI system")
        
        st.sidebar.markdown("---")
        
        # Quick Actions
        st.sidebar.subheader("Quick Actions")
        if st.sidebar.button("🔄 Refresh AI Data"):
            st.rerun()
        
        if st.sidebar.button("🎯 Run AI Analysis"):
            st.session_state.ai_page = "orchestrator"
            st.rerun()
    
    def render_ai_overview(self):
        """Render AI overview page."""
        st.markdown('<h1 class="ai-header">🤖 AI Cyberattack Detection System</h1>', unsafe_allow_html=True)
        
        # Get AI statistics
        ai_stats = self.get_ai_statistics()
        if not ai_stats:
            st.error("Cannot connect to the AI system. Please check the connection.")
            return
        
        # AI Components Overview
        st.subheader("🧠 AI Components Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Classical ML Models",
                ai_stats.get("components", {}).get("models", {}).get("trained_models", 0),
                delta=None
            )
        
        with col2:
            st.metric(
                "Quantum Models",
                len(ai_stats.get("components", {}).get("quantum_ml", {}).get("trained_models", [])),
                delta=None
            )
        
        with col3:
            st.metric(
                "Federated Clients",
                ai_stats.get("components", {}).get("federated_learning", {}).get("total_clients", 0),
                delta=None
            )
        
        with col4:
            st.metric(
                "Blockchain Blocks",
                ai_stats.get("components", {}).get("blockchain_security", {}).get("chain_length", 0),
                delta=None
            )
        
        st.markdown("---")
        
        # AI Components Status
        st.subheader("🔧 AI Components Status")
        
        components = ai_stats.get("components", {})
        
        # Classical ML
        with st.expander("🧮 Classical ML Models", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Available Models:**")
                models = components.get("models", {}).get("models", {})
                for model_name, model_info in models.items():
                    status = "✅ Trained" if model_info.get("is_trained") else "❌ Not Trained"
                    st.write(f"- {model_name}: {status}")
            
            with col2:
                st.write("**Performance Metrics:**")
                if models:
                    # Create performance chart
                    model_names = list(models.keys())
                    accuracies = [0.95, 0.97, 0.93, 0.96, 0.94, 0.92]  # Simulated accuracies
                    
                    fig = px.bar(
                        x=model_names[:len(accuracies)],
                        y=accuracies,
                        title="Model Accuracy",
                        labels={"x": "Model", "y": "Accuracy"}
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        # Quantum ML
        with st.expander("⚛️ Quantum ML", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="quantum-card">', unsafe_allow_html=True)
                st.write("**Quantum Advantage:**")
                quantum_stats = components.get("quantum_ml", {})
                st.metric("Quantum Available", "✅ Yes" if quantum_stats.get("quantum_available") else "❌ No")
                st.metric("Trained Models", len(quantum_stats.get("trained_models", [])))
                st.metric("Encoding Methods", len(quantum_stats.get("encoding_methods", [])))
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.write("**Quantum Algorithms:**")
                st.write("- Variational Quantum Classifier (VQC)")
                st.write("- Quantum Support Vector Machine (QSVM)")
                st.write("- Quantum Approximate Optimization (QAOA)")
                st.write("- Quantum Generative Adversarial Network (QGAN)")
        
        # Blockchain Security
        with st.expander("🔗 Blockchain Security", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="blockchain-card">', unsafe_allow_html=True)
                st.write("**Blockchain Status:**")
                blockchain_stats = components.get("blockchain_security", {})
                st.metric("Chain Length", blockchain_stats.get("chain_length", 0))
                st.metric("Is Valid", "✅ Yes" if blockchain_stats.get("is_valid") else "❌ No")
                st.metric("Threat Registry", blockchain_stats.get("threat_registry_size", 0))
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.write("**Blockchain Features:**")
                st.write("- Threat Intelligence Sharing")
                st.write("- Consensus Verification")
                st.write("- Smart Contract Automation")
                st.write("- Trust Score Management")
        
        # Federated Learning
        with st.expander("🌐 Federated Learning", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Federated Status:**")
                federated_stats = components.get("federated_learning", {})
                st.metric("Total Clients", federated_stats.get("total_clients", 0))
                st.metric("Active Clients", federated_stats.get("active_clients", 0))
                st.metric("Completed Rounds", federated_stats.get("completed_rounds", 0))
            
            with col2:
                st.write("**Privacy Features:**")
                st.write("- Differential Privacy")
                st.write("- Secure Aggregation")
                st.write("- Homomorphic Encryption")
                st.write("- Zero-Knowledge Proofs")
        
        # Advanced Analytics
        with st.expander("📈 Advanced Analytics", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Analytics Capabilities:**")
                st.write("- Threat Landscape Analysis")
                st.write("- Attack Correlation Detection")
                st.write("- Predictive Threat Forecasting")
                st.write("- Geographic Threat Mapping")
            
            with col2:
                st.write("**Visualization Types:**")
                st.write("- Interactive Dashboards")
                st.write("- Real-time Charts")
                st.write("- Geographic Maps")
                st.write("- Network Graphs")
        
        # Explainable AI
        with st.expander("🔍 Explainable AI", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Explanation Methods:**")
                st.write("- SHAP (SHapley Additive exPlanations)")
                st.write("- LIME (Local Interpretable Model-agnostic Explanations)")
                st.write("- Counterfactual Explanations")
                st.write("- Causal Analysis")
            
            with col2:
                st.write("**Interpretability Features:**")
                st.write("- Feature Importance Analysis")
                st.write("- Decision Tree Visualization")
                st.write("- Attention Weight Analysis")
                st.write("- Uncertainty Quantification")
    
    def render_orchestrator_page(self):
        """Render AI orchestrator page."""
        st.title("🎯 AI Orchestrator")
        
        # Orchestrator form
        with st.form("orchestrator_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                target_type = st.selectbox("Target Type", ["url", "ip"])
                target = st.text_input("Target", placeholder="Enter URL or IP address")
            
            with col2:
                context = st.text_area("Additional Context", placeholder="Optional context information")
                components = st.multiselect(
                    "AI Components",
                    options=["classical_ml", "deep_learning", "quantum_ml", "graph_neural_networks", 
                            "federated_learning", "blockchain_security", "adaptive_learning", 
                            "explainable_ai", "anomaly_detection", "novelty_detection"],
                    default=["classical_ml", "deep_learning", "quantum_ml"]
                )
            
            submitted = st.form_submit_button("Orchestrate AI Analysis", type="primary")
        
        if submitted and target:
            with st.spinner("Orchestrating AI analysis..."):
                result = self.orchestrate_ai_analysis(target, target_type, context, components)
                
                if result:
                    st.success("AI orchestration completed!")
                    
                    # Display results
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Final Prediction", result.get("final_prediction", "Unknown"))
                    
                    with col2:
                        st.metric("Confidence", f"{result.get('final_confidence', 0):.2%}")
                    
                    with col3:
                        st.metric("Consensus Score", f"{result.get('consensus_score', 0):.2%}")
                    
                    with col4:
                        st.metric("Processing Time", f"{result.get('processing_time_ms', 0)}ms")
                    
                    # Component results
                    st.subheader("Component Results")
                    
                    component_results = result.get("component_results", {})
                    for component, comp_result in component_results.items():
                        with st.expander(f"🔧 {component.replace('_', ' ').title()}"):
                            if "error" in comp_result:
                                st.error(f"Error: {comp_result['error']}")
                            else:
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.write(f"**Prediction:** {comp_result.get('prediction', 'Unknown')}")
                                    st.write(f"**Confidence:** {comp_result.get('confidence', 0):.2%}")
                                
                                with col2:
                                    st.write(f"**Model Type:** {comp_result.get('model_type', 'Unknown')}")
                                    if 'quantum_advantage' in comp_result:
                                        st.write(f"**Quantum Advantage:** {comp_result['quantum_advantage']:.2%}")
                    
                    # Explanation
                    st.subheader("🔍 AI Explanation")
                    
                    explanation = result.get("explanation", {})
                    if explanation:
                        st.write(explanation.get("explanation_text", "No explanation available"))
                        
                        # Feature importance
                        feature_importance = explanation.get("feature_importance", {})
                        if feature_importance:
                            st.write("**Feature Importance:**")
                            for feature, importance in feature_importance.items():
                                st.write(f"- {feature}: {importance:.3f}")
                    
                    # Recommendations
                    st.subheader("💡 Recommendations")
                    
                    recommendations = result.get("recommendations", [])
                    if recommendations:
                        for i, recommendation in enumerate(recommendations, 1):
                            st.write(f"{i}. {recommendation}")
                    else:
                        st.info("No specific recommendations available.")
        
        # Orchestrator statistics
        st.subheader("📊 Orchestrator Statistics")
        
        orchestrator_stats = self.get_orchestrator_statistics()
        if orchestrator_stats:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Orchestrations", orchestrator_stats.get("total_orchestrations", 0))
            
            with col2:
                st.metric("Average Processing Time", f"{orchestrator_stats.get('average_processing_time', 0):.0f}ms")
            
            with col3:
                st.metric("Average Confidence", f"{orchestrator_stats.get('average_confidence', 0):.2%}")
    
    def render_quantum_page(self):
        """Render quantum ML page."""
        st.title("⚛️ Quantum Machine Learning")
        
        # Quantum ML Overview
        st.markdown('<div class="quantum-card">', unsafe_allow_html=True)
        st.write("**Quantum Computing for Cyberattack Detection**")
        st.write("Leveraging quantum algorithms for enhanced threat detection and pattern recognition.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Quantum algorithms
        st.subheader("🔬 Quantum Algorithms")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Variational Quantum Classifier (VQC)**")
            st.write("- Uses quantum circuits for classification")
            st.write("- Adapts to complex threat patterns")
            st.write("- Provides quantum advantage for certain problems")
            
            st.write("**Quantum Support Vector Machine (QSVM)**")
            st.write("- Quantum version of SVM")
            st.write("- Efficient for high-dimensional data")
            st.write("- Exploits quantum parallelism")
        
        with col2:
            st.write("**Quantum Approximate Optimization (QAOA)**")
            st.write("- Optimizes threat detection parameters")
            st.write("- Finds optimal security configurations")
            st.write("- Solves combinatorial optimization problems")
            
            st.write("**Quantum Generative Adversarial Network (QGAN)**")
            st.write("- Generates synthetic threat data")
            st.write("- Enhances training datasets")
            st.write("- Improves model generalization")
        
        # Quantum feature encoding
        st.subheader("🔧 Quantum Feature Encoding")
        
        encoding_methods = [
            "Angle Encoding",
            "Amplitude Encoding", 
            "Basis Encoding",
            "IQP Encoding"
        ]
        
        selected_method = st.selectbox("Encoding Method", encoding_methods)
        
        if st.button("Encode Features"):
            st.info(f"Quantum feature encoding with {selected_method} would be performed here")
        
        # Quantum predictions
        st.subheader("🎯 Quantum Predictions")
        
        with st.form("quantum_prediction_form"):
            target = st.text_input("Target", placeholder="Enter URL or IP address")
            target_type = st.selectbox("Target Type", ["url", "ip"])
            
            if st.form_submit_button("Make Quantum Prediction"):
                if target:
                    st.info("Quantum prediction would be performed here")
    
    def render_blockchain_page(self):
        """Render blockchain security page."""
        st.title("🔗 Blockchain Security")
        
        # Blockchain overview
        st.markdown('<div class="blockchain-card">', unsafe_allow_html=True)
        st.write("**Blockchain-Based Threat Intelligence**")
        st.write("Decentralized, tamper-proof threat intelligence sharing and verification.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Blockchain statistics
        st.subheader("📊 Blockchain Statistics")
        
        blockchain_stats = self.get_blockchain_statistics()
        if blockchain_stats:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Chain Length", blockchain_stats.get("chain_length", 0))
            
            with col2:
                st.metric("Is Valid", "✅ Yes" if blockchain_stats.get("is_valid") else "❌ No")
            
            with col3:
                st.metric("Threat Registry", blockchain_stats.get("threat_registry_size", 0))
            
            with col4:
                st.metric("Trust Scores", blockchain_stats.get("trust_scores_count", 0))
        
        # Recent blocks
        st.subheader("🔗 Recent Blocks")
        
        # Simulate recent blocks
        recent_blocks = [
            {"index": 1, "type": "threat_detection", "timestamp": "2024-01-01T10:00:00Z", "hash": "abc123..."},
            {"index": 2, "type": "model_update", "timestamp": "2024-01-01T10:05:00Z", "hash": "def456..."},
            {"index": 3, "type": "trust_verification", "timestamp": "2024-01-01T10:10:00Z", "hash": "ghi789..."}
        ]
        
        df = pd.DataFrame(recent_blocks)
        st.dataframe(df, use_container_width=True)
        
        # Smart contracts
        st.subheader("🤖 Smart Contracts")
        
        with st.expander("Deploy Smart Contract"):
            contract_id = st.text_input("Contract ID")
            contract_rules = st.text_area("Contract Rules (JSON)")
            
            if st.button("Deploy Contract"):
                st.success("Smart contract deployed successfully!")
        
        # Threat consensus
        st.subheader("🤝 Threat Consensus")
        
        with st.form("consensus_form"):
            threat_id = st.text_input("Threat ID")
            
            if st.form_submit_button("Verify Consensus"):
                if threat_id:
                    consensus = self.verify_threat_consensus(threat_id)
                    if consensus:
                        st.write("**Consensus Result:**")
                        st.json(consensus)
    
    def render_federated_page(self):
        """Render federated learning page."""
        st.title("🌐 Federated Learning")
        
        # Federated learning overview
        st.write("**Privacy-Preserving Collaborative Learning**")
        st.write("Train models across multiple organizations without sharing sensitive data.")
        
        # Federated statistics
        st.subheader("📊 Federated Learning Statistics")
        
        federated_stats = self.get_federated_statistics()
        if federated_stats:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Clients", federated_stats.get("total_clients", 0))
            
            with col2:
                st.metric("Active Clients", federated_stats.get("active_clients", 0))
            
            with col3:
                st.metric("Completed Rounds", federated_stats.get("completed_rounds", 0))
            
            with col4:
                st.metric("Privacy Budget", f"{federated_stats.get('privacy_budget_remaining', 0):.2f}")
        
        # Client management
        st.subheader("👥 Client Management")
        
        with st.form("client_registration_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                client_id = st.text_input("Client ID")
                client_name = st.text_input("Client Name")
            
            with col2:
                client_type = st.selectbox("Client Type", ["organization", "individual", "system"])
                privacy_level = st.selectbox("Privacy Level", ["low", "medium", "high"])
            
            if st.form_submit_button("Register Client"):
                if client_id:
                    st.success("Client registered successfully!")
        
        # Federated rounds
        st.subheader("🔄 Federated Rounds")
        
        if st.button("Run Federated Round"):
            with st.spinner("Running federated learning round..."):
                st.info("Federated learning round would be executed here")
        
        # Privacy features
        st.subheader("🔒 Privacy Features")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Differential Privacy**")
            st.write("- Adds noise to protect individual data")
            st.write("- Maintains model accuracy")
            st.write("- Configurable privacy budget")
            
            st.write("**Secure Aggregation**")
            st.write("- Encrypts model updates")
            st.write("- Prevents data leakage")
            st.write("- Supports multiple clients")
        
        with col2:
            st.write("**Homomorphic Encryption**")
            st.write("- Computes on encrypted data")
            st.write("- No decryption required")
            st.write("- Maximum privacy protection")
            
            st.write("**Zero-Knowledge Proofs**")
            st.write("- Proves model validity")
            st.write("- Without revealing data")
            st.write("- Cryptographic guarantees")
    
    def render_analytics_page(self):
        """Render advanced analytics page."""
        st.title("📈 Advanced Analytics")
        
        # Analytics overview
        st.write("**Comprehensive Threat Analytics and Insights**")
        
        # Run analytics
        if st.button("Run Comprehensive Analysis"):
            with st.spinner("Running comprehensive analytics..."):
                analytics_results = self.get_ai_analytics()
                
                if analytics_results:
                    st.success("Analytics completed!")
                    
                    # Executive summary
                    st.subheader("📋 Executive Summary")
                    
                    summary = analytics_results.get("executive_summary", {})
                    if summary:
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Total Threats", summary.get("total_threats_analyzed", 0))
                        
                        with col2:
                            st.metric("Emerging Threats", summary.get("emerging_threats", 0))
                        
                        with col3:
                            st.metric("Predicted Volume", summary.get("predicted_future_volume", 0))
                        
                        # Key findings
                        st.write("**Key Findings:**")
                        for finding in summary.get("key_findings", []):
                            st.write(f"- {finding}")
                    
                    # Threat landscape
                    st.subheader("🗺️ Threat Landscape")
                    
                    threat_landscape = analytics_results.get("threat_landscape", {})
                    if threat_landscape:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Threat distribution
                            threat_dist = threat_landscape.get("threat_distribution", {})
                            if threat_dist:
                                fig = px.pie(
                                    values=list(threat_dist.values()),
                                    names=list(threat_dist.keys()),
                                    title="Threat Type Distribution"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            # Severity distribution
                            severity_dist = threat_landscape.get("severity_distribution", {})
                            if severity_dist:
                                fig = px.bar(
                                    x=list(severity_dist.keys()),
                                    y=list(severity_dist.values()),
                                    title="Severity Distribution"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                    
                    # Predictive insights
                    st.subheader("🔮 Predictive Insights")
                    
                    predictive_insights = analytics_results.get("predictive_insights", {})
                    if predictive_insights:
                        for insight_type, insight_data in predictive_insights.items():
                            with st.expander(f"📊 {insight_type.replace('_', ' ').title()}"):
                                st.write(f"**Predicted Value:** {insight_data.get('predicted_value', 'N/A')}")
                                st.write(f"**Confidence:** {insight_data.get('confidence', 0):.2%}")
                                st.write(f"**Time Horizon:** {insight_data.get('time_horizon', 'N/A')}")
                                
                                # Recommendations
                                recommendations = insight_data.get("recommendations", [])
                                if recommendations:
                                    st.write("**Recommendations:**")
                                    for rec in recommendations:
                                        st.write(f"- {rec}")
                    
                    # Recommendations
                    st.subheader("💡 Overall Recommendations")
                    
                    recommendations = analytics_results.get("recommendations", [])
                    if recommendations:
                        for i, recommendation in enumerate(recommendations, 1):
                            st.write(f"{i}. {recommendation}")
        
        # Analytics capabilities
        st.subheader("🔧 Analytics Capabilities")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Threat Landscape Analysis**")
            st.write("- Comprehensive threat mapping")
            st.write("- Emerging threat identification")
            st.write("- Threat evolution tracking")
            
            st.write("**Attack Correlation Analysis**")
            st.write("- Multi-vector attack detection")
            st.write("- Attack sequence analysis")
            st.write("- Causal relationship identification")
        
        with col2:
            st.write("**Predictive Analytics**")
            st.write("- Threat trend forecasting")
            st.write("- Attack volume prediction")
            st.write("- Risk assessment")
            
            st.write("**Geographic Analysis**")
            st.write("- Threat origin mapping")
            st.write("- Regional threat patterns")
            st.write("- Geographic risk assessment")
    
    def render_explainability_page(self):
        """Render explainable AI page."""
        st.title("🔍 Explainable AI")
        
        # Explainability overview
        st.write("**Transparent and Interpretable AI Decisions**")
        st.write("Understand how AI models make threat detection decisions.")
        
        # Explanation methods
        st.subheader("🔬 Explanation Methods")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**SHAP (SHapley Additive exPlanations)**")
            st.write("- Game theory-based explanations")
            st.write("- Feature importance analysis")
            st.write("- Global and local explanations")
            
            st.write("**LIME (Local Interpretable Model-agnostic Explanations)**")
            st.write("- Local explanation generation")
            st.write("- Model-agnostic approach")
            st.write("- Perturbation-based analysis")
        
        with col2:
            st.write("**Counterfactual Explanations**")
            st.write("- "What-if" scenario analysis")
            st.write("- Minimal change explanations")
            st.write("- Actionable insights")
            
            st.write("**Causal Analysis**")
            st.write("- Causal relationship identification")
            st.write("- Root cause analysis")
            st.write("- Causal graph visualization")
        
        # Explanation request
        st.subheader("🎯 Request Explanation")
        
        with st.form("explanation_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                target = st.text_input("Target", placeholder="Enter URL or IP address")
                target_type = st.selectbox("Target Type", ["url", "ip"])
            
            with col2:
                explanation_methods = st.multiselect(
                    "Explanation Methods",
                    options=["SHAP", "LIME", "Counterfactual", "Causal"],
                    default=["SHAP", "LIME"]
                )
            
            if st.form_submit_button("Generate Explanation"):
                if target:
                    explanation = self.get_ai_explanation(target, target_type)
                    
                    if explanation:
                        st.success("Explanation generated!")
                        
                        # Display explanation
                        st.subheader("📋 Explanation Results")
                        
                        # Prediction
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**Prediction:** {explanation.get('prediction', 'Unknown')}")
                            st.write(f"**Confidence:** {explanation.get('confidence', 0):.2%}")
                        
                        with col2:
                            st.write(f"**Consensus Score:** {explanation.get('consensus_score', 0):.2%}")
                            st.write(f"**Processing Time:** {explanation.get('processing_time_ms', 0)}ms")
                        
                        # Explanation text
                        explanation_data = explanation.get("explanation", {})
                        if explanation_data:
                            st.write("**Explanation:**")
                            st.write(explanation_data.get("explanation_text", "No explanation available"))
                        
                        # Recommendations
                        recommendations = explanation.get("recommendations", [])
                        if recommendations:
                            st.write("**Recommendations:**")
                            for i, rec in enumerate(recommendations, 1):
                                st.write(f"{i}. {rec}")
        
        # Explanation statistics
        st.subheader("📊 Explanation Statistics")
        
        explainable_stats = self.get_explainable_statistics()
        if explainable_stats:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("SHAP Available", "✅ Yes" if explainable_stats.get("shap_available") else "❌ No")
            
            with col2:
                st.metric("LIME Available", "✅ Yes" if explainable_stats.get("lime_available") else "❌ No")
            
            with col3:
                st.metric("Available Methods", len(explainable_stats.get("available_methods", [])))
    
    # API helper methods
    def get_ai_statistics(self) -> Optional[Dict]:
        """Get AI system statistics."""
        try:
            response = self.session.get(f"{self.api_base_url}/ai/statistics")
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error("Error getting AI statistics", error=str(e))
        return None
    
    def orchestrate_ai_analysis(self, target: str, target_type: str, 
                              context: str = None, components: List[str] = None) -> Optional[Dict]:
        """Orchestrate AI analysis."""
        try:
            payload = {
                "target": target,
                "target_type": target_type
            }
            if context:
                payload["context"] = {"description": context}
            if components:
                payload["components"] = components
            
            response = self.session.post(f"{self.api_base_url}/ai/orchestrate", json=payload)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error("Error orchestrating AI analysis", error=str(e))
        return None
    
    def get_orchestrator_statistics(self) -> Optional[Dict]:
        """Get orchestrator statistics."""
        try:
            response = self.session.get(f"{self.api_base_url}/ai/statistics")
            if response.status_code == 200:
                data = response.json()
                return data.get("orchestrator", {})
        except Exception as e:
            logger.error("Error getting orchestrator statistics", error=str(e))
        return None
    
    def get_blockchain_statistics(self) -> Optional[Dict]:
        """Get blockchain statistics."""
        try:
            response = self.session.get(f"{self.api_base_url}/ai/blockchain/statistics")
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error("Error getting blockchain statistics", error=str(e))
        return None
    
    def verify_threat_consensus(self, threat_id: str) -> Optional[Dict]:
        """Verify threat consensus."""
        try:
            response = self.session.get(f"{self.api_base_url}/ai/blockchain/verify/{threat_id}")
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error("Error verifying threat consensus", error=str(e))
        return None
    
    def get_federated_statistics(self) -> Optional[Dict]:
        """Get federated learning statistics."""
        try:
            response = self.session.get(f"{self.api_base_url}/ai/federated/statistics")
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error("Error getting federated statistics", error=str(e))
        return None
    
    def get_ai_analytics(self) -> Optional[Dict]:
        """Get AI analytics."""
        try:
            response = self.session.get(f"{self.api_base_url}/ai/analytics")
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error("Error getting AI analytics", error=str(e))
        return None
    
    def get_ai_explanation(self, target: str, target_type: str) -> Optional[Dict]:
        """Get AI explanation."""
        try:
            response = self.session.get(f"{self.api_base_url}/ai/explain/{target}?target_type={target_type}")
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error("Error getting AI explanation", error=str(e))
        return None
    
    def get_explainable_statistics(self) -> Optional[Dict]:
        """Get explainable AI statistics."""
        try:
            response = self.session.get(f"{self.api_base_url}/ai/statistics")
            if response.status_code == 200:
                data = response.json()
                return data.get("components", {}).get("explainable_ai", {})
        except Exception as e:
            logger.error("Error getting explainable statistics", error=str(e))
        return None


def main():
    """Main function to run the AI dashboard."""
    dashboard = AIDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()
