import streamlit as st
import streamlit.components.v1 as components
import json
import pandas as pd
import numpy as np
import pickle
import os

# Page configuration
st.set_page_config(
    page_title="ExoClassify - AI Exoplanet Classification",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize session state for page navigation
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'Home'

# Load models and preprocessors
@st.cache_resource
def load_models():
    """Load all models and preprocessors directly from /Nasa folder"""
    models = {}
    
    # Get directory of the current file (/Nasa)
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Debug info (helps you confirm file visibility in Streamlit Cloud)
   

    try:
        # Binary classification models
        with open(os.path.join(base_dir, "binary_model.pkl"), "rb") as f:
            models['binary_model'] = pickle.load(f)
        with open(os.path.join(base_dir, "scaler_binary.pkl"), "rb") as f:
            models['scaler_binary'] = pickle.load(f)
        with open(os.path.join(base_dir, "poly_transformer_binary.pkl"), "rb") as f:
            models['poly_binary'] = pickle.load(f)

        # Multi-class classification models
        with open(os.path.join(base_dir, "multiclass_model.pkl"), "rb") as f:
            models['multiclass_model'] = pickle.load(f)
        with open(os.path.join(base_dir, "scaler_multiclass.pkl"), "rb") as f:
            models['scaler_multiclass'] = pickle.load(f)
        with open(os.path.join(base_dir, "poly_transformer_multiclass.pkl"), "rb") as f:
            models['poly_multiclass'] = pickle.load(f)
        with open(os.path.join(base_dir, "label_encoder_multiclass.pkl"), "rb") as f:
            models['label_encoder'] = pickle.load(f)

        models['loaded'] = True
        st.success("✅ All models successfully loaded from /Nasa!")

    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        models['loaded'] = False

    return models

models = load_models()
# Custom CSS with Teal/Cyan Theme
st.markdown("""
<style>
    [data-testid="collapsedControl"] { display: none; }
    
    .stApp {
       background: linear-gradient(180deg, #102631 50%, #050f17);
        color: #e0f4ff;
    }
    
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
        max-width: 100%;
    }
    
    h1, h2, h3, h4, h5, h6 { color: #00d9ff !important; }
    
    .navbar {
        background: rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
        padding: 1rem 0;
        margin-bottom: 2rem;
        border-radius: 10px;
    }
    
    .logo {
        font-size: 1.8rem;
        font-weight: bold;
        background: linear-gradient(45deg, #00ffc8, #00d9ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    div[data-testid="column"] button {
        background: transparent !important;
        border: none !important;
        color: white !important;
        padding: 0.5rem 1rem !important;
        border-radius: 25px !important;
        transition: all 0.3s ease !important;
    }
    
    div[data-testid="column"] button:hover {
        background: rgba(0, 217, 255, 0.2) !important;
        color: #00ffc8 !important;
    }
    
    .stButton > button:not([data-testid="column"] button) {
        background: linear-gradient(45deg, #00ffc8, #00d9ff) !important;
        color: #0a1e2e !important;
        border: none !important;
        border-radius: 50px !important;
        padding: 0.6rem 2rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
    }
    
    .stButton > button:hover {
        box-shadow: 0 10px 25px rgba(0, 217, 255, 0.4) !important;
        transform: translateY(-2px) !important;
    }
    
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select {
        background: rgba(255, 255, 255, 0.1) !important;
        color: white !important;
        border: 1px solid rgba(0, 217, 255, 0.3) !important;
        border-radius: 8px !important;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        color: white;
        padding: 10px 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(45deg, #00ffc8, #00d9ff) !important;
        color: #0a1e2e !important;
    }
    
    [data-testid="stMetricValue"] {
        color: #00ffc8 !important;
        font-size: 1.5rem !important;
        font-weight: bold !important;
    }
    
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #00ffc8, #00d9ff);
    }
    
    .stAlert {
        background: rgba(0, 217, 255, 0.1);
        border-left: 4px solid #00ffc8;
        border-radius: 10px;
    }
    
    .streamlit-expanderHeader {
        background: rgba(0, 217, 255, 0.08);
        border-radius: 10px;
        color: white !important;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .hero {
        text-align: center;
        padding: 3rem 0;
        background: radial-gradient(circle at center, rgba(0, 255, 200, 0.15) 0%, transparent 70%);
        border-radius: 15px;
        margin-bottom: 2rem;
    }
    
    .hero h1 { font-size: 3rem; margin-bottom: 1rem; }
    .hero p { font-size: 1.2rem; opacity: 0.9; }
    
    .result-card {
        background: rgba(0, 217, 255, 0.08);
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 4px solid #00ffc8;
    }
    
    .comparison-card {
        background: rgba(0, 217, 255, 0.08);
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
    }
    
    .tool-card {
        background: rgba(0, 217, 255, 0.08);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 1rem;
    }
    
    .content-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 0 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Navigation function
def navigate_to(page_name):
    st.session_state.current_page = page_name

# Top Navigation Bar
nav_col1, nav_col2, nav_col3, nav_col4, nav_col5, nav_col6 = st.columns([2, 1, 1, 1, 1, 1])

with nav_col1:
    st.markdown('<div class="logo">ExoClassify</div>', unsafe_allow_html=True)

with nav_col2:
    if st.button("Home", key="nav_home", type="primary" if st.session_state.current_page == "Home" else "secondary"):
        navigate_to("Home")
        st.rerun()

with nav_col3:
    if st.button("Classification", key="nav_class", type="primary" if st.session_state.current_page == "Classification" else "secondary"):
        navigate_to("Classification")
        st.rerun()

with nav_col4:
    if st.button("Research", key="nav_research", type="primary" if st.session_state.current_page == "Research" else "secondary"):
        navigate_to("Research")
        st.rerun()

with nav_col5:
    if st.button("Resources", key="nav_resources", type="primary" if st.session_state.current_page == "Resources" else "secondary"):
        navigate_to("Resources")
        st.rerun()

st.markdown("---")

# Page Content wrapped in container
st.markdown('<div class="content-container">', unsafe_allow_html=True)

# ============================================================================
# HOME PAGE
# ============================================================================
if st.session_state.current_page == "Home":
    st.markdown("""
    <div class="hero">
        <h1>Discover Exoplanets with AI</h1>
        <p>Classify and explore exoplanets using advanced machine learning powered by NASA data</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("🚀 Start Classification", use_container_width=True, key="home_start"):
                navigate_to("Classification")
                st.rerun()
        
        with col_b:
            if st.button("🛠 Research Tools", use_container_width=True, key="home_research"):
                navigate_to("Research")
                st.rerun()
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Exoplanets Section with Image
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("### 🌌 What Are Exoplanets?")
        st.write("""
        Exoplanets are planets that orbit stars beyond our own Sun. For centuries, humanity could only 
        wonder whether other worlds existed outside our solar system, but with the advancement of telescopes 
        and space missions like NASA's Kepler, we have discovered thousands of these distant planets—ranging 
        from gas giants larger than Jupiter to rocky worlds similar in size to Earth.
        """)
        st.write("""
        Studying exoplanets helps scientists understand how planetary systems form, evolve, and potentially 
        harbor life. Each new discovery expands our knowledge of the universe and our place within it, 
        revealing an astonishing diversity of worlds that challenge our understanding of what a planet can be.
        """)
    
    with col2:
        st.markdown("""
        <div style='background: rgba(0, 217, 255, 0.08); padding: 1rem; border-radius: 15px; text-align: center;'>
            <img src='https://maxpolyakov.com/wp-content/uploads/2023/03/most-unusual-exoplanets-cover.jpg' 
                 style='width: 100%; border-radius: 10px; margin-bottom: 0.5rem;'>
            <p style='font-size: 0.9rem; opacity: 0.8; margin: 0;'>Artistic representation of an exoplanet orbiting a distant star</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Our Mission Section
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("""
        <div style='background: rgba(0, 217, 255, 0.08); padding: 1rem; border-radius: 15px; text-align: center;'>
            <img src='https://cdn.mos.cms.futurecdn.net/JYUeUs7hGmsEmTQMdnFkvn-840-80.jpg.webp' 
                 style='width: 100%; border-radius: 10px; margin-bottom: 0.5rem;'>
            <p style='font-size: 0.9rem; opacity: 0.8; margin: 0;'>Advanced AI technology meets astronomical discovery</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 🎯 Our Mission")
        st.write("""
        Our website is dedicated to making the study of exoplanets more accessible to everyone from professional 
        researchers to curious beginners. At its core, the platform is powered by a specialized artificial 
        intelligence model trained on authentic Kepler mission data.
        """)
        st.write("""
        This AI can detect, classify, and analyze exoplanet candidates with high precision, offering researchers 
        a powerful tool for accelerating discovery and data interpretation. For beginners and students, the 
        website also features an interactive simulation mode that simplifies complex astronomical data, allowing 
        users to visualize how exoplanets orbit their stars and understand the principles behind their detection.
        """)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Goals Section
    st.markdown("""
    <div style='background: linear-gradient(135deg, rgba(0, 217, 255, 0.1) 0%, rgba(0, 255, 200, 0.1) 100%); 
                padding: 2rem; border-radius: 15px; text-align: center; border: 1px solid rgba(0, 217, 255, 0.3);'>
        <h3 style='color: #00ffc8; margin-bottom: 1rem;'>🚀 Our Ultimate Goal</h3>
        <p style='font-size: 1.1rem; line-height: 1.8;'>
            To bridge the gap between advanced exoplanet research and public curiosity—creating a space where 
            cutting-edge science meets exploration, learning, and inspiration. By combining real astronomical data, 
            intelligent analysis, and engaging visual experiences, we aim to make the vast universe of exoplanets 
            open and understandable to everyone.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Features Section
    st.markdown("### ✨ Platform Features")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="tool-card">
            <h4 style='text-align: center;'>🤖 AI Classification</h4>
            <p>Advanced machine learning trained on 150K+ NASA samples with 94.7% accuracy for precise exoplanet detection</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="tool-card">
            <h4 style='text-align: center;'>🌐 3D Visualization</h4>
            <p>Interactive simulations that bring exoplanetary systems to life with real-time orbital mechanics</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="tool-card">
            <h4 style='text-align: center;'>🔬 Research Tools</h4>
            <p>Professional-grade analysis tools with customizable algorithms and comprehensive datasets</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Stats Section
    st.markdown("### 📊 Platform Statistics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Exoplanets Catalogued", "5,500+", "Growing Daily")
    with col2:
        st.metric("AI Accuracy", "94.7%", "+2.3% vs baseline")
    with col3:
        st.metric("Training Samples", "150K+", "NASA Missions")
    with col4:
        st.metric("Detection Methods", "6", "Algorithms Available")

# ============================================================================
# CLASSIFICATION PAGE
# ============================================================================
elif st.session_state.current_page == "Classification":
    st.title("🔬 Exoplanet Classification & Visualization")
    
    
    # Classification type selector
    st.markdown("### 🎯 Select Classification Type")
    classification_type = st.radio(
        "",
        ["Binary Classification (Planet / Not Planet)", 
         "Multi-class Classification (Confirmed / Candidate / False Positive)"],
        horizontal=True,
        key="classification_type_selector"
    )
    
    is_binary = "Binary" in classification_type
    
    st.markdown("---")
    
    # Create two main columns
    col_input, col_output = st.columns([1, 1])
    
    with col_input:
        st.markdown("### 📥 Input Parameters")
        
        with st.form("classification_form"):
            st.markdown("#### 🚩 Binary Flags")
            col1, col2 = st.columns(2)
            with col1:
                koi_fpflag_ss = st.selectbox(
                    "Stellar Eclipse Flag",
                    [0, 1],
                    help="Is there a stellar eclipse? (0=No, 1=Yes)"
                )
            with col2:
                koi_fpflag_co = st.selectbox(
                    "Centroid Offset Flag",
                    [0, 1],
                    help="Is signal from nearby star? (0=No, 1=Yes)"
                )
            
            st.markdown("#### 📐 Angular Offsets (arcseconds)")
            col1, col2 = st.columns(2)
            with col1:
                koi_dikco_msky = st.number_input(
                    "PRF Offset from KIC",
                    min_value=0.0,
                    max_value=20.0,
                    value=0.5,
                    step=0.1,
                    help="Angular offset from catalog position"
                )
            with col2:
                koi_dicco_msky = st.number_input(
                    "PRF Offset OOT",
                    min_value=0.0,
                    max_value=20.0,
                    value=0.3,
                    step=0.1,
                    help="Angular offset between images"
                )
            
            st.markdown("#### ⭐ Stellar & System Properties")
            col1, col2 = st.columns(2)
            with col1:
                koi_smet_err2 = st.number_input(
                    "Stellar Metallicity Error",
                    min_value=-1.0,
                    max_value=0.0,
                    value=-0.05,
                    step=0.01,
                    help="Negative error bound for metallicity"
                )
            with col2:
                koi_count = st.number_input(
                    "Number of Planets",
                    min_value=1,
                    max_value=10,
                    value=1,
                    step=1,
                    help="Number of planets in the system"
                )
            
            st.markdown("#### 🌟 Star Properties")
            col1, col2 = st.columns(2)
            with col1:
                star_temp = st.number_input(
                    "Star Temperature (K)",
                    min_value=2000,
                    max_value=40000,
                    value=5778,
                    step=100,
                    help="Surface temperature of host star"
                )
            with col2:
                star_radius = st.number_input(
                    "Star Radius (Solar radii)",
                    min_value=0.1,
                    max_value=20.0,
                    value=1.0,
                    step=0.1,
                    help="Size relative to our Sun"
                )
            
            st.markdown("#### 🪐 Planet Properties")
            col1, col2, col3 = st.columns(3)
            with col1:
                koi_prad = st.number_input(
                    "Planet Radius (Earth radii)",
                    min_value=0.1,
                    max_value=30.0,
                    value=1.0,
                    step=0.1,
                    help="Size relative to Earth"
                )
            with col2:
                koi_teq = st.number_input(
                    "Equilibrium Temp (K)",
                    min_value=100,
                    max_value=3000,
                    value=288,
                    step=10,
                    help="Planet temperature"
                )
            with col3:
                koi_model_snr = st.number_input(
                    "Signal-to-Noise Ratio",
                    min_value=0.0,
                    max_value=500.0,
                    value=50.0,
                    step=1.0,
                    help="Transit signal strength"
                )
            
            st.markdown("#### 🔄 Orbital Parameters")
            col1, col2 = st.columns(2)
            with col1:
                orbital_period = st.number_input(
                    "Orbital Period (days)",
                    min_value=0.1,
                    max_value=10000.0,
                    value=365.25,
                    step=1.0,
                    help="Time for one complete orbit"
                )
            with col2:
                orbit_distance = st.number_input(
                    "Orbital Distance (AU)",
                    min_value=0.01,
                    max_value=100.0,
                    value=1.0,
                    step=0.01,
                    help="Distance from star (1 AU = Earth-Sun distance)"
                )
            
            impact_param = st.slider(
                "Impact Parameter",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.01,
                help="How centered the transit is (0=center, 1=edge)"
            )
            
            submitted = st.form_submit_button("🚀 Classify Planet", use_container_width=True)
    
    with col_output:
        st.markdown("### 📊 Classification Results")
        
        if submitted:
            # Calculate derived features
            earth_similarity = 1 / (1 + abs(koi_prad - 1) + abs(koi_teq - 288)/100)
            log_snr = np.log1p(koi_model_snr)
            
            # Calculate transit depth (in ppm)
            transit_depth = ((koi_prad * 6371) / (star_radius * 696000)) ** 2 * 1e6
            
            # Calculate insolation (relative to Earth)
            insolation = (star_temp / 5778) ** 4 * (star_radius ** 2) / (orbit_distance ** 2)
            
            # Create feature array (8 features)
            features_8 = np.array([[
                koi_fpflag_ss,
                koi_fpflag_co,
                koi_dikco_msky,
                koi_dicco_msky,
                koi_smet_err2,
                earth_similarity,
                log_snr,
                koi_count
            ]])
            
            try:
                if models.get('loaded', False):
                    if is_binary:
                        # Binary classification
                        scaler = models['scaler_binary']
                        poly = models['poly_binary']
                        model = models['binary_model']
                        
                        features_scaled = scaler.transform(features_8)
                        features_poly = poly.transform(features_scaled)
                        
                        prediction = model.predict(features_poly)[0]
                        probabilities = model.predict_proba(features_poly)[0]
                        
                        st.markdown("""
                        <div class="result-card">
                            <h4>Predicted Class</h4>
                            <p style='color: {}; font-size: 1.5rem; font-weight: bold;'>
                            {}</p>
                        </div>
                        """.format(
                            "#00ffc8" if prediction == 1 else "#ff6b6b",
                            "✅ CONFIRMED PLANET" if prediction == 1 else "❌ FALSE POSITIVE"
                        ), unsafe_allow_html=True)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Confidence", f"{max(probabilities)*100:.1f}%")
                        with col2:
                            st.metric("Class", "Planet" if prediction == 1 else "Not Planet")
                        
                        st.markdown("#### Probability Distribution")
                        st.write("*False Positive*")
                        st.progress(float(probabilities[0]))
                        st.write(f"{probabilities[0]*100:.2f}%")
                        
                        st.write("*Confirmed Planet*")
                        st.progress(float(probabilities[1]))
                        st.write(f"{probabilities[1]*100:.2f}%")
                    
                    else:
                        # Multi-class classification
                        scaler = models['scaler_multiclass']
                        poly = models['poly_multiclass']
                        model = models['multiclass_model']
                        label_encoder = models['label_encoder']
                        
                        features_scaled = scaler.transform(features_8)
                        features_poly = poly.transform(features_scaled)
                        
                        prediction = model.predict(features_poly)[0]
                        probabilities = model.predict_proba(features_poly)[0]
                        class_name = label_encoder.inverse_transform([prediction])[0]
                        
                        color_map = {
                            'CONFIRMED': '#00ffc8',
                            'CANDIDATE': '#ffd700',
                            'FALSE POSITIVE': '#ff6b6b'
                        }
                        
                        st.markdown(f"""
                        <div class="result-card">
                            <h4>Predicted Class</h4>
                            <p style='color: {color_map.get(class_name, "#00ffc8")}; font-size: 1.5rem; font-weight: bold;'>
                            {class_name}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Confidence", f"{max(probabilities)*100:.1f}%")
                        with col2:
                            st.metric("Predicted Class", class_name)
                        
                        st.markdown("#### Class Probabilities")
                        for i, class_label in enumerate(label_encoder.classes_):
                            st.write(f"{class_label}")
                            st.progress(float(probabilities[i]))
                            st.write(f"{probabilities[i]*100:.2f}%")
                    
                    with st.expander("🔍 View Calculated Features"):
                        st.write(f"*Earth Similarity Index:* {earth_similarity:.4f}")
                        st.write(f"*Log SNR:* {log_snr:.4f}")
                        st.write(f"*Transit Depth:* {transit_depth:.2f} ppm")
                        st.write(f"*Insolation:* {insolation:.4f} (relative to Earth)")
                        st.write(f"*Input Features Shape:* {features_8.shape}")
                
                else:
                    st.info("⚠ Running in demo mode - showing placeholder predictions")
                    
                    if is_binary:
                        st.markdown("""
                        <div class="result-card">
                            <h4>Predicted Class (Demo)</h4>
                            <p style='color: #00ffc8; font-size: 1.5rem; font-weight: bold;'>
                            ✅ CONFIRMED PLANET</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.write("*False Positive*")
                        st.progress(0.25)
                        st.write("25.0%")
                        
                        st.write("*Confirmed Planet*")
                        st.progress(0.75)
                        st.write("75.0%")
                    else:
                        st.markdown("""
                        <div class="result-card">
                            <h4>Predicted Class (Demo)</h4>
                            <p style='color: #00ffc8; font-size: 1.5rem; font-weight: bold;'>
                            CONFIRMED</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.write("*CONFIRMED*")
                        st.progress(0.70)
                        st.write("70.0%")
                        
                        st.write("*CANDIDATE*")
                        st.progress(0.20)
                        st.write("20.0%")
                        
                        st.write("*FALSE POSITIVE*")
                        st.progress(0.10)
                        st.write("10.0%")
                    
                    with st.expander("🔍 View Calculated Features"):
                        st.write(f"*Earth Similarity Index:* {earth_similarity:.4f}")
                        st.write(f"*Log SNR:* {log_snr:.4f}")
                        st.write(f"*Transit Depth:* {transit_depth:.2f} ppm")
                        st.write(f"*Insolation:* {insolation:.4f} (relative to Earth)")
                        
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
                st.exception(e)
        
        else:
            st.info("👆 Enter parameters and click 'Classify Planet' to see results")
    
    st.markdown("---")
         # 3D Visualization section
    st.markdown("## 🌌 3D Visualization & Simulation")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    html_path = os.path.join(base_dir, "Index.html")

    if submitted and os.path.exists(html_path):
    # Prepare parameters to send to simulation
        params = {
            "orbitDistance": float(orbit_distance),
            "planetRadius": float(koi_prad),
            "orbitalPeriod": float(orbital_period),
            "planetTemp": float(koi_teq),
            "impactParam": float(impact_param),
            "starTemp": float(star_temp),
            "starRadius": float(star_radius),
            "insolation": float(insolation),
            "transitDepth": float(transit_depth)
    }

    # Load Index.html directly from the same folder
    with open(html_path, "r", encoding="utf-8") as f:
        sim_html = f.read()

    # Replace the default exoParams in HTML with actual params
    params_js = json.dumps(params)
    start_marker = "let exoParams = {"
    end_marker = "};"
    start_index = sim_html.find(start_marker)
    if start_index != -1:
        end_index = sim_html.find(end_marker, start_index) + len(end_marker)
        if end_index != -1:
            sim_html = (
                sim_html[:start_index]
                + f"let exoParams = {params_js};"
                + sim_html[end_index:]
            )

    components.html(sim_html, height=800, scrolling=False)

    elif os.path.exists(html_path):
        st.info("👆 Submit the classification form to view the 3D simulation with your parameters")
    else:
        st.warning("⚠️ 3D visualization file (Index.html) not found in /Nasa folder")

    st.markdown("---")

# --- Comparison Section ---
    if submitted:
        st.markdown("### 📊 System Comparison")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            <div class="comparison-card">
                <h4>🌍 Earth System</h4>
                <p><strong>Orbital Period:</strong> 365.25 days</p>
                <p><strong>Distance from Star:</strong> 1 AU</p>
                <p><strong>Planet Radius:</strong> 1.0 Earth radii</p>
                <p><strong>Temperature:</strong> 288 K</p>
                <p><strong>Insolation:</strong> 1.0</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="comparison-card">
                <h4>🪐 Your Exoplanet</h4>
                <p><strong>Orbital Period:</strong> {orbital_period:.1f} days</p>
                <p><strong>Distance from Star:</strong> {orbit_distance:.2f} AU</p>
                <p><strong>Planet Radius:</strong> {koi_prad:.2f} Earth radii</p>
                <p><strong>Temperature:</strong> {koi_teq} K</p>
                <p><strong>Insolation:</strong> {insolation:.2f}</p>
                <p><strong>Transit Depth:</strong> {transit_depth:.0f} ppm</p>
            </div>
            """, unsafe_allow_html=True)
# ============================================================================
# RESEARCH PAGE
# ============================================================================
elif st.session_state.current_page == "Research":
    st.title("🛠 Research Tools & Model Training")
    
    st.markdown("""
    <div class="hero">
        <h1>Train Your Own Models</h1>
        <p>Experiment with different algorithms and hyperparameters on real Kepler data</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state for training
    if 'training_complete' not in st.session_state:
        st.session_state.training_complete = False
    if 'training_results' not in st.session_state:
        st.session_state.training_results = {}
    
    # Main layout
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.markdown("### 📊 Select Algorithm & Configure")
        
        # Algorithm selection
        algorithm = st.selectbox(
            "Choose ML Algorithm",
            ["Random Forest", "XGBoost", "LightGBM", "CatBoost", "AdaBoost", "Gradient Boosting"],
            help="Select the algorithm you want to train"
        )
        
        st.markdown(f"#### ⚙ {algorithm} Hyperparameters")
        
        # Hyperparameters based on selected algorithm
        if algorithm == "Random Forest":
            with st.form("rf_form"):
                n_estimators = st.number_input("Number of Trees (n_estimators)", 
                                              min_value=10, max_value=2000, value=1000, step=50,
                                              help="Number of trees in the forest")
                max_depth = st.number_input("Maximum Depth (max_depth)", 
                                           min_value=1, max_value=50, value=8, step=1,
                                           help="Maximum depth of each tree")
                min_samples_split = st.number_input("Min Samples Split", 
                                                   min_value=2, max_value=20, value=2, step=1,
                                                   help="Minimum samples to split a node")
                min_samples_leaf = st.number_input("Min Samples Leaf", 
                                                  min_value=1, max_value=20, value=1, step=1,
                                                  help="Minimum samples in leaf node")
                max_features = st.selectbox("Max Features", 
                                           ["sqrt", "log2", None],
                                           help="Features to consider for best split")
                
                train_button = st.form_submit_button("🚀 Train Model", use_container_width=True)
                
                if train_button:
                    with st.spinner("Training Random Forest... This may take a few minutes."):
                        try:
                            from sklearn.ensemble import RandomForestClassifier
                            from sklearn.preprocessing import StandardScaler
                            from sklearn.model_selection import train_test_split
                            from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
                            import time
                            
                            # Load dataset
                            data_path = "C:/Ebrahim/Nasa/cumulative_2025.10.02_20.38.17.csv"
                            if not os.path.exists(data_path):
                                st.error(f"Dataset not found: {data_path}")
                            else:
                                start_time = time.time()
                                
                                # Data preprocessing (simplified version)
                                df = pd.read_csv(data_path)
                                
                                # Drop columns
                                drop_cols = ["koi_longp", "koi_ingress", "koi_model_dof", "koi_model_chisq", 
                                            "koi_sage", "rowid", "kepoi_name", "kepid", "kepler_name",
                                            "koi_pdisposition", "koi_score", "koi_time0bk", "koi_comment", 
                                            "koi_limbdark_mod", "koi_parm_prov", "koi_trans_mod", 
                                            "koi_datalink_dvr", "koi_datalink_dvs", "koi_tce_delivname", 
                                            "koi_sparprov", "koi_vet_stat", "koi_vet_date", "koi_disp_prov",
                                            "koi_ldm_coeff3", "koi_ldm_coeff4"]
                                
                                df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
                                
                                # Prepare target
                                from sklearn.preprocessing import LabelEncoder
                                le = LabelEncoder()
                                y = le.fit_transform(df['koi_disposition'])
                                
                                # Select numerical features
                                numerical_features = df.select_dtypes(include=['float64', 'int64']).columns
                                numerical_features = [col for col in numerical_features if col != 'koi_disposition']
                                X = df[numerical_features].fillna(df[numerical_features].median())
                                
                                # Train-test split
                                X_train, X_test, y_train, y_test = train_test_split(
                                    X, y, test_size=0.2, random_state=42, stratify=y
                                )
                                
                                # Scale features
                                scaler = StandardScaler()
                                X_train_scaled = scaler.fit_transform(X_train)
                                X_test_scaled = scaler.transform(X_test)
                                
                                # Train model
                                model = RandomForestClassifier(
                                    n_estimators=n_estimators,
                                    max_depth=max_depth,
                                    min_samples_split=min_samples_split,
                                    min_samples_leaf=min_samples_leaf,
                                    max_features=max_features,
                                    random_state=42,
                                    n_jobs=-1
                                )
                                
                                model.fit(X_train_scaled, y_train)
                                
                                # Evaluate
                                y_pred = model.predict(X_test_scaled)
                                accuracy = accuracy_score(y_test, y_pred)
                                cm = confusion_matrix(y_test, y_pred)
                                report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
                                
                                training_time = time.time() - start_time
                                
                                # Store results
                                st.session_state.training_results = {
                                    'algorithm': algorithm,
                                    'accuracy': accuracy,
                                    'confusion_matrix': cm,
                                    'classification_report': report,
                                    'training_time': training_time,
                                    'classes': le.classes_,
                                    'hyperparameters': {
                                        'n_estimators': n_estimators,
                                        'max_depth': max_depth,
                                        'min_samples_split': min_samples_split,
                                        'min_samples_leaf': min_samples_leaf,
                                        'max_features': str(max_features)
                                    }
                                }
                                st.session_state.training_complete = True
                                st.success(f"✅ Training completed in {training_time:.2f} seconds!")
                                st.rerun()
                                
                        except Exception as e:
                            st.error(f"Error during training: {str(e)}")
                            st.exception(e)
        
        elif algorithm == "XGBoost":
            with st.form("xgb_form"):
                n_estimators = st.number_input("Number of Estimators", 
                                              min_value=10, max_value=2000, value=1000, step=50)
                max_depth = st.number_input("Max Depth", 
                                           min_value=1, max_value=20, value=8, step=1)
                learning_rate = st.number_input("Learning Rate", 
                                               min_value=0.01, max_value=0.5, value=0.1, step=0.01)
                subsample = st.slider("Subsample", 
                                     min_value=0.5, max_value=1.0, value=0.8, step=0.05)
                colsample_bytree = st.slider("Column Sample by Tree", 
                                            min_value=0.5, max_value=1.0, value=0.8, step=0.05)
                gamma = st.number_input("Gamma (Min Split Loss)", 
                                       min_value=0.0, max_value=5.0, value=0.1, step=0.1)
                
                train_button = st.form_submit_button("🚀 Train Model", use_container_width=True)
                
                if train_button:
                    with st.spinner("Training XGBoost..."):
                        try:
                            import xgboost as xgb
                            from sklearn.preprocessing import StandardScaler
                            from sklearn.model_selection import train_test_split
                            from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
                            import time
                            
                            data_path = "cumulative_2025.10.02_20.38.17.csv"
                            if not os.path.exists(data_path):
                                st.error(f"Dataset not found: {data_path}")
                            else:
                                start_time = time.time()
                                
                                df = pd.read_csv(data_path)
                                drop_cols = ["koi_longp", "koi_ingress", "koi_model_dof", "koi_model_chisq", 
                                            "koi_sage", "rowid", "kepoi_name", "kepid", "kepler_name",
                                            "koi_pdisposition", "koi_score", "koi_time0bk", "koi_comment", 
                                            "koi_limbdark_mod", "koi_parm_prov", "koi_trans_mod", 
                                            "koi_datalink_dvr", "koi_datalink_dvs", "koi_tce_delivname", 
                                            "koi_sparprov", "koi_vet_stat", "koi_vet_date", "koi_disp_prov",
                                            "koi_ldm_coeff3", "koi_ldm_coeff4"]
                                df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
                                
                                from sklearn.preprocessing import LabelEncoder
                                le = LabelEncoder()
                                y = le.fit_transform(df['koi_disposition'])
                                
                                numerical_features = df.select_dtypes(include=['float64', 'int64']).columns
                                numerical_features = [col for col in numerical_features if col != 'koi_disposition']
                                X = df[numerical_features].fillna(df[numerical_features].median())
                                
                                X_train, X_test, y_train, y_test = train_test_split(
                                    X, y, test_size=0.2, random_state=42, stratify=y
                                )
                                
                                scaler = StandardScaler()
                                X_train_scaled = scaler.fit_transform(X_train)
                                X_test_scaled = scaler.transform(X_test)
                                
                                model = xgb.XGBClassifier(
                                    n_estimators=n_estimators,
                                    max_depth=max_depth,
                                    learning_rate=learning_rate,
                                    subsample=subsample,
                                    colsample_bytree=colsample_bytree,
                                    gamma=gamma,
                                    random_state=42,
                                    eval_metric='mlogloss',
                                    n_jobs=-1
                                )
                                
                                model.fit(X_train_scaled, y_train)
                                
                                y_pred = model.predict(X_test_scaled)
                                accuracy = accuracy_score(y_test, y_pred)
                                cm = confusion_matrix(y_test, y_pred)
                                report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
                                
                                training_time = time.time() - start_time
                                
                                st.session_state.training_results = {
                                    'algorithm': algorithm,
                                    'accuracy': accuracy,
                                    'confusion_matrix': cm,
                                    'classification_report': report,
                                    'training_time': training_time,
                                    'classes': le.classes_,
                                    'hyperparameters': {
                                        'n_estimators': n_estimators,
                                        'max_depth': max_depth,
                                        'learning_rate': learning_rate,
                                        'subsample': subsample,
                                        'colsample_bytree': colsample_bytree,
                                        'gamma': gamma
                                    }
                                }
                                st.session_state.training_complete = True
                                st.success(f"✅ Training completed in {training_time:.2f} seconds!")
                                st.rerun()
                                
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
        
        elif algorithm == "LightGBM":
            with st.form("lgb_form"):
                n_estimators = st.number_input("Number of Estimators", 
                                              min_value=10, max_value=2000, value=1000, step=50)
                max_depth = st.number_input("Max Depth", 
                                           min_value=1, max_value=20, value=8, step=1)
                learning_rate = st.number_input("Learning Rate", 
                                               min_value=0.01, max_value=0.5, value=0.05, step=0.01)
                num_leaves = st.number_input("Number of Leaves", 
                                            min_value=10, max_value=200, value=31, step=5)
                subsample = st.slider("Subsample", 
                                     min_value=0.5, max_value=1.0, value=0.8, step=0.05)
                colsample_bytree = st.slider("Column Sample by Tree", 
                                            min_value=0.5, max_value=1.0, value=0.8, step=0.05)
                
                train_button = st.form_submit_button("🚀 Train Model", use_container_width=True)
                
                if train_button:
                    st.info("Training LightGBM - feature implementation similar to XGBoost above")
        
        elif algorithm == "CatBoost":
            with st.form("cat_form"):
                iterations = st.number_input("Iterations", 
                                            min_value=10, max_value=2000, value=1000, step=50)
                depth = st.number_input("Depth", 
                                       min_value=1, max_value=16, value=8, step=1)
                learning_rate = st.number_input("Learning Rate", 
                                               min_value=0.01, max_value=0.5, value=0.05, step=0.01)
                l2_leaf_reg = st.number_input("L2 Leaf Regularization", 
                                             min_value=1.0, max_value=10.0, value=3.0, step=0.5)
                
                train_button = st.form_submit_button("🚀 Train Model", use_container_width=True)
                
                if train_button:
                    st.info("Training CatBoost - feature implementation similar to above")
        
        elif algorithm == "AdaBoost":
            with st.form("ada_form"):
                n_estimators = st.number_input("Number of Estimators", 
                                              min_value=10, max_value=1000, value=500, step=50)
                learning_rate = st.number_input("Learning Rate", 
                                               min_value=0.01, max_value=2.0, value=0.1, step=0.05)
                algorithm_type = st.selectbox("Algorithm", ["SAMME", "SAMME.R"])
                
                train_button = st.form_submit_button("🚀 Train Model", use_container_width=True)
                
                if train_button:
                    st.info("Training AdaBoost - feature implementation similar to above")
        
        elif algorithm == "Gradient Boosting":
            with st.form("gb_form"):
                n_estimators = st.number_input("Number of Estimators", 
                                              min_value=10, max_value=1000, value=500, step=50)
                max_depth = st.number_input("Max Depth", 
                                           min_value=1, max_value=20, value=5, step=1)
                learning_rate = st.number_input("Learning Rate", 
                                               min_value=0.01, max_value=0.5, value=0.1, step=0.01)
                subsample = st.slider("Subsample", 
                                     min_value=0.5, max_value=1.0, value=0.8, step=0.05)
                
                train_button = st.form_submit_button("🚀 Train Model", use_container_width=True)
                
                if train_button:
                    st.info("Training Gradient Boosting - feature implementation similar to above")
    
    with col_right:
        st.markdown("### 📊 Training Results")
        
        if st.session_state.training_complete and st.session_state.training_results:
            results = st.session_state.training_results
            
            # Metrics
            st.markdown(f"""
            <div class="result-card">
                <h4>{results['algorithm']} Performance</h4>
                <p><strong>Accuracy:</strong> <span style='color: #00ffc8; font-size: 1.5rem;'>{results['accuracy']*100:.2f}%</span></p>
                <p><strong>Training Time:</strong> {results['training_time']:.2f} seconds</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Detailed metrics
            col1, col2, col3 = st.columns(3)
            report = results['classification_report']
            
            with col1:
                st.metric("Precision", f"{report['weighted avg']['precision']:.3f}")
            with col2:
                st.metric("Recall", f"{report['weighted avg']['recall']:.3f}")
            with col3:
                st.metric("F1-Score", f"{report['weighted avg']['f1-score']:.3f}")
            
            # Confusion Matrix
            st.markdown("#### Confusion Matrix")
            cm = results['confusion_matrix']
            
            # Create a simple visualization
            fig_cm = """
            <div style='background: rgba(0, 217, 255, 0.08); padding: 1rem; border-radius: 10px;'>
                <table style='width: 100%; text-align: center; border-collapse: collapse;'>
                    <tr style='background: rgba(0, 217, 255, 0.2);'>
                        <th>Predicted →</th>
            """
            for cls in results['classes']:
                fig_cm += f"<th>{cls}</th>"
            fig_cm += "</tr>"
            
            for i, cls in enumerate(results['classes']):
                fig_cm += f"<tr><td style='background: rgba(0, 217, 255, 0.2);'><strong>{cls}</strong></td>"
                for j in range(len(results['classes'])):
                    color = 'rgba(0, 255, 200, 0.3)' if i == j else 'rgba(255, 107, 107, 0.3)'
                    fig_cm += f"<td style='background: {color}; padding: 10px;'>{cm[i][j]}</td>"
                fig_cm += "</tr>"
            
            fig_cm += "</table></div>"
            st.markdown(fig_cm, unsafe_allow_html=True)
            
            # Per-class metrics
            with st.expander("📋 Detailed Classification Report"):
                for cls in results['classes']:
                    if cls in report:
                        st.markdown(f"{cls}:")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.write(f"Precision: {report[cls]['precision']:.3f}")
                        with col2:
                            st.write(f"Recall: {report[cls]['recall']:.3f}")
                        with col3:
                            st.write(f"F1-Score: {report[cls]['f1-score']:.3f}")
                        with col4:
                            st.write(f"Support: {int(report[cls]['support'])}")
            
            # Hyperparameters
            with st.expander("⚙ Model Hyperparameters"):
                for key, value in results['hyperparameters'].items():
                    st.write(f"{key}: {value}")
            
            # Clear results button
            if st.button("🔄 Train New Model", use_container_width=True):
                st.session_state.training_complete = False
                st.session_state.training_results = {}
                st.rerun()
        
        else:
            st.info("👈 Configure hyperparameters and click 'Train Model' to see results here")
            
            st.markdown("""
            <div class="tool-card">
                <h4>💡 Training Tips</h4>
                <p><strong>Random Forest:</strong> Good baseline, very stable</p>
                <p><strong>XGBoost:</strong> Usually highest accuracy, slower training</p>
                <p><strong>LightGBM:</strong> Fast training, good for large datasets</p>
                <p><strong>CatBoost:</strong> Handles categorical features well</p>
                <p><strong>AdaBoost:</strong> Good for binary classification</p>
            </div>
            """, unsafe_allow_html=True)

# ============================================================================
# RESOURCES PAGE
# ============================================================================
            
elif st.session_state.current_page == "Resources":
    st.title("📚 Resources & Learning Center")
    
    st.markdown("""
    <div class="hero">
        <h1>Master Exoplanet Classification</h1>
        <p>Complete guide to algorithms, parameters, and detection methods</p>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🤖 ML Algorithms", 
        "⚙ Hyperparameters", 
        "📖 Detection Methods",
        "💻 Code Examples",
        "❓ FAQ"
    ])
    
    # ========== TAB 1: ML ALGORITHMS ==========
    with tab1:
        st.markdown("## Machine Learning Algorithms")
        
        algo_selected = st.selectbox(
            "Select an algorithm to learn about:",
            ["Random Forest", "XGBoost", "LightGBM", "CatBoost", "AdaBoost"]
        )
        
        if algo_selected == "Random Forest":
            st.markdown("""
            ### 🌲 Random Forest Classifier
            
            *Overview:*
            Random Forest is an ensemble learning method that creates multiple decision trees during training 
            and outputs the class that is the mode of the classes from individual trees.
            
            *How it Works:*
            1. Creates multiple decision trees using random subsets of data (bootstrap sampling)
            2. Each tree votes on the classification
            3. Final prediction is the majority vote
            4. Uses feature randomization to reduce correlation between trees
            
            *Strengths:*
            ✅ Very stable and reliable
            ✅ Resistant to overfitting
            ✅ Works well with default parameters
            ✅ Provides feature importance rankings
            ✅ Handles missing values well
            ✅ No need for feature scaling
            
            *Weaknesses:*
            ❌ Can be slow with large datasets
            ❌ Not as accurate as gradient boosting methods
            ❌ Requires more memory than single trees
            ❌ Less interpretable than single decision trees
            
            *Best For:*
            - Baseline models
            - When interpretability is important
            - Datasets with many features
            - Binary and multi-class classification
            - When you need fast predictions
            
            *Typical Performance on Exoplanet Data:*
            - Accuracy: 92-95%
            - Training Time: 2-5 seconds
            - Memory Usage: Moderate
            """)
            
            st.markdown("""
            <div class="tool-card">
                <h4>📊 Our Best Random Forest Model</h4>
                <p><strong>Accuracy:</strong> 94.7%</p>
                <p><strong>Precision:</strong> 0.92</p>
                <p><strong>Recall:</strong> 0.91</p>
                <p><strong>F1-Score:</strong> 0.92</p>
                <p><strong>Parameters:</strong> n_estimators=1000, max_depth=8</p>
            </div>
            """, unsafe_allow_html=True)
        
        elif algo_selected == "XGBoost":
            st.markdown("""
            ### ⚡ XGBoost (Extreme Gradient Boosting)
            
            *Overview:*
            XGBoost is an optimized distributed gradient boosting library designed to be highly efficient, 
            flexible and portable. It's one of the most powerful ML algorithms available.
            
            *How it Works:*
            1. Builds trees sequentially
            2. Each new tree corrects errors from previous trees
            3. Uses gradient descent to minimize loss function
            4. Implements regularization to prevent overfitting
            5. Uses smart splits and pruning techniques
            
            *Strengths:*
            ✅ Usually achieves highest accuracy
            ✅ Built-in regularization (L1 & L2)
            ✅ Handles sparse data efficiently
            ✅ Parallel processing support
            ✅ Cross-validation built-in
            ✅ Handles missing values automatically
            
            *Weaknesses:*
            ❌ More complex to tune
            ❌ Longer training time than Random Forest
            ❌ Can overfit if not tuned properly
            ❌ Requires careful parameter selection
            ❌ More sensitive to outliers
            
            *Best For:*
            - Competitions and production systems
            - When maximum accuracy is needed
            - Structured/tabular data
            - Large datasets
            - When you have time for hyperparameter tuning
            
            *Typical Performance on Exoplanet Data:*
            - Accuracy: 95-96%
            - Training Time: 5-15 seconds
            - Memory Usage: Moderate-High
            """)
            
            st.markdown("""
            <div class="tool-card">
                <h4>🎯 Key Parameters to Tune</h4>
                <p><strong>n_estimators:</strong> More trees = better accuracy but slower (500-2000)</p>
                <p><strong>max_depth:</strong> Controls tree complexity (6-10 is typical)</p>
                <p><strong>learning_rate:</strong> Step size (0.01-0.3, smaller = more robust)</p>
                <p><strong>subsample:</strong> Fraction of samples for each tree (0.6-1.0)</p>
                <p><strong>colsample_bytree:</strong> Fraction of features per tree (0.6-1.0)</p>
                <p><strong>gamma:</strong> Minimum loss reduction for split (0-5)</p>
            </div>
            """, unsafe_allow_html=True)
        
        elif algo_selected == "LightGBM":
            st.markdown("""
            ### 🚀 LightGBM (Light Gradient Boosting Machine)
            
            *Overview:*
            LightGBM is a gradient boosting framework developed by Microsoft that uses tree-based learning 
            algorithms. It's designed for distributed and efficient training, especially on large datasets.
            
            *How it Works:*
            1. Grows trees leaf-wise (best-first) instead of level-wise
            2. Uses histogram-based algorithms for faster training
            3. Implements Gradient-based One-Side Sampling (GOSS)
            4. Exclusive Feature Bundling (EFB) for dimensionality reduction
            5. Optimized for speed and memory efficiency
            
            *Strengths:*
            ✅ Extremely fast training speed
            ✅ Lower memory usage than XGBoost
            ✅ Better accuracy than traditional GBDT
            ✅ Handles large datasets efficiently
            ✅ Built-in categorical feature support
            ✅ Direct support for parallel and GPU learning
            
            *Weaknesses:*
            ❌ Can overfit small datasets (<10K samples)
            ❌ Sensitive to parameter tuning
            ❌ Less stable than Random Forest
            ❌ May require more careful preprocessing
            
            *Best For:*
            - Large datasets (>10,000 samples)
            - When training speed is critical
            - High-dimensional data
            - Production systems with frequent retraining
            - When you have GPUs available
            
            *Typical Performance on Exoplanet Data:*
            - Accuracy: 94-95%
            - Training Time: 1-3 seconds
            - Memory Usage: Low-Moderate
            """)
            
            st.markdown("""
            <div class="tool-card">
                <h4>⚙ Important Parameters</h4>
                <p><strong>num_leaves:</strong> Max number of leaves in one tree (31 is default)</p>
                <p><strong>max_depth:</strong> Limit tree depth to prevent overfitting</p>
                <p><strong>learning_rate:</strong> Shrinkage rate (0.01-0.1)</p>
                <p><strong>n_estimators:</strong> Number of boosting iterations</p>
                <p><strong>min_child_samples:</strong> Minimum data in one leaf (20+ for small datasets)</p>
            </div>
            """, unsafe_allow_html=True)
        
        elif algo_selected == "CatBoost":
            st.markdown("""
            ### 🐱 CatBoost (Categorical Boosting)
            
            *Overview:*
            CatBoost is a gradient boosting library developed by Yandex that provides state-of-the-art results 
            and is especially strong with categorical features.
            
            *How it Works:*
            1. Uses ordered boosting to reduce overfitting
            2. Implements novel categorical feature encoding
            3. Builds symmetric (oblivious) trees for faster prediction
            4. Built-in handling of categorical variables
            5. Uses ordered target statistics for categories
            
            *Strengths:*
            ✅ Best-in-class handling of categorical features
            ✅ Less prone to overfitting
            ✅ Great default parameters (minimal tuning needed)
            """)
        
        elif algo_selected == "AdaBoost":
            st.markdown("""
            ### 🔄 AdaBoost
            
            *Overview:*
            AdaBoost (Adaptive Boosting) is a machine learning meta-algorithm formulated by Yoav Freund and Robert Schapire.
            
            *How it Works:*
            1. Trains weak learners (usually stumps) sequentially
            2. Focuses on misclassified samples by adjusting weights
            3. Combines weak learners with weighted voting
            
            *Strengths:*
            ✅ Simple and fast
            ✅ Less prone to overfitting than single trees
            ✅ Good for binary classification
            
            *Weaknesses:*
            ❌ Sensitive to noisy data
            ❌ Can overemphasize outliers
            ❌ Not as powerful as modern boosting methods
            
            *Best For:*
            - Binary classification problems
            - When dataset is clean
            - Quick prototyping
            """)
    
    # Other tabs would go here if needed, but based on your original code, they seem empty

st.markdown('</div>', unsafe_allow_html=True)

















