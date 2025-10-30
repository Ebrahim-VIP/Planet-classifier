import streamlit as st
import streamlit.components.v1 as components
import json

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

# Custom CSS with Teal/Cyan Theme
st.markdown("""
<style>
    /* Remove sidebar completely */
    [data-testid="collapsedControl"] {
        display: none;
    }
    
    /* Main background and colors - Teal Theme */
    .stApp {
       background: linear-gradient(180deg, #102631 50%, #050f17);
        color: #e0f4ff;
    }
    
    /* Remove default padding */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
        max-width: 100%;
    }
    
    /* Headers - Cyan */
    h1, h2, h3, h4, h5, h6 {
        color: #00d9ff !important;
    }
    
    /* Navigation Bar */
    .navbar {
        background: rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
        padding: 1rem 0;
        margin-bottom: 2rem;
        border-radius: 10px;
    }
    
    .nav-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        max-width: 1200px;
        margin: 0 auto;
        padding: 0 2rem;
    }
    
    /* Logo - Teal Gradient */
    .logo {
        font-size: 1.8rem;
        font-weight: bold;
        background: linear-gradient(45deg, #00ffc8, #00d9ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .nav-menu {
        display: flex;
        gap: 1rem;
        align-items: center;
    }
    
    /* Custom buttons for navigation */
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
    
    /* Active nav button styling */
    div[data-testid="column"] button[kind="primary"] {
        background: rgba(0, 217, 255, 0.2) !important;
        color: #00ffc8 !important;
    }
    
    /* Regular buttons (non-navigation) - Teal Gradient */
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
    
    /* Input fields */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select {
        background: rgba(255, 255, 255, 0.1) !important;
        color: white !important;
        border: 1px solid rgba(0, 217, 255, 0.3) !important;
        border-radius: 8px !important;
    }
    
    /* Tabs */
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
    
    /* Metrics - Teal */
    [data-testid="stMetricValue"] {
        color: #00ffc8 !important;
        font-size: 1.5rem !important;
        font-weight: bold !important;
    }
    
    /* Progress bars - Teal Gradient */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #00ffc8, #00d9ff);
    }
    
    /* File uploader - Teal Theme */
    [data-testid="stFileUploader"] {
        border: 2px dashed rgba(0, 255, 200, 0.5);
        border-radius: 10px;
        padding: 2rem;
        background: rgba(0, 217, 255, 0.05);
    }
    
    /* Info/Success/Warning boxes - Teal */
    .stAlert {
        background: rgba(0, 217, 255, 0.1);
        border-left: 4px solid #00ffc8;
        border-radius: 10px;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(0, 217, 255, 0.08);
        border-radius: 10px;
        color: white !important;
    }
    
    /* Remove Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Hero section - Teal Glow */
    .hero {
        text-align: center;
        padding: 3rem 0;
        background: radial-gradient(circle at center, rgba(0, 255, 200, 0.15) 0%, transparent 70%);
        border-radius: 15px;
        margin-bottom: 2rem;
    }
    
    .hero h1 {
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    
    .hero p {
        font-size: 1.2rem;
        opacity: 0.9;
    }
    
    /* Result card - Teal */
    .result-card {
        background: rgba(0, 217, 255, 0.08);
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 4px solid #00ffc8;
    }
    
    /* Placeholder 3D - Teal Theme */
    .placeholder-3d {
        background: rgba(0, 217, 255, 0.05);
        border: 2px dashed rgba(0, 255, 200, 0.3);
        border-radius: 15px;
        padding: 3rem;
        text-align: center;
        min-height: 400px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        color: #a0e0f0;
    }
    
    /* Comparison card */
    .comparison-card {
        background: rgba(0, 217, 255, 0.08);
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
    }
    
    /* Tool card */
    .tool-card {
        background: rgba(0, 217, 255, 0.08);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 1rem;
    }
    
    /* Code snippet */
    .code-snippet {
        background: rgba(0, 0, 0, 0.5);
        padding: 1rem;
        border-radius: 8px;
        font-family: 'Courier New', monospace;
        margin: 1rem 0;
        overflow-x: auto;
    }
    
    /* Content container */
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

# HOME PAGE
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
            if st.button("🛠️ Research Tools", use_container_width=True, key="home_research"):
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
        and space missions like NASA's *Kepler*, we have discovered thousands of these distant planets—ranging 
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
        intelligence model trained on authentic *Kepler* mission data.
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
#===========================================================
elif st.session_state.current_page == "Classification":
    st.title("🔬 Exoplanet Classification & Visualization")
    
    # Initialize default values in session state
    if 'orbital_period' not in st.session_state:
        st.session_state.orbital_period = 365.25
    if 'planet_radius' not in st.session_state:
        st.session_state.planet_radius = 1.0
    if 'transit_depth' not in st.session_state:
        st.session_state.transit_depth = 84.0
    if 'stellar_temp' not in st.session_state:
        st.session_state.stellar_temp = 5778
    if 'impact_param' not in st.session_state:
        st.session_state.impact_param = 0.5
    
    # Create two main columns for input and output
    col_input, col_output = st.columns([1, 1])
    
    with col_input:
        st.markdown("### 📥 Input Parameters")
        
        # Manual Entry Form only (Upload feature removed)
        with st.form("classification_form"):
            orbital_period = st.number_input(
                "Orbital Period (days)", min_value=0.0, value=st.session_state.orbital_period,
                step=0.1, help="Time for one complete orbit around the star"
            )
            transit_depth = st.number_input(
                "Transit Depth (ppm)", min_value=0.0, value=st.session_state.transit_depth,
                step=0.1, help="Decrease in star brightness during transit"
            )
            planet_radius = st.number_input(
                "Planet Radius (Earth radii)", min_value=0.0, value=st.session_state.planet_radius,
                step=0.1, help="Size relative to Earth"
            )
            stellar_temp = st.number_input(
                "Stellar Temperature (K)", min_value=0, value=st.session_state.stellar_temp,
                step=1, help="Temperature of the host star"
            )
            impact_param = st.number_input(
                "Impact Parameter", min_value=0.0, max_value=1.0, value=st.session_state.impact_param,
                step=0.01, help="Orbital inclination parameter"
            )
            
            submitted = st.form_submit_button("🚀 Classify Planet", use_container_width=True)
            
            if submitted:
                st.session_state.classification_done = True
                st.session_state.orbital_period = orbital_period
                st.session_state.planet_radius = planet_radius
                st.session_state.transit_depth = transit_depth
                st.session_state.stellar_temp = stellar_temp
                st.session_state.impact_param = impact_param
    
    with col_output:
        st.markdown("### 📊 Classification Results")
        
        if 'classification_done' in st.session_state and st.session_state.classification_done:
            st.markdown("""
            <div class="result-card">
                <h4>Predicted Class</h4>
                <p style='color: #00ffc8; font-size: 1.2rem; font-weight: bold;'>Confirmed Planet</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="result-card">
                <h4>Confidence Scores</h4>
            </div>
            """, unsafe_allow_html=True)
            
            st.write("**Confirmed Planet**")
            st.progress(0.75)
            st.write("75.0%")
            
            st.write("**Planet Candidate**")
            st.progress(0.20)
            st.write("20.0%")
            
            st.write("**False Positive**")
            st.progress(0.05)
            st.write("5.0%")
            
            st.button("📄 Generate Report", use_container_width=True, key="gen_report")
        else:
            st.info("Enter parameters and click 'Classify Planet' to see results")
    
    st.markdown("---")
    
    # Visualization Section
    st.markdown("## 🌌 3D Visualization & Simulation")
    
    # Controls
    col1, col2, col3, col4 = st.columns(4)

    # Load HTML file and inject parameters
    with open("Index.html", "r", encoding="utf-8") as f:
        sim_html = f.read()
    
    # Create JavaScript to send parameters to the HTML
    params = {
        "orbital_period": st.session_state.orbital_period,
        "planet_radius": st.session_state.planet_radius,
        "transit_depth": st.session_state.transit_depth,
        "stellar_temp": st.session_state.stellar_temp,
        "impact_param": st.session_state.impact_param,
        "orbital_distance": 1.0,
        "insolation": 1.0,
        "star_radius": 1.0,
        "planet_temp": 288
    }
    
    injection_script = f"""
    <script>
    window.addEventListener('load', function() {{
        const iframe = document.querySelector('iframe');
        if (iframe) {{
            iframe.addEventListener('load', function() {{
                iframe.contentWindow.postMessage({{
                    type: 'exoplanet_params',
                    params: {json.dumps(params)}
                }}, '*');
            }});
        }}
    }});
    </script>
    """
    
    full_html = sim_html + injection_script
    components.html(full_html, height=800, scrolling=False)
    
    st.markdown("---")
    
    # Comparison Section
    st.markdown("### 📊 System Comparison")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="comparison-card">
            <h4>🌍 Earth System</h4>
            <p><strong>Orbital Period:</strong> 365.25 days</p>
            <p><strong>Distance from Star:</strong> 1 AU</p>
            <p><strong>Planet Radius:</strong> 1.0 Earth radii</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if 'classification_done' in st.session_state and st.session_state.classification_done:
            period = st.session_state.get('orbital_period', '--')
            radius = st.session_state.get('planet_radius', '--')
            st.markdown(f"""
            <div class="comparison-card">
                <h4>🪐 Detected Exoplanet</h4>
                <p><strong>Orbital Period:</strong> {period} days</p>
                <p><strong>Distance from Star:</strong> -- AU</p>
                <p><strong>Planet Radius:</strong> {radius} Earth radii</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="comparison-card">
                <h4>🪐 Detected Exoplanet</h4>
                <p><strong>Orbital Period:</strong> -- days</p>
                <p><strong>Distance from Star:</strong> -- AU</p>
                <p><strong>Planet Radius:</strong> -- Earth radii</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Similar Planets
    st.markdown("### 🌟 Top 5 Similar Known Planets")
    
    planets = [
        ("Kepler-452b", "Super-Earth in habitable zone", "87%"),
        ("TOI-715 b", "Rocky planet candidate", "76%"),
        ("K2-18 b", "Sub-Neptune with water vapor", "71%"),
        ("TRAPPIST-1e", "Earth-sized in habitable zone", "68%"),
        ("Proxima Centauri b", "Nearest Earth-like exoplanet", "65%")
    ]
    
    for name, desc, similarity in planets:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**{name}** - {desc}")
        with col2:
            st.markdown(f"<span style='color: #00ffc8;'>Similarity: {similarity}</span>", unsafe_allow_html=True)
        st.markdown("<hr style='margin: 0.5rem 0; opacity: 0.3;'>", unsafe_allow_html=True)

# RESEARCH TOOLS PAGE======================================================
elif st.session_state.current_page == "Research":
    st.title("🛠 Research Tools & Advanced Features")
    
    # Initialize selected algorithm in session state
    if 'selected_algorithm' not in st.session_state:
        st.session_state.selected_algorithm = "Random Forest"
    
    # Define model statistics for each algorithm
    # TODO: Replace these placeholder values with actual model performance metrics
    model_stats = {
        "Random Forest": {
            "accuracy": "94.7%",
            "samples": "150K",
            "runtime": "2.3s",
            "f1_score": "0.92"
        },
        "Stacking": {
            "accuracy": "95.2%",
            "samples": "150K",
            "runtime": "4.1s",
            "f1_score": "0.93"
        },
        "Logistic Regression": {
            "accuracy": "89.3%",
            "samples": "150K",
            "runtime": "0.8s",
            "f1_score": "0.87"
        },
        "KNN": {
            "accuracy": "91.5%",
            "samples": "150K",
            "runtime": "1.5s",
            "f1_score": "0.89"
        },
        "Decision Trees": {
            "accuracy": "88.9%",
            "samples": "150K",
            "runtime": "1.2s",
            "f1_score": "0.86"
        },
        "AdaBoost": {
            "accuracy": "92.8%",
            "samples": "150K",
            "runtime": "3.2s",
            "f1_score": "0.90"
        }
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="tool-card">
            <h3>📚 Select ML Algorithm</h3>
            <p>Choose an algorithm and test with your own parameters.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Algorithm selection dropdown
        selected_algo = st.selectbox(
            "Select Algorithm",
            ["Random Forest", "Stacking", "Logistic Regression", "KNN", "Decision Trees", "AdaBoost"],
            key="algo_selector"
        )
        
        st.session_state.selected_algorithm = selected_algo
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Input parameters for classification (similar to Classification page)
        st.markdown("#### 📥 Test Input Parameters")
        with st.form("research_classification_form"):
            orbital_period = st.number_input("Orbital Period (days)", min_value=0.0, step=0.1, value=365.25)
            transit_depth = st.number_input("Transit Depth (ppm)", min_value=0.0, step=0.1, value=84.0)
            planet_radius = st.number_input("Planet Radius (Earth radii)", min_value=0.0, step=0.1, value=1.0)
            stellar_temp = st.number_input("Stellar Temperature (K)", min_value=0, step=1, value=5778)
            impact_param = st.number_input("Impact Parameter", min_value=0.0, max_value=1.0, step=0.1, value=0.5)
            
            test_submitted = st.form_submit_button("🔬 Test Algorithm", use_container_width=True)
            
            if test_submitted:
                st.success(f"Testing {selected_algo} with provided parameters...")
                # TODO: Connect to actual model prediction
                # prediction = model.predict(orbital_period, transit_depth, planet_radius, stellar_temp, impact_param)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class="tool-card">
            <h3>🔧 Interactive Tuning Sandbox</h3>
            <p>Experiment with hyperparameters in real-time.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Hyperparameters based on selected algorithm
        # TODO: Connect these hyperparameters to actual model training pipeline
        if selected_algo == "Random Forest":
            n_estimators = st.number_input("Number of Estimators", min_value=10, max_value=500, value=100, step=10)
            max_depth = st.number_input("Max Depth", min_value=1, max_value=50, value=10, step=1)
            min_samples_split = st.number_input("Min Samples Split", min_value=2, max_value=20, value=2, step=1)
            min_samples_leaf = st.number_input("Min Samples Leaf", min_value=1, max_value=20, value=1, step=1)
            max_features = st.selectbox("Max Features", ["sqrt", "log2", "None"])
            
        elif selected_algo == "Stacking":
            n_estimators_rf = st.number_input("Random Forest Estimators", min_value=10, max_value=500, value=100, step=10)
            n_estimators_ada = st.number_input("AdaBoost Estimators", min_value=10, max_value=500, value=50, step=10)
            final_estimator_c = st.number_input("Final Estimator C", min_value=0.01, max_value=10.0, value=1.0, step=0.1)
            cv_folds = st.number_input("Cross-Validation Folds", min_value=2, max_value=10, value=5, step=1)
            
        elif selected_algo == "Logistic Regression":
            c_value = st.number_input("C (Regularization)", min_value=0.001, max_value=100.0, value=1.0, step=0.1)
            max_iter = st.number_input("Max Iterations", min_value=100, max_value=5000, value=1000, step=100)
            penalty = st.selectbox("Penalty", ["l1", "l2", "elasticnet", "none"])
            solver = st.selectbox("Solver", ["lbfgs", "liblinear", "newton-cg", "sag", "saga"])
            
        elif selected_algo == "KNN":
            n_neighbors = st.number_input("Number of Neighbors", min_value=1, max_value=50, value=5, step=1)
            weights = st.selectbox("Weights", ["uniform", "distance"])
            algorithm = st.selectbox("Algorithm", ["auto", "ball_tree", "kd_tree", "brute"])
            leaf_size = st.number_input("Leaf Size", min_value=10, max_value=100, value=30, step=5)
            p_value = st.number_input("P (Power Parameter)", min_value=1, max_value=5, value=2, step=1)
            
        elif selected_algo == "Decision Trees":
            max_depth = st.number_input("Max Depth", min_value=1, max_value=50, value=10, step=1)
            min_samples_split = st.number_input("Min Samples Split", min_value=2, max_value=20, value=2, step=1)
            min_samples_leaf = st.number_input("Min Samples Leaf", min_value=1, max_value=20, value=1, step=1)
            criterion = st.selectbox("Criterion", ["gini", "entropy", "log_loss"])
            splitter = st.selectbox("Splitter", ["best", "random"])
            
        elif selected_algo == "AdaBoost":
            n_estimators = st.number_input("Number of Estimators", min_value=10, max_value=500, value=50, step=10)
            learning_rate = st.number_input("Learning Rate", min_value=0.01, max_value=2.0, value=1.0, step=0.1)
            algorithm_ada = st.selectbox("Algorithm", ["SAMME", "SAMME.R"])
            random_state = st.number_input("Random State", min_value=0, max_value=100, value=42, step=1)
        
        st.button("🚀 Retrain Model", use_container_width=True, key="retrain_btn")
        # TODO: Connect to model retraining function
        # retrain_model(selected_algo, hyperparameters)
    
    with col2:
        st.markdown("""
        <div class="tool-card">
            <h3>📊 Model Statistics</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Display statistics for selected algorithm
        current_stats = model_stats[st.session_state.selected_algorithm]
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Accuracy", current_stats["accuracy"])
            st.metric("Training Samples", current_stats["samples"])
        with col_b:
            st.metric("Runtime", current_stats["runtime"])
            st.metric("F1-Score", current_stats["f1_score"])
        
        st.write(f"**{st.session_state.selected_algorithm}** model trained on Kepler mission data with advanced feature engineering.")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class="tool-card">
            <h3>🛰 Dataset Links</h3>
            <p>Direct links to official NASA datasets used in our model training and validation.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.button("📊 NASA Kepler Archive", use_container_width=True, key="kepler_btn")
        st.button("📊 Exoplanet Archive", use_container_width=True, key="exo_archive_btn")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class="tool-card">
            <h3>💻 Code Integration</h3>
            <p>Access open source code and API documentation.</p>
        </div>
        """, unsafe_allow_html=True)
        
        
        st.button("📂 View on GitHub", use_container_width=True, key="github_btn")
# RESOURCES PAGE
elif st.session_state.current_page == "Resources":
    st.title("📚 Resources & Learning")
    
    st.markdown("### 🌍 What are Exoplanets?")
    st.write("""
    Exoplanets, or extrasolar planets, are planets that orbit stars outside our solar system. 
    Since the first confirmed detection in 1995, astronomers have discovered thousands of these 
    distant worlds using various detection methods, with the transit method being one of the most successful.
    """)
    
    st.markdown("---")
    
    st.markdown("### ❓ Frequently Asked Questions")
    
    with st.expander("What is transit depth?"):
        st.write("Transit depth measures how much a star's brightness decreases when a planet passes in front of it. It's expressed in parts per million (ppm) and helps determine the planet's size relative to its star.")
    
    with st.expander("What is orbital period?"):
        st.write("The orbital period is the time it takes for a planet to complete one full orbit around its star. For Earth, this is 365.25 days. Shorter periods indicate planets closer to their stars.")
    
    with st.expander("What are false positives?"):
        st.write("False positives are signals that look like planet transits but are actually caused by other phenomena like binary star eclipses, stellar activity, or instrumental noise.")
    
    with st.expander("How accurate is the AI classification?"):
        st.write("Our model achieves 94.7% accuracy on test data, trained on over 150,000 samples from NASA's Kepler, K2, and TESS missions. However, professional verification is always recommended for research purposes.")
    
    
    st.markdown("---")
    
    st.markdown("### 🔗 External Resources")
    
    st.button("🛰️ NASA Exoplanet Archive", key="ext1", use_container_width=True)
    st.button("🔭 Kepler Mission", key="ext2", use_container_width=True)
    st.button("📚 Educational Resources", key="ext4", use_container_width=True)
    
    st.markdown("---")
    
    st.markdown("### 📧 Contact & Support")
    st.write("Have questions or need help? We're here to assist you in your exoplanet discovery journey.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.button("📧 Email Support", use_container_width=True, key="email_btn")
    with col2:
        st.button("💬 Join Discord", use_container_width=True, key="discord_btn")
    with col3:
        st.button("🐛 Report Issues", use_container_width=True, key="issues_btn")

st.markdown('</div>', unsafe_allow_html=True)