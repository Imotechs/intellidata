import streamlit as st
import importlib
from PIL import Image

side_bar_logo = Image.open("ui/assets/images/logo.jpg")
main_logo = Image.open("ui/assets/images/logo-bg.png")

# ========================
# Page Configuration
# ========================
st.set_page_config(
    page_title="Smart Synthetic Data Generator",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ========================
# Initialize session state
# ========================
if "page" not in st.session_state:
    query_params = st.query_params
    st.session_state.page = query_params.get("page", "home")

# ========================
# Sidebar Navigation (Buttons)
# ========================
with st.sidebar:
    st.image(side_bar_logo, width=200)
    st.title("🧭Navigation")
    st.markdown("### Select an App")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🧪 Test Data"):
            st.session_state.page = "test"
            st.query_params.page = "test"
            st.rerun()  
    with col2:
        if st.button("📊 Real Data"):
            st.session_state.page = "real"
            st.query_params.page = "real"
            st.rerun()

    st.markdown("---")
    st.header("🛠️ App Controls")
    st.markdown("Upload your dataset and fill up the missing gaps with our AI solution, also generate synthetic rows intelligently.")
    st.markdown("👨‍💻 Built for GenAI Hackathon 2025")
    st.markdown("[GitHub Repo](https://github.com/Imotechs/Synthetic-Data-Generator.git) • [Live Demo](https://youtu.be/91LI12Mwm3I)")

# ========================
# Main Content Area
# ========================
page = st.session_state.page

if page == "test":
    import app
    app.run()

elif page == "real":
    import  app2
    app2.run()

else:
    col1, col2 = st.columns([1, 5])  # Adjust ratios to resize layout

    with col1:
        st.image(main_logo, width=80)

    with col2:
        st.markdown("<h1 style='margin-top: 20px;'>IntelliData Hub</h1>", unsafe_allow_html=True)
    st.markdown("Fill the missing gaps in your dataset, and generate a clean, refined version from the original data to facilitate effective model training.")
    st.markdown("Effortlessly generate up to 10,000 rows and 600 columns of test data for your prototype simulations.")
    st.markdown("Use the sidebar to choose a tool.")
    st.image("https://cdn-icons-png.flaticon.com/512/4384/4384684.png", width=150)

# ========================
# Footer
# ========================
st.divider()
st.caption("Built with by Team [Co-Creators] for the GenAI Hackathon 2025")
