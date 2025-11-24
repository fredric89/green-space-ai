import streamlit as st
import ee
import geemap.foliumap as geemap
import json
from google.oauth2 import service_account

# --- 1. CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Green Space AI (Debug Mode)")
st.title("🌍 AI Green Space Analyzer (Debug Mode)")

# --- 2. AUTHENTICATION ---
try:
    if "earth_engine" in st.secrets and "service_account_json" in st.secrets["earth_engine"]:
         key_content = st.secrets["earth_engine"]["service_account_json"]
    else:
         key_content = st.secrets["service_account_json"]
         
    service_account_info = json.loads(key_content, strict=False)
    creds = service_account.Credentials.from_service_account_info(service_account_info)
    creds = creds.with_scopes(['https://www.googleapis.com/auth/earthengine'])
    ee.Initialize(creds, project="mystic-curve-479206-q2")
    
except Exception as e:
    st.error(f"Authentication Error: {e}")
    st.stop()

# --- 3. HELPER FUNCTIONS ---
def mask_s2_clouds(image):
    scl = image.select('SCL')
    # Very permissive mask (keep almost everything to ensure data shows up)
    mask = scl.neq(3).And(scl.neq(8)).And(scl.neq(9))
    return image.updateMask(mask).divide(10000)

def add_indices(image):
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    return image.addBands(ndvi)

# --- 4. SIDEBAR INPUTS ---
with st.sidebar:
    st.header("Settings")
    
    # We stick to "Select City" to minimize variables for now
    city = st.selectbox("Location:", ["Manila", "Surigao City", "Laguna Province", "Quezon City"])
    
    roi = None
    if city == "Manila":
        roi = ee.Geometry.Point([120.9842, 14.5995]).buffer(5000).bounds()
    elif city == "Surigao City":
        roi = ee.Geometry.Point([125.4933, 9.7828]).buffer(6000).bounds()
    elif city == "Laguna Province":
        roi = ee.Geometry.Point([121.25, 14.20]).buffer(10000).bounds()
    elif city == "Quezon City":
        roi = ee.Geometry.Point([121.0437, 14.6760]).buffer(6000).bounds()

    st.info("Click 'Run' to generate the map.")

# --- 5. MAIN EXECUTION ---
if st.button("🚀 Run Analysis", type="primary"):
    
    st.divider()
    
    try:
        with st.spinner("Fetching Data..."):
            
            # --- DEBUG STEP 1: VERIFY DATA EXISTS ---
            # Using Harmonized collection
            dataset_2024 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
                .filterDate('2024-01-01', '2024-06-30') \
                .filterBounds(roi) \
                .select(['B4', 'B3', 'B2', 'SCL']) \
                .map(mask_s2_clouds)
            
            count = dataset_2024.size().getInfo()
            if count > 0:
                st.success(f"✅ Data Connection Active: Found {count} satellite images.")
            else:
                st.error("❌ No satellite images found. The map will be empty.")
                st.stop()

            # --- DEBUG STEP 2: CREATE LAYERS ---
            # We create a simple median composite
            image_2024 = dataset_2024.median().clip(roi)
            
            # Simple NDVI (Vegetation) Layer
            ndvi = image_2024.normalizedDifference(['B8', 'B4']).rename('NDVI')

            # --- DEBUG STEP 3: RENDER MAP (THE FIX) ---
            st.subheader("Map Visualization")
            
            # FIX 1: CHANGE BASEMAP
            # Google 'HYBRID' often fails without an API key. 
            # 'Esri.WorldImagery' is free and reliable.
            m = geemap.Map(basemap="Esri.WorldImagery") 
            m.centerObject(roi, 13)
            
            # VISUALIZATION PARAMETERS
            real_vis = {'min': 0, 'max': 0.3, 'bands': ['B4', 'B3', 'B2']} # True Color
            
            # LAYER 1: 2024 True Color (Satellite View)
            m.add_layer(image_2024, real_vis, "2024 True Color")
            
            # LAYER 2: SRTM Elevation (This is the Sanity Check)
            # This dataset works 100% of the time. If you don't see this, the map is broken.
            dem = ee.Image('USGS/SRTMGL1_003').clip(roi)
            dem_vis = {'min': 0, 'max': 1000, 'palette': ['black', 'blue', 'purple', 'cyan', 'green', 'yellow', 'red', 'white']}
            m.add_layer(dem, dem_vis, "DEBUG: Elevation (Sanity Check)")
            
            m.add_layer_control()
            
            # FORCE RENDER
            m.to_streamlit(height=600, bidirectional=False)
            
            st.caption("Layer Guide: '2024 True Color' is the satellite. 'DEBUG: Elevation' is a test layer.")

    except Exception as e:
        st.error(f"Error: {e}")
