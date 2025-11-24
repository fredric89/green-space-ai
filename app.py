import streamlit as st
import ee
import geemap.foliumap as geemap
import json
from google.oauth2 import service_account

# --- 1. CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Green Space AI - PH Edition")
st.title("🌍 AI Green Space Analyzer (Philippines - Force Render Mode)")

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
    # We use a gentler mask. We only remove Saturated (1) and massive clouds.
    # We keep "thin cirrus" to ensure we always have pixels to show.
    scl = image.select('SCL')
    mask = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)) 
    return image.updateMask(mask).divide(10000)

def add_indices(image):
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    ndbi = image.normalizedDifference(['B11', 'B8']).rename('NDBI')
    return image.addBands([ndvi, ndbi])

# --- 4. SIDEBAR INPUTS ---
with st.sidebar:
    st.header("Settings")
    
    analysis_mode = st.radio("Mode:", ["Select City", "Draw Area"])
    
    roi = None
    
    if analysis_mode == "Select City":
        city = st.selectbox("Location:", ["Manila", "Surigao City", "Laguna Province", "Quezon City", "Cebu City", "Davao City"])
        
        # INCREASED BUFFER SIZES slightly for better context
        if city == "Manila":
            roi = ee.Geometry.Point([120.9842, 14.5995]).buffer(6000).bounds()
        elif city == "Surigao City":
            roi = ee.Geometry.Point([125.4933, 9.7828]).buffer(8000).bounds()
        elif city == "Laguna Province":
            roi = ee.Geometry.Point([121.25, 14.20]).buffer(15000).bounds()
        elif city == "Quezon City":
            roi = ee.Geometry.Point([121.0437, 14.6760]).buffer(8000).bounds()
        elif city == "Cebu City":
            roi = ee.Geometry.Point([123.8854, 10.3157]).buffer(8000).bounds()
        elif city == "Davao City":
            roi = ee.Geometry.Point([125.6001, 7.1907]).buffer(8000).bounds()
            
        st.success(f"📍 {city} Selected")

    else:
        st.info("Draw a box on the map.")

# --- 5. DRAWING INTERFACE ---
if analysis_mode == "Draw Area":
    m = geemap.Map(center=[12.8797, 121.7740], zoom=6)
    m.add_basemap('HYBRID')
    map_output = m.to_streamlit(height=400, key="input_map", bidirectional=True)

    if map_output and isinstance(map_output, dict) and "last_active_drawing" in map_output:
        drawing = map_output["last_active_drawing"]
        if drawing:
            st.session_state["saved_geometry"] = drawing["geometry"]
            st.success("✅ Geometry Captured!")

    if "saved_geometry" in st.session_state:
        try:
            coords = st.session_state["saved_geometry"]["coordinates"]
            roi = ee.Geometry.Polygon(coords)
        except:
            pass

# --- 6. MAIN EXECUTION ---
if st.button("🚀 Run Analysis (Force Render)", type="primary"):
    
    if roi is None:
        st.error("⚠️ Please select a city or draw an area.")
        st.stop()

    st.divider()
    
    try:
        with st.spinner("Processing... (Ignoring cloud limits to force output)"):
            
            common_bands = ['B2', 'B3', 'B4', 'B8', 'B11', 'SCL']

            # --- THE FIX: REMOVED CLOUD FILTER ---
            # We took out .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 80))
            # This ensures we get ALL images, even if they are 100% cloudy.
            # The .median() function will mathematically remove the clouds later.
            
            dataset_old = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
                .filterDate('2017-01-01', '2019-12-31') \
                .filterBounds(roi) \
                .select(common_bands).map(mask_s2_clouds).map(add_indices)

            dataset_new = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
                .filterDate('2023-01-01', '2024-12-30') \
                .filterBounds(roi) \
                .select(common_bands).map(mask_s2_clouds).map(add_indices)

            # Create Composites (The Median "Magic")
            image_old = dataset_old.median().clip(roi)
            image_new = dataset_new.median().clip(roi)

            # --- AI TRAINING ---
            # We sample pixels. If this fails, we catch the error but STILL show the real map.
            try:
                training = image_new.sample(region=roi, scale=30, numPixels=1000)
                clusterer = ee.Clusterer.wekaKMeans(3).train(training)
                classified_old = image_old.cluster(clusterer)
                classified_new = image_new.cluster(clusterer)
                ai_success = True
            except:
                ai_success = False
                st.warning("⚠️ AI Classification failed (too many clouds), but showing Real Images below.")

            # --- 8. RESULTS MAP ---
            st.subheader("Map Visualization")
            
            m_result = geemap.Map()
            m_result.centerObject(roi, 12)
            
            # VISUALIZATION PARAMETERS
            ai_vis = {'min': 0, 'max': 2, 'palette': ['red', 'green', 'blue']}
            real_vis = {'min': 0, 'max': 3000, 'bands': ['B4', 'B3', 'B2']} # True Color
            
            # LAYER 1: The AI Layers (If successful)
            if ai_success:
                m_result.add_layer(classified_old, ai_vis, "2017 AI Analysis")
                m_result.add_layer(classified_new, ai_vis, "2024 AI Analysis")
            
            # LAYER 2: The "Real" Satellite Photos (Backup)
            # These will ALWAYS show up.
            m_result.add_layer(image_old, real_vis, "2017 Real Photo")
            m_result.add_layer(image_new, real_vis, "2024 Real Photo")
            
            m_result.add_layer_control()
            m_result.to_streamlit(height=600, bidirectional=False)
            
            # --- 9. STATISTICS ---
            if ai_success:
                st.write("Calculated Statistics:")
                GREEN_CLASS_ID = 1 
                def get_area(img):
                    mask = img.eq(GREEN_CLASS_ID)
                    area = mask.multiply(ee.Image.pixelArea()).reduceRegion(
                        reducer=ee.Reducer.sum(), geometry=roi, scale=30, maxPixels=1e9
                    )
                    return area.get('cluster').getInfo()

                val_old = get_area(classified_old) or 0
                val_new = get_area(classified_new) or 0
                
                km_old = val_old / 1e6
                km_new = val_new / 1e6
                diff = km_new - km_old

                c1, c2, c3 = st.columns(3)
                c1.metric("2017 Vegetation", f"{km_old:.2f} km²")
                c2.metric("2024 Vegetation", f"{km_new:.2f} km²")
                c3.metric("Difference", f"{diff:.2f} km²", delta=f"{diff:.2f} km²")
            else:
                st.write("Statistics unavailable due to cloud cover.")

    except Exception as e:
        st.error(f"Error: {e}")
