import streamlit as st
import ee
import geemap.foliumap as geemap
import json
from google.oauth2 import service_account

# --- 1. CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Green Space AI - PH Edition")
st.title("🌍 AI Green Space Analyzer (Philippines Edition)")

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
    mask = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(1))
    return image.updateMask(mask).divide(10000)

def add_indices(image):
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    ndbi = image.normalizedDifference(['B11', 'B8']).rename('NDBI')
    return image.addBands([ndvi, ndbi])

# --- 4. INPUT SECTION (SIDEBAR) ---
with st.sidebar:
    st.header("1. Settings")
    
    # MODE SELECTOR
    analysis_mode = st.radio("Select Mode:", ["Use Test City (Stable)", "Draw Custom Area (Experimental)"])
    
    roi = None
    
    if analysis_mode == "Use Test City (Stable)":
        # UPDATED CITIES HERE
        city = st.selectbox("Pick a Location:", ["Manila", "Surigao City", "Laguna Province", "Quezon City"])
        
        if city == "Manila":
            # Manila Coordinates [Lon, Lat]
            roi = ee.Geometry.Point([120.9842, 14.5995]).buffer(5000).bounds()
            
        elif city == "Surigao City":
            # Surigao Coordinates
            roi = ee.Geometry.Point([125.4933, 9.7828]).buffer(6000).bounds()
            
        elif city == "Laguna Province":
            # Centered near Calamba/Los Baños with a larger buffer (15km) to capture more province area
            roi = ee.Geometry.Point([121.25, 14.20]).buffer(15000).bounds()
            
        elif city == "Quezon City":
            # QC Coordinates
            roi = ee.Geometry.Point([121.0437, 14.6760]).buffer(7000).bounds()
            
        st.success(f"Selected: {city}")

    else:
        st.info("Draw a box on the map below, then click the button.")

# --- 5. DRAWING INTERFACE (Custom Mode) ---
if analysis_mode == "Draw Custom Area (Experimental)":
    st.subheader("Draw Area")
    # Centered on Philippines by default
    m = geemap.Map(center=[12.8797, 121.7740], zoom=6)
    m.add_basemap('HYBRID')
    
    map_output = m.to_streamlit(height=400, key="input_map", bidirectional=True)

    if map_output is not None and isinstance(map_output, dict) and "last_active_drawing" in map_output:
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

# --- 6. RUN BUTTON ---
if st.button("🚀 Run AI Analysis", type="primary"):
    
    if roi is None:
        st.error("⚠️ No area selected! Please pick a city or draw a box.")
        st.stop()

    st.divider()
    
    # --- 7. ANALYSIS LOGIC ---
    try:
        with st.spinner("Processing satellite data..."):
            
            # Data Config
            CLOUD_LIMIT = 80
            common_bands = ['B2', 'B3', 'B4', 'B8', 'B11', 'SCL']

            # Load Collections (Harmonized)
            dataset_old = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
                .filterDate('2017-01-01', '2019-12-31') \
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', CLOUD_LIMIT)) \
                .filterBounds(roi).sort('CLOUDY_PIXEL_PERCENTAGE').limit(20) \
                .select(common_bands).map(mask_s2_clouds).map(add_indices)

            dataset_new = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
                .filterDate('2023-01-01', '2024-12-30') \
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', CLOUD_LIMIT)) \
                .filterBounds(roi).sort('CLOUDY_PIXEL_PERCENTAGE').limit(20) \
                .select(common_bands).map(mask_s2_clouds).map(add_indices)
            
            # Check for empty data
            if dataset_new.size().getInfo() == 0:
                st.error("No clear images found. Try a different city or drawing.")
                st.stop()

            image_old = dataset_old.median().clip(roi)
            image_new = dataset_new.median().clip(roi)

            # Unsupervised Classification
            training = image_new.sample(region=roi, scale=30, numPixels=1000)
            
            if training.size().getInfo() == 0:
                 st.error("Could not sample pixels (too cloudy).")
                 st.stop()
                 
            clusterer = ee.Clusterer.wekaKMeans(3).train(training)
            classified_old = image_old.cluster(clusterer)
            classified_new = image_new.cluster(clusterer)

            # --- 8. RESULTS DISPLAY ---
            st.subheader("Analysis Results")
            
            m_result = geemap.Map()
            m_result.centerObject(roi, 12)
            
            vis = {'min': 0, 'max': 2, 'palette': ['red', 'green', 'blue']}
            
            m_result.add_layer(classified_old, vis, "2017 Classification")
            m_result.add_layer(classified_new, vis, "2024 Classification")
            m_result.add_layer_control()
            
            # bidirectional=False fixes the disappearance bug
            m_result.to_streamlit(height=500, bidirectional=False)

            # --- 9. STATISTICS ---
            st.write("Calculating Stats...")
            
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
            c1.metric("2017 Green Space", f"{km_old:.2f} km²")
            c2.metric("2024 Green Space", f"{km_new:.2f} km²")
            c3.metric("Difference", f"{diff:.2f} km²", delta=f"{diff:.2f} km²")

    except Exception as e:
        st.error(f"Analysis Failed: {e}")
