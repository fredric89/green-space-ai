import streamlit as st
import ee
import geemap.foliumap as geemap
import json
from google.oauth2 import service_account

# --- 1. CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Green Space AI")
st.title("🌍 AI Green Space Analyzer")

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

# --- 4. MAP INTERFACE ---
col1, col2 = st.columns([3, 1])

with col1:
    st.info("1. Draw a box. 2. Wait for '✅ Saved'. 3. Click Run.")
    
    m = geemap.Map()
    m.add_basemap('HYBRID')
    
    # RENDER MAP
    map_output = m.to_streamlit(height=500, bidirectional=True)

    # SAFETY VAULT: Save drawing to memory
    if isinstance(map_output, dict) and "last_active_drawing" in map_output:
        drawing = map_output["last_active_drawing"]
        if drawing is not None:
            st.session_state["saved_geometry"] = drawing["geometry"]

with col2:
    if "saved_geometry" in st.session_state:
        st.success("✅ Region Saved!", icon="💾")
    else:
        st.warning("Draw a box on the map.", icon="✏️")

# --- 5. ANALYSIS LOGIC ---
if st.button("Run AI Analysis", type="primary"):
    
    roi = None
    
    if "saved_geometry" in st.session_state:
        try:
            coords = st.session_state["saved_geometry"]["coordinates"]
            roi = ee.Geometry.Polygon(coords)
        except Exception as e:
            st.error(f"Coordinates Error: {e}")
            st.stop()
    else:
        st.error("⚠️ No drawing found! Please draw a box.")
        st.stop()

    with st.spinner("Analyzing... (This may take 10-20 seconds)"):
        # Fixed Parameters
        CLOUD_LIMIT = 80
        common_bands = ['B2', 'B3', 'B4', 'B8', 'B11', 'SCL']

        # 1. Fetch Data (Harmonized)
        dataset_old = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterDate('2017-01-01', '2019-12-31') \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', CLOUD_LIMIT)) \
            .filterBounds(roi).sort('CLOUDY_PIXEL_PERCENTAGE').limit(30) \
            .select(common_bands).map(mask_s2_clouds).map(add_indices)

        dataset_new = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterDate('2023-01-01', '2024-12-30') \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', CLOUD_LIMIT)) \
            .filterBounds(roi).sort('CLOUDY_PIXEL_PERCENTAGE').limit(30) \
            .select(common_bands).map(mask_s2_clouds).map(add_indices)
        
        # Check if we actually found images
        count_new = dataset_new.size().getInfo()
        if count_new == 0:
             st.error("❌ No clear images found for this region/date. Try a different area.")
             st.stop()

        image_old = dataset_old.median().clip(roi)
        image_new = dataset_new.median().clip(roi)

        # 2. Train AI (K-Means)
        # We sample pixels to teach the AI what 'Water', 'Green', and 'Urban' look like
        training = image_new.sample(region=roi, scale=20, numPixels=1000) # Reduced scale for speed
        
        # SAFETY CHECK: Did we catch any valid pixels?
        if training.size().getInfo() == 0:
             st.error("❌ The AI couldn't find valid pixels to learn from (likely too many clouds).")
             st.stop()
             
        clusterer = ee.Clusterer.wekaKMeans(3).train(training)
        classified_old = image_old.cluster(clusterer)
        classified_new = image_new.cluster(clusterer)

    # --- 6. RESULTS (SIMPLIFIED VISUALIZATION) ---
    st.divider()
    st.subheader("Results")
    
    # We use a standard map with Layer Control (Toggle) instead of Split Map
    # This is much less likely to break.
    m2 = geemap.Map()
    m2.centerObject(roi, 13)
    
    vis_params = {'min': 0, 'max': 2, 'palette': ['red', 'green', 'blue']}
    
    # Add Layers
    m2.add_layer(classified_old, vis_params, "2017 AI Classification")
    m2.add_layer(classified_new, vis_params, "2024 AI Classification")
    
    # Add standard Layer Control so you can toggle them on/off
    m2.add_layer_control()
    
    m2.to_streamlit(height=500)

    # --- 7. METRICS ---
    st.caption("Calculation in progress...")
    GREEN_CLASS_ID = 1
    
    def get_area(img):
        mask = img.eq(GREEN_CLASS_ID)
        area = mask.multiply(ee.Image.pixelArea()).reduceRegion(
            reducer=ee.Reducer.sum(), geometry=roi, scale=10, maxPixels=1e10
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
    c3.metric("Change", f"{diff:.2f} km²", delta=f"{diff:.2f} km²")
