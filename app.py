import streamlit as st
import ee
import geemap.foliumap as geemap
import json
from google.oauth2 import service_account

# --- 1. CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Green Space AI")
st.title("🌍 AI Green Space Analyzer")
st.markdown("Compare 2017 vs 2024 satellite imagery.")

# --- 2. AUTHENTICATION ---
try:
    # Handle different secret structures safely
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

# --- 4. MAP INTERFACE & "SAFETY VAULT" LOGIC ---
col1, col2 = st.columns([3, 1])

with col1:
    st.info("1. Draw a box. 2. Wait for the '✅ Saved' message. 3. Click Run.")
    
    # Initialize Map
    m = geemap.Map()
    m.add_basemap('HYBRID')
    
    # Render Map and get output
    # We use a static key to try and keep the map stable
    map_output = m.to_streamlit(height=500, key="satellite_map")

    # --- THE SAFETY VAULT ---
    # Every time the script runs (even before you click the button),
    # we check if there is a drawing. If yes, we SAVE it to permanent memory.
    if map_output is not None and "last_active_drawing" in map_output:
        drawing = map_output["last_active_drawing"]
        if drawing is not None:
            # Save to session state
            st.session_state["saved_geometry"] = drawing["geometry"]

with col2:
    # Status Indicator
    if "saved_geometry" in st.session_state:
        st.success(f"✅ Region Saved!", icon="💾")
        st.write("Ready to analyze.")
    else:
        st.warning("Waiting for drawing...", icon="✏️")

# --- 5. ANALYSIS LOGIC ---
if st.button("Run AI Analysis", type="primary"):
    
    roi = None
    
    # We look inside the "Safety Vault", NOT at the map directly.
    # This prevents the "No drawing found" error if the map resets.
    if "saved_geometry" in st.session_state:
        try:
            coords = st.session_state["saved_geometry"]["coordinates"]
            roi = ee.Geometry.Polygon(coords)
        except Exception as e:
            st.error(f"Error reading coordinates: {e}")
            st.stop()
    else:
        st.error("⚠️ No drawing saved! Please draw a box first.")
        # Fallback to London just so it doesn't crash
        roi = ee.Geometry.Point([-0.12, 51.50]).buffer(5000).bounds()

    with st.spinner("Processing..."):
        # Fixed Parameters
        CLOUD_LIMIT = 80
        GREEN_CLASS_ID = 1
        common_bands = ['B2', 'B3', 'B4', 'B8', 'B11', 'SCL']

        # 1. Fetch Data
        dataset_old = ee.ImageCollection('COPERNICUS/S2_SR') \
            .filterDate('2017-01-01', '2019-12-31') \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', CLOUD_LIMIT)) \
            .filterBounds(roi).sort('CLOUDY_PIXEL_PERCENTAGE').limit(30) \
            .select(common_bands).map(mask_s2_clouds).map(add_indices)

        dataset_new = ee.ImageCollection('COPERNICUS/S2_SR') \
            .filterDate('2023-01-01', '2024-12-30') \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', CLOUD_LIMIT)) \
            .filterBounds(roi).sort('CLOUDY_PIXEL_PERCENTAGE').limit(30) \
            .select(common_bands).map(mask_s2_clouds).map(add_indices)

        # 2. Process
        image_old = dataset_old.median().clip(roi)
        image_new = dataset_new.median().clip(roi)

        # 3. Classify
        training = image_new.sample(region=roi, scale=10, numPixels=5000)
        clusterer = ee.Clusterer.wekaKMeans(3).train(training)
        classified_old = image_old.cluster(clusterer)
        classified_new = image_new.cluster(clusterer)

    # --- 6. RESULTS ---
    st.divider()
    st.subheader("Results")
    
    # Display Result Map
    m2 = geemap.Map()
    m2.centerObject(roi, 13)
    vis = {'min': 0, 'max': 2, 'palette': ['red', 'green', 'blue']}
    m2.split_map(
        geemap.ee_tile_layer(classified_old, vis, '2017'),
        geemap.ee_tile_layer(classified_new, vis, '2024')
    )
    m2.to_streamlit(height=500)

    # Calculate Metrics
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

    # Metrics Row
    c1, c2, c3 = st.columns(3)
    c1.metric("2017 Green Space", f"{km_old:.2f} km²")
    c2.metric("2024 Green Space", f"{km_new:.2f} km²")
    c3.metric("Change", f"{diff:.2f} km²", delta=f"{diff:.2f} km²")
