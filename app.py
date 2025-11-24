import streamlit as st
import ee
import geemap.foliumap as geemap
import matplotlib.pyplot as plt
import json
from google.oauth2 import service_account

# --- 1. CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Green Space AI")
st.title("🌍 AI Green Space Analyzer (2017 vs 2024)")
st.markdown("Compare satellite imagery to detect changes in urban green space.")

# --- 2. AUTHENTICATION ---
try:
    if "earth_engine" in st.secrets and "service_account_json" in st.secrets["earth_engine"]:
         key_content = st.secrets["earth_engine"]["service_account_json"]
    else:
         key_content = st.secrets["service_account_json"]
         
    service_account_info = json.loads(key_content, strict=False)
    my_scopes = ['https://www.googleapis.com/auth/earthengine']
    creds = service_account.Credentials.from_service_account_info(service_account_info)
    creds = creds.with_scopes(my_scopes)
    ee.Initialize(creds, project="mystic-curve-479206-q2")
    
except Exception as e:
    st.error(f"Authentication failed: {e}")
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

# --- 4. FIXED SETTINGS ---
CLOUD_LIMIT = 80
GREEN_CLASS_ID = 1

# --- 5. MAP INTERFACE ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("1. Select Region")
    st.info("Zoom to your city and draw a rectangle using the tool on the left.")
    
    m = geemap.Map()
    m.add_basemap('HYBRID')
    
    # --- THE FIX: ADD A UNIQUE KEY ---
    # We add key="satellite_map" so Streamlit doesn't reset it on every click.
    # The map data will automatically be stored in st.session_state["satellite_map"]
    map_output = m.to_streamlit(height=500, key="satellite_map")

# --- 6. ANALYSIS LOGIC ---
if st.button("Run AI Analysis"):
    
    roi = None
    
    # --- CHECK THE KEYED STATE DIRECTLY ---
    # We look directly into the session state using the key we defined above
    data = st.session_state.get("satellite_map", {})
    
    if data and data.get("last_active_drawing"):
        # We found the drawing!
        coords = data["last_active_drawing"]["geometry"]["coordinates"]
        roi = ee.Geometry.Polygon(coords)
        st.success("✅ Drawing detected! Starting analysis...")
    else:
        # Fallback
        st.warning("⚠️ No drawing found. Did you draw a rectangle? Using London (Default).")
        roi = ee.Geometry.Point([-0.12, 51.50]).buffer(10000).bounds()

    with st.spinner("Processing Satellite Data (2017 vs 2024)..."):
        # (Standard Analysis Code)
        common_bands = ['B2', 'B3', 'B4', 'B8', 'B11', 'SCL']
        
        dataset_old = ee.ImageCollection('COPERNICUS/S2_SR') \
            .filterDate('2017-01-01', '2019-12-31') \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', CLOUD_LIMIT)) \
            .filterBounds(roi) \
            .sort('CLOUDY_PIXEL_PERCENTAGE') \
            .limit(30) \
            .select(common_bands) \
            .map(mask_s2_clouds) \
            .map(add_indices)

        dataset_new = ee.ImageCollection('COPERNICUS/S2_SR') \
            .filterDate('2023-01-01', '2024-12-30') \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', CLOUD_LIMIT)) \
            .filterBounds(roi) \
            .sort('CLOUDY_PIXEL_PERCENTAGE') \
            .limit(30) \
            .select(common_bands) \
            .map(mask_s2_clouds) \
            .map(add_indices)

        image_old = dataset_old.median().clip(roi)
        image_new = dataset_new.median().clip(roi)

        training = image_new.sample(region=roi, scale=10, numPixels=5000)
        clusterer = ee.Clusterer.wekaKMeans(3).train(training)
        classified_old = image_old.cluster(clusterer)
        classified_new = image_new.cluster(clusterer)

    # --- 7. RESULTS ---
    st.subheader("Visual Comparison")
    m2 = geemap.Map()
    m2.centerObject(roi, 12)
    vis_params = {'min': 0, 'max': 2, 'palette': ['red', 'green', 'blue']}
    
    left_layer = geemap.ee_tile_layer(classified_old, vis_params, '2017', opacity=0.6)
    right_layer = geemap.ee_tile_layer(classified_new, vis_params, '2024', opacity=0.6)
    m2.split_map(left_layer, right_layer)
    m2.to_streamlit(height=500)

    # Calculate Areas
    def get_area(image, class_id):
        mask = image.eq(class_id)
        area_image = mask.multiply(ee.Image.pixelArea())
        stats = area_image.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=roi,
            scale=10,
            maxPixels=1e10
        )
        return stats.get('cluster').getInfo()

    area_old_sqm = get_area(classified_old, GREEN_CLASS_ID)
    area_new_sqm = get_area(classified_new, GREEN_CLASS_ID)
    
    if area_old_sqm is None: area_old_sqm = 0
    if area_new_sqm is None: area_new_sqm = 0

    area_old_km = area_old_sqm / 1e6
    area_new_km = area_new_sqm / 1e6
    
    colA, colB, colC = st.columns(3)
    colA.metric("2017 Green Space", f"{area_old_km:.2f} km²")
    colB.metric("2024 Green Space", f"{area_new_km:.2f} km²")
    diff = area_new_km - area_old_km
    colC.metric("Net Change", f"{diff:.2f} km²", delta=f"{diff:.2f} km²")

    # Plotting
    st.subheader("2030 Projections")
    years_elapsed = 2024 - 2017
    rate_per_year = diff / years_elapsed if years_elapsed > 0 else 0
    predicted_area_2030 = area_new_km + (rate_per_year * 6)

    fig, ax = plt.subplots(figsize=(10, 6))
    years_hist = [2017, 2024]
    values_hist = [area_old_km, area_new_km]
    years_pred = [2024, 2030]
    values_pred = [area_new_km, predicted_area_2030]

    ax.plot(years_hist, values_hist, marker='o', linestyle='-', color='green', linewidth=3, label='Observed')
    ax.plot(years_pred, values_pred, marker='o', linestyle='--', color='gray', linewidth=2.5, label='Projected')
    
    final_color = 'red' if predicted_area_2030 < area_new_km else 'blue'
    ax.scatter([2030], [predicted_area_2030], color=final_color, s=200, zorder=5)
    
    ax.set_title("Green Space Projection")
    ax.set_ylabel("Area (sq km)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
