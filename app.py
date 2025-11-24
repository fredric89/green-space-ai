import streamlit as st
import ee
import geemap.foliumap as geemap
import json
from google.oauth2 import service_account

# --- 1. CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Green Space AI")
st.title("🌍 AI Green Space Analyzer (Custom Draw)")

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
    mask = scl.neq(3).And(scl.neq(8)).And(scl.neq(9))
    return image.updateMask(mask).divide(10000)

def add_indices(image):
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    return image.addBands(ndvi)

# --- 4. DRAWING INTERFACE ---
col1, col2 = st.columns([3, 1])

with col1:
    st.info("1. Draw a box on the map. 2. Wait for '✅ Geometry Captured'. 3. Click Run.")
    
    # FIX: Initialize empty map (No basemap argument to avoid KeyError)
    m = geemap.Map(center=[12.8797, 121.7740], zoom=6)
    
    # FIX: Manually add Esri Satellite Tiles (The Bulletproof Way)
    esri_url = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'
    m.add_tile_layer(esri_url, name="Esri Satellite", attribution="Esri")
    
    # Render Map
    map_output = m.to_streamlit(height=500, key="input_map", bidirectional=True)

    # Save drawing
    if map_output is not None and isinstance(map_output, dict) and "last_active_drawing" in map_output:
        drawing = map_output["last_active_drawing"]
        if drawing:
            st.session_state["saved_geometry"] = drawing["geometry"]

with col2:
    if "saved_geometry" in st.session_state:
        st.success("✅ Geometry Captured!", icon="💾")
        st.write("Ready to analyze.")
    else:
        st.warning("Please draw a box.", icon="✏️")

# --- 5. MAIN EXECUTION ---
if st.button("🚀 Run AI Analysis", type="primary"):
    
    roi = None
    
    # Retrieve ROI
    if "saved_geometry" in st.session_state:
        try:
            coords = st.session_state["saved_geometry"]["coordinates"]
            roi = ee.Geometry.Polygon(coords)
        except Exception as e:
            st.error(f"Error reading drawing: {e}")
            st.stop()
    else:
        st.error("⚠️ No area selected! Please draw a box on the map first.")
        st.stop()

    st.divider()
    
    try:
        with st.spinner("Processing Satellite Data..."):
            
            # CONFIG
            collection_id = 'COPERNICUS/S2_SR_HARMONIZED'
            common_bands = ['B2', 'B3', 'B4', 'B8', 'SCL']

            # 1. Fetch Data
            dataset_old = ee.ImageCollection(collection_id) \
                .filterDate('2017-01-01', '2019-12-31') \
                .filterBounds(roi) \
                .select(common_bands).map(mask_s2_clouds).map(add_indices)

            dataset_new = ee.ImageCollection(collection_id) \
                .filterDate('2023-01-01', '2024-12-30') \
                .filterBounds(roi) \
                .select(common_bands).map(mask_s2_clouds).map(add_indices)

            # 2. Check Data
            count_new = dataset_new.size().getInfo()
            if count_new == 0:
                 st.error("❌ No satellite images found for this area. Try drawing a larger box.")
                 st.stop()
            
            # 3. Create Composites
            image_old = dataset_old.median().clip(roi)
            image_new = dataset_new.median().clip(roi)

            # 4. AI Classification
            training = image_new.sample(region=roi, scale=30, numPixels=1000) 
            
            if training.size().getInfo() == 0:
                 st.warning("⚠️ High cloud cover detected. AI results may be inaccurate.")
            
            clusterer = ee.Clusterer.wekaKMeans(3).train(training)
            classified_old = image_old.cluster(clusterer)
            classified_new = image_new.cluster(clusterer)

            # --- 6. RESULTS MAP ---
            st.subheader("Analysis Results")
            
            # FIX: Initialize Result Map with Manual Esri Layer
            m_result = geemap.Map()
            m_result.add_tile_layer(esri_url, name="Esri Satellite", attribution="Esri")
            m_result.centerObject(roi, 13)
            
            # VISUALIZATION
            real_vis = {'min': 0, 'max': 3000, 'bands': ['B4', 'B3', 'B2']}
            ai_vis = {'min': 0, 'max': 2, 'palette': ['red', 'green', 'blue']}
            
            # Add Layers
            m_result.add_layer(image_new, real_vis, "2024 Real Photo (Ref)")
            m_result.add_layer(classified_old, ai_vis, "2017 AI Map")
            m_result.add_layer(classified_new, ai_vis, "2024 AI Map")
            
            m_result.add_layer_control()
            
            # Force Render
            m_result.to_streamlit(height=600, bidirectional=False)
            
            # --- 7. STATISTICS ---
            st.write("### Statistics")
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

    except Exception as e:
        st.error(f"Analysis Failed: {e}")
