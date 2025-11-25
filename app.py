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
    """Improved cloud masking"""
    qa = image.select('SCL')
    # Keep only confident vegetation, bare soil, and water
    cloud_mask = qa.neq(3).And(qa.neq(8)).And(qa.neq(9))  # Remove clouds, cloud shadows, cirrus
    return image.updateMask(cloud_mask).divide(10000)

def add_indices(image):
    """Add vegetation indices"""
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    # Add more indices for better classification
    ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')
    return image.addBands([ndvi, ndwi])

# --- 4. DRAWING INTERFACE ---
col1, col2 = st.columns([3, 1])

with col1:
    st.info("1. Draw a box. 2. Wait for '✅ Geometry Captured'. 3. Click Run.")
    
    # Initialize map
    m = geemap.Map(center=[12.8797, 121.7740], zoom=6)
    esri_url = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'
    m.add_tile_layer(esri_url, name="Esri Satellite", attribution="Esri")
    
    # Add drawing control
    m.add_draw_control()
    
    # Render Map
    map_output = m.to_streamlit(height=500, key="input_map", bidirectional=True)

    # Save drawing with better error handling
    if map_output and "last_active_drawing" in map_output:
        drawing = map_output["last_active_drawing"]
        if drawing and "geometry" in drawing:
            st.session_state["saved_geometry"] = drawing["geometry"]
            st.rerun()  # Force refresh to update the status

with col2:
    if "saved_geometry" in st.session_state:
        st.success("✅ Geometry Captured!", icon="💾")
        st.write("Ready to analyze.")
        
        # Show geometry info for debugging
        coords = st.session_state["saved_geometry"]["coordinates"]
        st.write(f"Area vertices: {len(coords[0])}")
    else:
        st.warning("Please draw a box.", icon="✏️")

# --- 5. MAIN EXECUTION ---
if st.button("🚀 Run AI Analysis", type="primary"):
    
    if "saved_geometry" not in st.session_state:
        st.error("⚠️ No area selected! Please draw a box.")
        st.stop()
    
    try:
        # Convert geometry with better error handling
        coords = st.session_state["saved_geometry"]["coordinates"]
        roi = ee.Geometry.Polygon(coords)
        
        # Debug: Show ROI bounds
        bounds = roi.bounds().getInfo()
        st.write(f"Analysis Area: {bounds}")
        
    except Exception as e:
        st.error(f"Error reading drawing: {e}")
        st.stop()

    st.divider()
    
    try:
        with st.spinner("Processing Satellite Data..."):
            
            # Use more flexible date ranges and better filtering
            collection_id = 'COPERNICUS/S2_SR_HARMONIZED'
            
            # Define bands including vegetation indices
            bands = ['B2', 'B3', 'B4', 'B8', 'SCL']
            
            # 1. Fetch Data with better filtering
            def get_collection(start_date, end_date):
                return (ee.ImageCollection(collection_id)
                        .filterDate(start_date, end_date)
                        .filterBounds(roi)
                        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
                        .select(bands)
                        .map(mask_s2_clouds)
                        .map(add_indices))
            
            dataset_old = get_collection('2018-01-01', '2019-12-31')
            dataset_new = get_collection('2023-01-01', '2024-07-31')  # Updated to ensure data availability
            
            # 2. Check Data Availability
            count_old = dataset_old.size().getInfo()
            count_new = dataset_new.size().getInfo()
            
            st.write(f"Found {count_old} images (2018-2019)")
            st.write(f"Found {count_new} images (2023-2024)")
            
            if count_new == 0:
                st.error("❌ No recent satellite images found. Try a different location or larger area.")
                st.stop()
            
            if count_old == 0:
                st.warning("⚠️ No historical images found. Using recent data for both comparisons.")
                dataset_old = dataset_new  # Fallback to same period
            
            # 3. Create composites
            image_old = dataset_old.median().clip(roi)
            image_new = dataset_new.median().clip(roi)
            
            # 4. Improved AI Classification using NDVI for training
            # Use NDVI to identify potential vegetation samples
            ndvi_old = image_old.select('NDVI')
            ndvi_new = image_new.select('NDVI')
            
            # Create training data based on NDVI thresholds
            def get_training_data(image):
                # Sample from high NDVI (vegetation), medium (mixed), low (non-vegetation)
                vegetation = image.select(['B4', 'B3', 'B2', 'B8', 'NDVI']).updateMask(image.select('NDVI').gt(0.4))
                non_veg = image.select(['B4', 'B3', 'B2', 'B8', 'NDVI']).updateMask(image.select('NDVI').lt(0.2))
                mixed = image.select(['B4', 'B3', 'B2', 'B8', 'NDVI']).updateMask(
                    image.select('NDVI').gte(0.2).And(image.select('NDVI').lte(0.4))
                
                # Sample from each class
                samples = ee.FeatureCollection([
                    vegetation.sample(region=roi, scale=30, numPixels=100, seed=1),
                    non_veg.sample(region=roi, scale=30, numPixels=100, seed=2),
                    mixed.sample(region=roi, scale=30, numPixels=100, seed=3)
                ]).flatten()
                
                return samples

            training = get_training_data(image_new)
            
            if training.size().getInfo() == 0:
                st.warning("⚠️ Using fallback sampling method.")
                training = image_new.sample(region=roi, scale=30, numPixels=500)
            
            # Train classifier
            clusterer = ee.Clusterer.wekaKMeans(3).train(training)
            
            # Classify both periods
            classified_old = image_old.cluster(clusterer)
            classified_new = image_new.cluster(clusterer)

            # --- 6. RESULTS MAP ---
            st.subheader("🌿 Analysis Results")
            
            # Create result map
            m_result = geemap.Map()
            esri_url = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y/{x}'
            m_result.add_tile_layer(esri_url, name="Esri Satellite", attribution="Esri")
            m_result.centerObject(roi, 12)
            
            # Visualization parameters
            real_vis = {'min': 0, 'max': 0.3, 'bands': ['B4', 'B3', 'B2']}
            ai_vis = {'min': 0, 'max': 2, 'palette': ['red', 'blue', 'green']}  # Adjusted colors
            
            # Add layers in logical order
            m_result.add_layer(image_new, real_vis, "2023-2024 Satellite")
            m_result.add_layer(classified_new, ai_vis, "2024 AI Classification", shown=False)
            m_result.add_layer(classified_old, ai_vis, "2018-2019 AI Classification", shown=False)
            
            m_result.add_layer_control()
            
            # Display the map
            st.write("### Interactive Map")
            m_result.to_streamlit(height=600, key="result_map")
            
            # --- 7. STATISTICS ---
            st.write("### 📊 Vegetation Statistics")
            
            # Improved vegetation detection using NDVI
            def calculate_vegetation_area(image):
                ndvi = image.select('NDVI')
                # Consider pixels with NDVI > 0.3 as vegetation
                vegetation_mask = ndvi.gt(0.3)
                area = vegetation_mask.multiply(ee.Image.pixelArea()).reduceRegion(
                    reducer=ee.Reducer.sum(), 
                    geometry=roi, 
                    scale=30, 
                    maxPixels=1e9,
                    bestEffort=True
                )
                return area.get('NDVI')
            
            try:
                veg_area_old = calculate_vegetation_area(image_old).getInfo() or 0
                veg_area_new = calculate_vegetation_area(image_new).getInfo() or 0
                
                km_old = veg_area_old / 1e6
                km_new = veg_area_new / 1e6
                diff = km_new - km_old
                
                # Display metrics
                c1, c2, c3 = st.columns(3)
                c1.metric("2018-2019 Vegetation", f"{km_old:.2f} km²")
                c2.metric("2023-2024 Vegetation", f"{km_new:.2f} km²")
                c3.metric("Change", f"{diff:+.2f} km²", delta=f"{diff:+.2f} km²")
                
                # Interpretation
                if diff > 0:
                    st.success(f"🎉 Vegetation increased by {diff:.2f} km² ({diff/km_old*100:.1f}%)")
                elif diff < 0:
                    st.error(f"📉 Vegetation decreased by {abs(diff):.2f} km² ({abs(diff)/km_old*100:.1f}%)")
                else:
                    st.info("📊 No significant change in vegetation area")
                    
            except Exception as stats_error:
                st.warning(f"Could not calculate precise statistics: {stats_error}")

    except Exception as e:
        st.error(f"Analysis Failed: {str(e)}")
        st.info("💡 Tips: Try drawing a larger area, or select a different location with less cloud cover.")
