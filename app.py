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

# Initialize session state
if "saved_geometry" not in st.session_state:
    st.session_state.saved_geometry = None
if "analysis_run" not in st.session_state:
    st.session_state.analysis_run = False

# --- 4. DRAWING INTERFACE ---
col1, col2 = st.columns([3, 1])

with col1:
    st.info("1. Draw a rectangle using the toolbar. 2. Wait for '✅ Geometry Captured'. 3. Click Run.")
    
    # Initialize map with drawing enabled
    m = geemap.Map(
        center=[12.8797, 121.7740], 
        zoom=6,
        draw_export=True,
        locate_control=True,
    )
    
    # Add base layers
    esri_url = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'
    m.add_tile_layer(esri_url, name="Esri Satellite", attribution="Esri")
    
    # Add default rectangle if no geometry exists (for testing)
    if st.session_state.saved_geometry is None:
        # Add a default rectangle around Manila for testing
        default_roi = ee.Geometry.Rectangle([120.9, 14.5, 121.1, 14.7])
        m.add_layer(default_roi, {'color': 'red'}, "Default Area")
    
    # Add existing geometry if it exists
    if st.session_state.saved_geometry:
        existing_roi = ee.Geometry.Polygon(st.session_state.saved_geometry["coordinates"])
        m.add_layer(existing_roi, {'color': 'blue', 'fillColor': '00000000'}, "Selected Area")
    
    # Render Map with drawing enabled
    map_output = m.to_streamlit(height=500, key="input_map", bidirectional=True)

    # Process drawing output
    if map_output and "last_active_drawing" in map_output:
        drawing = map_output["last_active_drawing"]
        if drawing and "geometry" in drawing:
            # Only update if it's a new drawing
            new_coords = str(drawing["geometry"]["coordinates"])
            if (st.session_state.saved_geometry is None or 
                str(st.session_state.saved_geometry["coordinates"]) != new_coords):
                st.session_state.saved_geometry = drawing["geometry"]
                st.session_state.analysis_run = False
                st.rerun()

with col2:
    if st.session_state.saved_geometry:
        st.success("✅ Geometry Captured!", icon="💾")
        
        # Show geometry info
        coords = st.session_state.saved_geometry["coordinates"]
        area_info = f"Area vertices: {len(coords[0])}"
        st.write(area_info)
        
        # Add a clear button
        if st.button("🗑️ Clear Drawing", type="secondary"):
            st.session_state.saved_geometry = None
            st.session_state.analysis_run = False
            st.rerun()
    else:
        st.warning("Please draw a rectangle using the drawing tools in the top-right corner of the map.", icon="✏️")
        st.info("💡 Click the rectangle tool, then draw on the map.")

# --- 5. MAIN EXECUTION ---
if st.session_state.saved_geometry:
    if st.button("🚀 Run AI Analysis", type="primary"):
        st.session_state.analysis_run = True
        st.rerun()

# Only run analysis if we have geometry and the button was clicked
if st.session_state.saved_geometry and st.session_state.analysis_run:
    
    try:
        # Convert geometry with better error handling
        coords = st.session_state.saved_geometry["coordinates"]
        roi = ee.Geometry.Polygon(coords)
        
        # Show analysis area info
        st.divider()
        st.subheader("📊 Analysis in Progress...")
        
        bounds = roi.bounds().getInfo()
        st.write(f"**Analysis Area:** {bounds}")
        
    except Exception as e:
        st.error(f"Error reading drawing: {e}")
        st.stop()

    try:
        with st.spinner("Processing Satellite Data... This may take a few moments."):
            
            # Use more flexible date ranges and better filtering
            collection_id = 'COPERNICUS/S2_SR_HARMONIZED'
            
            # Define bands including vegetation indices
            bands = ['B2', 'B3', 'B4', 'B8', 'SCL']
            
            # 1. Fetch Data with better filtering
            def get_collection(start_date, end_date):
                return (ee.ImageCollection(collection_id)
                        .filterDate(start_date, end_date)
                        .filterBounds(roi)
                        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 50))  # More lenient cloud filter
                        .select(bands)
                        .map(mask_s2_clouds)
                        .map(add_indices))
            
            dataset_old = get_collection('2018-01-01', '2019-12-31')
            dataset_new = get_collection('2023-01-01', '2024-07-31')
            
            # 2. Check Data Availability
            count_old = dataset_old.size().getInfo()
            count_new = dataset_new.size().getInfo()
            
            st.write(f"**Images Found:** {count_old} (2018-2019), {count_new} (2023-2024)")
            
            if count_new == 0 and count_old == 0:
                st.error("❌ No satellite images found for this area. Try a different location or larger area.")
                st.stop()
            
            if count_old == 0:
                st.warning("⚠️ No historical images found. Using recent data for both comparisons.")
                dataset_old = dataset_new
            
            # 3. Create composites
            image_old = dataset_old.median().clip(roi)
            image_new = dataset_new.median().clip(roi)
            
            # 4. AI Classification
            # Use simpler approach for reliability
            training = image_new.sample(region=roi, scale=30, numPixels=500, seed=42)
            
            if training.size().getInfo() == 0:
                st.warning("⚠️ Limited data for training. Results may be approximate.")
                # Try with smaller sample size
                training = image_new.sample(region=roi, scale=30, numPixels=100, seed=42)
            
            clusterer = ee.Clusterer.wekaKMeans(3).train(training)
            classified_old = image_old.cluster(clusterer)
            classified_new = image_new.cluster(clusterer)

            # --- 6. RESULTS MAP ---
            st.subheader("🌿 Analysis Results")
            
            # Create result map
            m_result = geemap.Map()
            esri_url = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'
            m_result.add_tile_layer(esri_url, name="Esri Satellite", attribution="Esri")
            m_result.centerObject(roi, 12)
            
            # Visualization parameters
            real_vis = {'min': 0, 'max': 0.3, 'bands': ['B4', 'B3', 'B2']}
            ai_vis = {'min': 0, 'max': 2, 'palette': ['red', 'blue', 'green']}
            
            # Add layers
            m_result.add_layer(image_new, real_vis, "2023-2024 Satellite")
            m_result.add_layer(classified_new, ai_vis, "2024 AI Classification", shown=False)
            m_result.add_layer(classified_old, ai_vis, "2018-2019 AI Classification", shown=False)
            m_result.add_layer(roi, {'color': 'yellow'}, "Analysis Area")
            
            m_result.add_layer_control()
            
            # Display the map
            st.write("### Interactive Results Map")
            st.info("💡 Use the layer control (top-right) to toggle different views")
            m_result.to_streamlit(height=600, key="result_map")
            
            # --- 7. STATISTICS ---
            st.write("### 📊 Vegetation Statistics")
            
            # Simple NDVI-based vegetation calculation
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
                result = area.get('NDVI').getInfo()
                return result if result else 0
            
            try:
                veg_area_old = calculate_vegetation_area(image_old)
                veg_area_new = calculate_vegetation_area(image_new)
                
                km_old = veg_area_old / 1e6
                km_new = veg_area_new / 1e6
                diff = km_new - km_old
                
                # Display metrics
                c1, c2, c3 = st.columns(3)
                c1.metric("2018-2019 Vegetation", f"{km_old:.2f} km²")
                c2.metric("2023-2024 Vegetation", f"{km_new:.2f} km²")
                c3.metric("Change", f"{diff:+.2f} km²", delta=f"{diff:+.2f} km²")
                
                # Interpretation
                if diff > 0.01:  # Significant increase
                    st.success(f"🎉 Vegetation increased by {diff:.2f} km²")
                elif diff < -0.01:  # Significant decrease
                    st.error(f"📉 Vegetation decreased by {abs(diff):.2f} km²")
                else:
                    st.info("📊 No significant change in vegetation area")
                    
            except Exception as stats_error:
                st.warning(f"Could not calculate precise statistics: {stats_error}")

    except Exception as e:
        st.error(f"Analysis Failed: {str(e)}")
        st.info("💡 Tips: Try drawing a larger area, or select a different location with less cloud cover.")

# Footer
st.divider()
st.write("---")
st.write("🌱 **Green Space AI** - Monitor vegetation changes using satellite imagery and AI")
