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
    cloud_mask = qa.neq(3).And(qa.neq(8)).And(qa.neq(9))
    return image.updateMask(cloud_mask).divide(10000)

def add_indices(image):
    """Add vegetation indices"""
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    return image.addBands(ndvi)

# Initialize session state
if "saved_geometry" not in st.session_state:
    st.session_state.saved_geometry = None
if "analysis_run" not in st.session_state:
    st.session_state.analysis_run = False

# --- 4. DRAWING INTERFACE ---
col1, col2 = st.columns([3, 1])

with col1:
    st.info("1. Draw a rectangle using the toolbar. 2. Wait for '✅ Geometry Captured'. 3. Click Run.")
    
    # Initialize map
    m = geemap.Map(center=[14.5995, 120.9842], zoom=10)  # Centered on Manila
    
    # Add base layer using a different approach
    m.add_basemap("SATELLITE")
    
    # Render Map with drawing enabled
    map_output = m.to_streamlit(height=500, key="input_map", bidirectional=True)

    # Process drawing output
    if map_output and "last_active_drawing" in map_output:
        drawing = map_output["last_active_drawing"]
        if drawing and "geometry" in drawing:
            new_coords = str(drawing["geometry"]["coordinates"])
            if (st.session_state.saved_geometry is None or 
                str(st.session_state.saved_geometry["coordinates"]) != new_coords):
                st.session_state.saved_geometry = drawing["geometry"]
                st.session_state.analysis_run = False
                st.rerun()

with col2:
    if st.session_state.saved_geometry:
        st.success("✅ Geometry Captured!", icon="💾")
        
        # Add a clear button
        if st.button("🗑️ Clear Drawing", type="secondary"):
            st.session_state.saved_geometry = None
            st.session_state.analysis_run = False
            st.rerun()
    else:
        st.warning("Please draw a rectangle using the drawing tools.", icon="✏️")

# --- 5. MAIN EXECUTION ---
if st.session_state.saved_geometry:
    if st.button("🚀 Run AI Analysis", type="primary"):
        st.session_state.analysis_run = True

# Only run analysis if we have geometry and the button was clicked
if st.session_state.saved_geometry and st.session_state.analysis_run:
    
    try:
        # Convert geometry
        coords = st.session_state.saved_geometry["coordinates"]
        roi = ee.Geometry.Polygon(coords)
        
        st.divider()
        st.subheader("📊 Analysis in Progress...")
        
    except Exception as e:
        st.error(f"Error reading drawing: {e}")
        st.stop()

    try:
        with st.spinner("Processing Satellite Data... This may take 30-60 seconds."):
            
            # Use simpler approach with guaranteed data
            collection_id = 'COPERNICUS/S2_SR_HARMONIZED'
            bands = ['B2', 'B3', 'B4', 'B8', 'SCL']
            
            # Get collections with more lenient filters
            def get_collection(start_date, end_date):
                return (ee.ImageCollection(collection_id)
                        .filterDate(start_date, end_date)
                        .filterBounds(roi)
                        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 80))
                        .select(bands)
                        .map(mask_s2_clouds)
                        .map(add_indices))
            
            # Use seasonal dates for better data availability
            dataset_old = get_collection('2018-03-01', '2018-05-31')  # Spring
            dataset_new = get_collection('2024-03-01', '2024-05-31')  # Spring
            
            # Check data availability
            count_old = dataset_old.size().getInfo()
            count_new = dataset_new.size().getInfo()
            
            st.write(f"**Images found:** {count_old} (2018), {count_new} (2024)")
            
            if count_new == 0:
                st.error("❌ No recent images found. Trying alternative dates...")
                dataset_new = get_collection('2023-01-01', '2023-12-31')
                count_new = dataset_new.size().getInfo()
                st.write(f"Alternative images found: {count_new}")
            
            if count_new == 0:
                st.error("❌ No satellite data available for this area. Please try a different location.")
                st.stop()
            
            if count_old == 0:
                st.warning("⚠️ No historical images found. Using recent data for comparison.")
                dataset_old = dataset_new
            
            # Create composites
            image_old = dataset_old.median().clip(roi)
            image_new = dataset_new.median().clip(roi)
            
            # --- FIXED MAP RENDERING SECTION ---
            st.subheader("🌿 Analysis Results")
            
            # Create a NEW map for results - this is crucial
            result_map = geemap.Map()
            result_map.add_basemap("SATELLITE")
            
            # Set the map center to the ROI
            roi_center = roi.centroid().getInfo()['coordinates']
            result_map.set_center(roi_center[0], roi_center[1], 12)
            
            # Add ROI boundary
            result_map.add_layer(roi, {"color": "yellow", "fillColor": "00000000"}, "Analysis Area")
            
            # Add imagery with proper visualization
            visualization = {
                'bands': ['B4', 'B3', 'B2'],
                'min': 0,
                'max': 0.3
            }
            
            result_map.add_layer(image_new, visualization, "Satellite Image 2024")
            
            # Simple NDVI visualization for vegetation
            ndvi_vis = {
                'min': -0.2,
                'max': 0.8,
                'palette': ['red', 'yellow', 'green']
            }
            
            result_map.add_layer(image_new.select('NDVI'), ndvi_vis, "NDVI 2024", shown=False)
            result_map.add_layer(image_old.select('NDVI'), ndvi_vis, "NDVI 2018", shown=False)
            
            # Add layer control
            result_map.add_layer_control()
            
            # Display the map - THIS IS CRITICAL
            st.write("### Interactive Results Map")
            result_map.to_streamlit(height=600, key="results_display")
            
            # --- STATISTICS SECTION ---
            st.write("### 📊 Vegetation Statistics")
            
            def calculate_vegetation_area(image):
                """Calculate vegetation area using NDVI"""
                ndvi = image.select('NDVI')
                # Vegetation threshold
                vegetation = ndvi.gt(0.3)
                area = vegetation.multiply(ee.Image.pixelArea())
                stats = area.reduceRegion(
                    reducer=ee.Reducer.sum(),
                    geometry=roi,
                    scale=30,
                    maxPixels=1e9
                )
                return stats.get('NDVI').getInfo() or 0
            
            veg_2018 = calculate_vegetation_area(image_old)
            veg_2024 = calculate_vegetation_area(image_new)
            
            km_2018 = veg_2018 / 1000000  # Convert to km²
            km_2024 = veg_2024 / 1000000
            change = km_2024 - km_2018
            
            # Display results
            col1, col2, col3 = st.columns(3)
            col1.metric("2018 Vegetation", f"{km_2018:.2f} km²")
            col2.metric("2024 Vegetation", f"{km_2024:.2f} km²")
            col3.metric("Change", f"{change:+.2f} km²", delta=f"{change:+.2f} km²")
            
            # Interpretation
            if change > 0.1:
                st.success(f"🎉 Significant vegetation increase: {change:.2f} km²")
            elif change < -0.1:
                st.error(f"📉 Significant vegetation decrease: {change:.2f} km²")
            else:
                st.info("📊 Minimal change in vegetation area")
                
            st.info("💡 **Tip:** Use the layer control in the top-right corner of the map to toggle between different views.")

    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")
        import traceback
        st.error(f"Detailed error: {traceback.format_exc()}")

# Footer
st.divider()
st.caption("🌱 Green Space AI - Vegetation Monitoring Tool")
