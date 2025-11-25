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
    qa = image.select('SCL')
    cloud_mask = qa.neq(3).And(qa.neq(8)).And(qa.neq(9))
    return image.updateMask(cloud_mask).divide(10000)

def add_indices(image):
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
    m = geemap.Map(center=[14.5995, 120.9842], zoom=10)
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
        
        # Show ROI info
        bounds = roi.bounds().getInfo()
        st.write(f"Analysis area bounds: {bounds}")
        
    except Exception as e:
        st.error(f"Error reading drawing: {e}")
        st.stop()

    try:
        with st.spinner("Processing Satellite Data... This may take 30-60 seconds."):
            
            # SIMPLIFIED DATA FETCH - Using a more reliable approach
            collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            
            # Get a single recent image for testing
            recent_image = (collection
                          .filterBounds(roi)
                          .filterDate('2024-01-01', '2024-06-01')
                          .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
                          .first())
            
            # Check if we got an image
            try:
                image_id = recent_image.getInfo()
                if not image_id:
                    st.error("No satellite image found for the selected area and date range.")
                    st.stop()
            except:
                st.error("No satellite image found. Try a larger area or different location.")
                st.stop()
            
            # Process the image
            image_processed = mask_s2_clouds(recent_image)
            image_processed = add_indices(image_processed)
            image_clipped = image_processed.clip(roi)
            
            # --- CRITICAL: TEST MAP RENDERING WITH SIMPLE APPROACH ---
            st.subheader("🌿 Analysis Results")
            
            # Create a completely new map
            result_map = geemap.Map()
            
            # Add basemap first
            result_map.add_basemap("SATELLITE")
            
            # Center the map on ROI
            roi_center = roi.centroid().coordinates().getInfo()
            result_map.set_center(roi_center[0], roi_center[1], 12)
            
            # TEST 1: Add the ROI boundary (this should always work)
            result_map.add_layer(roi, {"color": "red", "fillColor": "00000000"}, "Analysis Area")
            
            # TEST 2: Try adding the satellite image with very simple parameters
            try:
                # Use RGB visualization
                vis_params = {
                    'bands': ['B4', 'B3', 'B2'],
                    'min': 0,
                    'max': 0.3
                }
                
                # Add the Earth Engine image to the map
                result_map.addLayer(image_clipped, vis_params, "Satellite Image")
                st.success("✅ Satellite image added to map")
                
            except Exception as layer_error:
                st.error(f"Could not add satellite layer: {layer_error}")
                
                # Fallback: Try NDVI visualization
                try:
                    ndvi_params = {
                        'min': -1,
                        'max': 1,
                        'palette': ['red', 'yellow', 'green']
                    }
                    result_map.addLayer(image_clipped.select('NDVI'), ndvi_params, "NDVI")
                    st.info("Using NDVI visualization instead")
                except:
                    st.warning("Using only base map and ROI boundary")
            
            # Add layer control
            result_map.add_layer_control()
            
            # --- DISPLAY THE MAP ---
            st.write("### Interactive Results Map")
            
            # CRITICAL: Use a unique key for the result map
            result_map.to_streamlit(height=600, key="result_map_unique")
            
            # --- SIMPLE STATISTICS ---
            st.write("### 📊 Basic Analysis")
            
            try:
                # Calculate NDVI statistics
                ndvi_stats = image_clipped.select('NDVI').reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=roi,
                    scale=30,
                    maxPixels=1e9
                ).getInfo()
                
                mean_ndvi = ndvi_stats.get('NDVI', 0)
                
                st.metric("Average NDVI", f"{mean_ndvi:.3f}")
                
                if mean_ndvi > 0.3:
                    st.success("Good vegetation coverage (NDVI > 0.3)")
                elif mean_ndvi > 0.1:
                    st.info("Moderate vegetation coverage")
                else:
                    st.warning("Low vegetation coverage")
                    
            except Exception as stats_error:
                st.warning(f"Could not calculate statistics: {stats_error}")

    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")

# Debug information
with st.expander("Debug Information"):
    st.write("Session state:", st.session_state)
    if st.session_state.saved_geometry:
        st.write("Geometry coordinates length:", len(st.session_state.saved_geometry["coordinates"][0]))
