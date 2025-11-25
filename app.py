import streamlit as st
import ee
import geemap.foliumap as geemap
import json
from google.oauth2 import service_account
import requests
from PIL import Image
import io

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

def get_static_map_image(image, vis_params, geometry, dimensions=800):
    """Get a static map image from Earth Engine"""
    try:
        # Get the thumbnail URL
        url = image.getThumbURL({
            'region': geometry,
            'dimensions': dimensions,
            'format': 'png',
            **vis_params
        })
        
        # Download the image
        response = requests.get(url)
        if response.status_code == 200:
            return Image.open(io.BytesIO(response.content))
        else:
            return None
    except Exception as e:
        st.error(f"Error generating map image: {e}")
        return None

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
        
    except Exception as e:
        st.error(f"Error reading drawing: {e}")
        st.stop()

    try:
        with st.spinner("🛰️ Downloading and processing satellite data..."):
            
            # Get satellite data - using simpler approach
            def get_best_image(year):
                collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                            .filterBounds(roi)
                            .filterDate(f'{year}-01-01', f'{year}-12-31')
                            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
                            .sort('CLOUDY_PIXEL_PERCENTAGE'))
                
                image_count = collection.size().getInfo()
                if image_count > 0:
                    return collection.first()
                else:
                    # If no low-cloud images, try with higher cloud tolerance
                    return (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                           .filterBounds(roi)
                           .filterDate(f'{year}-01-01', f'{year}-12-31')
                           .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 50))
                           .sort('CLOUDY_PIXEL_PERCENTAGE')
                           .first())
            
            # Get images for both periods
            image_2024 = get_best_image(2024)
            image_2018 = get_best_image(2018)
            
            # Check if we have valid images
            try:
                image_2024.getInfo()
                image_2018.getInfo()
            except:
                st.error("❌ No satellite images found for the selected area. Please try a larger area or different location.")
                st.stop()
            
            # Process images
            image_2024_processed = add_indices(mask_s2_clouds(image_2024)).clip(roi)
            image_2018_processed = add_indices(mask_s2_clouds(image_2018)).clip(roi)
            
            # --- STATIC MAP GENERATION ---
            st.subheader("🌿 Analysis Results")
            st.info("📷 Displaying static satellite imagery analysis")
            
            # Define visualization parameters
            rgb_params = {
                'bands': ['B4', 'B3', 'B2'],
                'min': 0,
                'max': 0.3
            }
            
            ndvi_params = {
                'bands': ['NDVI'],
                'min': -0.2,
                'max': 0.8,
                'palette': ['red', 'yellow', 'green', 'darkgreen']
            }
            
            # Generate static maps
            with st.spinner("🖼️ Generating visualizations..."):
                
                # 2024 Images
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("### 📅 2024 Data")
                    
                    # RGB Image 2024
                    rgb_2024 = get_static_map_image(image_2024_processed, rgb_params, roi)
                    if rgb_2024:
                        st.image(rgb_2024, caption="2024 Satellite Image (Natural Color)", use_column_width=True)
                    else:
                        st.error("Could not generate 2024 satellite image")
                    
                    # NDVI Image 2024
                    ndvi_2024 = get_static_map_image(image_2024_processed, ndvi_params, roi)
                    if ndvi_2024:
                        st.image(ndvi_2024, caption="2024 Vegetation Index (NDVI)", use_column_width=True)
                    else:
                        st.error("Could not generate 2024 vegetation map")
                
                with col2:
                    st.write("### 📅 2018 Data")
                    
                    # RGB Image 2018
                    rgb_2018 = get_static_map_image(image_2018_processed, rgb_params, roi)
                    if rgb_2018:
                        st.image(rgb_2018, caption="2018 Satellite Image (Natural Color)", use_column_width=True)
                    else:
                        st.error("Could not generate 2018 satellite image")
                    
                    # NDVI Image 2018
                    ndvi_2018 = get_static_map_image(image_2018_processed, ndvi_params, roi)
                    if ndvi_2018:
                        st.image(ndvi_2018, caption="2018 Vegetation Index (NDVI)", use_column_width=True)
                    else:
                        st.error("Could not generate 2018 vegetation map")
            
            # --- VEGETATION ANALYSIS ---
            st.write("### 📊 Vegetation Change Analysis")
            
            def calculate_vegetation_area(image):
                """Calculate vegetation area in square kilometers"""
                ndvi = image.select('NDVI')
                # Consider NDVI > 0.3 as vegetation
                vegetation_mask = ndvi.gt(0.3)
                area_image = vegetation_mask.multiply(ee.Image.pixelArea())
                
                area_stats = area_image.reduceRegion(
                    reducer=ee.Reducer.sum(),
                    geometry=roi,
                    scale=30,
                    maxPixels=1e9,
                    bestEffort=True
                )
                
                area_sq_m = area_stats.get('NDVI').getInfo()
                if area_sq_m:
                    return area_sq_m / 1000000  # Convert to km²
                else:
                    return 0
            
            # Calculate vegetation areas
            veg_area_2018 = calculate_vegetation_area(image_2018_processed)
            veg_area_2024 = calculate_vegetation_area(image_2024_processed)
            area_change = veg_area_2024 - veg_area_2018
            percent_change = (area_change / veg_area_2018 * 100) if veg_area_2018 > 0 else 0
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "2018 Vegetation Area",
                    f"{veg_area_2018:.2f} km²"
                )
            
            with col2:
                st.metric(
                    "2024 Vegetation Area", 
                    f"{veg_area_2024:.2f} km²"
                )
            
            with col3:
                st.metric(
                    "Change (2018-2024)",
                    f"{area_change:+.2f} km²",
                    f"{percent_change:+.1f}%"
                )
            
            # Change interpretation
            st.write("### 📈 Change Interpretation")
            
            if area_change > 0.5:
                st.success(f"🎉 **Significant Vegetation Increase!** The area gained {area_change:.2f} km² of vegetation ({percent_change:+.1f}% increase).")
            elif area_change > 0.1:
                st.success(f"📈 **Vegetation Improvement!** The area gained {area_change:.2f} km² of vegetation ({percent_change:+.1f}% increase).")
            elif area_change < -0.5:
                st.error(f"📉 **Significant Vegetation Loss!** The area lost {abs(area_change):.2f} km² of vegetation ({percent_change:+.1f}% decrease).")
            elif area_change < -0.1:
                st.warning(f"🔻 **Vegetation Decrease!** The area lost {abs(area_change):.2f} km² of vegetation ({percent_change:+.1f}% decrease).")
            else:
                st.info(f"📊 **Stable Vegetation!** Minimal change ({area_change:+.2f} km², {percent_change:+.1f}%) in vegetation area.")
            
            # Additional statistics
            with st.expander("📋 Detailed Statistics"):
                st.write("**Area Information:**")
                st.write(f"- Analysis region area: {roi.area().getInfo() / 1000000:.2f} km²")
                st.write(f"- 2018 vegetation coverage: {(veg_area_2018 / (roi.area().getInfo() / 1000000)) * 100:.1f}%")
                st.write(f"- 2024 vegetation coverage: {(veg_area_2024 / (roi.area().getInfo() / 1000000)) * 100:.1f}%")
                
                st.write("**Image Details:**")
                st.write(f"- 2018 image date: {image_2018.date().format('YYYY-MM-dd').getInfo()}")
                st.write(f"- 2024 image date: {image_2024.date().format('YYYY-MM-dd').getInfo()}")
                
            # Legend explanation
            st.write("### 🎨 Map Legend")
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Satellite Images (RGB):**")
                st.write("- Shows natural color view")
                st.write("- Red tones: Urban areas, bare soil")
                st.write("- Green tones: Vegetation")
                st.write("- Blue tones: Water")
            
            with col2:
                st.write("**Vegetation Index (NDVI):**")
                st.write("- 🟥 Red: No vegetation (water, urban)")
                st.write("- 🟨 Yellow: Sparse vegetation")
                st.write("- 🟩 Green: Moderate vegetation")
                st.write("- 🟩 Dark Green: Dense vegetation")

    except Exception as e:
        st.error(f"❌ Analysis failed: {str(e)}")
        st.info("💡 Try selecting a different area or making your drawing larger.")

# Footer
st.divider()
st.caption("🌱 Green Space AI | Sentinel-2 Satellite Data Analysis | 2018-2024 Comparison")
