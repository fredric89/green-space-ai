import streamlit as st
import ee
import geemap.foliumap as geemap
import json
from google.oauth2 import service_account
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO

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

def get_thumbnail_url(image, vis_params, region, dimensions='500'):
    """Get a thumbnail URL from Earth Engine"""
    try:
        url = image.getThumbURL({
            'region': region,
            'dimensions': dimensions,
            'format': 'png',
            **vis_params
        })
        return url
    except:
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
        with st.spinner("Processing Satellite Data... This may take 30-60 seconds."):
            
            # Get satellite data
            collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            
            # Get recent image
            recent_image = (collection
                          .filterBounds(roi)
                          .filterDate('2024-01-01', '2024-06-01')
                          .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
                          .sort('CLOUDY_PIXEL_PERCENTAGE')
                          .first())
            
            # Get historical image
            historical_image = (collection
                              .filterBounds(roi)
                              .filterDate('2018-01-01', '2018-12-31')
                              .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
                              .sort('CLOUDY_PIXEL_PERCENTAGE')
                              .first())
            
            # Check if images exist
            try:
                recent_info = recent_image.getInfo()
                historical_info = historical_image.getInfo()
            except:
                st.error("No satellite images found for the selected area. Try a larger area or different location.")
                st.stop()
            
            # Process images
            recent_processed = add_indices(mask_s2_clouds(recent_image)).clip(roi)
            historical_processed = add_indices(mask_s2_clouds(historical_image)).clip(roi)
            
            # --- STATIC VISUALIZATION SECTION ---
            st.subheader("🌿 Analysis Results")
            
            # Visualization parameters
            rgb_vis = {
                'bands': ['B4', 'B3', 'B2'],
                'min': 0,
                'max': 0.3
            }
            
            ndvi_vis = {
                'bands': ['NDVI'],
                'min': -0.2,
                'max': 0.8,
                'palette': ['red', 'yellow', 'green', 'darkgreen']
            }
            
            # Get region bounds for thumbnails
            region_bounds = roi.bounds().getInfo()['coordinates'][0]
            
            # Create thumbnails
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("### 2024 Satellite Image")
                try:
                    thumbnail_url = get_thumbnail_url(recent_processed, rgb_vis, roi)
                    if thumbnail_url:
                        st.image(thumbnail_url, use_column_width=True, caption="Recent Satellite View (RGB)")
                    else:
                        st.warning("Could not generate 2024 thumbnail")
                except Exception as e:
                    st.error(f"Error generating 2024 image: {e}")
                
                st.write("### 2024 Vegetation (NDVI)")
                try:
                    ndvi_url = get_thumbnail_url(recent_processed, ndvi_vis, roi)
                    if ndvi_url:
                        st.image(ndvi_url, use_column_width=True, caption="Vegetation Index 2024 (Red=Low, Green=High)")
                    else:
                        st.warning("Could not generate 2024 NDVI thumbnail")
                except Exception as e:
                    st.error(f"Error generating 2024 NDVI: {e}")
            
            with col2:
                st.write("### 2018 Satellite Image")
                try:
                    hist_thumbnail_url = get_thumbnail_url(historical_processed, rgb_vis, roi)
                    if hist_thumbnail_url:
                        st.image(hist_thumbnail_url, use_column_width=True, caption="Historical Satellite View 2018 (RGB)")
                    else:
                        st.warning("Could not generate 2018 thumbnail")
                except Exception as e:
                    st.error(f"Error generating 2018 image: {e}")
                
                st.write("### 2018 Vegetation (NDVI)")
                try:
                    hist_ndvi_url = get_thumbnail_url(historical_processed, ndvi_vis, roi)
                    if hist_ndvi_url:
                        st.image(hist_ndvi_url, use_column_width=True, caption="Vegetation Index 2018 (Red=Low, Green=High)")
                    else:
                        st.warning("Could not generate 2018 NDVI thumbnail")
                except Exception as e:
                    st.error(f"Error generating 2018 NDVI: {e}")
            
            # --- VEGETATION STATISTICS ---
            st.write("### 📊 Vegetation Change Analysis")
            
            def calculate_vegetation_stats(image):
                """Calculate vegetation statistics"""
                ndvi = image.select('NDVI')
                
                # Calculate different vegetation levels
                total_pixels = ndvi.reduceRegion(
                    reducer=ee.Reducer.count(),
                    geometry=roi,
                    scale=30,
                    maxPixels=1e9
                ).getInfo().get('NDVI', 1)
                
                dense_veg = ndvi.gt(0.6).reduceRegion(
                    reducer=ee.Reducer.sum(),
                    geometry=roi,
                    scale=30,
                    maxPixels=1e9
                ).getInfo().get('NDVI', 0)
                
                moderate_veg = ndvi.gt(0.3).reduceRegion(
                    reducer=ee.Reducer.sum(),
                    geometry=roi,
                    scale=30,
                    maxPixels=1e9
                ).getInfo().get('NDVI', 0)
                
                # Calculate percentages
                dense_percent = (dense_veg / total_pixels) * 100
                moderate_percent = (moderate_veg / total_pixels) * 100
                total_veg_percent = dense_percent + moderate_percent
                
                return {
                    'dense_vegetation': dense_percent,
                    'moderate_vegetation': moderate_percent,
                    'total_vegetation': total_veg_percent,
                    'area_km2': (total_pixels * 900) / 1000000  # 30m pixel = 900m²
                }
            
            try:
                # Calculate statistics
                stats_2018 = calculate_vegetation_stats(historical_processed)
                stats_2024 = calculate_vegetation_stats(recent_processed)
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Total Vegetation Area", 
                        f"{stats_2024['area_km2']:.2f} km²",
                        f"{(stats_2024['area_km2'] - stats_2018['area_km2']):+.2f} km²"
                    )
                
                with col2:
                    st.metric(
                        "Vegetation Coverage", 
                        f"{stats_2024['total_vegetation']:.1f}%",
                        f"{(stats_2024['total_vegetation'] - stats_2018['total_vegetation']):+.1f}%"
                    )
                
                with col3:
                    st.metric(
                        "Dense Vegetation", 
                        f"{stats_2024['dense_vegetation']:.1f}%",
                        f"{(stats_2024['dense_vegetation'] - stats_2018['dense_vegetation']):+.1f}%"
                    )
                
                # Vegetation change analysis
                st.write("#### Vegetation Change Summary")
                
                veg_change = stats_2024['total_vegetation'] - stats_2018['total_vegetation']
                dense_change = stats_2024['dense_vegetation'] - stats_2018['dense_vegetation']
                
                if veg_change > 5:
                    st.success(f"🎉 **Significant vegetation increase!** Total vegetation increased by {veg_change:.1f}%")
                elif veg_change > 1:
                    st.success(f"📈 **Vegetation improvement!** Total vegetation increased by {veg_change:.1f}%")
                elif veg_change < -5:
                    st.error(f"📉 **Significant vegetation loss!** Total vegetation decreased by {abs(veg_change):.1f}%")
                elif veg_change < -1:
                    st.warning(f"🔻 **Vegetation decrease!** Total vegetation decreased by {abs(veg_change):.1f}%")
                else:
                    st.info(f"📊 **Stable vegetation!** Minimal change ({veg_change:+.1f}%) in vegetation coverage")
                
                # Detailed breakdown
                with st.expander("Detailed Statistics"):
                    st.write("**2018 Vegetation:**")
                    st.write(f"- Dense vegetation: {stats_2018['dense_vegetation']:.1f}%")
                    st.write(f"- Moderate vegetation: {stats_2018['moderate_vegetation']:.1f}%")
                    st.write(f"- Total vegetation: {stats_2018['total_vegetation']:.1f}%")
                    st.write(f"- Total area: {stats_2018['area_km2']:.2f} km²")
                    
                    st.write("**2024 Vegetation:**")
                    st.write(f"- Dense vegetation: {stats_2024['dense_vegetation']:.1f}%")
                    st.write(f"- Moderate vegetation: {stats_2024['moderate_vegetation']:.1f}%")
                    st.write(f"- Total vegetation: {stats_2024['total_vegetation']:.1f}%")
                    st.write(f"- Total area: {stats_2024['area_km2']:.2f} km²")
                    
            except Exception as stats_error:
                st.warning(f"Could not calculate detailed statistics: {stats_error}")
                
                # Fallback: Simple NDVI comparison
                try:
                    mean_ndvi_2018 = historical_processed.select('NDVI').reduceRegion(
                        ee.Reducer.mean(), roi, 30).getInfo().get('NDVI', 0)
                    mean_ndvi_2024 = recent_processed.select('NDVI').reduceRegion(
                        ee.Reducer.mean(), roi, 30).getInfo().get('NDVI', 0)
                    
                    st.metric("Average NDVI Change", 
                             f"{mean_ndvi_2024:.3f}",
                             f"{(mean_ndvi_2024 - mean_ndvi_2018):+.3f}")
                except:
                    st.error("Unable to calculate vegetation statistics")

    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")

# Footer
st.divider()
st.caption("🌱 Green Space AI - Vegetation Monitoring Tool | Uses Sentinel-2 Satellite Data")
