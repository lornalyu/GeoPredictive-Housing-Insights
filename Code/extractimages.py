# import json
import geopandas as gpd
import shapely.geometry as sg
import ee
import geemap
from shapely.geometry import Polygon

# Initialize the Earth Engine library.
ee.Initialize()

# Define the function to extract GeoJSON information
def extract_geojson_info(feature):
    geometry = feature['geometry']
    parcel_id = (feature['properties']['ParcelID']).replace(' ','')
    print(parcel_id)
    print(type(parcel_id))
    
    if geometry['type'] == 'Polygon':
        coordinates = geometry['coordinates']
        # Convert the coordinates into a Shapely Polygon object
        polygon = sg.Polygon(coordinates[0])  # Assuming the first list in coordinates contains the exterior ring
        print(polygon)
        
        # Compute the minimum rotated rectangle
        #obb = polygon.minimum_rotated_rectangle
        obb = polygon
        
        #Create a GeoDataFrame with the polygon geometry
        gdf = gpd.GeoDataFrame({'geometry': [polygon]}, crs='EPSG:4326')
        # Extract the bounding box coordinates
        bounds = gdf.bounds
        min_lon, min_lat, max_lon, max_lat = bounds.minx.values[0], bounds.miny.values[0], bounds.maxx.values[0], bounds.maxy.values[0]
        
        # Convert the coordinates into an Earth Engine polygon geometry
        ee_polygon = ee.Geometry.Polygon(coordinates)
    else:
        raise ValueError("Unsupported geometry type. Only 'Polygon' is supported.")
    
    return min_lon, min_lat, max_lon, max_lat, obb, ee_polygon, parcel_id

# Example usage:
file_path = 'geojson_file.geojson'

### One can use the following to select a specific set of images, sampled using a condition present in the GeoJson

# Load GeoJSON data
with open(file_path, 'r') as f:
    data = json.load(f)
    
# Filter features where 'HouseCondition' equals 1
features = [feature for feature in data['features'] if feature['properties'].get('HouseCondition') == 0]

''' Alternatively, one can load features using the following code to create a random sample
# Randomly sample feature indexes
with open(file_path, 'r') as f:
    data = json.load(f)
    num_features = len(data['features'])
    sampled_indexes = random.sample(range(num_features), min(20, num_features))  #If random sampling
'''

# Initialize a map.
m = geemap.Map()

# Iterate over filtered features
for i, feature in enumerate(features):
    
    try:
        # Extract GeoJSON information for the current feature
        min_lon, min_lat, max_lon, max_lat, obb, houseCoords, parcel_id = extract_geojson_info(feature)
    
    except:
        continue
    
    # Extract the coordinates of the minimum rotated rectangle
    obb_coords = list(obb.exterior.coords)

    # Create a bounding box for the area of interest (around the house).
    boundingBox = ee.Geometry.Polygon(obb_coords)

    # Load the image collection.
    dataset = ee.ImageCollection('USDA/NAIP/DOQQ').filter(ee.Filter.date('2021-01-01', '2023-12-31'))

    # Create an image composite or use the first image.
    firstImage = dataset.median()

    # Clip the image to the bounding box.
    clippedImage = firstImage.clip(boundingBox)

    # Create a mask from the houseCoords polygon.
    mask = ee.Image.constant(1).clip(houseCoords).mask()

    # Apply the mask to the clipped image.
    blackImage = ee.Image.constant([0, 0, 0, 255]).clip(boundingBox)
    maskedImage = clippedImage.updateMask(mask).blend(blackImage.updateMask(mask.Not()))
   
    # Add the masked image layer to the map.
    m.add_layer(maskedImage, trueColorVis, f'Feature {i}')

    # Print information about the current feature
    print(f"Feature {i}:")
    print("Max Latitude:", max_lat)
    print("Min Latitude:", min_lat)
    print("Max Longitude:", max_lon)
    print("Min Longitude:", min_lon)
    print("============================================")
    
    # Visualization parameters.
    trueColorVis = {
    'bands': ['R', 'G', 'B'],
    'min': 0,
    'max': 255,
    'gamma': [1.0, 1.0, 1.0]
    }
    
    visualizedImage = maskedImage.visualize(**trueColorVis)
    geemap.ee_export_image_to_drive(visualizedImage, description=f'{parcel_id}', folder="GoogleDriveFolder", region=boundingBox, scale=0.05)
    print("Exported Image")
