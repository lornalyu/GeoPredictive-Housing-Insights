# Predicting Damaged Roofs Through Aerial Imagery and Machine Learning
This Project is sponsored by Emory's QTM department and Neighborhood Nexus. Our goal is to harness Fulton County tax records and high-resolution aerial imagery to predict property code violations and potential abandonment in Atlanta. The initiative will utilize machine learning to analyze patterns and aid in the proactive maintenance of affordable housing, contributing meaningful insights to urban housing studies.

![image](https://github.com/lornalyu/QTM-Project/assets/157392307/d1d9deee-2d70-4d06-8683-da73d9032be7)

![image](https://github.com/lornalyu/QTM-Project/assets/157392307/0d5a4df3-d46c-4f60-bdaf-68015f6a2481)

## Table of Contents

- [Motivation](#motivation)
- [Project Introduction](#project-introduction)
  - [Objective](#objective)
  - [Intended Use](#intended-use)
- [Project Partners](#project-partners)
- [Methods and Technologies](#methods-and-technologies)
  - [Methods Used](#methods-used)
  - [Technologies Employed](#technologies-employed)
- [Project Workflow](#project-workflow)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Results and Discussion](#results-and-discussion)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)
- [References](#references)

## Project Introduction

### Objective

The objective of this project is to use machine learning to analyze patterns in aerial imagery and tax records to identify properties at risk of neglect or violation, facilitating strategic interventions.

### Intended Use

This system is designed for use by city planners, housing authorities, and non-profit organizations focused on housing quality and urban development. It will enable these stakeholders to identify at-risk properties more efficiently and act before issues escalate.

## Project Partners

This project represents a collaborative effort between Emory University's [Quantitative Theory and Methods (QTM)](https://quantitative.emory.edu/index.html) department and  [Neighborhood Nexus](https://neighborhoodnexus.org/), a non-profit dedicated to data-driven decision-making in Georgia. The QTM department provided the analytical and methodological expertise through faculty oversight and student participation, focusing on the development and application of predictive models. Neighborhood Nexus contributed critical local data insights and facilitated connections with local government and community organizations, ensuring the project's findings were practical and accessible. This partnership exemplified the integration of academic research with community data to address real-world challenges, setting a precedent for future initiatives that leverage data science for social good and highlighting the benefits of collaborative approaches in urban development and policy making.


## Motivation

Urban housing maintenance remains a formidable challenge, especially within economically disadvantaged communities where many homeowners are burdened by the high costs associated with property upkeep. This leads to a degradation of living conditions and, ultimately, the aesthetic and functional decline of neighborhoods. These issues are particularly acute in Atlanta’s Fulton County, where the disparity in housing maintenance can lead to significant social and economic consequences.

### Problem Definition

The project identifies a critical gap in the current approach to managing urban housing maintenance. Traditional methods—relying heavily on physical inspections and resident reports—are not only slow and reactive but also fail to effectively target resources towards the homes most in need. This inefficiency is exacerbated in communities where residents may be less likely to report issues due to various socio-economic barriers.

### Anticipated Impact

The project is designed to transform how communities address the upkeep of affordable housing. By integrating advanced data analytics, the initiative will enable a more strategic allocation of resources, ensuring that help reaches those who need it most. Furthermore, by providing a scalable model that can be adapted to other regions, the project has the potential to make a broad and lasting impact on urban development and housing sustainability.


## Methods and Technologies

### Methods Used

- Machine Learning and Deep Learning for predictive modeling
- Data Mining and Visualization for insights
- Predictive Modeling to forecast potential housing violations

### Technologies Employed

- **Google Earth Engine**: For acquiring high-resolution aerial images.
- **Google Drive and Excel**: For data storage and preprocessing.
- **Python and Jupyter Notebook**: For data analysis, modeling, and creating reproducible research documentation.
- **GitHub**: For version control and sharing of project code and documentation.

## Project Workflow

Detailed steps from data acquisition to model training and validation are outlined, focusing on the use of machine learning algorithms to process and analyze imagery data for detecting roof damages. Below is a code chunk which demonstrates one way in which we pulled images from Google Earth Engine using coordinates contained in a tax parcel geojson file.

```
# Import neccesary libraries
import json
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
```

## Exploratory Data Analysis

Visualizations created from the dataset provide insights into the patterns of housing conditions, aiding in the identification of areas most in need of intervention.

## Results and Discussion
![image](https://github.com/lornalyu/QTM-Project/assets/157392307/fe02956a-e4ec-4273-9e2d-cd0f3a39f67a)
![image](https://github.com/lornalyu/QTM-Project/assets/157392307/c0fb2f29-0ee3-494a-b83a-0ad1c3abc014)
![image](https://github.com/lornalyu/QTM-Project/assets/157392307/4dcefcbb-3cff-4e0e-8b42-1e3f47bfd777)

The model currently achieves 70% accuracy in classifying roof conditions. Ongoing work aims to improve this metric and expand the dataset to cover more regions.

## Acknowledgements

We are grateful for the sponsorship of Emory's QTM department and Neighborhood Nexus. Special thanks to Dr. Kevin McAlister for his invaluable mentorship and to DamageMap for their support.

## Contact

For inquiries, please contact any of the collaborators via email.

## References

- **Code Repository**: [Damaged Structures Detector](https://github.com/your_repository_here)
- **Relevant Paper**: Alidoost, F., & Arefi, H. (2018). "A CNN-Based Approach for Automatic Building Detection and Recognition of Roof Types Using a Single Aerial Image," *PFG – Journal of Photogrammetry, Remote Sensing and Geoinformation Science*, 86(5–6), 235–248. [DOI](https://doi.org/10.1007/s41064-018-0060-5)
