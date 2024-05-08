#  GeoPredictive-Housing-Insights
## Predicting Damaged Roofs Through Aerial Imagery and Machine Learning
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

Detailed steps from data acquisition to model training and validation are outlined, focusing on the use of machine learning algorithms to process and analyze imagery data for detecting roof damages. 

## Project Workflow
![image](https://github.com/lornalyu/GeoPredictive-Housing-Insights/assets/140224372/62597558-bf89-494c-8b2a-6daae2daa525)


The workflow of this project leverages advanced AI and machine learning techniques to assess and classify roof damage from aerial imagery. The process is broken down into several key stages:

1. **AI Model for Building Footprint Segmentation**: Initially, drone imagery is utilized to create detailed maps of building footprints. An AI model developed by our [twin project](https://github.com/itsyunnie/Remote-Assessment-of-Residential-Properties-for-Maintenance-Needs-Using-Satellite-Image) processes these images to segment and identify individual buildings accurately. This foundational step is crucial as it ensures that subsequent analyses are precisely targeted at specific structures.

2. **Segmented Image Preparation**: Once building footprints are identified, the next step involves segmenting these images to isolate the buildings from the surrounding environment. This segmentation facilitates focused analysis and ensures that the classification algorithms are applied directly to the relevant areas of interest. Codes below this section show the function developed to enable such cleaning processes.

3. **Damage Classification with AI Classifier**: The core of our project focuses on this stage. We use a pre-trained ResNet34 model from Microsoft (US Building Footprints) to classify all the roofs on the input landscape. Using the segmented images, the specialized classifier is employed to determine the condition of each roof. This classifier has been trained to recognize various damage levels, which allows for detailed assessments of the structural integrity of each roof.

4. **ROI Cropping for Classification**: For enhanced accuracy, Regions of Interest (ROIs) within each segmented image are cropped further to refine the input for the damage classification model. This step ensures that the classifier focuses precisely on potential damage indicators, thereby improving the reliability and precision of the output.

5. **Output Generation**: The final output of the workflow is a classified map that visually represents the condition of roofs across the surveyed area. This map is instrumental for stakeholders, such as housing authorities and maintenance teams, to prioritize interventions and repairs.


### Extracting Images from Tax Parcel Coordinates
Below we walk through what it generally looks like to pull images from Google Earth Engine using a set of coordinates and exporting these to a Google Drive Folder. 

#### Import necessary libraries
```
# Import neccesary libraries
import geopandas as gpd
import ee
import geemap

# Plus any additional libraries required to create a polygon object from a set of coordinates, for example:
from shapely.geometry import Polygon
import shapely.geometry as sg
import json
```

#### Initialize Earth Engine library
This will require an authentication token that will be generated upon running this line

```
# Initialize the Earth Engine library.
ee.Initialize()
```

#### Example Operationalization
```
# Initialize a map.
m = geemap.Map()

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

- **Code Repositories**:
- [Remote Assessment of Residential Properties for Maintenance and Repair Needs Using Satellite Image Analysis: A Case Study in Fulton County, GA](https://github.com/itsyunnie/Remote-Assessment-of-Residential-Properties-for-Maintenance-Needs-Using-Satellite-Image)
- [Damaged Structures Detector](https://github.com/your_repository_here](https://github.com/kkraoj/damaged_structures_detector))
- **Relevant Paper**: Alidoost, F., & Arefi, H. (2018). "A CNN-Based Approach for Automatic Building Detection and Recognition of Roof Types Using a Single Aerial Image," *PFG – Journal of Photogrammetry, Remote Sensing and Geoinformation Science*, 86(5–6), 235–248. [DOI](https://doi.org/10.1007/s41064-018-0060-5)
