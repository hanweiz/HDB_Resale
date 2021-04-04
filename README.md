
# Predicting HDB Sale price using Machine Learning

## Problem Statement
### How can I predict the sale price of a 4-room HDB flat in 2024

Goal: To come up with the predicted sale price of a 4-room flat in Yishun using 
Machine Learning Techniques using models within the scikit-learn module in 
Python. Other supporting libraries include pandas, numpy, matplotlib and seaborn
for EDA. 

## Data
### Obtaining the data
Steps:

* We collected the HDB resale transaction data can be obtained from the website 
maintained by the Singapore government [here](https://data.gov.sg/dataset/resale-flat-prices).

	* Only data from 2017 onwards has been collected for increased relevance
	  to the current course of events.

* We examined the features provided and determined that additional
  factors can potentially affect the sale price e.g. proximity to:
	* 1) MRT and/or LRT stations
	* 2) Bus interchanges
	* 3) Shopping Centres
	
* Calls were made to the [OneMapSG API](https://www.onemap.gov.sg/docs/)
  for the required proximity information.

This project aims to come up with a Machine Learning (ML) model to predict the 
resale prices of HDB flats in Singapore using python libraries such as scikit-learn,
matplotlib, seaborn, pandas and numpy.

## Modelling
### First Model
1. The prepared data is decomposed with PCA. Sufficient components were retained to
capture 95% of the label variance while the rest are discarded.

2. The modified data are then split into 2 separate sets: training (80%) and
   validation (20%). The training using linear regression is performed on the
   modified features suggested by PCA.

3. The MSE of the fitted model is then calculated against the validation data.

### Second Model
1. Selected relevant non-numerical features are imputed, with the understanding
   of the potential bias introduced to the data from doing so. 

2. The modified dataset is then split into training (80%) and validation (20%).
   All imputed and numerical features are used to train the model.

3. The MSE of the fitted model is calculated against the validation data and
   compared to that from the First Model to ascertain suitability.
