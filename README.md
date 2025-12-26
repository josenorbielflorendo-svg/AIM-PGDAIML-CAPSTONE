# AIM-PGDAIML-CAPSTONE
This repository contains all the files, jupyter notebook and other outputs for AIM PGDAIML Capstone by Jose Norbiel G. Florendo

Synthetic dataset comprises Mill, Mine Advance and Mine Production data from a real gold mining company in the Philippines.

## Dataset
The data dictionaries for the 4 input files are stored under /dataset.
It contains 4 synthetic dataset from a Philippine Mining Company:
1. Capstone Dataset_MCF.csv
2. Capstone Dataset_Mill Reports.csv
3. Capstone Dataset_Mine Advance.csv
4. Capstone Dataset_Mine Production.csv

## Jupyter Notebook
The jupyter notebook is placed inside the root of this directory: PGDAIML_Capstone_Florendo_rev2.ipynb
This jupyter notebook requires the following python libraries:
- pandas
- numpy
- matplotlib.pyplot
- seaborn
- re
- sklearn
- joblib
- copy
- xgboost
-lightgbm
- catboost
- sys
- shap

## ML Web-app (Taipy-based, Taipy is built on top of Flask framework)
The taipy web-app is placed inside the root of this directory: PGDAIML_Capstone_Florendo_Taipy.py
### the Demo video is placed in the root of the directory: Taipy Webapp - Demo.mkv
This web-app requires the following python libraries:
os
joblib
pandas
from taipy.gui import Gui, notify, download

## Presentations
The technical presentation is stored under /presentation/PGDAIML - Capstone - Jose Norbiel Florendo - Technical - Application of Supervised ML for Au-Ag Throughput, Recovery, and Mine Production in an Operating Philippine Gold Mine.pdf
The technical presentation is stored under /presentation/PGDAIML - Capstone - Jose Norbiel Florendo - Business - Application of Supervised ML for Au-Ag Throughput, Recovery, and Mine Production in an Operating Philippine Gold Mine.pdf

## Data Dictionaries
The data dictionaries for the 4 input files are stored under /data_dictionary.
It contains the data dictionaries for the 4 synthetic datasets:
1. MCF - Data Dictionary.csv
2. Mill Reports - Data Dictionary.csv
3. Mine Advance - Data Dictioanary.csv
4. Mine Production - Data Dictionary.csv

## Models
The saved ML models are stored under /models.
It contains 10 saved ML models which include the baseline and tuned models:
1. best_baseline_model_Gold_Ounces_Produced.joblib
2. best_baseline_model_Gold_Recovery.joblib
3. best_baseline_model_Mine_Tonnage.joblib
4. best_baseline_model_Silver_Ounces_Produced.joblib
5. best_baseline_model_Silver_Recovery.joblib\
6. best_model_Gold_Ounces_Produced.joblib
7. best_model_Gold_Recovery.joblib
8. best_model_Mine_Tonnage.joblib
9. best_model_Silver_Ounces_Produced.joblib
10. best_model_Silver_Recovery.joblib

## Sample Dataset to use for Web-app
The web-app allows for inputing bulk data for prediction within the Taipy Webapp. 
There are 2 sample input files are stored under /sample_dataset_for_dashboard.
1. Dashboard_Taipy_SampleData_AuAgOutput-Recovery.csv
2. Dashboard_Taipy_SampleData_MineTonnage.csv

Notes: 
The Jupyter Notebook had difficulty in exporting an html or pdf file of the JupyterNotebook so I'm unable to submit those, but the JupyterNotebook file is uploaded in this repo.

