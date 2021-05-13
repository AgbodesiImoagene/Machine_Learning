import pandas as pd
import numpy as np

df = pd.read_csv("healthcare-dataset-stroke-data.csv")
df.loc[:, "gender"].replace({"Other":0, "Male":1, "Female":2}, inplace=True)
df.loc[:, "ever_married"].replace({"No":0, "Yes":1}, inplace=True)
df.loc[:, "work_type"].replace({"children":0, "Never_worked":1, "Self-employed":2, "Govt_job":3, "Private":4}, inplace=True)
df.loc[:, "Residence_type"].replace({"Rural":0, "Urban":1}, inplace=True)
df.loc[:, "smoking_status"].replace({"Unknown":0, "never smoked":1, "formerly smoked":2, "smokes":3}, inplace=True)
df.to_csv("cleaned_data.csv")
