from datetime import datetime
import numpy as np # linear algebra
import pandas as pd # data processing


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')


from sklearn.ensemble import RandomForestRegressor

import pickle

data = pd.read_csv('../../../data/CO2 Emissions_Canada.csv')
data

renamed_columns = {
    'Make' : "make",
    'Model' : 'model',
    'Vehicle Class': 'vehicle_class',
    'Engine Size(L)': 'engine_size',
    'Cylinders': 'cylinders',
    'Transmission' : 'transmission',
    'Fuel Type': 'fuel_type',
    'Fuel Consumption City (L/100 km)': 'fuel_cons_city',
    'Fuel Consumption Hwy (L/100 km)': 'fuel_cons_hwy',
    'Fuel Consumption Comb (L/100 km)': 'fuel_cons_comb',
    'Fuel Consumption Comb (mpg)': 'mpgfuel_cons_comb',
    'CO2 Emissions(g/km)': 'co2' }
data.rename(renamed_columns, axis='columns', inplace=True)

data = data.drop(['engine_size', 'cylinders','transmission', 'fuel_type', 'fuel_cons_city', 'fuel_cons_hwy', 'fuel_cons_comb', 'mpgfuel_cons_comb'], axis = 1)

categorical_columns = ['make', 'model', 'vehicle_class']
label_encoders = {}
for col in categorical_columns:
    label_encoder = LabelEncoder()
    data[col] = label_encoder.fit_transform(data[col])
    label_encoders[col] = label_encoder
    
X = data[['make','model', 'vehicle_class']]
y = data["co2"]

class MLService:

    def __init__(self):
        pass

    def predict(identity):

        d = {'make': X['make'][identity], 'model': X['model'][identity], 'vehicle_class': X['vehicel_class'][identity]}
        dataframe = pd.DataFrame(data=d)

        with open('filename.pkl', 'rb') as f:
            model = pickle.load(f)

        output = model.predict(dataframe)

        return output