"""

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
import streamlit as st

# Step 1: Load and prepare the dataset
data = pd.read_excel('data.xlsx')

# Features and target
X = data[['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10',
          'P11', 'P12', 'P13', 'P14', 'P15', 'P16', 'P17', 'P18', 'perc', 'transformer']]
y = data['thd']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Train the CatBoost model
model = CatBoostRegressor(iterations=1000, learning_rate=0.1, depth=6)
model.fit(X_train, y_train, verbose=False)

# Step 3: Build the Streamlit app

# Streamlit app title
st.title("THD Prediction App")

# Input fields for P1 to P18 and W1 to W18
p_values = []
w_values = []
for i in range(1, 19):
    p = st.number_input(f'P{i}', value=0.0)
    w = st.number_input(f'W{i}', value=0.0)
    p_values.append(p)
    w_values.append(w)

# Input field for transformer
trans = st.number_input('Transformer', value=1.0)

# Calculate 'perc' feature
perc = sum([p * w for p, w in zip(p_values, w_values)]) / (trans * 8)

# Combine inputs into the feature array
features = p_values + [perc, trans]

# Predict THD and display result
if st.button('Predict THD'):
    prediction = model.predict(np.array([features]))
    st.write(f'Predicted THD: {prediction[0]}')


# To run the app, use: streamlit run app.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import streamlit as st

# Step 1: Load and prepare the dataset
data = pd.read_excel('data.xlsx')

# Features and target
X = data[['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10',
          'P11', 'P12', 'P13', 'P14', 'P15', 'P16', 'P17', 'P18', 'perc', 'transformer']]
y = data['thd']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Train the RandomForest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 3: Build the Streamlit app

# Streamlit app title
st.title("THD Prediction App")

# Input fields for P1 to P18 and W1 to W18
p_values = []
w_values = []
for i in range(1, 19):
    p = st.number_input(f'P{i}', value=0.0)
    w = st.number_input(f'W{i}', value=0.0)
    p_values.append(p)
    w_values.append(w)

# Input field for transformer
trans = st.number_input('Transformer', value=1.0)

# Calculate 'perc' feature
perc = sum([p * w for p, w in zip(p_values, w_values)]) / (trans * 8)

# Combine inputs into the feature list
features = p_values + [perc, trans]

# Predict THD and display result
if st.button('Predict THD'):
    prediction = model.predict([features])
    st.write(f'Predicted THD: {prediction[0]}')

# To run the app, use: streamlit run app.py
# 
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import streamlit as st

# Step 1: Load and prepare the dataset
data = pd.read_excel('data.xlsx')

# Features and target
X = data[['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10',
          'P11', 'P12', 'P13', 'P14', 'P15', 'P16', 'P17', 'P18', 'perc', 'transformer']]
y = data['thd']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Train the RandomForest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 3: Build the Streamlit app

# Streamlit app title
st.title("THD Prediction App")

# Input fields with updated labels
p_values = []
w_values = []
p1 = st.checkbox("I agree")
if p1:
    st.write("Great!")
    p1 = st.number_input(f'{p_labels[i]}', value=0.0)
p_labels = [
    "Number of Television", "Number of Compact Flourescent Lamp", "Number of Refrigerator",
    "Number of Air Conditioner", "Number of Phones", "Number of Laptops", "Number of Fans",
    "Number of Washing Machines", "Number of Microwave", "Number of Personal Computers",
    "Number of UPS", "Number of Flourescent Lamp", "Number of Hot Plate", "Number of Iron",
    "Number of Electric Cooker", "Number of Printer", "Photocopier", "Number of Linear Load"
]
w_labels = [
    "Wattage of Television", "Wattage of Compact Flourescent Lamp", "Wattage of Refrigerator",
    "Wattage of Air Conditioner", "Wattage of Phones", "Wattage of Laptops", "Wattage of Fans",
    "Wattage of Washing Machine", "Wattage of Microwave", "Wattage of Personal Computer",
    "Wattage of UPS", "Wattage of Flourescent Lamp", "Wattage of Hot Plate", "Wattage of Iron",
    "Wattage of Electric Cooker", "Wattage of Printer", "Wattage of Photocopier", "Wattage of Linear Loads"
]

for i in range(18):
    p = st.number_input(f'{p_labels[i]}', value=0.0)
    w = st.number_input(f'{w_labels[i]}', value=0.0)
    p_values.append(p)
    w_values.append(w)

# Input field for transformer
trans = st.number_input('Transformer Ratings', value=1.0)

# Calculate 'perc' feature
perc = sum([p * w for p, w in zip(p_values, w_values)]) / (trans * 8)

# Combine inputs into the feature list
features = p_values + [perc, trans]

# Predict THD and display result
if st.button('Predict THD'):
    prediction = model.predict([features])
    st.write(f'Predicted THD: {prediction[0]}')

# To run the app, use: streamlit run app.py
