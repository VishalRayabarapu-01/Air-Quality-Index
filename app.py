# import streamlit as st
# import requests
# import pandas as pd
# import numpy as np
# import pickle
# import time
# import datetime
# import tensorflow as tf
# from tensorflow.keras.models import load_model

# # ----- Streamlit Page Config -----
# st.set_page_config(page_title="AQI Prediction", layout="wide")
# st.title("Hybrid AQI Prediction Using TLSTM & XGBoost")

# # ----- User Inputs -----
# api_key = '8d36c87986e710cf4104e889f6aebf17'
# zip_code = st.text_input("Enter ZIP Code", value="")
# country_code = st.text_input("Enter Country Code", value="IN")

# lat,lon=None,None
# # ----- Get Latitude and Longitude via Geo API -----
# if st.button("Get Location"):
#     geo_url = f"https://api.openweathermap.org/geo/1.0/zip?zip={zip_code},{country_code}&appid={api_key}"
#     geo_response = requests.get(geo_url)
#     if geo_response.status_code == 200:
#         geo_data = geo_response.json()
#         lat = geo_data.get("lat")
#         lon = geo_data.get("lon")
#         st.success("Location Data Retrieved and loacation is : " + geo_data.get('name'))
#     else:
#         st.error("Error fetching location data.")
#         lat, lon = None, None

# class TLSTMCell(tf.keras.layers.Layer):
#     def __init__(self, units, **kwargs):
#         super(TLSTMCell, self).__init__(**kwargs)
#         self.units = units
#         self.lstm_cell = tf.keras.layers.LSTMCell(units)

#     @property
#     def state_size(self):
#         return self.lstm_cell.state_size

#     @property
#     def output_size(self):
#         return self.lstm_cell.output_size

#     def call(self, inputs, states, training=None):
#         # Assume inputs shape: (batch, features + 1) where the last channel is delta_t.
#         feature_dim = tf.shape(inputs)[-1] - 1
#         features = inputs[:, :feature_dim]
#         delta_t = inputs[:, feature_dim:]
#         # Apply time decay to features
#         adjusted_features = features * tf.exp(-delta_t)
#         return self.lstm_cell(adjusted_features, states, training=training)

#     def get_config(self):
#         config = super(TLSTMCell, self).get_config()
#         config.update(
#             {
#                 "units": self.units,
#             }
#         )
#         return config

# # Proceed if lat and lon are available
# if lat and lon:
#     # ----- Get Pollutant Forecast Data -----
#     # For forecasting, we use the current time as start and get 10 forecast points.
#     poll_url = (f"https://api.openweathermap.org/data/2.5/air_pollution/forecast?"
#                 f"lat={lat}&lon={lon}&appid={api_key}")
#     poll_response = requests.get(poll_url)
#     if poll_response.status_code == 200:
#         poll_data = poll_response.json()
#         forecast_list = poll_data.get("list", [])
#         if len(forecast_list) < 10:
#             st.error("Not enough forecast data available.")
#         else:
#             forecast_list = forecast_list[:10]
#         rows = []
#         for entry in forecast_list:
#             components = entry.get("components", {}) 
#             row = {
#                 "PM2.5 (µg/m³)": components.get("pm2_5", np.nan),
#                 "PM10 (µg/m³)": components.get("pm10", np.nan),
#                 "Ozone (µg/m³)": components.get("o3", np.nan),
#                 "NO2 (µg/m³)": components.get("no2", np.nan),
#                 "NO (µg/m³)": components.get("no", np.nan),
#                 "SO2 (µg/m³)": components.get("so2", np.nan),
#                 "CO (mg/m³)": components.get("co", np.nan),
#                 "NH3 (µg/m³)": components.get("nh3", np.nan)
#             }
#             rows.append(row)
        
#         df_forecast = pd.DataFrame(rows)
#         features=df_forecast.columns
#         st.write("Pollutant Data (First 10 Forecast Points):", df_forecast)
        
        
#         with open("./training_data/scaler_features.pkl", "rb") as f:
#             loaded_scaler = pickle.load(f)
#             df_forecast[features] = loaded_scaler.fit_transform(df_forecast[features])
            
#         with open("./training_data/scaler_target.pkl", "rb") as f:
#             scaler_target = pickle.load(f)
        
#         X_features = np.array(df_forecast[features].iloc[-10:].values).reshape(1, 10, len(features))
#         dt=[0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
#         X_time = np.array(dt).reshape(1, 10, 1).astype("float32")

#         model = load_model("training_data/model", custom_objects={"TLSTMCell": TLSTMCell})
#         predicted_scaled_aqi = model.predict([X_features, X_time])
#         predicted_aqi=scaler_target.inverse_transform(predicted_scaled_aqi)[0][0]
#         print(f"Predicted AQI for Next Day: {predicted_aqi}")
#         st.write(f"Predicted AQI for Next Forecast Time Point: {predicted_aqi}")
        
#     else:
#         st.error("Error fetching pollutant forecast data from API.")
        
        
        
# import streamlit as st
# import requests
# import pandas as pd
# import numpy as np
# import pickle
# import tensorflow as tf
# from tensorflow.keras.models import load_model

# # ----- Streamlit Page Config -----
# st.set_page_config(page_title="AQI Prediction", layout="wide")
# st.markdown("""
#     <style>
#         body {
#             background-color: white;
#         }
#     </style>
# """, unsafe_allow_html=True)
# st.title("Hybrid AQI Prediction Using TLSTM & XGBoost")

# # Display placeholder image below the title
# placeholder_img = st.image("./images/hd-human-icon.png")  # Replace with an appropriate image path

# # ----- User Inputs -----
# api_key = '8d36c87986e710cf4104e889f6aebf17'
# zip_code = st.text_input("Enter ZIP Code", value="")
# country_code = st.text_input("Enter Country Code", value="IN")

# lat, lon = None, None

# # ----- Get Latitude and Longitude via Geo API -----
# if st.button("Get Location"):
#     geo_url = f"https://api.openweathermap.org/geo/1.0/zip?zip={zip_code},{country_code}&appid={api_key}"
#     geo_response = requests.get(geo_url)
#     if geo_response.status_code == 200:
#         geo_data = geo_response.json()
#         lat = geo_data.get("lat")
#         lon = geo_data.get("lon")
#         st.success("Location Data Retrieved: " + geo_data.get('name'))
#     else:
#         st.error("Error fetching location data.")
#         lat, lon = None, None

# class TLSTMCell(tf.keras.layers.Layer):
#     def __init__(self, units, **kwargs):
#         super(TLSTMCell, self).__init__(**kwargs)
#         self.units = units
#         self.lstm_cell = tf.keras.layers.LSTMCell(units)

#     @property
#     def state_size(self):
#         return self.lstm_cell.state_size

#     @property
#     def output_size(self):
#         return self.lstm_cell.output_size

#     def call(self, inputs, states, training=None):
#         feature_dim = tf.shape(inputs)[-1] - 1
#         features = inputs[:, :feature_dim]
#         delta_t = inputs[:, feature_dim:]
#         adjusted_features = features * tf.exp(-delta_t)
#         return self.lstm_cell(adjusted_features, states, training=training)

#     def get_config(self):
#         config = super(TLSTMCell, self).get_config()
#         config.update({"units": self.units})
#         return config

# if lat and lon:
#     poll_url = f"https://api.openweathermap.org/data/2.5/air_pollution/forecast?lat={lat}&lon={lon}&appid={api_key}"
#     poll_response = requests.get(poll_url)
#     if poll_response.status_code == 200:
#         poll_data = poll_response.json()
#         forecast_list = poll_data.get("list", [])[:10]
#         rows = []
#         for entry in forecast_list:
#             components = entry.get("components", {})
#             rows.append({
#                 "PM2.5 (µg/m³)": components.get("pm2_5", np.nan),
#                 "PM10 (µg/m³)": components.get("pm10", np.nan),
#                 "Ozone (µg/m³)": components.get("o3", np.nan),
#                 "NO2 (µg/m³)": components.get("no2", np.nan),
#                 "NO (µg/m³)": components.get("no", np.nan),
#                 "SO2 (µg/m³)": components.get("so2", np.nan),
#                 "CO (mg/m³)": components.get("co", np.nan),
#                 "NH3 (µg/m³)": components.get("nh3", np.nan)
#             })
#         df_forecast = pd.DataFrame(rows)
#         st.write("Pollutant Data (First 10 Forecast Points):", df_forecast)
        
#         with open("./training_data/scaler_features.pkl", "rb") as f:
#             loaded_scaler = pickle.load(f)
#             df_forecast[df_forecast.columns] = loaded_scaler.transform(df_forecast[df_forecast.columns])
        
#         with open("./training_data/scaler_target.pkl", "rb") as f:
#             scaler_target = pickle.load(f)
        
#         X_features = np.array(df_forecast.iloc[-10:].values).reshape(1, 10, len(df_forecast.columns))
#         X_time = np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1]).reshape(1, 10, 1).astype("float32")
        
#         model = load_model("training_data/model", custom_objects={"TLSTMCell": TLSTMCell})
        
#         # Show loading animation
#         st.image("./images/row-0-column-3.png")  # Replace with actual loading GIF
#         predicted_scaled_aqi = model.predict([X_features, X_time])
#         predicted_aqi = scaler_target.inverse_transform(predicted_scaled_aqi)[0][0]
        
#         # Show final image
#         st.image("./images/row-0-column-1.png")  # Replace with an appropriate result image
#         st.write(f"Predicted AQI for Next Forecast Time Point: {predicted_aqi}")
#     else:
#         st.error("Error fetching pollutant forecast data from API.")
import time
import streamlit as st
import requests
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model

st.set_page_config(page_title="AQI Prediction", layout="wide")
# Function to inject custom CSS
def apply_custom_css():
    custom_css = """
    <style>
       
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

apply_custom_css()


st.title("Hybrid AQI Prediction Using TLSTM & XGBoost")

# ----- Display Initial Image -----
st.image("images/hd-human-icon.png", use_container_width =True)

# ----- User Inputs -----
api_key = "8d36c87986e710cf4104e889f6aebf17"
zip_code = st.text_input("Enter ZIP Code", value="")
country_code = st.text_input("Enter Country Code", value="IN")

lat, lon = None, None

# ----- Get Latitude and Longitude -----
if st.button("Get Location"):
    geo_url = f"https://api.openweathermap.org/geo/1.0/zip?zip={zip_code},{country_code}&appid={api_key}"
    geo_response = requests.get(geo_url)
    if geo_response.status_code == 200:
        geo_data = geo_response.json()
        lat = geo_data.get("lat")
        lon = geo_data.get("lon")
        st.success("Location Data Retrieved: " + geo_data.get("name"))
    else:
        st.error("Error fetching location data.")
        lat, lon = None, None


class TLSTMCell(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(TLSTMCell, self).__init__(**kwargs)
        self.units = units
        self.lstm_cell = tf.keras.layers.LSTMCell(units)

    @property
    def state_size(self):
        return self.lstm_cell.state_size

    @property
    def output_size(self):
        return self.lstm_cell.output_size

    def call(self, inputs, states, training=None):
        feature_dim = tf.shape(inputs)[-1] - 1
        features = inputs[:, :feature_dim]
        delta_t = inputs[:, feature_dim:]
        adjusted_features = features * tf.exp(-delta_t)
        return self.lstm_cell(adjusted_features, states, training=training)

    def get_config(self):
        config = super(TLSTMCell, self).get_config()
        config.update({"units": self.units})
        return config


# Proceed if lat and lon are available
if lat and lon:
    # ----- Get Pollutant Forecast Data -----
    poll_url = (
        f"https://api.openweathermap.org/data/2.5/air_pollution/forecast?"
        f"lat={lat}&lon={lon}&appid={api_key}"
    )
    poll_response = requests.get(poll_url)

    if poll_response.status_code == 200:
        poll_data = poll_response.json()
        forecast_list = poll_data.get("list", [])

        if len(forecast_list) < 10:
            st.error("Not enough forecast data available.")
        else:
            forecast_list = forecast_list[:10]
            
        rows = []
        for entry in forecast_list:
            components = entry.get("components", {})
            row = {
                    "PM2.5 (µg/m³)": components.get("pm2_5", np.nan),
                    "PM10 (µg/m³)": components.get("pm10", np.nan),
                    "Ozone (µg/m³)": components.get("o3", np.nan),
                    "NO2 (µg/m³)": components.get("no2", np.nan),
                    "NO (µg/m³)": components.get("no", np.nan),
                    "SO2 (µg/m³)": components.get("so2", np.nan),
                    "CO (mg/m³)": components.get("co", np.nan),
                    "NH3 (µg/m³)": components.get("nh3", np.nan),
            }
            rows.append(row)

        df_forecast = pd.DataFrame(rows)
        features = df_forecast.columns
        st.write("Pollutant Data (First 10 Forecast Points):", df_forecast)
        
        with st.spinner("Predicting AQI..."):
            with open("./training_data/scaler_features.pkl", "rb") as f:
                loaded_scaler = pickle.load(f)
                df_forecast[features] = loaded_scaler.transform(df_forecast[features])

            with open("./training_data/scaler_target.pkl", "rb") as f:
                scaler_target = pickle.load(f)

            X_features = np.array(df_forecast[features].iloc[-10:].values).reshape(
                1, 10, len(features)
            )
            dt = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            X_time = np.array(dt).reshape(1, 10, 1).astype("float32")

            model = load_model("training_data/final_tlstm_model.keras", custom_objects={"TLSTMCell": TLSTMCell})
            predicted_scaled_aqi = model.predict([X_features, X_time])
            predicted_aqi = scaler_target.inverse_transform(predicted_scaled_aqi)[0][0]
            image_map = {
                (0, 50): "0-50.png",
                (50, 100): "50-100.png",
                (100, 150): "100-150.png",
                (150, 200): "150-200.png",
                (200, 300): "200-300.png",
                (300, 500): "300-500.png",
            }
            selected_image = None
            for (low, high), img_file in image_map.items():
                if low <= predicted_aqi < high:
                    selected_image = img_file
                    break
            st.image('./images/'+selected_image,width=600)
            st.success(f"Predicted AQI : {predicted_aqi}")

    else:
        st.error("Error fetching pollutant forecast data from API.")
