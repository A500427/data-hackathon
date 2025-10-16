# hackathon_app.py
import streamlit as st
import xarray as xr
import pandas as pd
import numpy as np
import pydeck as pdk
import lightgbm as lgb
import tempfile
import os

st.title("Extreme Weeromstandigheden Voorspelling 🌩️")

# --- Upload Zarr-bestand ---
uploaded_file = st.file_uploader(
    "Upload een Zarr-bestand", type=["zarr"], help="Upload een Zarr bestand met weersdata"
)

if uploaded_file is not None:
    st.info("Bestand ontvangen, laden…")
    
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmp_path = os.path.join(tmpdirname, "uploaded.zarr")
        uploaded_file.seek(0)
        with open(tmp_path, "wb") as f:
            f.write(uploaded_file.read())

        # Open Zarr dataset
        ds = xr.open_zarr(tmp_path)
        st.success("Dataset geladen! ✅")

        # --- Variabele en tijd selecteren ---
        selected_var = st.selectbox("Kies variabele:", list(ds.data_vars))
        time_vals = pd.to_datetime(ds['valid_time'].values)
        selected_time = st.slider(
            "Kies tijdstip:",
            min_value=time_vals.min(),
            max_value=time_vals.max(),
            value=time_vals.max()
        )

        # Data slice
        data_slice = ds[selected_var].sel(valid_time=selected_time).compute()

        # --- Subset voor visualisatie ---
        lat_sub = ds['latitude'].values
        lon_sub = ds['longitude'].values

        lon_grid, lat_grid = np.meshgrid(lon_sub, lat_sub)
        vals = data_slice.values

        df = pd.DataFrame({
            "latitude": lat_grid.ravel(),
            "longitude": lon_grid.ravel(),
            "value": vals.ravel()
        }).dropna()

        st.write("📊 Data voorbeeld:")
        st.dataframe(df.head())

        # --- PyDeck visualisatie ---
        st.write("🌍 Kaart van geselecteerde variabele:")
        layer = pdk.Layer(
            "HeatmapLayer",
            data=df,
            get_position='[longitude, latitude]',
            get_weight='value',
            radiusPixels=30
        )
        view_state = pdk.ViewState(
            latitude=float(lat_sub.mean()),
            longitude=float(lon_sub.mean()),
            zoom=2,
            pitch=0
        )
        r = pdk.Deck(layers=[layer], initial_view_state=view_state)
        st.pydeck_chart(r)

        # --- LightGBM voorspellingsvoorbeeld ---
        st.write("🤖 Voorspelling van extreme waarden:")

        # Feature voorbereiding (lichte subset, kan uitgebreid)
        X = df[['latitude', 'longitude']]
        y = df['value']

        # Train-test split (klein, voor voorbeeld)
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train eenvoudig LightGBM model
        model = lgb.LGBMRegressor(n_estimators=50, max_depth=5)
        model.fit(X_train, y_train)

        # Voorspelling
        y_pred = model.predict(X_test)

        # Resultaten tonen
        st.write("📈 Voorbeeld voorspellingen:")
        st.dataframe(pd.DataFrame({"latitude": X_test['latitude'], 
                                   "longitude": X_test['longitude'], 
                                   "actual": y_test, 
                                   "predicted": y_pred}).head())
