import streamlit as st
import xarray as xr
import pandas as pd
import numpy as np
import lightgbm as lgb
import pydeck as pdk
import os

# --- TITEL ---
st.title("🌪️ Voorspelling van Extreme Weeromstandigheden")

# --- PAD NAAR DATA ---
data_dir = r"C:\Users\andri\Downloads\Opdracht Minor dat science\hackathon\processed_zarr"
zarr_files = [f for f in os.listdir(data_dir) if f.endswith(".zarr")]

if not zarr_files:
    st.error("⚠️ Geen Zarr-bestanden gevonden.")
    st.stop()

selected_zarr = st.selectbox("📂 Kies een Zarr-bestand:", zarr_files)
zarr_path = os.path.join(data_dir, selected_zarr)

# --- DATA LADEN ---
try:
    ds = xr.open_zarr(zarr_path, chunks={})
except Exception as e:
    st.error(f"❌ Fout bij laden: {e}")
    st.stop()

# --- VARIABELEN ---
variables = list(ds.data_vars)
if not variables:
    st.error("Geen data-variabelen gevonden in dit bestand.")
    st.stop()

# Kies de variabele voor voorspelling
selected_var = st.selectbox("🌡️ Kies een variabele voor analyse:", variables)

# --- TIJD ---
if "valid_time" not in ds.coords:
    st.error("Geen 'valid_time' dimensie gevonden in dataset.")
    st.stop()

time_vals = pd.to_datetime(ds["valid_time"].values)
selected_time = st.select_slider("🕓 Kies tijdstip:", options=list(time_vals), value=time_vals[len(time_vals)//2])

# --- DATA SUBSET ---
# Helper om voor een gegeven tijd en variabele een dataframe te bouwen (met subsampling)
def _build_df_for_time(t, var):
    try:
        ds_slice = ds[var].sel(valid_time=t).compute()
    except Exception:
        return None, None, None, None, None

    # Zorg dat we een 2D array (latitude x longitude) krijgen:
    if "valid_time" in ds_slice.dims:
        if ds_slice.sizes.get("valid_time", 0) > 1:
            ds_slice = ds_slice.isel(valid_time=0)
        else:
            ds_slice = ds_slice.squeeze("valid_time", drop=True)

    lat_local = ds["latitude"].values
    lon_local = ds["longitude"].values
    data_np_local = ds_slice.values

    # Controle: data_np moet 2D zijn
    if data_np_local.ndim != 2:
        return None, None, None, None, None

    # RAM-vriendelijke subset
    if lat_local.size == 0 or lon_local.size == 0:
        return None, None, None, None, None

    lat_sub_local = lat_local[::5]
    lon_sub_local = lon_local[::5]
    vals_local = data_np_local[::5, ::5]
    lon_grid_local, lat_grid_local = np.meshgrid(lon_sub_local, lat_sub_local)

    # Zorg dat shapes gelijk zijn (trim naar de kleinste common shape)
    min_shape_local = (min(vals_local.shape[0], lat_grid_local.shape[0]), min(vals_local.shape[1], lon_grid_local.shape[1]))
    if min_shape_local[0] == 0 or min_shape_local[1] == 0:
        return None, None, None, None, None

    vals_local = vals_local[:min_shape_local[0], :min_shape_local[1]]
    lat_grid_local = lat_grid_local[:min_shape_local[0], :min_shape_local[1]]
    lon_grid_local = lon_grid_local[:min_shape_local[0], :min_shape_local[1]]

    # Controleer of er überhaupt niet-NaN waardes zijn
    if np.all(np.isnan(vals_local)):
        return None, None, None, None, None

    # Maak dataframe en verwijder rijen zonder geldige waarde
    df_local = pd.DataFrame({
        "latitude": lat_grid_local.ravel(),
        "longitude": lon_grid_local.ravel(),
        "value": vals_local.ravel()
    })
    df_local = df_local.dropna(subset=["value"]).reset_index(drop=True)

    if df_local.empty:
        return None, None, None, None, None

    return df_local, ds_slice, lat_grid_local, lon_grid_local, vals_local

# Probeer geselecteerde tijd eerst, anders zoek naar de dichtstbijzijnde tijd met geldige data
df = None
lat_grid = lon_grid = vals = None

# Sorteer tijden op afstand tot gekozen tijd
times_sorted = sorted(list(time_vals), key=lambda t: abs(t - selected_time))

# 1) Probeer met de door de gebruiker gekozen variabele
for t in times_sorted:
    df_candidate, ds_candidate, lat_grid_candidate, lon_grid_candidate, vals_candidate = _build_df_for_time(t, selected_var)
    if df_candidate is not None:
        df = df_candidate
        data_slice = ds_candidate
        lat_grid = lat_grid_candidate
        lon_grid = lon_grid_candidate
        vals = vals_candidate
        selected_time = t
        break

# 2) Indien geen geldige data gevonden: probeer andere variabelen als fallback
if df is None or df.empty:
    for var in variables:
        if var == selected_var:
            continue
        for t in times_sorted:
            df_candidate, ds_candidate, lat_grid_candidate, lon_grid_candidate, vals_candidate = _build_df_for_time(t, var)
            if df_candidate is not None:
                df = df_candidate
                data_slice = ds_candidate
                lat_grid = lat_grid_candidate
                lon_grid = lon_grid_candidate
                vals = vals_candidate
                selected_time = t
                prev_var = selected_var
                selected_var = var
                st.warning(f"Geen geldige data gevonden voor '{prev_var}'. Valt terug op variabele '{var}' (tijd: {pd.to_datetime(t)}).")
                break
        if df is not None:
            break

if df is None or df.empty:
    st.error("❌ Geen geldige data gevonden voor de gekozen tijd/variabele en nabije tijden. Kies een ander bestand/variabele/tijdstip.")
    # Zorg dat we de uitvoering echt stoppen, zowel in Streamlit- als Jupyter-omgeving
    raise RuntimeError("Geen geldige data gevonden (df is None of empty)")

st.write(f"📊 Data voorbeeld (tijd: {pd.to_datetime(selected_time)}):", df.head())

# --- SIMULEER EXTREEM WEER LABEL ---
# (Bijv. alles boven 90e percentiel is "extreem")
# Veiligheidscheck: voorkomen van np.percentile op lege array
if df["value"].size == 0:
    st.error("❌ Geen geldige data om drempel te berekenen (leeg). Stop.")
    # In een notebook/streamlit-run wil je de uitvoering echt stoppen
    raise RuntimeError("Geen gegevens beschikbaar (df is empty)")

threshold = np.percentile(df["value"].values, 90)
df["extreem"] = (df["value"] >= threshold).astype(int)

st.write(f"⚙️ Drempel voor extreem weer: {threshold:.2f}")

# --- LIGHTGBM MODEL ---
st.subheader("🤖 LightGBM Model Training")

# Input features en target
X = df[["latitude", "longitude", "value"]]
y = df["extreem"]

# Controleer of target meer dan één klasse bevat (anders kan het model niet getraind worden)
if y.nunique() < 2:
    st.warning("⚠️ Doelvariabele heeft geen variatie (allemaal dezelfde klasse). Modeltraining wordt overgeslagen. Voorspellingen worden op basis van de labelwaarde gezet.")
    # Zet voorspellingskolommen consistent met het label
    df["voorspelling"] = y.astype(float)
    df["voorspelling_binair"] = y
else:
    # Train eenvoudig model (geen splitsing voor demo)
    train_data = lgb.Dataset(X, label=y)
    params = {
        "objective": "binary",
        "metric": "binary_error",
        "verbosity": -1
    }
    model = lgb.train(params, train_data, num_boost_round=50)

    # --- VOORSPELLING ---
    df["voorspelling"] = model.predict(X)
    df["voorspelling_binair"] = (df["voorspelling"] > 0.5).astype(int)

# --- VISUALISATIE ---
st.subheader("🗺️ Voorspelde Extreme Gebieden")

mid_lat = float(np.nanmean(df["latitude"]))
mid_lon = float(np.nanmean(df["longitude"]))

layer = pdk.Layer(
    "ScatterplotLayer",
    data=df[df["voorspelling_binair"] == 1],
    get_position=["longitude", "latitude"],
    get_color=[255, 0, 0, 160],
    get_radius=25000,
)

view_state = pdk.ViewState(latitude=mid_lat, longitude=mid_lon, zoom=3)
st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state))