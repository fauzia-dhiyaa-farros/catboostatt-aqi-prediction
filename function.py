# function.py
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


# ======================================================
# Data Loader
# ======================================================
def load_data(path: str):
    """
    Load dataset CSV.
    """
    df = pd.read_csv(path, parse_dates=["tanggal"], low_memory=False)
    # Bersihkan NaN/None
    df = df.dropna(how="all").reset_index(drop=True)
    return df


# ======================================================
# Main Analyzer Class
# ======================================================
class AirQualityAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    # ---------- General Info ----------
    def dataset_summary(self):
        """Ringkasan dataset"""
        return self.df.describe(include="all")

    def missing_values(self):
        """Cek missing values"""
        return self.df.isnull().sum().reset_index().rename(columns={"index": "Column", 0: "Missing"})

    def available_years(self):
        if "tanggal" in self.df.columns:
            return sorted(self.df["tanggal"].dt.year.unique())
        return []

    def filter_year_range(self, start, end):
        if "tanggal" not in self.df.columns:
            return self.df
        return self.df[(self.df["tanggal"].dt.year >= start) & (self.df["tanggal"].dt.year <= end)]

    def filter_period(self, start=None, end=None):
        if "tanggal" not in self.df.columns:
            return self.df
        df = self.df.copy()
        if start:
            df = df[df["tanggal"] >= start]
        if end:
            df = df[df["tanggal"] <= end]
        return df

    # ---------- WHO / BMKG Reference ----------
    def who_reference_table(self):
        """Tabel referensi WHO/BMKG AQI"""
        data = {
            "Parameter": ["PM2.5", "PM10", "SO₂", "CO", "O₃", "NO₂", "temperature", "relative_humidity", "wind_speed"],
            "BMKG/WHO Reference": [
                "< 15 µg/m³ (24h, BMKG)",   # PM2.5
                "< 50 µg/m³ (24h, BMKG)",   # PM10
                "< 365 µg/m³ (24h, BMKG)",  # SO2
                "< 10 mg/m³ (8h, BMKG)",    # CO
                "< 120 µg/m³ (8h, BMKG)",   # O3
                "< 200 µg/m³ (1h, BMKG)",   # NO2
                "20–25 °C",                 # Temp
                "40–60%",                   # RH
                "0–15 m/s"                  # Wind
            ]
        }
        return pd.DataFrame(data)

    def get_aqi_table(self):
        """Kategori AQI sederhana (BMKG)"""
        data = {
            "Category": ["Good", "Moderate", "Unhealthy", "Very Unhealthy", "Hazardous"],
            "AQI Range": ["0–50", "51–100", "101–199", "200–299", "300+"],
            "Color": ["Green", "Yellow", "Red", "Purple", "Maroon"],
        }
        return pd.DataFrame(data)

    # ---------- Intro Text ----------
    def get_introduction(self):
        return """
        This dashboard provides **Spatio-Temporal Air Quality Index (AQI) Analysis** for DKI Jakarta.  
        It integrates pollutant concentration data (PM2.5, PM10, SO₂, CO, NO₂, O₃) from BMKG Jakarta monitoring stations,  
        combined with meteorological variables from NOAA/NCEI (temperature, humidity, wind speed, precipitation).  
        """

    def get_business_understanding(self):
        return """
        The goal of this project is to build a predictive model for **Jakarta's AQI** using  
        multivariate pollutant & meteorological data. This will help decision-makers and citizens  
        to understand air pollution trends and take preventive actions.
        """

    def get_references(self):
        return """
        - World Health Organization (WHO) Air Quality Guidelines, 2021  
        - BMKG (Badan Meteorologi, Klimatologi, dan Geofisika) — Jakarta AQI data  
        - NOAA/NCEI Global Historical Climatology Network (GHCN)  
        """

    # ======================================================
    # Visualization Functions
    # ======================================================
    def plot_time_series_plotly(self, var="PM2.5"):
        if var not in self.df.columns or "tanggal" not in self.df.columns:
            return go.Figure()
        fig = px.line(self.df, x="tanggal", y=var, color=self.df.get("station", None),
                      title=f"{var} Time Series")
        return fig

    def plot_peak_trends_plotly(self, df=None, var="PM2.5"):
        data = df if df is not None else self.df
        if var not in data.columns or "tanggal" not in data.columns:
            return go.Figure()
        dfg = data.groupby([data["tanggal"].dt.year, "station"])[var].max().reset_index()
        dfg.rename(columns={"tanggal": "year", var: f"max_{var}"}, inplace=True)
        fig = px.bar(dfg, x="year", y=f"max_{var}", color="station", barmode="group",
                     title=f"Peak {var} per Year by Station")
        return fig

    def peak_per_station(self, df=None, var="PM2.5"):
        data = df if df is not None else self.df
        if var not in data.columns or "station" not in data.columns:
            return pd.DataFrame()
        return data.groupby("station")[var].max().reset_index().rename(columns={var: f"max_{var}"})

    def plot_station_means_plotly(self, var="PM2.5"):
        if var not in self.df.columns or "station" not in self.df.columns:
            return go.Figure()
        dfg = self.df.groupby("station")[var].mean().reset_index()
        fig = px.bar(dfg, x="station", y=var, color="station", title=f"Average {var} per Station")
        return fig

    def plot_correlation_heatmap_plotly(self):
        if self.df.empty:
            return go.Figure()
        num_df = self.df.select_dtypes(include=np.number)
        if num_df.empty:
            return go.Figure()
        corr = num_df.corr()
        fig = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Heatmap")
        return fig

    def strong_correlations(self, threshold=0.3):
        num_df = self.df.select_dtypes(include=np.number)
        if num_df.empty:
            return pd.DataFrame()
        corr = num_df.corr()
        strong = []
        for i in corr.columns:
            for j in corr.columns:
                if i != j and abs(corr.loc[i, j]) >= threshold:
                    strong.append((i, j, corr.loc[i, j]))
        return pd.DataFrame(strong, columns=["Var1", "Var2", "Correlation"])
