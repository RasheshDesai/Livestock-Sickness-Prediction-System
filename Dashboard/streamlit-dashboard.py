import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import joblib
from pathlib import Path
import keras

# Page configuration
st.set_page_config(
    page_title="Calf Health Monitoring System",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 6.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .alert-high {
        background-color: #ffcccc;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #ff0000;
    }
    .alert-medium {
        background-color: #fff4cc;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #ffaa00;
    }
    .alert-low {
        background-color: #e6f3ff;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #0066cc;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'predictions_made' not in st.session_state:
    st.session_state.predictions_made = False
if 'results_df' not in st.session_state:
    st.session_state.results_df = None
if 'data' not in st.session_state:
    st.session_state.data = None
if 'selected_calf' not in st.session_state:
    st.session_state.selected_calf = None

# Model loading function with caching
@st.cache_resource
def load_models():
    """Load ML models and artifacts"""
    try:
        lstm_model = keras.models.load_model('C:/Users/rashe/Desktop/Cattle Scan/Calfs and Heifers/models/LSTM_model.keras')
        scaler = joblib.load('C:/Users/rashe/Desktop/Cattle Scan/Calfs and Heifers/LSTM_deployment_artifacts/scaler.pkl')
        
        optimal_threshold = np.load('C:/Users/rashe/Desktop/Cattle Scan/Calfs and Heifers/LSTM_deployment_artifacts/optimal_threshold.npy')
        
        catboost_model = joblib.load('C:/Users/rashe/Desktop/Cattle Scan/Calfs and Heifers/models/catboost_fever_model_6h.joblib')
        return lstm_model, scaler, optimal_threshold, catboost_model, None
    except Exception as e:
        return None, None, None, None, str(e)

# Data loading function with caching
@st.cache_data
def load_data(file_path):
    """Load and preprocess data"""
    try:
        df = pd.read_csv(file_path)
        df['datetime'] = pd.to_datetime(df['datetime'])
        return df, None
    except Exception as e:
        return None, str(e)

def predict_lstm(df, model, scaler, feature_names, lookback=96):
    """Run LSTM predictions"""
    data_values = df[feature_names].values
    X = []
    indices = []
    
    for i in range(lookback, len(df)):
        window = data_values[i-lookback:i]
        X.append(window)
        indices.append(i)
    
    X = np.array(X)
    
    if len(X) == 0:
        return np.full(len(df), np.nan)
    
    n_samples, n_steps, n_features = X.shape
    X_reshaped = X.reshape(-1, n_features)
    X_scaled = scaler.transform(X_reshaped)
    X_scaled = X_scaled.reshape(n_samples, n_steps, n_features)
    
    preds = model.predict(X_scaled, verbose=0)
    preds = preds.flatten()
    
    full_probs = np.full(len(df), np.nan)
    full_probs[indices] = preds
    
    return full_probs

def add_rolling_features(df, window_hours=6):
    """Add rolling window features"""
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values(['id', 'datetime'])
    
    numeric_cols = ['temp', 'airTemp', 'humidity', 'accel', 'rumination', 
                   'ageInDays', 'aveHerdTemp', 'THI', 'rumination_per_day']
    
    if 'id' in df.columns and pd.api.types.is_numeric_dtype(df['id']):
        numeric_cols.append('id')
    
    rolling_features = []
    
    for calf_id, group in df.groupby('id'):
        group = group.set_index('datetime')
        for col in numeric_cols:
            if col in group.columns:
                group[f'{col}_{window_hours}h_mean'] = group[col].rolling(
                    window=f'{window_hours}h', min_periods=1
                ).mean()
                group[f'{col}_{window_hours}h_std'] = group[col].rolling(
                    window=f'{window_hours}h', min_periods=1
                ).std()
        rolling_features.append(group.reset_index())
    
    df_with_features = pd.concat(rolling_features, ignore_index=True)
    rolling_cols = [col for col in df_with_features.columns if f'{window_hours}h' in col]
    df_with_features[rolling_cols] = df_with_features[rolling_cols].fillna(0)
    
    return df_with_features

def calculate_persistence(df, boolean_series, window_str):
    """Calculate persistence using rolling windows"""
    temp_df = df.set_index('datetime').copy()
    temp_df['bool_col'] = boolean_series.values
    
    result = temp_df.groupby('id')['bool_col'].transform(
        lambda x: x.rolling(window=window_str, min_periods=1).min()
    )
    return result.values.astype(bool)

def apply_alert_logic(test_data):
    """Apply improved alert logic with persistence"""
    test_data = test_data.sort_values(['id', 'datetime']).reset_index(drop=True)
    
    # 1. Define Base Conditions
    is_lstm_high = test_data['probability_lstm'] >= 0.7
    is_cat_high = test_data['probability_catboost'] >= 0.6
    is_lstm_med = test_data['probability_lstm'] >= 0.5
    is_cat_med = test_data['probability_catboost'] >= 0.4
    is_lstm_low_threshold = test_data['probability_lstm'] >= 0.4
    
    # 2. Calculate Persistence (Rolling Windows)
    persist_cat_high_3h = calculate_persistence(test_data, is_cat_high, '3h')
    persist_lstm_high_60m = calculate_persistence(test_data, is_lstm_high, '60min')
    persist_lstm_low_60m = calculate_persistence(test_data, is_lstm_low_threshold, '60min')
    
    # 3. Evaluate Priorities using np.select
    conditions = [
        (is_lstm_high & is_cat_high),       # Priority 1: HIGH
        persist_cat_high_3h,                # Priority 2: HIGH
        (is_lstm_med & is_cat_med),         # Priority 3: MEDIUM
        persist_lstm_high_60m,              # Priority 4: LOW
        persist_lstm_low_60m,               # Priority 5: LOW
        pd.Series([True] * len(test_data))  # Priority 6: NONE (Default)
    ]
    
    choices_level = [
        "HIGH",
        "HIGH",
        "MEDIUM",
        "LOW",
        "LOW",
        "NONE"
    ]
    
    choices_action = [
        "URGENT veterinary exam",
        "URGENT veterinary exam (Persistent High CatBoost)",
        "Schedule exam within 24h",
        "Watch closely (Persistent High LSTM)",
        "Watch closely (Persistent Low LSTM)",
        "Routine monitoring"
    ]
    
    test_data['combined_alert_level'] = np.select(conditions, choices_level, default="NONE")
    test_data['recommended_action'] = np.select(conditions, choices_action, default="Routine monitoring")
    
    # Add persistence flags for debugging/verification
    test_data['flag_persist_cat_high_3h'] = persist_cat_high_3h
    test_data['flag_persist_lstm_high_60m'] = persist_lstm_high_60m
    test_data['flag_persist_lstm_low_60m'] = persist_lstm_low_60m
    
    return test_data

def get_individual_alert(p, thresholds=(0.8, 0.5, 0.3)):
    """Get individual alert level"""
    if pd.isna(p):
        return "INSUFFICIENT"
    if p >= thresholds[0]:
        return "HIGH"
    if p >= thresholds[1]:
        return "MEDIUM"
    if p >= thresholds[2]:
        return "LOW"
    return "NORMAL"

def run_predictions(df, lstm_model, scaler, catboost_model):
    """Run both LSTM and CatBoost predictions with improved alert logic"""
    
    feature_names_lstm = ['temp', 'airTemp', 'humidity', 'accel', 'rumination', 
                         'ageInDays', 'aveHerdTemp', 'THI', 'rumination_per_day']
    
    # LSTM predictions
    lstm_probs = predict_lstm(df, lstm_model, scaler, feature_names_lstm, lookback=96)
    df['probability_lstm'] = lstm_probs
    df['confidence_lstm'] = abs(df['probability_lstm'] - 0.5) * 2
    
    # Prepare for CatBoost
    df_catboost = df.copy()
    df_catboost['id'] = pd.to_numeric(df_catboost['id'], errors='coerce').fillna(0)
    df_catboost = add_rolling_features(df_catboost, window_hours=6)
    
    expected_features = [
        'temp', 'airTemp', 'humidity', 'accel', 'rumination', 'ageInDays', 
        'aveHerdTemp', 'THI', 'rumination_per_day', 'temp_6h_mean', 'temp_6h_std',
        'airTemp_6h_mean', 'airTemp_6h_std', 'humidity_6h_mean', 'humidity_6h_std',
        'accel_6h_mean', 'accel_6h_std', 'rumination_6h_mean', 'rumination_6h_std',
        'ageInDays_6h_mean', 'ageInDays_6h_std', 'aveHerdTemp_6h_mean', 
        'aveHerdTemp_6h_std', 'id_6h_mean', 'id_6h_std', 'THI_6h_mean', 
        'THI_6h_std', 'rumination_per_day_6h_mean', 'rumination_per_day_6h_std'
    ]
    
    X_catboost = pd.DataFrame()
    for feature in expected_features:
        if feature in df_catboost.columns:
            X_catboost[feature] = df_catboost[feature]
        else:
            X_catboost[feature] = 0.0
    
    X_catboost = X_catboost.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    
    # CatBoost predictions
    catboost_probs = catboost_model.predict_proba(X_catboost)[:, 1]
    df['probability_catboost'] = catboost_probs
    df['confidence_catboost'] = abs(df['probability_catboost'] - 0.5) * 2
    
    # Apply improved alert logic with persistence
    df = apply_alert_logic(df)
    
    # Individual alert levels for reference
    df['LSTM_alert_level'] = df['probability_lstm'].apply(get_individual_alert)
    df['Catboost_alert_level'] = df['probability_catboost'].apply(get_individual_alert)
    
    # Filter out insufficient data
    return df[~df['probability_lstm'].isna() & ~df['probability_catboost'].isna()]


def calculate_hourly_rolling_average(df, window_hours=24):
    """Calculate rolling average for activity data"""
    df = df.copy()
    df = df.sort_values('datetime')
    
    # Aggregate to hourly averages first (from 15-min data)
    df['hour'] = df['datetime'].dt.floor('H')
    hourly_df = df.groupby('hour').agg({
        'accel': ['mean', 'max', 'min'],
        'datetime': 'first'
    }).reset_index()
    
    # Flatten column names
    hourly_df.columns = ['hour', 'accel_avg', 'accel_max', 'accel_min', 'datetime']
    
    # Calculate rolling average
    hourly_df[f'rolling_{window_hours}h'] = hourly_df['accel_avg'].rolling(
        window=window_hours, 
        min_periods=1
    ).mean()
    
    return hourly_df

def get_daily_activity_summary(df):
    """Get daily activity summary"""
    df = df.copy()
    df['date'] = df['datetime'].dt.date
    
    daily_summary = df.groupby('date').agg({
        'accel': ['mean', 'sum', 'max', 'min', 'std']
    }).reset_index()
    
    daily_summary.columns = ['date', 'avg', 'total', 'max', 'min', 'std']
    return daily_summary

# Main Dashboard
st.markdown('<p class="main-header">🐄 Calf Health Monitoring Dashboard</p>', unsafe_allow_html=True)

# Sidebar - Data Selection
with st.sidebar:
    st.header("📊 Data Selection")
    
    # Data folder path
    data_folder = st.text_input(
        "Data Folder Path", 
        value="C:/Users/rashe/Desktop/Cattle Scan/Visualizations/Data/New"
    )
    
    # List available CSV files
    try:
        data_path = Path(data_folder)
        csv_files = list(data_path.glob("*.csv"))
        file_names = [f.name for f in csv_files]
        
        selected_file = st.selectbox("Select Data File", file_names)
        selected_file_path = data_path / selected_file
        
    except Exception as e:
        st.error(f"Error accessing data folder: {e}")
        st.stop()
    
    st.divider()
    
    # Load data
    if st.button("📥 Load Data", type="primary"):
        data, error = load_data(selected_file_path)
        if error:
            st.error(f"Error loading data: {error}")
        else:
            st.session_state.data = data
            st.success(f"Loaded {len(data)} records")

# Check if data is loaded and show filters only if data exists
if st.session_state.data is not None:
    data = st.session_state.data
    
    # Sidebar - Filters
    with st.sidebar:
        st.header("🔍 Filters")
        
        # Calf ID selection
        available_calves = sorted(data['id'].unique())
        selected_calf = st.selectbox("Select Calf ID", available_calves)
    
    # Date range selection
    min_date = data['datetime'].min().date()
    max_date = data['datetime'].max().date()
    
    date_range = st.date_input(
        "Select Date Range",
        value=(max_date - timedelta(days=7), max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    st.divider()
    
    # Run prediction button
    if st.button("Run Prediction", type="primary"):
        with st.spinner("Loading models and running predictions..."):
            # Load models
            lstm_model, scaler, optimal_threshold, catboost_model, error = load_models()
            
            if error:
                st.error(f"Error loading models: {error}")
                st.stop()
            
            # Filter data
            filtered_data = data[data['id'] == selected_calf].copy()
            
            if len(date_range) == 2:
                start_date = pd.Timestamp(date_range[0])
                end_date = pd.Timestamp(date_range[1]) + pd.Timedelta(days=1)
                filtered_data = filtered_data[
                    (filtered_data['datetime'] >= start_date) & 
                    (filtered_data['datetime'] < end_date)
                ]
            
            if len(filtered_data) < 96:
                st.error("⚠️ Not enough data points (minimum 96 required for LSTM)")
                st.stop()
            
            # Run predictions
            results = run_predictions(filtered_data, lstm_model, scaler, catboost_model)
            
            st.session_state.results_df = results
            st.session_state.predictions_made = True
            st.session_state.selected_calf = selected_calf
            
            st.success("Predictions completed!")

# Main Content - Only show if data is loaded
if st.session_state.data is None:
    st.info("👈 Please load data from the sidebar to begin")
    st.markdown("""
        Combine Model Architecture:
    """)
    st.image("C:/Users/rashe/Desktop/Cattle Scan/Calfs and Heifers/output/Combined_Model_Architecture.jpeg", width=600)
    st.stop()

# Show predictions if available
if st.session_state.predictions_made:
    results_df = st.session_state.results_df
    
    # Summary Metrics
    st.header("📈 Summary Metrics")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Total Records", 
            len(results_df),
            help="Total number of data points analyzed"
        )
    
    with col2:
        high_alerts = len(results_df[results_df['combined_alert_level'] == 'HIGH'])
        st.metric(
            "High Alerts", 
            high_alerts,
            delta=f"{high_alerts/len(results_df)*100:.1f}%" if len(results_df) > 0 else "0%",
            delta_color="inverse"
        )
    
    with col3:
        medium_alerts = len(results_df[results_df['combined_alert_level'] == 'MEDIUM'])
        st.metric(
            "Medium Alerts", 
            medium_alerts,
            delta=f"{medium_alerts/len(results_df)*100:.1f}%" if len(results_df) > 0 else "0%"
        )
    
    with col4:
        avg_lstm_prob = results_df['probability_lstm'].mean()
        st.metric(
            "Avg LSTM Sickness Probability", 
            f"{avg_lstm_prob*100:.1f}%",
            help="Average LSTM prediction probability"
        )
    with col5:
        avg_catboost_prob = results_df['probability_catboost'].mean()
        st.metric(
            "Avg CatBoost Sickness Probability", 
            f"{avg_catboost_prob*100:.1f}%",
            help="Average CatBoost prediction probability"
        )
    st.divider()
    
    latest_record = results_df.iloc[-1]
    alert_level = latest_record['combined_alert_level']
    
    if alert_level == 'HIGH':
        st.markdown(f"""
        <div class="alert-high">
            <h3>🚨 HIGH ALERT - Calf ID: {st.session_state.selected_calf}</h3>
            <p><strong>Action:</strong> {latest_record['recommended_action']}</p>
            <p><strong>LSTM Probability:</strong> {latest_record['probability_lstm']:.3f}</p>
            <p><strong>CatBoost Probability:</strong> {latest_record['probability_catboost']:.3f}</p>
            <p><strong>Latest Temperature:</strong> {latest_record['temp']:.2f}°C</p>
            <p><strong>Time:</strong> {latest_record['datetime']}</p>
        </div>
        """, unsafe_allow_html=True)
    elif alert_level == 'MEDIUM':
        st.markdown(f"""
        <div class="alert-medium">
            <h3>⚠️ MEDIUM ALERT - Calf ID: {st.session_state.selected_calf}</h3>
            <p><strong>Action:</strong> {latest_record['recommended_action']}</p>
            <p><strong>LSTM Probability:</strong> {latest_record['probability_lstm']:.3f}</p>
            <p><strong>CatBoost Probability:</strong> {latest_record['probability_catboost']:.3f}</p>
            <p><strong>Latest Temperature:</strong> {latest_record['temp']:.2f}°C</p>
            <p><strong>Time:</strong> {latest_record['datetime']}</p>
        </div>
        """, unsafe_allow_html=True)
    elif alert_level == 'LOW':
        st.markdown(f"""
        <div class="alert-low">
            <h3>ℹ️ LOW ALERT - Calf ID: {st.session_state.selected_calf}</h3>
            <p><strong>Action:</strong> {latest_record['recommended_action']}</p>
            <p><strong>LSTM Probability:</strong> {latest_record['probability_lstm']:.3f}</p>
            <p><strong>CatBoost Probability:</strong> {latest_record['probability_catboost']:.3f}</p>
            <p><strong>Latest Temperature:</strong> {latest_record['temp']:.2f}°C</p>
            <p><strong>Time:</strong> {latest_record['datetime']}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.success(f"No alerts - Calf ID {st.session_state.selected_calf} is healthy")
    
    st.divider()
    
    # Visualizations
    st.header("📊 Visualizations")
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Predictions Timeline", 
    "Temperature & Rumination Signs",
    "Activity Analysis", 
    "Alert Distribution",
    "Persistence Flags",
    "Data Table"
])
    
    with tab1:
        # Predictions over time
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=results_df['datetime'],
            y=results_df['probability_lstm'],
            mode='lines',
            name='LSTM Probability',
            line=dict(color='#1f77b4', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=results_df['datetime'],
            y=results_df['probability_catboost'],
            mode='lines',
            name='CatBoost Probability',
            line=dict(color='#ff7f0e', width=2)
        ))
        
        # Add threshold lines
        fig.add_hline(y=0.7, line_dash="dash", line_color="red", 
                     annotation_text="High Alert (LSTM)", annotation_position="right")
        fig.add_hline(y=0.5, line_dash="dash", line_color="orange", 
                     annotation_text="Medium Alert (LSTM)", annotation_position="right")
        
        fig.update_layout(
            title=f"Prediction Probabilities Over Time - Calf {st.session_state.selected_calf}",
            xaxis_title="Date & Time",
            yaxis_title="Probability",
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    

    with tab2:
        # Temperature and vital signs
        fig2 = go.Figure()
        
        fig2.add_trace(go.Scatter(
            x=results_df['datetime'],
            y=results_df['temp'],
            mode='lines',
            name='Body Temperature',
            line=dict(color='red', width=2)
        ))
        
        fig2.add_trace(go.Scatter(
            x=results_df['datetime'],
            y=results_df['airTemp'],
            mode='lines',
            name='Air Temperature',
            line=dict(color='blue', width=1.5),
            yaxis='y2'
        ))
        
        # Add alert markers
        # Q40.5 alerts (clock symbol)
        q40_5_alerts = results_df[results_df['alert'] == 'Q40.5']
        if not q40_5_alerts.empty:
            fig2.add_trace(go.Scatter(
                x=q40_5_alerts['datetime'],
                y=q40_5_alerts['temp'],
                mode='markers+text',
                name='Q40.5 Alert',
                marker=dict(size=12, color='orange', symbol='circle'),
                text='🕐',
                textposition='top center',
                textfont=dict(size=16),
                hovertemplate='<b>Q40.5 Alert</b><br>Time: %{x}<br>Temp: %{y:.2f}°C<extra></extra>'
            ))
        
        # Q41 alerts (exclamation symbol)
        q41_alerts = results_df[results_df['alert'] == 'Q41']
        if not q41_alerts.empty:
            fig2.add_trace(go.Scatter(
                x=q41_alerts['datetime'],
                y=q41_alerts['temp'],
                mode='markers+text',
                name='Q41 Alert',
                marker=dict(size=12, color='red', symbol='circle'),
                text='❗',
                textposition='top center',
                textfont=dict(size=16),
                hovertemplate='<b>Q41 Alert</b><br>Time: %{x}<br>Temp: %{y:.2f}°C<extra></extra>'
            ))
        
        # Add fever threshold
        fig2.add_hline(y=40, line_dash="dash", line_color="darkred", 
                    annotation_text="Fever Threshold (40°C)", annotation_position="left")
        
        fig2.update_layout(
            title=f"Temperature Monitoring - Calf {st.session_state.selected_calf}",
            xaxis_title="Date & Time",
            yaxis_title="Body Temperature (°C)",
            yaxis2=dict(
                title="Air Temperature (°C)",
                overlaying='y',
                side='right'
            ),
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig2, use_container_width=True)
        
        # Rumination chart - no column wrapper needed
        fig_rum = px.line(
            results_df, 
            x='datetime', 
            y='rumination_per_day',
            title='Rumination Activity'
        )
        fig_rum.update_layout(height=300)
        st.plotly_chart(fig_rum, use_container_width=True)

    with tab3:
        st.subheader("🏃 Activity Monitoring (Acceleration)")
        
        # Calculate rolling averages and daily summary
        hourly_activity = calculate_hourly_rolling_average(results_df, window_hours=24)
        daily_activity = get_daily_activity_summary(results_df)
        
        # Top section: 3-week overview
        st.markdown("##### 📊 Daily Activity Overview")
        
        fig_daily = go.Figure()
        
        # Bar chart for daily totals
        fig_daily.add_trace(go.Bar(
            x=daily_activity['date'],
            y=daily_activity['avg'],
            name='Daily Average',
            marker_color=['#ef4444' if avg < 0.15 else '#3b82f6' for avg in daily_activity['avg']],
            hovertemplate='Date: %{x}<br>Avg Activity: %{y:.3f}<extra></extra>'
        ))
        
        # Line for trend
        fig_daily.add_trace(go.Scatter(
            x=daily_activity['date'],
            y=daily_activity['avg'],
            mode='lines+markers',
            name='Trend',
            line=dict(color='#1e40af', width=2),
            marker=dict(size=6)
        ))
        
        fig_daily.update_layout(
            title=f"Daily Activity Levels - Calf {st.session_state.selected_calf}",
            xaxis_title="Date",
            yaxis_title="Average Activity (accel)",
            hovermode='x unified',
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig_daily, use_container_width=True)
        
        st.divider()
        
        # Bottom section: Hourly detail with 24h rolling average
        st.markdown("Hourly Activity with 24-Hour Rolling Average")
        
        fig_hourly = go.Figure()
        
        # Raw hourly data (lighter)
        fig_hourly.add_trace(go.Scatter(
            x=hourly_activity['datetime'],
            y=hourly_activity['accel_avg'],
            mode='lines',
            name='Hourly Average',
            line=dict(color='#cbd5e1', width=1),
            hovertemplate='%{x}<br>Hourly Avg: %{y:.3f}<extra></extra>'
        ))
        
        # 24-hour rolling average (prominent)
        fig_hourly.add_trace(go.Scatter(
            x=hourly_activity['datetime'],
            y=hourly_activity['rolling_24h'],
            mode='lines',
            name='24h Rolling Average',
            line=dict(color='#1e40af', width=3),
            hovertemplate='%{x}<br>24h Rolling Avg: %{y:.3f}<extra></extra>'
        ))
        
        # Add reference line for low activity threshold
        fig_hourly.add_hline(
            y=0.1, 
            line_dash="dash", 
            line_color="red",
            annotation_text="Low Activity Threshold", 
            annotation_position="right"
        )
        
        fig_hourly.update_layout(
            title="Hourly Activity Pattern with 24-Hour Smoothing",
            xaxis_title="Date & Time",
            yaxis_title="Activity Level (accel)",
            hovermode='x unified',
            height=300,
            showlegend=True
        )
        
        st.plotly_chart(fig_hourly, use_container_width=True)

    with tab4:
        # Alert distribution
        col1, col2 = st.columns(2)
        
        with col1:
            alert_counts = results_df['combined_alert_level'].value_counts()
            fig_pie = px.pie(
                values=alert_counts.values,
                names=alert_counts.index,
                title='Combined Alert Distribution',
                color=alert_counts.index,
                color_discrete_map={
                    'HIGH': '#ff4444',
                    'MEDIUM': '#ffaa00',
                    'LOW': '#4444ff',
                    'NONE': '#44ff44'
                }
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Alert timeline
            alert_numeric = results_df['combined_alert_level'].map({
                'HIGH': 3,
                'MEDIUM': 2,
                'LOW': 1,
                'NONE': 0
            })
            
            fig_alert_time = go.Figure()
            fig_alert_time.add_trace(go.Scatter(
                x=results_df['datetime'],
                y=alert_numeric,
                mode='lines+markers',
                name='Alert Level',
                line=dict(color='purple', width=2),
                marker=dict(size=4)
            ))
            
            fig_alert_time.update_layout(
                title='Alert Level Over Time',
                xaxis_title='Date & Time',
                yaxis_title='Alert Level',
                yaxis=dict(
                    tickmode='array',
                    tickvals=[0, 1, 2, 3],
                    ticktext=['NONE', 'LOW', 'MEDIUM', 'HIGH']
                ),
                height=300
            )
            
            st.plotly_chart(fig_alert_time, use_container_width=True)
    
    with tab5:
        st.subheader("Persistence Flags Analysis")
        
        st.markdown("""
        **Persistence Logic:**
        - 🔴 **CatBoost High 3h**: CatBoost ≥ 0.6 persisting for 3+ hours
        - 🟠 **LSTM High 60min**: LSTM ≥ 0.7 persisting for 60+ minutes
        - 🟡 **LSTM Low 60min**: LSTM ≥ 0.4 persisting for 60+ minutes
        """)
        
        # Create persistence visualization
        fig_persist = go.Figure()
        
        # Add traces for each persistence flag
        fig_persist.add_trace(go.Scatter(
            x=results_df['datetime'],
            y=results_df['flag_persist_cat_high_3h'].astype(int),
            mode='lines',
            name='CatBoost High 3h',
            line=dict(color='red', width=2),
            fill='tozeroy',
            fillcolor='rgba(255,0,0,0.2)'
        ))
        
        fig_persist.add_trace(go.Scatter(
            x=results_df['datetime'],
            y=results_df['flag_persist_lstm_high_60m'].astype(int) * 0.66,
            mode='lines',
            name='LSTM High 60min',
            line=dict(color='orange', width=2),
            fill='tozeroy',
            fillcolor='rgba(255,165,0,0.2)'
        ))
        
        fig_persist.add_trace(go.Scatter(
            x=results_df['datetime'],
            y=results_df['flag_persist_lstm_low_60m'].astype(int) * 0.33,
            mode='lines',
            name='LSTM Low 60min',
            line=dict(color='yellow', width=2),
            fill='tozeroy',
            fillcolor='rgba(255,255,0,0.2)'
        ))
        
        fig_persist.update_layout(
            title='Persistence Flags Over Time',
            xaxis_title='Date & Time',
            yaxis_title='Active Flag',
            yaxis=dict(
                tickmode='array',
                tickvals=[0, 0.33, 0.66, 1],
                ticktext=['None', 'LSTM Low', 'LSTM High', 'CatBoost High']
            ),
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig_persist, use_container_width=True)
        
        # Summary statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            cat_persist_pct = (results_df['flag_persist_cat_high_3h'].sum() / len(results_df) * 100)
            st.metric(
                "CatBoost High 3h Persistence",
                f"{cat_persist_pct:.1f}%",
                help="Percentage of time CatBoost ≥ 0.6 for 3+ hours"
            )
        
        with col2:
            lstm_high_persist_pct = (results_df['flag_persist_lstm_high_60m'].sum() / len(results_df) * 100)
            st.metric(
                "LSTM High 60min Persistence",
                f"{lstm_high_persist_pct:.1f}%",
                help="Percentage of time LSTM ≥ 0.7 for 60+ minutes"
            )
        
        with col3:
            lstm_low_persist_pct = (results_df['flag_persist_lstm_low_60m'].sum() / len(results_df) * 100)
            st.metric(
                "LSTM Low 60min Persistence",
                f"{lstm_low_persist_pct:.1f}%",
                help="Percentage of time LSTM ≥ 0.4 for 60+ minutes"
            )
    with tab6:
        st.subheader("Prediction Results")

        # =============================
        # COLUMN SELECT + FILTER UI
        # =============================
        col1, col2 = st.columns([3, 1])
        with col1:
            show_columns = st.multiselect(
                "Select columns to display",
                options=results_df.columns.tolist(),
                default=[
                    'datetime', 'temp', 'probability_lstm', 'probability_catboost',
                    'LSTM_alert_level', 'Catboost_alert_level',
                    'combined_alert_level', 'recommended_action'
                ]
            )

        with col2:
            filter_alert = st.selectbox(
                "Filter by Alert",
                options=['All'] + results_df['combined_alert_level'].unique().tolist()
            )
        
        display_df = results_df.copy()
        if filter_alert != 'All':
            display_df = display_df[display_df['combined_alert_level'] == filter_alert]
        
        st.dataframe(
            display_df[show_columns],
            use_container_width=True,
            height=400
        )

        st.subheader("Hourly Alert Heatmap")

        heatmap_df = results_df.copy()

        # Clean datetime parts
        heatmap_df["datetime"] = pd.to_datetime(heatmap_df["datetime"], errors="coerce")
        heatmap_df = heatmap_df.dropna(subset=["datetime"])

        # Extract clean date + hour
        heatmap_df["date"] = heatmap_df["datetime"].dt.date
        heatmap_df["hour"] = heatmap_df["datetime"].dt.hour + 1

        # Convert date → formatted strings ("Nov 5, 2025")
        heatmap_df["date_str"] = heatmap_df["date"].apply(
            lambda d: pd.to_datetime(d).strftime("%b %d, %Y")
        )

        # Remove any blank values
        heatmap_df = heatmap_df[heatmap_df["date_str"].str.strip() != ""]

        # Map alert levels to numeric
        alert_map = {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}
        heatmap_df["alert_numeric"] = (
            heatmap_df["combined_alert_level"].map(alert_map).fillna(0)
        )

        # Pivot: rows = dates, cols = hours
        heatmap_data = heatmap_df.pivot_table(
            values="alert_numeric",
            index="date_str",
            columns="hour",
            aggfunc="max",
            fill_value=0
        )

        # Ensure no blank index remains
        heatmap_data = heatmap_data.reset_index()
        heatmap_data = heatmap_data[heatmap_data["date_str"].str.strip() != ""]
        heatmap_data = heatmap_data.set_index("date_str")

        # Sort by the actual date column (chronological), not the string
        heatmap_data = heatmap_data.sort_index(
            key=lambda x: pd.to_datetime(x, format="%b %d, %Y"),
            ascending=False
        )

        # Create text annotations for Q40.5 and Q41 alerts
        # Use text characters instead of emojis
        alert_annotations = heatmap_df.groupby(['date_str', 'hour'])['alert'].apply(
            lambda x: '⏰' if 'Q40.5' in x.values else ('!' if 'Q41' in x.values else '')
        ).unstack(fill_value='')

        # Align with heatmap_data
        alert_annotations = alert_annotations.reindex(index=heatmap_data.index, columns=heatmap_data.columns, fill_value='')

        # Build heatmap
        fig_heatmap = go.Figure(
            data=go.Heatmap(
                z=heatmap_data.values,
                x=list(range(1, 25)),
                y=heatmap_data.index,
                text=alert_annotations.values,
                texttemplate='%{text}',
                textfont=dict(size=20, color='white', family='Arial Black'),  # Bold white text
                colorscale=[
                    [0, "#2ecc71"], 
                    [0.25, "#3498db"],
                    [0.55, "#f1c40f"],
                    [1, "#e74c3c"]
                ],
                zmin=0,
                zmax=3,
                colorbar=dict(
                    title="Alert Level",
                    tickvals=[0, 1, 2, 3],
                    ticktext=["None", "Low", "Medium", "High"]
                ),
                xgap=2,
                ygap=2,
                hovertemplate="Date: %{y}<br>Hour: %{x}:00<br>Alert Level: %{z}<extra></extra>"
            )
        )

        fig_heatmap.update_layout(
            title="Alert Intensity by Date and Hour (⏰ = Q40.5, ! = Q41)",
            xaxis=dict(
                title="Hour of Day",
                tickmode="linear",
                dtick=1,
                side="top"
            ),
            yaxis=dict(title="Date"),
            height=500 + (len(heatmap_data) * 22),
            margin=dict(l=40, r=40, t=70, b=40)
        )

        fig_heatmap.update_layout(
            title="Alert Intensity by Date and Hour (🕐 = Q40.5, ❗= Q41)",
            xaxis=dict(
                title="Hour of Day",
                tickmode="linear",
                dtick=1,
                side="top"
            ),
            yaxis=dict(title="Date"),
            height=500 + (len(heatmap_data) * 22),
            margin=dict(l=40, r=40, t=70, b=40)
        )

        st.plotly_chart(fig_heatmap, use_container_width=True)

else:
    # Show this when data is loaded but predictions haven't been run yet
    st.info("👈 Select a calf and date range, then click 'Run Prediction' to see results")



# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <small>Calf Health Monitoring System | Powered by Cattle Scan</small>
</div>
""", unsafe_allow_html=True)