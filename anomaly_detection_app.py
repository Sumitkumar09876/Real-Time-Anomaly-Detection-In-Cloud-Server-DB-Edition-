import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import time
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
# Import our database module
import database as db

# Page configuration - use full width
st.set_page_config(
    page_title="Real-time Anomaly Detection",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern look - with improved visibility and contrast
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem !important;
        color: white; /* Changed text color to white */
        margin-bottom: 0.5rem;
    }
    .subheader {
        font-size: 1.5rem;
        color: white; /* Changed text color to white */
        margin-bottom: 1rem;
    }
    .card {
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        padding: 20px;
        margin-bottom: 20px;
        background-color: transparent; /* Removed white background */
    }
    .metric-container {
        display: flex;
        justify-content: space-between;
    }
    .metric {
        text-align: center;
        padding: 10px;
        border-radius: 5px;
        background-color: transparent; /* Removed white background */
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        margin-bottom: 0;
        color: white; /* Changed text color to white */
    }
    .metric-label {
        font-size: 0.85rem;
        color: #475569;
        font-weight: 500;
    }
    .stButton>button {
        border-radius: 20px;
        padding: 10px 25px;
        font-weight: 500;
        border: none;
        transition: all 0.3s ease;
    }
    .control-button {
        width: 100%;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .anomaly-alert {
        background-color: rgba(254, 226, 226, 0.9);
        border-left: 5px solid #EF4444;
        padding: 10px 20px;
        border-radius: 5px;
        color: #991B1B;
        font-weight: 500;
    }
    .info-text {
        background-color: rgba(219, 234, 254, 0.9);
        border-left: 5px solid #3B82F6;
        padding: 10px 20px;
        border-radius: 5px;
        color: #1E40AF;
    }
    /* Improve sidebar aesthetics */
    .sidebar .sidebar-content {
        background-color: #F8FAFC;
    }
    /* Full width content */
    .block-container {
        max-width: 100%;
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    /* Custom progress bar styling */
    .progress-bar-bg {
        height: 8px;
        background-color: #f1f5f9;
        border-radius: 4px;
        margin-bottom: 15px;
    }
    .progress-bar-fill {
        height: 100%;
        border-radius: 4px;
    }
    /* Button container styling */
    .button-container {
        display: flex;
        gap: 10px;
        margin-bottom: 15px;
    }
    /* Status indicator dot */
    .status-dot {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 8px;
    }
    /* Compact header */
    .header-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 15px;
    }
    /* Options container */
    .options-container {
        background-color: #f8fafc;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
        border: 1px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

# Title of the app with custom styling - moved to a compact header
st.markdown(
    '<div class="header-container">'
    '<div><h1 class="main-header">Real-time Anomaly Detection in Cloud Servers</h1>'
    '<p>Monitor and detect unusual behavior in your cloud infrastructure</p></div>'
    '</div>', 
    unsafe_allow_html=True
)

# Initialize session state variables if not already initialized
if 'is_paused' not in st.session_state:
    st.session_state.is_paused = False
if 'start' not in st.session_state:
    st.session_state.start = 0  # Start index for simulation
if 'current_batch' not in st.session_state:
    st.session_state.current_batch = None  # To keep the last batch of data visible
if 'db_session' not in st.session_state:
    # Create a database session when the app starts
    st.session_state.db_session = db.get_database_connection()
if 'anomaly_count' not in st.session_state:
    st.session_state.anomaly_count = 0
if 'total_points' not in st.session_state:
    st.session_state.total_points = 0
if 'last_anomaly_time' not in st.session_state:
    st.session_state.last_anomaly_time = None
# Initialize data buffer for smooth transitions
if 'data_buffer' not in st.session_state:
    st.session_state.data_buffer = pd.DataFrame()
if 'buffer_size' not in st.session_state:
    st.session_state.buffer_size = 100  # Number of points to show in the live view

# Function to generate synthetic data
def generate_synthetic_data(n_samples=1000):
    time_values = np.arange(n_samples)
    # Base patterns with some randomness
    cpu_usage = 0.5 + 0.2 * np.sin(time_values / 50) + np.random.normal(0, 0.05, n_samples)
    memory_usage = 0.3 + 0.1 * np.cos(time_values / 30) + np.random.normal(0, 0.03, n_samples)
    network_activity = 0.4 + 0.15 * np.sin(time_values / 40 + 2) + np.random.normal(0, 0.06, n_samples)
    
    # Add some anomalies randomly
    for i in range(n_samples // 50):  # Create anomalies at ~2% rate
        anomaly_idx = np.random.randint(n_samples)
        anomaly_type = np.random.choice(['cpu', 'memory', 'network', 'all'])
        
        if anomaly_type == 'cpu' or anomaly_type == 'all':
            cpu_usage[anomaly_idx] = np.random.choice([np.random.uniform(0.9, 1.0), np.random.uniform(0, 0.1)])
        if anomaly_type == 'memory' or anomaly_type == 'all':
            memory_usage[anomaly_idx] = np.random.choice([np.random.uniform(0.8, 1.0), np.random.uniform(0, 0.05)])
        if anomaly_type == 'network' or anomaly_type == 'all':
            network_activity[anomaly_idx] = np.random.choice([np.random.uniform(0.9, 1.0), np.random.uniform(0, 0.1)])
    
    # Ensure values are within reasonable ranges
    cpu_usage = np.clip(cpu_usage, 0, 1)
    memory_usage = np.clip(memory_usage, 0, 1)
    network_activity = np.clip(network_activity, 0, 1)
    
    # Create dataframe
    data = pd.DataFrame({
        'time': time_values,
        'cpu_usage': cpu_usage,
        'memory_usage': memory_usage,
        'network_activity': network_activity
    })
    
    return data

# Generate the data once and store it globally
if 'synthetic_data' not in st.session_state:
    st.session_state.synthetic_data = generate_synthetic_data(1000)

# Debug function to print diagnostics
def debug_print(message):
    if debug_mode:
        st.sidebar.write(f"DEBUG: {message}")

# Create a compact control panel above the visualization
control_col1, control_col2, control_col3, control_col4 = st.columns([1, 1, 1, 2])

with control_col1:
    start_button = st.button(
        "🚀 Start Detection", 
        key="start_button",
        help="Begin real-time anomaly detection",
        use_container_width=True
    )
    if start_button:
        st.session_state.start = 1
        if 'data_buffer' not in st.session_state or st.session_state.data_buffer.empty:
            # Initialize buffer with first few points
            start_idx = 0
            end_idx = min(st.session_state.buffer_size // 2, len(st.session_state.synthetic_data))
            st.session_state.data_buffer = st.session_state.synthetic_data[start_idx:end_idx].copy()

with control_col2:
    pause_button = st.button(
        "⏯️ Pause/Resume" if not st.session_state.is_paused else "⏯️ Resume",
        key="pause_button",
        help="Pause or resume the detection process",
        use_container_width=True
    )
    if pause_button:
        st.session_state.is_paused = not st.session_state.is_paused

with control_col3:
    view_history_button = st.button(
        "📊 View History",
        key="history_button",
        help="View historical anomaly data",
        use_container_width=True
    )

# Status indicator - inline with buttons
with control_col4:
    status = "Running" if st.session_state.start > 0 and not st.session_state.is_paused else "Paused" if st.session_state.is_paused else "Ready"
    status_color = "#15B525" if status == "Running" else "#FFA500" if status == "Paused" else "#7D7C7C"
    
    st.markdown(f"""
    <div style="display:flex; align-items:center; height:48px; padding:0 15px; background-color:#f8fafc; border-radius:5px;">
        <div class="status-dot" style="background-color:{status_color};"></div>
        <span style="font-weight:500; color:#334155;">Status: <span style="color:{status_color}; font-weight:bold;">{status}</span></span>
        <span style="margin-left:auto; font-weight:500; color:#334155;">Anomalies: <span style="font-weight:bold; color:#1E3A8A;">{st.session_state.anomaly_count}</span></span>
    </div>
    """, unsafe_allow_html=True)

# Create a two-column layout for the main content - use full width effectively
main_col1, main_col2 = st.columns([4, 1])

# Main visualization area in the left column (larger)
with main_col1:
    # Alert placeholder for anomalies
    alert_placeholder = st.empty()
    
    # Enhanced visualization section - moved up and made larger
    st.markdown('<div class="card" style="padding:15px;">', unsafe_allow_html=True)
    
    # Main visualization placeholder
    visualization_placeholder = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)
    
# System overview and metrics in the right column (smaller)
with main_col2:
    # System overview card - streamlined
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h3 class="subheader">System Overview</h3>', unsafe_allow_html=True)
    
    # Anomaly metrics in a clearer format
    cols = st.columns(1)
    
    with cols[0]:
        # Anomaly rate
        if st.session_state.total_points > 0:
            anomaly_rate = f"{(st.session_state.anomaly_count / st.session_state.total_points) * 100:.2f}%"
        else:
            anomaly_rate = "0.00%"
            
        # Last anomaly time
        last_time = "Never" if not st.session_state.last_anomaly_time else st.session_state.last_anomaly_time.strftime("%H:%M:%S")
        
        # Improved metrics display
        st.markdown(f"""
        <div style="margin-bottom:15px;">
            <div style="display:flex; justify-content:space-between;">
                <span style="color:#475569; font-weight:500;">Anomaly Rate:</span>
                <span style="font-weight:bold; color:#1E3A8A;">{anomaly_rate}</span>
            </div>
        </div>
        <div style="margin-bottom:15px;">
            <div style="display:flex; justify-content:space-between;">
                <span style="color:#475569; font-weight:500;">Last Anomaly:</span>
                <span style="font-weight:bold; color:#1E3A8A;">{last_time}</span>
            </div>
        </div>
        <div style="margin-bottom:15px;">
            <div style="display:flex; justify-content:space-between;">
                <span style="color:#475569; font-weight:500;">Total Points:</span>
                <span style="font-weight:bold; color:#1E3A8A;">{st.session_state.total_points}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Current data point card - improved visibility
    st.markdown('<div class="card" style="margin-top: 20px;">', unsafe_allow_html=True)
    st.markdown('<h3 class="subheader">Current Metrics</h3>', unsafe_allow_html=True)
    
    if st.session_state.current_batch is not None and len(st.session_state.current_batch) > 0:
        latest_point = st.session_state.current_batch.iloc[-1]
        # Show current metrics
        for feature in ['cpu_usage', 'memory_usage', 'network_activity']:
            if feature in latest_point:
                value = latest_point[feature]
                # Color code based on value
                color = "#15B525"  # Green for normal values
                if value > 0.8:
                    color = "#EF4444"  # Red for high values
                elif value > 0.6:
                    color = "#F59E0B"  # Orange for moderately high values
                    
                st.markdown(f"""
                <div style="margin-bottom:15px;">
                    <div style="display:flex; justify-content:space-between; margin-bottom:5px;">
                        <span style="color:#475569; font-weight:500;">{feature.replace('_', ' ').title()}</span>
                        <span style="font-weight:bold; color:{color};">{value:.2f}</span>
                    </div>
                    <div class="progress-bar-bg">
                        <div class="progress-bar-fill" style="width:{value*100}%; background-color:{color};"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Show if current point is anomaly
        if 'predicted_anomaly' in latest_point and latest_point['predicted_anomaly'] == 1:
            st.markdown("""
            <div style="background-color:#FEE2E2; border-left:5px solid #EF4444; padding:10px; margin-top:10px; border-radius:5px;">
                <span style="font-weight:bold; color:#991B1B;">⚠️ Anomaly Detected!</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("<p style='color:#475569;'>No data available yet</p>", unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Sidebar for user input with improved styling
with st.sidebar:
    st.markdown('<h3 class="subheader">Detection Configuration</h3>', unsafe_allow_html=True)
    
    # Model Parameters
    st.markdown('<div class="options-container">', unsafe_allow_html=True)
    st.markdown('<p style="font-weight:bold; color:#334155;">Model Parameters</p>', unsafe_allow_html=True)
    contamination = st.slider("Anomaly Threshold", 0.01, 0.1, 0.02, 
                            help="Lower values mean fewer anomalies")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Simulation Parameters
    st.markdown('<div class="options-container">', unsafe_allow_html=True)
    st.markdown('<p style="font-weight:bold; color:#334155;">Visualization Settings</p>', unsafe_allow_html=True)
    simulation_speed = st.slider("Update Speed", 0.1, 2.0, 0.5, 
                               help="Time between data points (seconds)")
    buffer_size = st.slider("Display Buffer Size", 50, 300, 100,
                          help="Number of points to show in graph")
    if buffer_size != st.session_state.buffer_size:
        st.session_state.buffer_size = buffer_size
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Feature Selection
    st.markdown('<div class="options-container">', unsafe_allow_html=True)
    st.markdown('<p style="font-weight:bold; color:#334155;">Features</p>', unsafe_allow_html=True)
    features = st.multiselect("Select Metrics to Monitor", 
                            ['cpu_usage', 'memory_usage', 'network_activity'], 
                            default=['cpu_usage', 'memory_usage', 'network_activity'])
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Database options
    st.markdown('<div class="options-container">', unsafe_allow_html=True)
    st.markdown('<p style="font-weight:bold; color:#334155;">Storage Options</p>', unsafe_allow_html=True)
    save_to_db = st.checkbox("Save anomalies to database", value=True)
    db_type = st.selectbox("Database Type", ["sqlite", "mysql"], index=0)
    st.markdown('</div>', unsafe_allow_html=True)

    # Debug mode toggle
    debug_mode = st.checkbox("Debug Mode", value=False)

# Historical data view - shown only when requested
view_history = view_history_button  # Set from button press

if view_history:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h3 class="subheader">Historical Anomalies</h3>', unsafe_allow_html=True)
    
    # Time range selection
    hours_to_look_back = st.slider("Hours to look back", 1, 72, 24)
    historical_data = db.get_historical_anomalies(hours=hours_to_look_back, session=st.session_state.db_session)
    
    if not historical_data.empty:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Enhanced histogram
            fig_hist = px.histogram(
                historical_data, 
                x='timestamp', 
                nbins=20,
                title="Anomaly Distribution Over Time",
                color_discrete_sequence=['#4361EE'],  # Consistent with main graph
                opacity=0.8  # Add some transparency
            )

            fig_hist.update_layout(
                title=dict(
                    font=dict(size=20, family="Arial, sans-serif"),
                    x=0.5,  # Center title
                ),
                xaxis_title=dict(text="Time", font=dict(size=14, family="Arial, sans-serif")),
                yaxis_title=dict(text="Number of Anomalies", font=dict(size=14, family="Arial, sans-serif")),
                plot_bgcolor='rgba(250, 250, 250, 0.9)',
                paper_bgcolor='rgba(255, 255, 255, 0)',
                font=dict(family="Arial, sans-serif"),
                height=300,
                margin=dict(l=20, r=20, t=40, b=20)
            )

            # Add grid lines for better readability
            fig_hist.update_xaxes(
                showgrid=True,
                gridwidth=0.5,
                gridcolor='rgba(226, 232, 240, 0.6)'
            )
            fig_hist.update_yaxes(
                showgrid=True,
                gridwidth=0.5,
                gridcolor='rgba(226, 232, 240, 0.6)'
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            # Feature correlation during anomalies
            corr_features = ['cpu_usage', 'memory_usage', 'network_activity']
            corr_data = historical_data[corr_features].corr()
            # Enhanced correlation heatmap
            fig_corr = px.imshow(
                corr_data, 
                text_auto=True, 
                color_continuous_scale=[
                    [0, "#F9F9F9"],
                    [0.3, "#C2D3F9"],
                    [0.6, "#6B93E6"],
                    [0.8, "#3B65B8"],
                    [1, "#1E3A8A"]
                ],  # Custom color scale for better contrast
                title="Feature Correlation Analysis",
                labels=dict(color="Correlation")
            )

            fig_corr.update_layout(
                title=dict(
                    font=dict(size=20, family="Arial, sans-serif"),
                    x=0.5,  # Center title
                ),
                height=300,
                margin=dict(l=20, r=20, t=50, b=20),
                coloraxis_colorbar=dict(
                    title="Correlation",
                    tickvals=[-1, -0.5, 0, 0.5, 1],
                    tickfont=dict(size=12)
                ),
                font=dict(family="Arial, sans-serif")
            )

            # Improve text display on heatmap
            fig_corr.update_traces(
                textfont=dict(
                    size=14,
                    family="Arial, sans-serif",
                    color="black"
                ),
                texttemplate="%{text:.2f}"  # Format to 2 decimal places
            )
            st.plotly_chart(fig_corr, use_container_width=True)
        
        # Show the data in an interactive table
        st.markdown('<p style="font-weight: 500; color: #334155; margin-top: 20px;">Anomaly Records</p>', unsafe_allow_html=True)
        st.dataframe(historical_data, use_container_width=True)
        
    else:
        st.markdown('<div class="info-text">No historical anomalies found in the selected time period.</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Real-time anomaly detection implementation
if st.session_state.start > 0 and not st.session_state.is_paused:
    debug_print(f"Processing batch starting at index {st.session_state.start}")
    
    # Get batch of data to process
    start_idx = min(st.session_state.start, len(st.session_state.synthetic_data) - 30)
    end_idx = min(start_idx + 5, len(st.session_state.synthetic_data))  # Process smaller batches for smoother updates
    
    debug_print(f"Batch range: {start_idx} to {end_idx}")
    
    if start_idx < end_idx:
        current_batch = st.session_state.synthetic_data[start_idx:end_idx].copy()
        
        # Make sure we have selected features for detection
        if not features:
            features = ['cpu_usage', 'memory_usage', 'network_activity']
        
        debug_print(f"Using features: {features}")
        
        # Train Isolation Forest model
        model = IsolationForest(contamination=contamination, random_state=42)
        model.fit(current_batch[features])
        
        # Predict anomalies (-1 for anomalies, 1 for normal)
        predictions = model.predict(current_batch[features])
        
        # Convert predictions to boolean (True for anomalies)
        current_batch['predicted_anomaly'] = (predictions == -1).astype(int)
        
        # Store current batch in session state
        st.session_state.current_batch = current_batch
        
        # Update metrics
        anomalies_in_batch = current_batch['predicted_anomaly'].sum()
        st.session_state.anomaly_count += anomalies_in_batch
        st.session_state.total_points += len(current_batch)
        
        debug_print(f"Found {anomalies_in_batch} anomalies in this batch")
        
        if anomalies_in_batch > 0:
            st.session_state.last_anomaly_time = datetime.now()
        
        # Save to database if enabled
        if save_to_db:
            try:
                debug_print("Saving to database...")
                db.save_batch_to_database(current_batch, session=st.session_state.db_session)
                debug_print("Database save successful")
            except Exception as e:
                st.error(f"Database error: {str(e)}")
                debug_print(f"Database error details: {str(e)}")
        
        # Update data buffer for smooth visualization (rolling window)
        st.session_state.data_buffer = pd.concat([st.session_state.data_buffer, current_batch]).iloc[-st.session_state.buffer_size:]
        buffer_data = st.session_state.data_buffer  # Add this line to define buffer_data

        # Define y-axis range variables
        y_min = 0
        y_max = 1.1
        
        # Create classic-looking visualization
        fig = go.Figure()

        # Use traditional, classic colors
        colors = {
            'cpu_usage': '#4F81BD',       # Classic Excel blue
            'memory_usage': '#C0504D',    # Classic Excel red
            'network_activity': '#9BBB59'  # Classic Excel green
        }

        # Add traces for each feature with traditional styling
        for feature in features:
            fig.add_trace(go.Scatter(
                x=buffer_data['time'],
                y=buffer_data[feature],
                mode='lines',
                name=feature.replace('_', ' ').title(),
                line=dict(
                    width=2, 
                    color=colors.get(feature, '#000000'),
                    shape='linear',  # Straight lines for classic look
                    dash=None        # Solid line
                ),
                hoverinfo='y+name'
            ))

        # Mark anomalies with classic styling
        if 'predicted_anomaly' in current_batch.columns:
            anomalies = current_batch[current_batch['predicted_anomaly'] == 1]
            if not anomalies.empty:
                for feature in features:
                    fig.add_trace(go.Scatter(
                        x=anomalies['time'],
                        y=anomalies[feature],
                        mode='markers',
                        marker=dict(
                            symbol='circle',  # Classic circle markers
                            size=10,
                            color='red',      # Traditional red for anomalies
                            line=dict(width=1, color='darkred')
                        ),
                        name=f'Anomaly in {feature.replace("_", " ").title()}',
                        showlegend=False,
                        hoverinfo='y+name'
                    ))

        # Classic layout styling - with improved text visibility
        fig.update_layout(
            title=dict(
                text="Server Resource Monitoring",
                font=dict(size=20, family="Times New Roman", color="#000000")  # Black text
            ),
            xaxis=dict(
                title=dict(
                    text="Time",
                    font=dict(size=15, family="Times New Roman", color="#000000")  # Black text
                ),
                showgrid=True,
                gridwidth=1,
                gridcolor='#E5E5E5',
                zeroline=True,
                zerolinecolor='#000000',
                tickfont=dict(family="Times New Roman", size=12, color="#000000")  # Black text
            ),
            yaxis=dict(
                title=dict(
                    text="Resource Usage",
                    font=dict(size=15, family="Times New Roman", color="#000000")  # Black text
                ),
                showgrid=True,
                gridwidth=1,
                gridcolor='#E5E5E5',
                range=[y_min, y_max],
                zeroline=True,
                zerolinecolor='#000000',
                tickformat=".0%",  # Show as percentage
                tickfont=dict(family="Times New Roman", size=12, color="#000000")  # Black text
            ),
            plot_bgcolor='white',  # Classic white background
            paper_bgcolor='white',
            legend=dict(
                font=dict(family="Times New Roman", size=12, color="#000000"),  # Black text
                bgcolor='white',
                bordercolor='#CCCCCC',
                borderwidth=1
            ),
            margin=dict(l=20, r=20, t=50, b=50),
            height=500
        )

        # Display the plot
        visualization_placeholder.plotly_chart(fig, use_container_width=True, config={
            'displayModeBar': False,  # Hide the modern display bar for classic look
            'scrollZoom': True
        })
        
        debug_print("Plot rendered successfully")
        
        # Increment for next batch
        st.session_state.start += 1
        
        # Sleep to control simulation speed
        time.sleep(simulation_speed)
        
        # Rerun to update UI
        debug_print("Rerunning for next batch...")
        st.rerun()
    else:
        st.session_state.start = 0  # Reset when we reach the end
        st.markdown('<div class="info-text">Simulation complete. Click "Start Detection" to restart.</div>', unsafe_allow_html=True)

# Display the last batch when paused
elif st.session_state.is_paused and st.session_state.data_buffer is not None and not st.session_state.data_buffer.empty:
    debug_print("Showing paused state")
    
    # Use the entire buffer for the paused state
    buffer_data = st.session_state.data_buffer

    # Define y-axis range variables for paused state
    y_min = 0
    y_max = 1.1

    # Traditional, classic colors
    colors = {
        'cpu_usage': '#4F81BD',       # Classic Excel blue
        'memory_usage': '#C0504D',    # Classic Excel red
        'network_activity': '#9BBB59'  # Classic Excel green
    }

    fig = go.Figure()

    # Traditional, classic colors
    colors = {
        'cpu_usage': '#4F81BD',       # Classic Excel blue
        'memory_usage': '#C0504D',    # Classic Excel red
        'network_activity': '#9BBB59'  # Classic Excel green
    }

    # Add traces with classic styling
    for feature in features:
        fig.add_trace(go.Scatter(
            x=buffer_data['time'],
            y=buffer_data[feature],
            mode='lines',
            name=feature.replace('_', ' ').title(),
            line=dict(
                width=2, 
                color=colors.get(feature, '#000000'),
                shape='linear'  # Straight lines for classic look
            ),
            hoverinfo='y+name'
        ))

    # Classic anomaly markers
    if st.session_state.current_batch is not None and 'predicted_anomaly' in st.session_state.current_batch.columns:
        anomalies = st.session_state.current_batch[st.session_state.current_batch['predicted_anomaly'] == 1]
        if not anomalies.empty:
            for feature in features:
                fig.add_trace(go.Scatter(
                    x=anomalies['time'],
                    y=anomalies[feature],
                    mode='markers',
                    marker=dict(
                        symbol='circle',
                        size=10,
                        color='red',
                        line=dict(width=1, color='darkred')
                    ),
                    name=f'Anomaly in {feature.replace("_", " ").title()}',
                    showlegend=False,
                    hoverinfo='y+name'
                ))

    # Classic layout styling - with improved text visibility
    fig.update_layout(
        title=dict(
            text="Server Resource Monitoring",
            font=dict(size=20, family="Times New Roman", color="#000000")  # Black text
        ),
        xaxis=dict(
            title=dict(
                text="Time",
                font=dict(size=15, family="Times New Roman", color="#000000")  # Black text
            ),
            showgrid=True,
            gridwidth=1,
            gridcolor='#E5E5E5',
            zeroline=True,
            zerolinecolor='#000000',
            tickfont=dict(family="Times New Roman", size=12, color="#000000")  # Black text
        ),
        yaxis=dict(
            title=dict(
                text="Resource Usage",
                font=dict(size=15, family="Times New Roman", color="#000000")  # Black text
            ),
            showgrid=True,
            gridwidth=1,
            gridcolor='#E5E5E5',
            range=[y_min, y_max],
            zeroline=True,
            zerolinecolor='#000000',
            tickformat=".0%",  # Show as percentage
            tickfont=dict(family="Times New Roman", size=12, color="#000000")  # Black text
        ),
        plot_bgcolor='white',  # Classic white background
        paper_bgcolor='white',
        legend=dict(
            font=dict(family="Times New Roman", size=12, color="#000000"),  # Black text
            bgcolor='white',
            bordercolor='#CCCCCC',
            borderwidth=1
        ),
        margin=dict(l=20, r=20, t=50, b=50),
        height=500
    )

    # Display the plot
    visualization_placeholder.plotly_chart(fig, use_container_width=True, config={
        'displayModeBar': False,  # Hide the modern display bar for classic look
        'scrollZoom': True
    })

else:
    debug_print("Showing initial state")
    # Show placeholder instruction when not started
    visualization_placeholder.markdown(
        '<div class="info-text" style="text-align:center; padding:40px; font-size:1.2rem;">'
        '<p style="font-size:3rem; margin:0;">📊</p>'
        '<p style="font-weight:500; margin-top:20px;">Click "Start Detection" to begin monitoring server metrics</p>'
        '</div>',
        unsafe_allow_html=True
    )

# Add a save_batch_to_database function if needed
def save_batch_to_database(batch_df, session=None):
    """
    Save a batch of data to the database
    batch_df: DataFrame with the current batch
    session: SQLAlchemy session (if None, a new session will be created)
    """
    close_session = False
    if session is None:
        session = get_database_connection()
        close_session = True
    
    try:
        # Convert DataFrame rows to AnomalyRecord objects
        records = []
        for _, row in batch_df.iterrows():
            # Check if 'predicted_anomaly' column exists, otherwise use 'anomaly' or default to False
            is_anomaly = False
            if 'predicted_anomaly' in row:
                is_anomaly = bool(row['predicted_anomaly'])
            elif 'anomaly' in row:
                is_anomaly = bool(row['anomaly'])
            
            record = AnomalyRecord(
                time_value=int(row['time']),
                cpu_usage=float(row['cpu_usage']),
                memory_usage=float(row['memory_usage']),
                network_activity=float(row['network_activity']),
                is_anomaly=is_anomaly
            )
            records.append(record)
        
        # Add all records to database
        session.add_all(records)
        session.commit()
        
    except Exception as e:
        session.rollback()
        print(f"Error saving to database: {e}")
        raise e
    finally:
        if close_session:
            session.close()