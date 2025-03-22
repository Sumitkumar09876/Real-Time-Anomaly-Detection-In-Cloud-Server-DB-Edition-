# Real-time Anomaly Detection in Cloud Servers

This project implements real-time anomaly detection using `IsolationForest` in a `Streamlit` app. The system monitors CPU, memory, and network activity metrics in cloud servers and detects anomalies in real-time.

## User Interface
![Screenshot 2025-03-22 130004](https://github.com/user-attachments/assets/346aae39-a33e-4041-a583-55562159ee30)
![Screenshot 2025-03-22 130037](https://github.com/user-attachments/assets/45956452-5dba-408d-bf20-c33182cfc78d)
![Screenshot 2025-03-22 130058](https://github.com/user-attachments/assets/de5deefc-a6d7-4879-a3ed-15a91ea31722)

## Database
![Screenshot 2025-03-22 125759](https://github.com/user-attachments/assets/52f932af-b5d2-4212-9212-791c2cba884b)

## Features

- **Real-time Data Streaming & Anomaly Detection**: Detects anomalies in server metrics using the `IsolationForest` algorithm.
- **Classic Visualization**: Traditional Excel-style charts with clear visibility and readability.
- **Interactive Dashboard**: Professional monitoring interface with system overview metrics.
- **Pause/Resume Functionality**: Toggle the real-time simulation at any point.
- **Anomaly Alerts**: Visual alerts when anomalies are detected.
- **Historical Data Analysis**: View and analyze previously detected anomalies.
- **Customizable Parameters**: Adjust the contamination level, simulation speed, and selected features.
- **Database Integration**: Store and retrieve anomaly data using SQLite or MySQL.

## Tech Stack

- **Python 3.8+**
- **Streamlit**: For creating the interactive web app
- **Scikit-learn**: For the IsolationForest algorithm
- **Pandas & NumPy**: For data manipulation
- **Plotly**: For interactive visualizations
- **SQLite**: Lightweight, file-based database (default)
- **SQLAlchemy**: For database operations

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/real-time-anomaly-detection.git
   cd real-time-anomaly-detection
   ```

2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit app:
   ```bash
   streamlit run anomaly_detection_app.py
   ```

2. Open your web browser and go to `http://localhost:8501` to view the app.

## Project Structure

real-time-anomaly-detection/
│
├── anomaly_detection_app.py   # Main Streamlit application
├── database.py                # Database handling module
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation

## Database Configuration

The application supports SQLite and MySQL databases. Configure your database settings in the `database.py` file. The stored data includes detected anomalies and historical metrics.

## Customization Guide

Adjust the following parameters in the app:

- **Contamination Level**: Set the proportion of outliers in the data.
- **Simulation Speed**: Control the speed of the real-time simulation.
- **Selected Features**: Choose which metrics to monitor for anomalies.

This updated README provides:

1. **Expanded Features Section**: Highlights the classic visualization style and database features
2. **Complete Installation Instructions**: Step-by-step guide including virtual environment setup
3. **Usage Instructions**: Clear guidance on running and using the application
4. **Tech Stack Details**: Lists all major technologies used
5. **Project Structure**: Shows file organization
6. **Database Configuration**: Explains database options and stored data
7. **Customization Guide**: Information on adjustable parameters

The README now provides comprehensive information for users to understand, install, and use your application.

