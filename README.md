# Real-time Anomaly Detection in Cloud Servers

This project implements real-time anomaly detection using `IsolationForest` in a `Streamlit` app. The system monitors CPU, memory, and network activity metrics in cloud servers and detects anomalies in real-time.

![Screenshot 2024-10-23 211553](https://github.com/user-attachments/assets/c125a851-f7f3-42f4-9b87-e645346d6f69)
![Screenshot (16)](https://github.com/user-attachments/assets/4ad0c82e-ded9-4cfa-a305-ce3fdc0f309b)

## Features

- **Real-time Data Streaming & Anomaly Detection**: Detects anomalies in server metrics using the `IsolationForest` algorithm.
- **Pause/Resume Functionality**: Toggle the real-time simulation at any point.
- **Customizable Parameters**: Adjust the contamination level, simulation speed, and selected features (CPU usage, memory usage, network activity).
- **Live Plotting**: Real-time visualization of the server metrics and detected anomalies.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/real-time-anomaly-detection.git
   cd real-time-anomaly-detection
   
