from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import pandas as pd

# Load the synthetic dataset
data = pd.read_csv('cloud_server_synthetic_data.csv')

# Select the features and target
X = data[['cpu_usage', 'memory_usage', 'network_activity']]
y_true = data['anomaly']  # This is the actual anomaly label

# Initialize Isolation Forest model
iso_forest = IsolationForest(contamination=0.02, random_state=42)

# Fit the model to the data (training)
iso_forest.fit(X)

# Predict the anomalies (-1 for anomaly, 1 for normal)
y_pred = iso_forest.predict(X)
# Convert predictions: -1 -> 1 (anomaly), 1 -> 0 (normal)
y_pred = [1 if p == -1 else 0 for p in y_pred]

# Evaluate the model
print(classification_report(y_true, y_pred))
print(f"Accuracy: {accuracy_score(y_true, y_pred) * 100:.2f}%")

# Plot a sample of true vs predicted anomalies
plt.figure(figsize=(10, 6))
plt.plot(data['time'][:500], data['cpu_usage'][:500], label='CPU Usage')
plt.scatter(data['time'][:500], data['cpu_usage'][:500] * y_pred[:500], color='blue', label='Predicted Anomaly', marker='o')
plt.scatter(data['time'][:500], data['cpu_usage'][:500] * data['anomaly'][:500], color='red', label='True Anomaly', marker='x')
plt.legend()
plt.title('Cloud Server Anomalies (Predicted vs True Anomalies)')
plt.xlabel('Time')
plt.ylabel('CPU Usage')
plt.show()
