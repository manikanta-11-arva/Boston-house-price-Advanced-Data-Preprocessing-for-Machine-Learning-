import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

# Load data
data = pd.read_csv("train.csv")

# ---------------- FEATURE ENGINEERING ----------------
data['CrimeLevel'] = pd.qcut(data['CRIM'], q=3, labels=[1, 2, 3]).astype(int)
data['OldBuildings'] = data['AGE'] / 100
data['PropertyWealth'] = data['TAX'] / 1000
data['Accessibility'] = data['DIS']
data['Rooms'] = data['RM']
data['SocioEconomicStatus'] = data['LSTAT'] / 100

# Features for clustering
features = [
    'CrimeLevel',
    'Rooms',
    'SocioEconomicStatus',
    'OldBuildings',
    'Accessibility',
    'PropertyWealth'
]

X = data[features]

# ---------------- SCALING ----------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------- KMEANS CLUSTERING ----------------
kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(X_scaled)

# ---------------- CLUSTER LABELING ----------------
centers = scaler.inverse_transform(kmeans.cluster_centers_)
cluster_mapping = {}

for i, center in enumerate(centers):
    rooms = center[1]
    ses = center[2]

    if rooms > 6 and ses < 0.15:
        cluster_mapping[i] = 'Premium Residential'
    elif rooms < 5 and ses > 0.25:
        cluster_mapping[i] = 'High-Crime Low-Income'
    else:
        cluster_mapping[i] = 'Middle-Income Neighborhood'

data['Cluster_Label'] = data['Cluster'].map(cluster_mapping)

# ---------------- SAVE CSV ----------------
data.to_csv("boston_housing_clustered.csv", index=False)
print("✅ File saved as boston_housing_clustered.csv")

# ---------------- 3D VISUALIZATION ----------------
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(
    data['Rooms'],
    data['SocioEconomicStatus'],
    data['CrimeLevel'],
    c=data['Cluster']
)

ax.set_xlabel("Average Rooms (RM)")
ax.set_ylabel("Socio-Economic Status (LSTAT)")
ax.set_zlabel("Crime Level")
ax.set_title("3D Clustering of Neighborhoods (Real Estate Segmentation)")

plt.legend(*scatter.legend_elements(), title="Clusters")
plt.show()
