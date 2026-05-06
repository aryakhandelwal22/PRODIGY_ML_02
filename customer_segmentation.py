import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load dataset
df = pd.read_csv("Mall_Customers.csv")

# Display basic info
print("PRODIGY_ML_02 - Customer Segmentation")
print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Shape:", df.shape)

# Select features
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]


# ELBOW METHOD
wcss = []

for i in range(1, 11):
    kmeans = KMeans(
        n_clusters=i,
        init='k-means++',
        random_state=42
    )
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plot Elbow Graph
plt.figure(figsize=(10, 6), facecolor='#F8C8DC')
plt.plot(range(1, 11), wcss, marker='o', linewidth=3)

plt.title("PRODIGY_ML_02", fontsize=18, fontweight='bold')
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.grid(True)

plt.savefig("elbow_method.png")
plt.show()


# APPLY KMEANS
kmeans = KMeans(
    n_clusters=5,
    init='k-means++',
    random_state=42
)

clusters = kmeans.fit_predict(X)

# Add cluster labels
df['Cluster'] = clusters

# CUSTOMER SEGMENTATION GRAPH

plt.figure(figsize=(12, 8), facecolor='#F8C8DC')

for i in range(5):
    plt.scatter(
        X.iloc[clusters == i, 0],
        X.iloc[clusters == i, 1],
        s=80,
        label=f'Cluster {i+1}'
    )

# Centroids
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s=300,
    marker='X',
    c='black',
    label='Centroids'
)

plt.title("PRODIGY_ML_02", fontsize=20, fontweight='bold')
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.grid(True)

plt.savefig("cluster_output.png")
plt.show()

# Cluster summary
print("\nCluster Distribution:")
print(df['Cluster'].value_counts())