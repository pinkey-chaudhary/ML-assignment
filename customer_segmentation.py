import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

n = 200

customer_data = pd.DataFrame({
    "CustomerID": range(1, n + 1),
    "Gender": np.random.choice(["Male", "Female"], n),
    "Age": np.random.randint(18, 65, n),
    "Annual Income (k$)": np.random.randint(15, 140, n),
    "Spending Score (1-100)": np.random.randint(1, 101, n)
})

print("\nCUSTOMER DATASET")
print(customer_data.head())

X = customer_data[
    ["Annual Income (k$)", "Spending Score (1-100)"]
]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nData preprocessing completed!")


wcss = []

for i in range(1, 11):
    kmeans = KMeans(
        n_clusters=i,
        random_state=42,
        n_init=10
    )

    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker="o")

plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.grid(True)
plt.savefig("elbow_method.png")
plt.show()


kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

customer_data["Cluster"] = kmeans.fit_predict(X_scaled)

print("\nCUSTOMER SEGMENTATION RESULT")
print(customer_data.head(10))


plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=customer_data,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    hue="Cluster",
    palette="viridis",
    s=100
)

plt.title("Customer Segmentation Using K-Means")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score")
plt.legend(title="Customer Cluster")
plt.grid(True)

plt.savefig("customer_clusters.png")
plt.show()


cluster_summary = customer_data.groupby("Cluster")[
    ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
].mean().round(2)

print("\nCUSTOMER GROUP ANALYSIS")
print(cluster_summary)

cluster_summary["Spending Score (1-100)"].plot(
    kind="bar",
    title="Average Spending Score by Cluster",
    figsize=(8, 5)
)

plt.xlabel("Customer Cluster")
plt.ylabel("Average Spending Score")
plt.tight_layout()
plt.savefig("spending_analysis.png")
plt.show()


customer_data.to_csv(
    "customer_segmentation_result.csv",
    index=False
)

print("\nCustomer data saved successfully!")


print("\n----- NEW CUSTOMER PREDICTION -----")

income = float(input("Enter Annual Income (k$): "))
spending = float(input("Enter Spending Score (1-100): "))

if income <= 0 or not 1 <= spending <= 100:
    print("Please enter valid customer details.")

else:
    new_customer = pd.DataFrame({
        "Annual Income (k$)": [income],
        "Spending Score (1-100)": [spending]
    })

    new_customer_scaled = scaler.transform(new_customer)

    predicted_cluster = kmeans.predict(new_customer_scaled)

    print("\nCUSTOMER PREDICTION RESULT")
    print("Annual Income:", income, "k$")
    print("Spending Score:", spending)
    print("Customer belongs to Cluster:", predicted_cluster[0])


print("\n----- PROJECT SUMMARY -----")
print("Total Customers:", len(customer_data))
print("Total Clusters:", customer_data["Cluster"].nunique())
print("Algorithm Used: K-Means Clustering")

print("\nPROJECT COMPLETED SUCCESSFULLY!")