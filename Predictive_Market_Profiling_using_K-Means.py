import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Display settings for cleaner output

pd.set_option('display.max_columns',None)
sns.set(style="whitegrid")

print("Libraries imported successfully.")

# ==========================================
#  Data Loading
# ==========================================
df = pd.read_csv("data/ifood_df.csv")

print(df.head())

# ===============================
# Exploratory Data Analysis
# ===============================

# Display the shape (rows, columns)
print("Dataset Shape:", df.shape)

# Display detailed info
print("\n--- Dataset Info ---")
print(df.info())

# ==========================================
#  Data Quality Check: Missing Values, Duplicates, Summary Stats
# ==========================================

# 1. Check for missing values
print(" MISSING VALUES PER COLUMN:")
print(df.isnull().sum())

# 2. Check for duplicate rows
duplicates = df.duplicated().sum()
print(f" DUPLICATE ENTRIES: {duplicates}")

# Remove duplicates if found
if duplicates > 0:
    df = df.drop_duplicates()
    print(f" Duplicates removed. New dataset shape: {df.shape}")
else:
    print(" No duplicate rows found.")

# 3. Summary statistics after duplicate removal
print(" SUMMARY STATISTICS (After Cleaning):")
print(df.describe())

# ==========================================
#  Outlier Detection using Boxplot & Histograms
# ==========================================

# Select key numerical columns for visualization
num_cols = [
    "Income", "Age", "MntWines", "MntFruits", "MntMeatProducts",
    "MntFishProducts", "MntSweetProducts", "MntGoldProds",
    "MntTotal", "MntRegularProds"
]

# ==========================================
#  Boxplot for Each Feature (Individually)
# ==========================================

plt.figure(figsize=(18, 16))

for i, col in enumerate(num_cols,1):
  plt.subplot(5,2,i)
  sns.boxplot(x=df[col], color='skyblue')
  plt.title(f"{col}", fontsize=12)
  plt.xlabel("")  # cleaner look
plt.tight_layout()
plt.show()

# Visualize histograms
df[num_cols].hist(bins=20, figsize=(16,10), color='skyblue')
plt.suptitle("Distribution of Numerical Features", fontsize=16)
plt.tight_layout()
plt.show()

# ==========================================
#  Cleaning Data: Remove Impossible Values and Duplicates
# ==========================================

# 1. Remove negative spending (impossible values)
neg_values = (df["MntRegularProds"] < 0).sum()
print(f"Negative MntRegularProds rows: {neg_values}")

df = df[df["MntRegularProds"] >= 0]
print("Negative values removed.")

# 2. Remove rows where spending exceeds income
illogical = (df["MntTotal"] > df["Income"]).sum()
print(f"Rows where MntTotal > Income: {illogical}")

df=df[df["MntTotal"]<= df["Income"]]
print("Illogical spending rows removed.")

# Final shape
print("\nFinal dataset shape after cleaning:", df.shape)

# ==========================================
#  Feature Selection for Customer Segmentation with K-Means
# ==========================================

selected_features = [
    # Spending behavior
    "MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts",
    "MntSweetProducts", "MntGoldProds", "MntRegularProds", "MntTotal",

    # Engagement behavior
    "NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases", "NumWebVisitsMonth",

    # Value indicators
    "Recency", "Income", "Age"
]

df_selected=df[selected_features]

print(df_selected.head())

# ==========================================
#  Scaling Features for K-Means
# ==========================================

scaler = StandardScaler()
scaled_data= scaler.fit_transform(df_selected)

# Convert back to a DataFrame for easier viewing
df_scaled= pd.DataFrame(scaled_data, columns=df_selected.columns)

print(df_scaled.head())

# ==========================================
#  Elbow Method to Determine Optimal k -  Finding the Optimal Number of Clusters
# ==========================================

from sklearn.cluster import KMeans
from matplotlib.lines import lineStyles
wcss=[]

# Try different values of k
for k in range(1,11):
  kmeans= KMeans(n_clusters=k, random_state=42)
  kmeans.fit(df_scaled)
  wcss.append(kmeans.inertia_)


# Plot
plt.figure(figsize=(8,4))
plt.plot(range(1,11),wcss, marker='o', linestyle='-')
plt.title("Elbow Method - Optimal Number of Clusters")
plt.xlabel("Number of clusters (k)")
plt.ylabel("WCSS")
plt.grid(True)
plt.show()

# Determining Optimal k using Silhouette Score

for k in range(2,7):
  kmeans=KMeans(n_clusters=k, random_state=42)
  preds = kmeans.fit_predict(df_scaled)
  score = silhouette_score(df_scaled, preds)
  print(f"Silhouette Score for k={k}: {score:.3f}")

# ==========================================
#  Apply K-Means Clustering (k = 2)
# ==========================================

kmeans = KMeans(n_clusters=2, random_state=42)
cluster_labels=kmeans.fit_predict(df_scaled)

# Add cluster labels to the dataset
df['Cluster']= cluster_labels
df_selected= df_selected.copy()
df_selected['Cluster']= cluster_labels

df.head()

# Value counts of the cluster column
print(df['Cluster'].value_counts())

# ==========================================
#  Cluster Profiling & Customer Behavior Analysis: Mean Values per Cluster
# ==========================================

cluster_profile = df.groupby("Cluster")[selected_features].mean()

print(cluster_profile)

# ==========================================
#  Visualizing Customer Segments: Income vs. Total Spending (Scatter Plot Insights)
# ==========================================

plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="Income", y="MntTotal", hue="Cluster", palette="viridis", s=60, alpha=0.7)
plt.title("Cluster Separation by Income vs Total Spending")
plt.show()

# NumWebVisitsMonth vs NumWebPurchases

plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="NumWebVisitsMonth", y="NumWebPurchases", hue="Cluster", palette="viridis", s=60,
                  alpha=0.7)
plt.title("Cluster Separation: Browsing vs Buying Behavior")
plt.show()

# Age vs Total Spending

plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="Age", y="MntTotal", hue="Cluster", palette="viridis", s=60, alpha=0.7)
plt.title("Cluster Separation: Age vs Total Spending")
plt.show()

# ==========================================
#  Cluster Analysis: Behavioral & Demographic Differences
# ==========================================

# Bar Chart: Cluster Comparison of Key Features

# Define groups
spending = ["MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts",
              "MntSweetProducts", "MntGoldProds", "MntRegularProds", "MntTotal"]

engagement = ["NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases", "NumWebVisitsMonth"]

#  Spending Patterns Comparison

# Plot spending
plt.figure(figsize=(12, 5))
cluster_profile[spending].T.plot(kind="bar", figsize=(12, 5), colormap="viridis")
plt.title("Cluster Comparison — Spending Features (Avg)")
plt.ylabel("Average Value")
plt.xticks(rotation=45)
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.show()

# Engagement Behavior Comparison

plt.figure(figsize=(10, 4))
cluster_profile[engagement].T.plot(kind="bar", figsize=(10, 4), colormap="viridis")
plt.title("Cluster Comparison — Engagement Features (Avg)")
plt.ylabel("Average Value")
plt.xticks(rotation=45)
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.show()

#  Income Comparison

# Values
income_values = cluster_profile['Income'].values  # [value_cluster0, value_cluster1]

# Bar labels (cluster numbers)
clusters = ['0', '1']

# Colors for each bar
colors = ["#4B0082", "#FFD700"]  # Purple, Yellow

plt.figure(figsize=(6, 4))
plt.bar(clusters, income_values, color=colors, width=0.6)

plt.title("Cluster Comparison — Income")
plt.ylabel("Average Income")
plt.xlabel("Cluster")
plt.grid(axis="y", linestyle="--", alpha=0.6)

plt.show()

# Demographics: Age & Recency

plt.figure(figsize=(6, 4))
cluster_profile[['Age', 'Recency']].T.plot(kind="bar", figsize=(6, 4), colormap="viridis")
plt.title("Cluster Comparison — Age & Recency")
plt.ylabel("Average Value")
plt.xticks(rotation=0)
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.show()