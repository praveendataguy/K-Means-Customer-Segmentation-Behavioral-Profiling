K-Means Customer Segmentation & Behavioral Profiling:

An end-to-end unsupervised machine learning pipeline engineered to segment customers based on their purchasing habits, digital engagement metrics, and baseline demographics. By implementing K-Means Clustering optimized via Elbow Method and Silhouette Analysis, this workflow translates raw transactional records into actionable consumer personas.

📌 Project Overview:

Understanding distinct consumer patterns is critical for targeted marketing and resource optimization. This repository processes historical customer data from iFood, audits data quality, normalizes multivariant feature spaces, and builds an intuitive statistical model to identify high-value buyers versus highly engaged browsers.

Key Analytical Pillars:

Data Auditing & Validation: Programmatic drop-handling for duplicate tracking and strict execution of domain-specific business rules (e.g., filtering out negative values or spend-to-income logical errors).Feature Scaling: Uniform transformation of heavily skewed financial data and integer frequencies using StandardScaler to prevent distance-metric bias.Hyperparameter Tuning: Dual-verification of optimal cluster density $k$ through Within-Cluster Sum of Squares (WCSS) minimization and mean silhouette coefficients.Persona Profiling: Post-hoc statistical aggregation coupled with dual-axis scatter mapping and categorical bar-chart diagnostics.

🛠️ Tech Stack & Dependencies:

The pipeline is designed using standard production-grade python data science libraries:

Core Operations: Python 3.8+, Pandas

Mathematical Operations: NumPy

Machine Learning Framework: Scikit-Learn

Visualization Engine: Matplotlib, Seaborn

📂 Pipeline Architecture:

The workflow is structured sequentially into 10 decoupled sections for optimal interpretability and maintenance:

Graph TD

    A[1. Data Ingestion] --> B[2. Data Quality Audit]
    B --> C[3. EDA & Outlier Detection]
    C --> D[4. Business Logic Validation]
    D --> E[5. Feature Selection]
    E --> F[6. Matrix Standardization]
    F --> G[7. Hyperparameter Tuning]
    G --> H[8. K-Means Execution]
    H --> I[9. Cluster Profiling]
    I --> J[10. Segment Diagnostics]

1. Data Ingestion & Inspection:

Initial import and structural evaluation (.shape, .info(), and header verification) of the raw ifood_df.csv file.

2. Data Quality Audit:

Automated identification of missing array entries, summary statistic generations, and handling of duplicate records:

if duplicates > 0:
    df = df.drop_duplicates()

3. Exploratory Data Analysis (EDA):

Parallel visualization of numerical value spreads using custom multi-axis Matplotlib subplots for individual boxplots and master feature histograms.

4. Business Logic Validation:

Enforces programmatic constraints to filter out mathematically impossible anomalies:

Drops records where $\text{MntRegularProds} < 0$.

Drops records where $\text{MntTotal} > \text{Income}$.

5. Feature Engineering & Selection:

Isolates 15 core parameters distributed across three primary behavioral dimensions:

Spending Dimensions: MntWines, MntFruits, MntMeatProducts, MntFishProducts, MntSweetProducts, MntGoldProds, MntRegularProds, MntTotal.

Engagement Channels: NumWebPurchases, NumCatalogPurchases, NumStorePurchases, NumWebVisitsMonth.

Value Indicators: Recency, Income, Age.

6. Matrix Standardization:

Scales structural components to a standard normal distribution (mean $\mu = 0$, variance $\sigma^2 = 1$) to ensure the distance calculations in K-Means weigh all attributes equitably.

7. Hyperparameter Tuning:

Runs an iterative optimization loop ($1 \le k \le 10$) mapping out the inertia curve (Elbow Method) alongside calculating Silhouette Scores across candidate spaces.

8. Model Execution:

Instantiates the final configuration of the K-Means algorithm using a locked random_state for perfect pipeline reproducibility, writing the cluster labels directly back to the active dataframe.

9. Post-Hoc Profiling:

Group-by aggregations that parse out the arithmetic means of every metric per cluster to clearly delineate group boundaries.

10. Segment Visualization & Diagnostics:

Generates multi-variant scatter plots and comparative bar charts evaluating spending power vs web conversion metrics across segments.