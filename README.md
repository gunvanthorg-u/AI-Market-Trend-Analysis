AI-Market-Trend-Analysis
AI-Driven Market Trend Analysis and Customer Behavior Prediction  
Dataset:Marketing Data for Customer Behavior Analysis (Kaggle)
1. Problem Definition & Objective

Selected Project Track
AI Applications – Market Analytics

Problem Statement
In today’s competitive business environment, organizations collect large volumes of marketing and customer data. However, transforming this raw data into actionable insights for understanding customer behavior, identifying market trends, and predicting purchasing decisions remains a major challenge.
 Objective
The objective of this project is to develop an AI-driven system that:
- Analyzes customer marketing data
- Identifies market trends and customer segments
- Predicts customer response to marketing campaigns
Real-World Relevance
This project is relevant to:
- Retail and e-commerce analytics
- Digital marketing optimization
- Customer segmentation and targeting
- Business decision support systems


2. Data Understanding & Preparation
 Dataset Source
The dataset used is a publicly available Kaggle dataset titled
Marketing Data for Customer Behavior Analysis

Dataset Description
The dataset contains customer-related attributes such as demographic information,
spending behavior, and campaign response indicators.
 Data Preparation Steps
- Data loading and exploration
- Handling missing values
- Encoding categorical features
- Feature engineering (Age and Total Spending)
- Preparing numerical features for AI models

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("Marketing_data.csv")
print("Dataset Loaded:", df.shape)

# Create Age safely
if "Age" not in df.columns:
    age_col = [c for c in df.columns if "age" in c.lower()]
    if age_col:
        df["Age"] = df[age_col[0]]
    else:
        df["Age"] = df.select_dtypes(include=np.number).mean(axis=1)

Create Total Spending
spending_cols = [c for c in df.columns if c.lower().startswith("mnt")]
if spending_cols:
    df["MntTotal"] = df[spending_cols].sum(axis=1)
else:
    df["MntTotal"] = df.select_dtypes(include=np.number).sum(axis=1)

# Encode categorical columns
for col in df.select_dtypes(include="object").columns:
    df[col] = LabelEncoder().fit_transform(df[col])

df.head()
 3. Model / System Design

AI Techniques Used
- Machine Learning (ML)
- Unsupervised Learning – K-Means Clustering
- Supervised Learning – Random Forest Classifier

 System Pipeline
1. Data Loading
2. Data Cleaning and Feature Engineering
3. Customer Segmentation using K-Means
4. Market Response Prediction using Random Forest
5. Visualization and Evaluation

 Design Justification
K-Means clustering is used to identify customer segments efficiently.
Random Forest is chosen for prediction due to its robustness and ability to handle
non-linear relationships.
2 — Customer Segmentation


income_col = next((c for c in df.columns if "income" in c.lower()),
                  df.select_dtypes(include=np.number).columns[0])

features = df[[income_col, "Age", "MntTotal"]].fillna(0)
scaled_features = StandardScaler().fit_transform(features)

kmeans = KMeans(n_clusters=4, random_state=42)
df["Customer_Segment"] = kmeans.fit_predict(scaled_features)

plt.figure(figsize=(7,5))
plt.scatter(df[income_col], df["MntTotal"], c=df["Customer_Segment"], alpha=0.7)
plt.xlabel("Income Level")
plt.ylabel("Total Spending")
plt.title("Customer Segmentation using K-Means")
plt.colorbar(label="Customer Segment")
plt.grid(True)
plt.show()
4. Core Implementation

The core implementation includes:
- Feature scaling using StandardScaler
- Customer segmentation using K-Means clustering
- Market response prediction using Random Forest
- Visualization of market trends and segments

target_col = "Response" if "Response" in df.columns else df.columns[-1]

X = df.drop(target_col, axis=1)
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

 5. Evaluation & Analysis
- Accuracy
- Precision
- Recall
- F1-Score
 Analysis
The model demonstrates good predictive performance.
Customer segments show distinct spending behavior patterns.

Limitations
- Dataset size is limited
- No real-time data integration

  6. Ethical Considerations & Responsible AI

- The dataset does not contain sensitive personal identifiers.
- Potential bias may exist due to limited demographic diversity.
- The system is intended for decision support, not automated decision-making.
- Responsible AI practices were followed to ensure fairness and transparency.

 7. Conclusion & Future Scope

Conclusion
This project successfully demonstrates how AI can be used to analyze market trends,
segment customers, and predict purchasing behavior using marketing data.

 Future Scope
- Integration with real-time data sources
- Use of deep learning models for forecasting
- Deployment as an interactive dashboard
- Incorporation of LLM-based market insights
