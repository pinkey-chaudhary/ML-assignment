Customer Segmentation Using Machine Learning

1. Introduction

Customer Segmentation is a machine learning project that groups customers based on their annual income and spending score. It helps businesses understand customer behaviour and identify different customer groups.

2. Objective

- Analyze customer data.
- Group customers into different segments.
- Visualize customer clusters.
- Compare average spending scores.
- Predict the cluster of a new customer.

3. Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- K-Means Clustering

4. Algorithm

This project uses the K-Means Clustering algorithm, an unsupervised machine learning technique. The Elbow Method is used to help select the number of clusters.

5. Project Features

- Customer data generation
- Data preprocessing and feature scaling
- Elbow Method graph
- Customer segmentation graph
- Average spending score bar chart
- New customer cluster prediction
- CSV result export

6. Project Files

- "customer_segmentation.py" – Main Python code
- "elbow_method.png" – Elbow graph
- "customer_clusters.png" – Customer clusters graph
- "spending_analysis.png" – Spending analysis bar chart
- "customer_segmentation_result.csv" – Clustered customer data

7. How to Run

Install the required libraries:

pip install pandas numpy matplotlib seaborn scikit-learn

Run the project:

python customer_segmentation.py

Enter the annual income and spending score when prompted to predict a new customer's cluster.

8. Conclusion

The project demonstrates how K-Means Clustering can be used to segment customers based on their income and spending patterns. It also provides visual analysis and a simple prediction feature.
