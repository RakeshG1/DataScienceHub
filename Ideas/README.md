# Identify busy and non-busy hours for machine sensor telemetry over the past 60 days

## Statistical Approaches

### Time Aggregation and Descriptive Analysis:

    - Aggregate telemetry data by hour over 60 days (e.g., average or total per hour).
    - Calculate the mean and standard deviation for hourly telemetry counts.
    - Define thresholds:
        - Busy hours: ``Above mean + k × std``
        - Non-busy hours: ``Below mean - k×std``

### IQR Outlier Detection:

    - Use the Interquartile Range (IQR) method to identify hours with unusually high telemetry counts:
        - Busy hours: Above ``Q3+1.5×IQR``
        - Non-busy hours: Below ``Q1−1.5×IQR``

### Peak Detection:
    
    - Apply rolling averages or z-scores to flag hours with significant peaks in telemetry counts.

## Machine Learning Approaches

### Clustering:

    - Use k-means clustering to group hours into "busy" and "non-busy" categories based on telemetry counts.
    - Features: Hourly telemetry counts over 60 days.

### Time-Series Decomposition:

    - Decompose the time-series into trend, seasonality, and residuals.
    - Busy hours are likely aligned with high seasonal or residual components.

### Anomaly Detection:

    - Apply anomaly detection models like Isolation Forest or One-Class SVM to flag unusually busy hours.

### Supervised Classification:

    - Label some hours manually as "busy" or "non-busy" based on domain knowledge.
    - Train a model (e.g., decision tree or logistic regression) to predict these categories for other hours.