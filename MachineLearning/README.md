# <span style='color:steelblue'> Machine Learning </span>

Empowering machines or systems to learn from data and make predictions or decisions without explicit rules or programming, revolutionizing tasks like pattern recognition, natural language processing, and predictive modeling across various domains.

## <span style='color:orange'> Supervised Learning </span>

Supervised learning is a machine learning paradigm where the algorithm learns from labeled data, which consists of input-output pairs. The algorithm learns to map input data to the corresponding output labels, making predictions or decisions when given new, unseen data. The goal is to generalize from the training data to accurately predict the output for new, unseen instances.

### Regression

- **`Simple / Multiple Linear Regression`**

    - In linear regression, we assume that the relationship between the dependent variables and the independent variables is linear.
    - Mathematically, this relationship can be represented as: `y = β0 + β1x + β2x + ε`(This equation represents simple linear regression) or `y = mx + c` (This equation represents the equation of a straight line in Cartesian coordinates used in geometry to describe the relationship between x and y in a plane).
        - Here, 𝑦 is the dependent variable.
        - 𝑥1 is the independent variable.
        - 𝛽0 is the represents the y-intercept, which is the value of y when x is zero.
            - Let's say you're plotting data on a graph where the horizontal axis (x-axis) represents time, and the vertical axis (y-axis) represents temperature. If you have a y-intercept of 10, it means that when time (x) is zero, the temperature (y) starts at 10.
            - So, in simpler terms, the y-intercept (𝛽0) tells you where the line starts on the y-axis, regardless of what's happening with the x-axis.
        - 𝛽1 is the coefficient/slope representing the effect of first independent variable on the dependent variable, and ε is the error term.
        - 𝛽2 is the coefficient/slope representing the effect of second independent variable on the dependent variable, and ε is the error term.
        - m(slope/coefficient/weight) = cov(x, y) / var(x)
        - c(constant/intercept/bias) = y - m*x
    - The goal of simple linear regression is to find the values of β0 + β1 + β2 that minimize the sum of squared differences between the actual and predicted values of the dependent variable (which is a performance metric to find how good the model is).
    - `Optimisation Method[Gradient Descent]`
        - Gradient descent can be used to optimize the coefficients (parameters) of the linear regression model. It iteratively updates the coefficients in the direction that minimizes the chosen loss function.
    - `Loss Function`
        - **Ordinary Least Squares (OLS)**: Is method used to estimate the parameters (coefficients) of a linear regression model by minimizing the sum of the squared differences between the observed and predicted values.
            - OLS => sum i->n (yi-y`i)2    
        - **Mean Squared Error (MSE)**: Is the most common loss function for linear regression, which measures the average squared difference between the predicted and actual values. The objective is to minimize the MSE.
            - MSE => 1/n sum i->n (yi-y`i)2
                - observed y values and predicted y` values
        - Both OLS and MSE serve as cost functions in the context of linear regression, with OLS being the specific method used to optimize the model parameters and MSE being a metric used to evaluate the model's performance.
    - `Performance Metrics`
        - **Root Mean Squared Error (RMSE)**: RMSE is the square root of the MSE and provides a measure of the average magnitude of errors in the same units as the target variable.
            - RMSE = root(MSE)
        - **Coefficient of Determination (R-squared):**: R-squared measures the proportion of the variance in the dependent variable that is predictable from the independent variables. It ranges from 0 to 1, with higher values indicating better model fit.
            - R2 => 1 - (sum i->n (yi-yˉi)2 / sum i->n (yi-y'i)2 ). yˉ represents the mean of the predicted values. y' represents the mean of the observed values.
        - **Adjusted R-squared:**: Adjusted R-squared is a modified version of R-squared that penalizes the addition of unnecessary predictors in the model. It adjusts for the number of predictors and provides a more reliable measure of model fit. The formula for adjusted R-squared varies slightly depending on the number of predictors in the model.
    - `Assumptions`
        - **Linear Relationship**: There should be a linear relationship between the independent variable(s) and the dependent variable.
        - **Independence**: The residuals (the differences between the observed and predicted values) should be independent of each other. In other words, there should be no systematic pattern in the residuals.
        - **Homoscedasticity**: The variance of the residuals should be constant across all levels of the independent variable(s). This means that the spread of the residuals should be uniform along the regression line.
        - **Normality**: The residuals should be normally distributed. This means that the distribution of the residuals should resemble a bell-shaped curve when plotted.
        - **No Multicollinearity**: If there are multiple independent variables, they should not be highly correlated with each other. High multicollinearity can lead to unstable estimates of the coefficients and inflated standard errors.
        - **No Autocorrelation**: For time-series data, there should be no autocorrelation in the residuals. Autocorrelation occurs when the residuals are correlated with themselves across time.

    Meeting these expectations helps ensure that the linear regression model provides reliable and interpretable results. Violations of these assumptions may lead to biased estimates and inaccurate predictions. Therefore, it's essential to assess and address any violations before interpreting the results of a linear regression model.

### Classification

- **`Naive Bayes`**
    - Naive Bayes is a classification algorithm used for predicting the class (or category) of something based on a set of features (predictors). It works using the principles of probability, specifically Bayes' theorem.
    - `Classification Task`
        - Imagine you want to classify emails as spam or not spam. The "spam" and "not spam" are the classes, and features could be words appearing in the email, sender information, etc.
    - `Bayes Theorem`
        - Naive Bayes leverages Bayes' theorem, which allows us to calculate the probability of an event (email being spam) given some evidence (words in the email).
    - `Naive Assumption`
        - The key concept in Naive Bayes is the assumption of independence between features. This means it assumes features don't influence each other (e.g., the appearance of the word "money" doesn't affect the influence of the word "urgent"). While this might not always be true in reality, it simplifies the calculations.
    - `MultinomialNB and GaussianNB`: These are specific implementations of Naive Bayes designed for different data types
        - **MultinomialNB**: Suitable for discrete features (data with limited categories) and often used for count data. It uses the multinomial distribution to model the probability of features given a class.
            - Refers to the multinomial distribution, which is a probability distribution for discrete outcomes with a fixed number of possibilities (like categories in text classification).
            - Data type: Discrete (often count data)
            - Usage: Classifying text (spam/not spam), sentiment analysis
            - `The multinomial distribution concept provides a powerful tool for modeling and analyzing scenarios with discrete outcomes in various Machine Learning applications`.
                - Text Classification: Classifying documents into categories (sports, politics, etc.) based on word frequencies. Here, the categories are the classes (sports, politics), and the trials are word occurrences in the document.
        - **GaussianNB**: Suitable for continuous features (numerical data that can take any value within a range). It assumes features follow a Gaussian (normal) distribution.
            - Data type: Continuous
            - Classification: Involves predicting discrete categories (e.g., spam/not spam email, cat/dog image).
            - GaussianNB Assumptions: Assumes features (data points) for each class follow a Gaussian (normal) distribution. This means the data for each class tends to cluster around a central value with a bell-shaped curve.
            - Predictions: Based on these assumptions, GaussianNB calculates the probability of a new data point belonging to each class and assigns it to the class with the highest probability.
            - `GaussianNB cannot directly model the continuous relationship between features and the target variable.`
    - `Pros`
        - Simple to understand and implement.
        - Efficient for large datasets.
        - Performs well even with limited data compared to some other algorithms.
    - `Cons`
        - The assumption of feature independence can be unrealistic in some cases.
        - Sensitive to irrelevant features.

Below mathematical formula is core concept behind the Naive Bayes pricipal for both MultinomialNB and GaussianNB classifiers.

```md
Prior Probability => `(P(Play = yes))`
This is general chance of playing_tennis[yes] on any given day (regardless of weather evidence/condition)

Likelihood => `(P(Evidence/Condition | Play = yes))`
Chance of having the specific weather i.e., (sunny, hot, high humidity, no wind) given that someone will playing_tennis[yes].


Predict probability => `P(Play = yes | Evidence) = [Prior Probability] * [Likelihood]`
Predict probability of playing_tennis[yes] on given weather condition
```

In essence, Naive Bayes is a powerful and versatile classification algorithm that leverages probability for effective predictions. While its assumptions might not always hold perfectly, it often provides good results and is a popular choice for various classification tasks.

## <span style='color:orange'> Unsupervised Learning </span>

Extracting patterns from unlabeled data, enabling automatic discovery of hidden structures without explicit guidance. Essential for clustering, dimensionality reduction, and anomaly detection in diverse datasets.

### Anomaly-Detection

It helps in unveiling of unusual patterns in data, vital for fraud detection, fault diagnosis, and uncovering outliers, enhancing data integrity and decision-making processes.

#### 3 Sigma/Standard Deviation (or) 3 Z-score Method


**Sigma/Standard Deviation**: This method identifies outliers based on their deviation from the mean. Data points that are a certain number of standard deviations i.e., 3(in most cases) away from the mean are considered outliers.

```text
anomaly <= if abs(x - mean) > stdev*3
```

**Z-score**: It's a statistical measure that tells you how many standard deviations a specific point is away from the mean (average) of the data. We calculate the Z-score by subtracting the mean (μ) from a data point (x) and then dividing by the standard deviation (σ).

```text
Z-score = (x - μ) / σ.

A Z-score of 0 indicates the data point is exactly at the mean. Positive Z-scores represent values above the mean, and negative Z-scores represent values below the mean.

anomaly <= if abs(z-score) > z-score*3 
```

In other words, that 3 sigma rule suggests that most data points (around 99.7%) will have Z-scores within ±3 (positive or negative). Data points with Z-scores outside this range (absolute value greater than 3) are considered potential anomalies because they deviate significantly from the typical pattern.

Assumption: Both above method assumes data follows a normal distribution. As Sigma / Z-score are based on normal distribution.

#### IQR Method

This method uses the Interquartile Range (IQR) to identify outliers. Data points beyond a certain range from the first and third quartiles are considered outliers.

IQR tells us about the variability of the middle 50% of the data: Unlike measures like the range or standard deviation, which consider all data points, the IQR focuses on the central portion of the data distribution. It provides information about how the values are spread out around the median.
Identifying potential outliers: By using the IQR, one can identify potential outliers in the dataset. Typically, data points that fall below Q1 - 1.5 * IQR or above Q3 + 1.5 * IQR are considered outliers and may warrant further investigation.

Assumptions: No assumptions about the data distribution are made.

#### DBSCAN

Density-Based Spatial Clustering of Applications with Noise (DBSCAN) is a clustering algorithm that can also be used for outlier detection. It identifies outliers as points that are in low-density regions.

Assumptions: Assumes data points form clusters and outliers are isolated points.


## <span style='color:orange'> Forecasting </span>

Time series forecasting is a branch of predictive analytics that focuses on predicting future values of a variable based on its past values, often represented as a sequence of data points collected at successive, equally spaced time intervals. In simpler terms, it involves using historical data to make predictions about future values of a time-dependent variable.

- Methods and Models:

    Various methods and models are used for time series forecasting, including:

    - **Statistical Methods**: These include simple methods like moving averages and exponential smoothing, as well as more complex models like ARIMA (Autoregressive Integrated Moving Average) and seasonal decomposition.
    - **Machine Learning Techniques**: Machine learning algorithms such as neural networks, support vector machines, and random forests can also be used for time series forecasting.
    - **Deep Learning**: Deep learning models like recurrent neural networks (RNNs) and Long Short-Term Memory (LSTM) networks are particularly effective for capturing complex temporal dependencies in time series data.

### Timeseries Smoothing

Time series smoothing is a technique used to remove noise or fluctuations from a time series data while preserving the underlying trends or patterns. It involves applying a mathematical filter or algorithm to the data to create a smoother representation.

Use Cases

- **Noise Reduction**: Smoothing helps to filter out random fluctuations or noise in the data, making it easier to identify underlying trends or patterns.
- **Trend Identification**: By removing short-term fluctuations, smoothing techniques make it easier to identify long-term trends or cycles in the data.
- **Data Visualization**: Smoothed time series plots provide a clearer and more interpretable representation of the data, facilitating visual analysis and decision-making.
- **Forecasting**: Smoothing can improve the accuracy of forecasting models by providing a cleaner input dataset with reduced noise.
- **Data Preprocessing**: Smoothing is often used as a preprocessing step before applying more complex time series analysis or forecasting techniques.
- **Anomaly Detection**: Smoothing helps to highlight anomalies or unusual patterns in the data by removing normal fluctuations, making it easier to detect outliers.

`Moving Average`

- **Definition**: Moving average is a simple smoothing technique that calculates the average of a sliding window of data points across the time series.
- **Usage**: It is effective for filtering out short-term fluctuations or noise in the data while preserving long-term trends.

```text
yt = 1/n (sum i->n (yti+1))
```

`Loess (Locally Weighted Scatterplot Smoothing)`

- **Definition**: Loess smoothing is a non-parametric technique that fits a curve to the data by locally weighted regression, assigning higher weights to nearby points and lower weights to distant points.
- **Usage**: It is effective for capturing complex patterns and relationships in the data, especially when there are nonlinear trends or seasonal variations.

```text
yt = ß0 + ß1x1 + et
```
