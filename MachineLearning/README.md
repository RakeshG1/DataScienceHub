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

## <span style='color:orange'> Unsupervised Learning </span>

Extracting patterns from unlabeled data, enabling automatic discovery of hidden structures without explicit guidance. Essential for clustering, dimensionality reduction, and anomaly detection in diverse datasets.

### Anomaly-Detection

It helps in unveiling of unusual patterns in data, vital for fraud detection, fault diagnosis, and uncovering outliers, enhancing data integrity and decision-making processes.


## <span style='color:orange'> Timeseries </span>

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
