# <span style='color:steelblue'> Machine Learning </span>

Empowering machines or systems to learn from data and make predictions or decisions without explicit rules or programming, revolutionizing tasks like pattern recognition, natural language processing, and predictive modeling across various domains.

## <span style='color:orange'> Supervised Learning </span>

Supervised learning is a machine learning paradigm where the algorithm learns from labeled data, which consists of input-output pairs. The algorithm learns to map input data to the corresponding output labels, making predictions or decisions when given new, unseen data. The goal is to generalize from the training data to accurately predict the output for new, unseen instances.

### Regression

#### Simple / Multiple Linear Regression

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

#### Naive Bayes
    - Naive Bayes is a classification algorithm used for predicting the class (or category) of something based on a set of features (predictors). It works using the principles of probability, specifically Bayes' theorem.
- ``Classification Task``
    - Imagine you want to classify emails as spam or not spam. The "spam" and "not spam" are the classes, and features could be words appearing in the email, sender information, etc.
- `Bayes Theorem`
    - It is a fundamental concept in probability theory and statistics. It deals with conditional probability, which is the likelihood of an event occurring given that another event has already happened.
    - Understanding Bayes' theorem empowers you to make informed decisions by considering both prior knowledge.
    - Machine learning: Used in spam filtering, image recognition, medical diagnosis, and other classification tasks.
    - Risk assessment: Helps calculate the probability of an event happening based on historical data and new information.

```text
Formula:

P(A | B) = ( P(B | A) * P(A) ) / P(B)

where:

P(A | B) is the posterior probability of event A occurring, given that event B has already happened. This is what we're trying to calculate.
P(B | A) is the likelihood of event B occurring, given that event A is true.
P(A) is the prior probability of event A occurring, independent of any other event. This is our initial belief about how likely A is to happen.
P(B) is the total probability of event B occurring, regardless of A.
```
- `Naive Assumption`
    - The key concept in Naive Bayes is the assumption of independence between features. This means it assumes features don't influence each other (e.g., the appearance of the word "money" doesn't affect the influence of the word "urgent"). While this might not always be true in reality, it simplifies the calculations.
- `MultinomialNB and GaussianNB`: These are specific implementations of Naive Bayes designed for different data types
    - **MultinomialNB**: Suitable for discrete features (data with limited categories) and often used for count data. It uses the multinomial distribution to model the probability of features given a class.
        - Refers to the multinomial distribution, which helps us to understand what is a probability distribution for discrete outcomes with a fixed number of possibilities (like categories in text classification).
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

#### Decision Tree

A Decision Tree classifier is a powerful machine learning algorithm used for classification tasks. It works by building a tree-like model that splits the data based on features (attributes) to predict a target variable (class label).

Idealogy

``Features``: The characteristics used for prediction (e.g., Outlook, Temperature, Humidity, Wind).

``Target Variable``: The category you're trying to predict (e.g., Play Tennis: Yes/No).

``Splitting``: At each node of the tree, the data is split based on a feature value that best separates the data points belonging to different classes.

``Information gain``:

```
Information Gain(S, A) = Entropy(S) - Σ [ |S_v| / |S| * Entropy(S_v) ]

S: Represents the entire dataset (all samples) for which you want to calculate the information gain.
A: Represents a specific attribute (feature) in your data (e.g., Outlook, Temperature, Humidity, Wind in the Play Tennis example).
Entropy(S): This finds the impurity/randomness of the dataset S, measured using the Entropy function. It reflects how mixed up the classes (Yes/No for playing tennis) are in the entire dataset.
Σ: This is a summation symbol.
|S_v|: This represents the number of data points in a specific branch (subset) of the tree created by splitting on attribute A (e.g., the number of days with Sunny outlook).
|S|: This is the total number of data points in the entire dataset (S).
Entropy(S_v): This is the Entropy calculated for each branch (subset) S_v created by splitting on attribute A (e.g., Entropy for days with Sunny outlook).
```

- It measures the reduction in uncertainty about the target variable after a split. The feature with the highest information gain becomes the splitting criterion at that node.

- This process continues recursively, creating branches based on different feature values. Each branch represents a set of data points that share similar characteristics.

- At the terminal nodes (leaves) of the tree, the most frequent class label in that branch becomes the prediction.

Essentially, Information Gain measures the difference in impurity between the entire dataset (S) and the subsets (S_v) created after splitting on a particular attribute (A).  The higher the Information Gain i.e., ``variable who's split can have less impurity/randomness/more counts belong to one class``, the more the chosen attribute (A) helps in separating the data points belonging to different classes (Yes/No for playing tennis).

- `In the essence, |S_v| / |S| i.e., subset count is multiplied with Entropy(S_v) of subset, because it is helps in emphasising, if there is higher entropy with higher subset counts i.e., attribute each unique category counts. Then this multiplication results in high value / higher magnitude, think like penalising the outlier due to having extreme value`.

- `Total Entropy(S) is substracted with above explained emphasised Subset Entropy to see whether subset entropy is lesser than average overall entropy or not. If Entropy(S) - Σ [ |S_v| / |S| * Entropy(S_v) ] results in higher value i.e., +ve, it means Total Entropy(S) (is having higher entropy value) and  Σ [ |S_v| / |S| * Entropy(S_v) ] (is having low entropy value), in other words this subset attribute entropy (i.e., at each unique value of this attribute) is better(less random/certain) than overall uncertainity. Hence this is good attribute for splitting as it is having positive Information gain / Less entropy value than overall Entropy`.

``Entropy``:
The entropy function in machine learning, particularly in decision trees, measures the uncertainty or randomness within a dataset. It tells you how mixed up the different classes (categories) are in your data.
Imagine a bag of balls:

- ``Low Entropy``: If the bag only contains red balls, you can be very certain of picking a red ball (low uncertainty).
- ``High Entropy``: If the bag has a mix of red, blue, and green balls, it's less certain which color you'll pick (high uncertainty).
- ``Interpretation``: The result is a non-negative number. A value closer to zero indicates high certainty (e.g., all days recommend playing tennis - low entropy). A value closer to 1 (logarithm of 1/2) indicates maximum uncertainty (equal mix of Yes and No recommendations - high entropy).
- ``Keypoints``: Entropy helps us understand how well our data is separated into distinct classes. It provides a basis for choosing the best features to split on in a decision tree, aiming to reduce uncertainty in each branch.

```
Entropy(S) = -Σ [p(i) * log2(p(i))]

Σ (sigma symbol): This represents a sum over all possible classes (e.g., Yes/No for playing tennis).
p(i): This represents the probability of encountering class 'i' (the probability of picking a red, blue, or green ball).
log2(p(i)): This is a mathematical function (logarithm base 2) that helps penalize very low or very high probabilities. 
```

Formula understanding

- ``Penalizes Very High and Low Probabilities (log2(p(i)) ~ 1 => 0, or log2(p(i)) ~ 0 => -3.321)``
    - This makes sense because if a class is very likely, it contributes less to the overall uncertainty, vise-versa if a class is very unlikely, it contributes high to the overall uncertainty. 
    - In other words, ``if there are 9 red balls and 1 orange ball out 10 balls``
    - Red class contribute less to randomness/uncertainity to overall randomness of the whole data. As our intention to find how much uncertainity(how many diffetent color counts or all ball colors are same) exists in the overwall data interms of target variable i.e., ball color. 
    - So here we wanted to penalise or avoid colors which are very certain/high in counts i.e., 0, and want to highlight/emphasise the colors which are very uncertain/rare i.e., -3.321. 
    - For this we use log2, which serves this purpose. It penalise high probability i.e., 1 / 0.99 / 0.98 .. value as ~0 (as we don't to know about red color where it is already certain), but log2 penalise less probability i.e., 0.1 / 0.2 .. as ~ -3.321 (as we want to know about ex: orange color where it is very rare in the data)

- ``Magnifies Penalised Value by Multiplying``
    - The logarithm of values less than 1 is negative, and multiplying by the probability strengthens this negative value. This emphasizes the contribution of classes with moderate or lower probabilities to the overall uncertainty.
    - In other words, complete same color class gets log2(p(var)) = 0 hence results in low uncertainity, but when there is differen color class and different counts out of total counts then this is less < probability 1, which penalises log2 function and results in -ve values and multplying it with probability value of that color(0<p<1) further emphasises its value (if less minus => less effected with multiplication, more minus value => more effected with multiplication) to see which color has more uncertanities.
```
lets say target class i.e., color

Ex:- If red 10 = total balls 10
probability of red color ball => -(1 * math.log2(1)) => -(1 * 0) => 0 => Low uncertainity for red color class

Ex:- If red 9, orange 1 = total balls 10 
probability of red color ball => -(0.99 * math.log2(0.99)) => 0.014 => Low uncertainity for red color class
probability of orange color ball => -(0.01 * math.log2(0.01)) => 0.066 => Low uncertainity red color class

Ex:- If red 5, orange 5 = total balls 10
probability of red color ball => -(0.5 * math.log2(0.5)) => 0.5 => High uncertainity for red color class
```

``Pros``:

- Easy to interpret: The tree structure visually depicts the decision-making process.

- Handles both categorical and numerical features.

- Can work well with missing data.

``Cons``:

- Prone to overfitting if not carefully tuned.

- Sensitive to small changes in the data.


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
