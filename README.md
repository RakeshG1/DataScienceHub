# <span style='color:steelblue'> Data Science</span>

Data science is the field of study that uses scientific methods, algorithms, and systems to extract knowledge and insights from structured and unstructured data, enabling informed decision-making and predictive analysis across various domains.

## Statistics

### Mean

Mean is the average of a data set.

```text
Mean = (x1+x2+..xn) / (n) 
     = (Total number of numbers) / (Sum of all the numbers)
     = 2 + 4 + 6 + 8 / 4 => 5 
```

### Mode

It is a measure to find central tendency just like mean, but instead of avg. it focus on most frequently occuring value.

```text
Mode = {2, 3, 4, 4, 4, 5, 6} => 4
```

### Median

It is another measure to find central tendency just like mean & mode, but instead of avg. or most frequent value, it focuses on middle value in the dataset.

```text
If data set length is odd => value at x((n+1)/2)  

data set = {1, 3, 4, 6, 8}; length = 5

Median => x((5+1)/2) => x(6/2) => x(3) => 4

If data set length is even => (value at x(n/2) + value at x(n/2 + 1)) / 2

data set = {2, 4, 6, 8}; length = 4

Median => (value at x(4/2) + value at x(4/2 + 1)) / 2 => (x(2) + x(3)) / 2 => (4+6) / 2 => 5
```

### Standard Deviation

The standard deviation measures the amount of variation or dispersion in a set of values. It tells you how spread out the values are from the mean.

```text
Standard Deviation => sqrt(sum i->n (square(xi-𝑥ˉ)) / n-1)

xi: represents each individual value in the dataset.
𝑥ˉ: represents the mean (average) of the dataset.
𝑛: represents the total number of values in the dataset.

Steps:
1. Find the mean i.e., 𝑥ˉ of the dataset.
2. Subtract the mean from each individual value to get the deviation from the mean for each value.
3. Square each deviation to eliminate negative values and emphasize differences from the mean.
4. Find the average of these squared deviations.
5. Take the square root of this average to get the standard deviation.

Note: 
1. It's a way to measure the average distance of each data point from the mean, but now in the original units of the data. As square and square root get cancelled. So it becomes overall average
2. A low standard deviation indicates that the values tend to be close to the mean (also called the expected value) of the set, while a high standard deviation indicates that the values are spread out over a wider range.
```

### Variance

Variance is a measure of how much the values in a dataset differ from the mean. It quantifies the dispersion or spread of the data points.

```text
Standard Deviation => sum i->n (square(xi-𝑥ˉ)) / n-1

xi: represents each individual value in the dataset.
𝑥ˉ: represents the mean (average) of the dataset.
𝑛: represents the total number of values in the dataset.

Steps:
1. Find the mean i.e., 𝑥ˉ of the dataset.
2. Subtract the mean from each individual value to get the deviation from the mean for each value.
3. Square each deviation to eliminate negative values and emphasize differences from the mean.
4. Find the average of these squared deviations.

Note: 
1. Since variance is in squared units i.e., squared average, its interpretation might not be as intuitive as standard deviation (which is like original units as we cancelled sqrt and sqaure. so it becomes overall average).
2. A larger variance indicates that the data points are more spread out from the mean, while a smaller variance indicates that the data points are closer to the mean.
```

### Covariance

Covariance measures how much two variables change together. It tells you whether the variables tend to increase or decrease at the same time. Here's a simpler breakdown:

```text
Positive Covariance: If one variable tends to increase as the other increases, the covariance is positive.

Negative Covariance: If one variable tends to decrease as the other increases, the covariance is negative.

Zero Covariance: If there's no clear trend in how the variables change together, the covariance is close to zero.

Covariance is a bit trickier to interpret because it depends on the scale of the variables. If the variables are in different units or have different ranges, it might be hard to compare covariances directly.

Cov(X,Y) => sum i->n ((xi-𝑥ˉ)(yi-yˉ)) / (n-1)

xi = represents each individual value of variable 𝑋
𝑦𝑖 = represents each individual value of variable Y 
𝑥ˉ = represents the mean (average) of variable 𝑋
𝑦ˉ = represents the mean (average) of variable 𝑌
n = represents the total number of observations (values) for both variables.

Divide the sum by 𝑛−1, where n is the number of observations. This is because we are estimating the population covariance from a sample, so we use n−1 instead of 𝑛 to provide an unbiased estimate.
```

### Correlation

Correlation is a measure of the relationship between two variables. It tells you how much and in what direction they are related. Here's a simpler explanation:

```text
Positive Correlation: When one variable increases, the other tends to increase as well. It's like saying they move in the same direction.

Negative Correlation: When one variable increases, the other tends to decrease. They move in opposite directions.

No Correlation: When there's no clear trend in how the variables relate to each other. Changes in one variable don't predict changes in the other.

Correlation is a more standardized measure compared to covariance. It's expressed as a value between -1 and 1, where:

* 1 means perfect positive correlation,
* -1 means perfect negative correlation, and
* 0 means no correlation.

Correlation helps you understand how changes in one variable might affect the other, and it's easier to compare across different datasets because it's not affected by the scale of the variables.

Corr(X,Y) = Cov(X,Y) / (σX × σY)
​
Corr(X,Y) represents the correlation coefficient between variables X and 𝑌.
Cov(X,Y) represents the covariance coefficient between variables X and 𝑌.
σX represents the standard deviation of variable X.
σY represents the standard deviation of variable Y.
```
