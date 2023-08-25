# DATA ANALYSIS WITH PYTHON
## WEEK1:importing data set
## WEEK2: Data Wrangling  
### 2.1 Dealing with missing values  
1. Drop
   ```python
      '''
    Drop NA values in dataframe
    axis = 0: drop the entire row
    axis = 1: drop the entire col
    inline = True:  just writes the result back into the data frame
    '''
    
    df.dropna(axis = 0)	# drop all rows containing NaN 
    
    # Adding condition
    df.dropna(subset=["price"], axis=0, inplace=True)
    
    # The above is equivalent to this below code:
    df = df.dropna(subset=["price"], axis=0)
    
    
   ```
2. Replace  
   ```python
    '''
    Replace missing values:
    df.replace(missing_value, new_value)
    '''
    
    # Example: Replace NA by mean
    mean = df["normalized-losses"].mean()
    df["normalized-losses"].replace(np.nan, mean, inplace=True)
   ```
### 2.2 Data formatting  
  ```python
  df["city-mpg"] = 235/df["city-mpg"]
  df.rename(columns={"city-mpg": "city-L/100km"}, inplace=True) # rename df column names
  
  # Convert object to integer:
  df.dtypes() 		# to check the type of df
  df["price"] = df["price"].astypes("int")	# to convert df to integer
  ```
### 2.3 Data Normalization
**Normalization methods:**
1. *Simple Feature Scaling*: `x_new = x_old/x_max`, results range in `[0, 1]`
2. *Min-max*: `x_new = (x_old - x_min)/(x_max - x_min)`, results range in `[0, 1]`
3. *Z-score (standard score)*: `x_new = (x_old - mu)/sigma`, where (mu, sigma): average and standard deviation of the feature. Results hover around `0`, and typically range between `-3` and `+3` but can be higher or lower

```python
# Simple feature scaling
df["age"] = df["age"]/df["age"].max()	

# Min-max
df["age"] = (df["age"] - df["age"].min()) / (df["age"].max() - df["age"].min())
			
# Z-score
df["age"] = (df["age"] - df["age"].mean()) / df["age"].std() 

``` 
### 2.4. Binning in Python

Binning:
* Grouping of values into `bins`
* Converts numeric into categorical variables
* Group a set of numeric variables into a set of `bins`

Examples: 
* Bin age into `0-5`, `6-10`, `11-15`, and so on
* Bin prices (e.g., 5000; 10,000;... 39,000; 44,500, etc.) into categories: `low`, `mid`, `high`

```python
binwidth = int((max(df["price"]) - min(df["price"]))/4)
bins = range(min(df["price"]), max(df["price"]), binwidth)

group_names = ['low', 'medium', 'high']
df["price-binned"] = pd.cut(df["price"], bins, labels=group_names)
```

### 2.5. Turning categorical variables into quantitative variables in Python
Problem:
* Most statistical models cannot take in objects/strings as input  

Categorical to numeric solution:
* Add dummy variables for each unique category
* Assign `0` or `1` in each category (called **One-hot Encoding**)

Example:

Car | Fuel 		| gas 	| diesel
----|-----------|-------|--------
A 	| gas 		| 1 	| 0
B 	| diesel 	| 0 	| 1

```python
'''
Use pandas.get_dummies() method
Convert categorical variables to dummy variables (0 or 1)
'''
pd.get_dummies(df["fuel"])
```

## WEEK3: Exploratory Data Analysis
* Learning objectives:
	* Descriptive statistics: Describe basic features of a data set, and obtain a short summary about the sample and measures of the data
	* GroupBy: Grouping data -> transform the dataset
	* ANOVA: Analysis of variance, a statistical method in which the variation in a set of observations is divided into distinct components
	* Correlation between different variables
	* Advance correlation: various correlations statistical methods namely, Pearson correlation, and correlation heatmaps

 ### 3.1 Descriptive Statistics
 `df.describe()`
 ```python
df.describe()

'''
- Show stats of dataframe: count, mean, std, min, 25%, 75%, max
- NaN is automatically skipped 
'''
```

`df.value_count()`
```python
drive_wheels_counts = df["drive-wheels"].value_count()

# Change column name
drive_wheels_counts.rename(columns = {'drive-wheels': 'value_count' inplace=True})
drive_wheels_counts.index.name = 'drive-wheels'
```

#### `visualization methods regarding descriptive statistics:
`Boxplot`: Median, Upper quartile `75%`, Lower quartile `25%`, Lower extreme, upper extreme, outliers as individual dots  
```python
sns.boxplot(x='drive-wheels', y='price', data=df)
```

`Scatter plot`:  
```python
x = df["engine-size"]
y = df["price"]
plt.scatter(x,y)
```

### 3.2 GroupBy in Python
Example:
```python
# First pick out the three data columns we are interested in
df_test = df[['drive-wheels', 'body-style', 'price']]

# Group the reduced data according to 'drive-wheels' and 'body-style' 
# Since we are interested in knowing how the average price differs 
# across the board, we can take the mean of each group and append 
# it this bit at the very end of the line too
df_grp = df_test.groupby(['drive-wheels', 'body-style'], as_index=False).mean()

# Use the groupby function to find the average "price" of each car based on "body-style"
df[['price','body-style']].groupby(['body-style'],as_index= False).mean()
```

Result:
no.	| drive-wheels 	| body-style 	| price
----|---------------|---------------|------------
0	| 4wd			| hatchback		| **7603.00**
1	| 4wd			| sedan			| 12647
2	| 4wd			| wagon			| 9095
3	| fwd			| convertible	| 11595
4	| fwd			| hardtop		| 8249
5	| fwd			| hatchback		| 8396
6	| fwd			| sedan			| 9811
7	| fwd			| wagon			| 9997
8	| rwd			| convertible	| 23949
9	| rwd			| hardtop		| 24202
10	| rwd			| hatchback		| 14337
11	| rwd			| sedan			| 21711
12	| rwd			| wagon			| 16994

#### Pandas method: `pivot()`  
df_pivot = df_grp.pivot(index='drive-wheels', columns='body-style')
**body-style: along the columns, drive-wheels: along the rows**
Result:

| price 	 | 	         |	     | 		 |       |        |
|----------------|---------------|-----------|-----------|-------|--------|
|body-style	 | convertible 	 | hardtop   | hatchback | sedan | wagon  |
|drive-wheels 	 |  		 |  	     | 		 |       |        |
|4wd		 | 20239 	 | 20239     | 7603 	 | 12647 | 9095   |
|fwd		 | 11595 	 | 8249	     | 8396 	 | 9811	 | 9997   |
|rwd		 | 23949 	 | 24202     | 14337 	 | 21711 | 16994  |

#### visualization method: `heatmap`  

### 3.3 Correlation
`Pearson Correlation`
* Measure the strength of the correlation between 2 features:
	1. Pearson correlation coefficient
		* Close to `+1`: large positive relationship
		* Close to `-1`: Large negative relationship
		* Close to `0`: No relationship
	2. `P-value`
		* P-value < 0.001: *Strong* certainty in the result
		* P-value < 0.05: *Moderate* certainty in the result
		* P-value < 0.1: *Weak* certainty in the result
		* P-value > 0.001: *No* certainty in the result
* Strong correlation:
	* Correlation coefficient close to `1` or `-1`
	* P-value < 0.001

Calculate the Pearson correlation between `'horsepower'` and `'price'`:
```python
pearson_coef, p_value = stats.pearsonr(df['horsepower'], df['price'])
# pearson_coef = 0.81: close to 1 -> strong positive correlation
# p_value = 9.35e-48 << 0.001 indicates: stong certainty in the result
```

`visualization` :Heatmap

## WEEK 4: Model Development
**Define the explanatory variable and the response variable:** Define the response variable (y) as the focus of the experiment and the explanatory variable (x) as a variable used to explain the change of the response variable. Understand the differences between Simple Linear Regression because it concerns the study of only one explanatory variable and Multiple Linear Regression because it concerns the study of two or more explanatory variables.

**Evaluate the model using Visualization:** By visually representing the errors of a variable using scatterplots and interpreting the results of the model.

**identify alternative regression approaches:** Use a **Polynomial Regression** when the Linear regression does not capture the curvilinear relationship between variables and how to pick the optimal order to use in a model.

**Interpret the R-square and the Mean Square Error:** Interpret R-square (x 100) as the percentage of the variation in the response variable y  that is explained by the variation in explanatory variable(s) x. The Mean Squared Error tells you how close a regression line is to a set of points. It does this by taking the average distances from the actual points to the predicted points and squaring them.


### 4.1. Linear Regression(LR) and Multiple Linear Regression(MLR)
#### `Linear Regression`
```python
# Import linear_model from scikit-learn
from sklearn.linear_model import LinearRegression

# Create a Linear Regression object using the constructor
lm = LinearRegression()

# Define the predictor variable and target variable
X = df[['highway-mpg']]
Y = df['price']

# Then use lm.fit(X,Y) to fit the model, i.e., find the parameters b0 and b1
lm.fit(X,Y)

# Then obtain a prediction
Yhat = lm.predict(X) # Yhat is an array having same number of samples as input X 

# Get b0
lm.intercept_

# Get b1
lm.coef_

# Relationship between Price and Highway MPG is given by
Price = lm.intercept_ - lm.coef_*highway-mpg
```

#### `Multiple Linear Regression`  $$\hat(Y) = b_0 + b_1*x_1 + b_2*x_2 + ...$$
```python
# Extract 4 predictor variables and store them in the variable Z  
Z = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]

# Train the model as before
lm.fit(Z, df['price'])

# Obtain prediction
Yhat = lm.predict(X)

# MLR - Estimated Linear Model
lm.intercept_ 	# b0
lm.coef_ 		# coefficients array: array([b1, b2, b3, b4])
```

### 4.2 Model Evaluation using Visualization
#### `Regression Plot`
#### `SLR`
```python
width = 12
height = 10
plt.figure(figsize=(width, height))
sns.regplot(x="highway-mpg", y="price", data=df)
plt.ylim(0,)
```
verifiction by corr parameters
```python
df[["peak-rpm","highway-mpg","price"]].corr()
```
#### `MLR`
```python
ax1 = sns.distplot(df['price'], hist=False, color="r", label="Actual Value")
sns.distplot(Y_hat, hist=False, color="b", label="Fitted Values" , ax=ax1)
```
#### `Residual Plot`
A residual plot is a graph that shows the residuals on the vertical y-axis and the independent variable on the horizontal x-axis.  
We look at the spread of the residuals:  
- If the points in a residual plot are **randomly spread out around the x-axis**, then a **linear model** is appropriate for the data.  
Why is that? Randomly spread out residuals means that **the variance is constant**, and thus the linear model is a good fit for this data.  

### 4.3 Polynomial Regression
```python
x = df['highway-mpg']
y = df['price']

f = np.polyfit(x, y, 3)
p = np.poly1d(f)
print(p)
```
###### `PolynomialFeatures`

### 4.3 Data pipeline

There are many steps to getting a prediction. For example: (1) Normalization -> (2) Polynomial Transform -> (3) Linear Regression  
We simplify the process using a pipeline  
Pipelines sequentially perform a series of transformations (1 and 2). The last step (3) carries out a prediction  
```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

'''
create a list of tuples
* First element: name of the estimator model: 'scale', 'polynomial', 'model'
* Second element: contains model constructor: StandardScaler(), etc.
'''
Input = [('scale', StandardScaler()), ('polynomial', PolynomialFeatures(degrees=2)), ... ('model', LinearRegression())]

# Input the list in the pipeline constructor
pipe = Pipeline(Input) # A Pipeline object

# Now we can train the pipeline by applying the train method to the pipeline 
# object. We can also produce a prediction as well. The method normalizes 
# the data, performs a polynomial transform, then outputs of prediction.

pipe.train(X['horsepower', 'curb-weight', 'engine-size', 'highway-mpg'], y)
yhat = Pipe.predict(X['horsepower', 'curb-weight', 'engine-size', 'highway-mpg'])

# X -> Normalization -> Polynomial Transform -> Linear Regression -> yhat
```

### 4.4 Measures for In-Sample Evaluation
#### `Mean Square Error (MSE)`
```python
mse = mean_squared_error(df['price'], Yhat)
print('The mean square error of price and predicted value is: ', mse)
```
#### `R-squared (R^2)`
```python
#highway_mpg_fit
lm.fit(X, Y)
# Find the R^2
print('The R-square is: ', lm.score(X, Y))
```

 import the function r2_score from the module metrics as we are using a different function.  
```python
from sklearn.metrics import r2_score
r_squared = r2_score(y, p(x))
print('The R-square value is: ', r_squared)
```

