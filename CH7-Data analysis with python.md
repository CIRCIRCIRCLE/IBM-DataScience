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

| price 	 | 	         |	     | 		 |       |
|----------------|---------------|-----------|-----------|-------|--------
|body-style	 | convertible 	 | hardtop   | hatchback | sedan | wagon
|drive-wheels 	 |  		 |  	     | 		 |       | 
|4wd		 | 20239 	 | 20239     | 7603 	 | 12647 | 9095
|fwd		 | 11595 	 | 8249	     | 8396 	 | 9811	 | 9997
|rwd		 | 23949 	 | 24202     | 14337 	 | 21711 | 16994 

#### visualization method: `heatmap`  
