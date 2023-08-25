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
