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
## 2.3 Data Normalization
