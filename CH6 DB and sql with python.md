# Lab1: Create & Access SQLite database using Python
## Objectives
>1. Create a database  
>2. Create a table  
>3. Insert data into the table  
>4. Query data from the table  
>5. Retrieve the result set into a pandas dataframe  
>6. Close the database connection

### The connection methods are:
>The **cursor()** method, which returns a new cursor object using the connection.  
>The **commit()** method, which is used to commit any pending transaction to the database.  
>The **rollback()** method, which causes the database to roll-back to the start of any pending transaction.  
>The **close()** method, which is used to close a database connection.  

## Task 1.Create database using SQLite
```python
#Install & load sqlite3
#!pip install sqlite3  ##Uncomment the code to install sqlite3
import sqlite3
#connection object
conn = sqlite3.connect('INSTRUCTOR.db')
```

## Task 2: Create a table in the database
```python
# Drop the table if already exists.
cursor_obj.execute("DROP TABLE IF EXISTS INSTRUCTOR")
```
```python
# Creating table
table = """ create table IF NOT EXISTS INSTRUCTOR(ID INTEGER PRIMARY KEY NOT NULL, FNAME VARCHAR(20), LNAME VARCHAR(20), CITY VARCHAR(20), CCODE CHAR(2));"""
cursor_obj.execute(table)
print("Table is Ready")
```

## Task 3: Insert data into the table
<img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DB0201EN-SkillsNetwork/labs/FinalModule_Coursera_V5/images/table1.png" align="center">

Inserting the first row
```python
cursor_obj.execute('''insert into INSTRUCTOR values (1, 'Rav', 'Ahuja', 'TORONTO', 'CA')''')
```
 use a single query to insert the remaining two rows of data
 ```python
cursor_obj.execute('''insert into INSTRUCTOR values (2, 'Raul', 'Chong', 'Markham', 'CA'), (3, 'Hima', 'Vasudevan', 'Chicago', 'US')''')
```
## Task 4: Query data in the table
```python
statement = '''SELECT * FROM INSTRUCTOR'''
cursor_obj.execute(statement)

print("All the data")
output_all = cursor_obj.fetchall()
for row_all in output_all:
  print(row_all)
```
All the data  
(1, 'Rav', 'Ahuja', 'TORONTO', 'CA')  
(2, 'Raul', 'Chong', 'Markham', 'CA')  
(3, 'Hima', 'Vasudevan', 'Chicago', 'US')  

```python
## Fetch few rows from the table
statement = '''SELECT * FROM INSTRUCTOR'''
cursor_obj.execute(statement)
  
print("All the data")
# If you want to fetch few rows from the table we use fetchmany(numberofrows) and mention the number how many rows you want to fetch
output_many = cursor_obj.fetchmany(2) 
for row_many in output_many:
  print(row_many)
```
All the data  
(1, 'Rav', 'Ahuja', 'TORONTO', 'CA')  
(2, 'Raul', 'Chong', 'Markham', 'CA')  

```python
# Fetch only FNAME from the table
statement = '''SELECT FNAME FROM INSTRUCTOR'''
cursor_obj.execute(statement)
  
print("All the data")
output_column = cursor_obj.fetchall()
for fetch in output_column:
  print(fetch)
```
All the data  
('Rav',)  
('Raul',)  
('Hima',)  

__update message:__
```python
query_update='''update INSTRUCTOR set CITY='MOOSETOWN' where FNAME="Rav"'''
cursor_obj.execute(query_update)
```

## Task 5: Retrieve data into Pandas
```python
import pandas as pd
#retrieve the query results into a pandas dataframe
df = pd.read_sql_query("select * from instructor;", conn)

#print the dataframe
df
```

## Task 6: Close the Connection
```conn.close()```

# Analyzing a real world data-set with SQL and Python
## Objectives:
>1. Understand a dataset of selected socioeconomic indicators in Chicago
>2. Learn how to store data in an SQLite database.
>3. Solve example problems to practice your SQL skill

# Lab2: 
The city of Chicago released a dataset of socioeconomic data to the Chicago City Portal. This dataset contains a selection of six socioeconomic indicators of public health significance and a “hardship index,” for each Chicago community area, for the years 2008 – 2012.  
[the city of Chicago's website](https://data.cityofchicago.org/Health-Human-Services/Census-Data-Selected-socioeconomic-indicators-in-C/kn9c-c2s2?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkDB0201ENSkillsNetwork20127838-2021-01-01)  

### **SQL Magic** commands to execute queries more easily from Jupyter Notebooks
Magic commands have the general format **%sql select * from tablename**  
**Cell magics**:start with a double %% (percent) sign and apply to the entire cell.  
**Line magics**:start with a single % (percent) sign and apply to a particular line in a cell.  

## Task1:Socioeconomic in Chicago：connect to the database
Let us first load the SQL extension and establish a connection with the database  
The syntax for connecting to magic sql using sqllite is  
%sql sqlite://DatabaseName  
where DatabaseName will be your .db file  
```python
%load_ext SQL
import csv, sqlite3

con = sqlite3.connect("socioeconomic.db")
cur = con.cursor()
!pip install -q pandas==1.1.5

%sql sqlite:///socioeconomic.db
```
```python
# cursor object
cursor_obj = conn.cursor()
```

## Task 2: Store the dataset in the table
```python
import pandas
df = pandas.read_csv('https://data.cityofchicago.org/resource/jcxq-k9xf.csv')
df.to_sql("chicago_socioeconomic_data", con, if_exists='replace', index=False,method="multi")
```

__Problem 1__
_How many rows are in the dataset?_  
_data  
```%sql SELECT COUNT(*) FROM chicago_socioeconomic_data```
 * sqlite:///socioeconomic.db  
Done.  
COUNT(*)  
78  

__Problem 2__
_How many community areas in Chicago have a hardship index greater than 50.0?_  
%sql SELECT COUNT(*) FROM chicago_socioeconomic_data WHERE hardship_index > 50.0  
 * sqlite:///socioeconomic.db  
Done.  

__Problem 3__
_What is the maximum value of hardship index in this dataset?_
%sql SELECT MAX(hardship_index) FROM chicago_socioeconomic_data;  
 * sqlite:///socioeconomic.db  
Done.  

__Problem 4__
_Create a scatter plot using the variables per_capita_income_ and hardship_index. Explain the correlation between the two variables._
```python
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

income_vs_hardship = %sql SELECT per_capita_income_, hardship_index FROM chicago_socioeconomic_data;
plot = sns.jointplot(x='per_capita_income_',y='hardship_index', data=income_vs_hardship.DataFrame())
```
