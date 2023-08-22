# Create & Access SQLite database using Python
## Objectives
>1. Create a database  
>2. Create a table  
>3. Insert data into the table  
>4. Query data from the table  
>5. Retrieve the result set into a pandas dataframe  
>6. Close the database connection  

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
