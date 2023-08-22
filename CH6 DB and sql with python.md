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

## Task3: Insert data into the table
Inserting the first row
```python
cursor_obj.execute('''insert into INSTRUCTOR values (1, 'Rav', 'Ahuja', 'TORONTO', 'CA')''')
```
 use a single query to insert the remaining two rows of data
 ```python
cursor_obj.execute('''insert into INSTRUCTOR values (2, 'Raul', 'Chong', 'Markham', 'CA'), (3, 'Hima', 'Vasudevan', 'Chicago', 'US')''')
```
```
