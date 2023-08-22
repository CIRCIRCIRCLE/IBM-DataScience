# Create & Access SQLite database using Python
## Objectives
>1. Create a database  
>2. Create a table  
>3. Insert data into the table  
>4. Query data from the table  
>5. Retrieve the result set into a pandas dataframe  
>6. Close the database connection  

## 1.Create database using SQLite
```python
#Install & load sqlite3
#!pip install sqlite3  ##Uncomment the code to install sqlite3
import sqlite3
#connection object
conn = sqlite3.connect('INSTRUCTOR.db')
```
