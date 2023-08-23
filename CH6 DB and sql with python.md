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
```PIC```

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
>1.Magic commands have the general format **%sql select * from tablename**  
>2.**Cell magics**:start with a double %% (percent) sign and apply to the entire cell.  
>3.**Line magics**:start with a single % (percent) sign and apply to a particular line in a cell.  

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
```%sql SELECT COUNT(*) FROM chicago_socioeconomic_data WHERE hardship_index > 50.0 ``` 
 * sqlite:///socioeconomic.db  
Done.  

__Problem 3__  
_What is the maximum value of hardship index in this dataset?_
```%sql SELECT MAX(hardship_index) FROM chicago_socioeconomic_data;  ```
 * sqlite:///socioeconomic.db  
Done.  

## Task3: Visualization
_Create a scatter plot using the variables per_capita_income_ and hardship_index. Explain the correlation between the two variables._
```python
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

income_vs_hardship = %sql SELECT per_capita_income_, hardship_index FROM chicago_socioeconomic_data;
plot = sns.jointplot(x='per_capita_income_',y='hardship_index', data=income_vs_hardship.DataFrame())
```
```PIC```

## PS: Create and insert using sql magic

```python
import csv, sqlite3
con = sqlite3.connect("SQLiteMagic.db")
cur = con.cursor()

%load_ext SQL
%sql sqlite:///SQLiteMagic.db
```

```python
%%sql

CREATE TABLE INTERNATIONAL_STUDENT_TEST_SCORES (
	country VARCHAR(50),
	first_name VARCHAR(50),
	last_name VARCHAR(50),
	test_score INT
);
INSERT INTO INTERNATIONAL_STUDENT_TEST_SCORES (country, first_name, last_name, test_score)
VALUES
('United States', 'Marshall', 'Bernadot', 54),
('Ghana', 'Celinda', 'Malkin', 51),
('Ukraine', 'Guillermo', 'Furze', 53),
('Greece', 'Aharon', 'Tunnow', 48),
('Russia', 'Bail', 'Goodwin', 46),
('Poland', 'Cole', 'Winteringham', 49),
('Sweden', 'Emlyn', 'Erricker', 55),
('Russia', 'Cathee', 'Sivewright', 49),
('China', 'Barny', 'Ingerson', 57),
('Uganda', 'Sharla', 'Papaccio', 55)
```
## Using Python Variables in SQL Statements
You can use python variables in your SQL statements by adding a ":" prefix to your python variable names.  
For example, if I have a python variable country with a value of "Canada", I can use this variable in a SQL query to find all the rows of students from Canada.  
```python
country = "Canada"
%sql select * from INTERNATIONAL_STUDENT_TEST_SCORES where country = :country
```
## Assigning the Results of Queries to Python Variables and Converting Query Results to DataFrames
describe test score frequency, visualize the results  
You can easily convert a SQL query result to a pandas dataframe using the DataFrame() method. Dataframe objects are much more versatile than SQL query result objects. For example, we can easily graph our test score distribution after converting to a dataframe.  

```python
#get the distribution results
test_score_distribution = %sql SELECT test_score as "Test_Score", count(*) as "Frequency" from INTERNATIONAL_STUDENT_TEST_SCORES GROUP BY test_score;

#visualization
dataframe = test_score_distribution.DataFrame()
%matplotlib inline
import seaborn
plot = seaborn.barplot(x='Test_Score',y='Frequency', data=dataframe)
```

```PIC```

# lab3: Real world data with Chicago Public Schools
## Connect to database
```python
import csv, sqlite3
con = sqlite3.connect("RealWorldData.db")
cur = con.cursor()

%load_ext SQL
%sql sqlite:///RealWorldData.db
```

## Store the data in the table
```python
df = pandas.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DB0201EN-SkillsNetwork/labs/FinalModule_Coursera_V5/data/ChicagoPublicSchools.csv")

df.to_sql("CHICAGO_PUBLIC_SCHOOLS_DATA", con, if_exists='replace', index=False, method="multi")
```
```
%sql SELECT name FROM sqlite_master WHERE type='table'
```
|name|
|-----|
|CENSUS_DATA|
|CHICAGO_CRIME_DATA|
|CHICAGO_PUBLIC_SCHOOLS_DATA|

## Query the database system catalog to retrieve column metadata
```
%sql SELECT name,type,Length(type) FROM PRAGMA_TABLE_INFO('CHICAGO_PUBLIC_SCHOOLS_DATA')
```

|name|type|Length(type)|
|----|----|------------|
|School_ID|INTEGER|7|
|NAME_OF_SCHOOL|TEXT|4|
Elementary, Middle, or High School	TEXT	4  
Street_Address	TEXT	4  
City	TEXT	4  
State	TEXT	4  
ZIP_Code	INTEGER	7  
Phone_Number	TEXT	4  
Link	TEXT	4  
Network_Manager	TEXT	4  
Collaborative_Name	TEXT	4  
Adequate_Yearly_Progress_Made_	TEXT	4  
Track_Schedule	TEXT	4  
CPS_Performance_Policy_Status	TEXT	4  
CPS_Performance_Policy_Level	TEXT	4  
HEALTHY_SCHOOL_CERTIFIED	TEXT	4  
Safety_Icon	TEXT	4  
SAFETY_SCORE	REAL	4  
Family_Involvement_Icon	TEXT	4  
Family_Involvement_Score	TEXT	4  
Environment_Icon	TEXT	4  
Environment_Score	REAL	4  
Instruction_Icon	TEXT	4  
Instruction_Score	REAL	4  
Leaders_Icon	TEXT	4  
Leaders_Score	TEXT	4  
Teachers_Icon	TEXT	4  
Teachers_Score	TEXT	4  
Parent_Engagement_Icon	TEXT	4  
Parent_Engagement_Score	TEXT	4  
Parent_Environment_Icon	TEXT	4  
Parent_Environment_Score	TEXT	4  
AVERAGE_STUDENT_ATTENDANCE	TEXT	4  
Rate_of_Misconducts__per_100_students_	REAL	4  
Average_Teacher_Attendance	TEXT	4  
Individualized_Education_Program_Compliance_Rate	TEXT	4  
Pk_2_Literacy__	TEXT	4  
Pk_2_Math__	TEXT	4  
Gr3_5_Grade_Level_Math__	TEXT	4  
Gr3_5_Grade_Level_Read__	TEXT	4  
Gr3_5_Keep_Pace_Read__	TEXT	4  
Gr3_5_Keep_Pace_Math__	TEXT	4  
Gr6_8_Grade_Level_Math__	TEXT	4  
Gr6_8_Grade_Level_Read__	TEXT	4  
Gr6_8_Keep_Pace_Math_	TEXT	4  
Gr6_8_Keep_Pace_Read__	TEXT	4  
Gr_8_Explore_Math__	TEXT	4  
Gr_8_Explore_Read__	TEXT	4  
ISAT_Exceeding_Math__	REAL	4  
ISAT_Exceeding_Reading__	REAL	4  
ISAT_Value_Add_Math	REAL	4  
ISAT_Value_Add_Read	REAL	4  
ISAT_Value_Add_Color_Math	TEXT	4  
ISAT_Value_Add_Color_Read	TEXT	4  
Students_Taking__Algebra__	TEXT	4  
Students_Passing__Algebra__	TEXT	4  
9th Grade EXPLORE (2009)	TEXT	4  
9th Grade EXPLORE (2010)	TEXT	4  
10th Grade PLAN (2009)	TEXT	4  
10th Grade PLAN (2010)	TEXT	4  
Net_Change_EXPLORE_and_PLAN	TEXT	4  
11th Grade Average ACT (2011)	TEXT	4  
Net_Change_PLAN_and_ACT	TEXT	4  
College_Eligibility__	TEXT	4  
Graduation_Rate__	TEXT	4  
College_Enrollment_Rate__	TEXT	4  
COLLEGE_ENROLLMENT	INTEGER	7  
General_Services_Route	INTEGER	7  
Freshman_on_Track_Rate__	TEXT	4  
X_COORDINATE	REAL	4  
Y_COORDINATE	REAL	4  
Latitude	REAL	4  
Longitude	REAL	4  
COMMUNITY_AREA_NUMBER	INTEGER	7  
COMMUNITY_AREA_NAME	TEXT	4  
Ward	INTEGER	7  
Police_District	INTEGER	7  
Location	TEXT	4  

## Problem 1:  
Which schools have highest Safety Score?  
```python
%%sql 
SELECT NAME_OF_SCHOOL 
FROM CHICAGO_PUBLIC_SCHOOLS_DATA 
WHERE SAFETY_SCORE = (SELECT MAX(Safety_Score) from CHICAGO_PUBLIC_SCHOOLS_DATA)
```
## Problem 2:  
Retrieve the list of 5 Schools with the lowest Average Student Attendance sorted in ascending order based on attendance  
```python
%%sql
SELECT Name_of_School, Average_Student_Attendance
FROM CHICAGO_PUBLIC_SCHOOLS_DATA
WHERE Average_Student_Attendance NOT null
ORDER BY Average_Student_Attendance
LIMIT 5
```

## Problem 3:  
Now remove the '%' sign from the above result set for Average Student Attendance column  
```python
%%sql
SELECT Name_of_School, REPLACE(Average_Student_Attendance, '%', '') AS AVERAGE_ATTENDANCE
FROM CHICAGO_PUBLIC_SCHOOLS_DATA
WHERE Average_Student_Attendance NOT null
ORDER BY Average_Student_Attendance
LIMIT 5
```

## Problem 4:  
Which Schools have Average Student Attendance lower than 70%?  
___convert text type to double, then compare with integers___
```python
%%sql
SELECT Name_of_School, Average_Student_Attendance
FROM CHICAGO_PUBLIC_SCHOOLS_DATA
WHERE CAST(REPLACE(Average_Student_Attendance, '%', '') AS DOUBLE ) < 70 
ORDER BY Average_Student_Attendance
```

## Problem 5:  
Get the hardship index for the community area which has the highest value for College Enrollment.  
```python
%sql select community_area_number, community_area_name, hardship_index from CENSUS_DATA \
   where community_area_number in \
   (select community_area_number from CHICAGO_PUBLIC_SCHOOLS_DATA order by college_enrollment desc limit 1)
```
