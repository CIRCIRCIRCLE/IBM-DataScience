## Tools: `Matplotlib`,`Seaborn`, `Folium`  
# Week1: Introduction to Data Visualization Tools
example: Canada immigration--matplotlib
```python
#data preprocessing
df_can.drop(['AREA','REG','DEV','Type','Coverage'], axis=1, inplace=True)
df_can.rename(columns={'OdName':'Country', 'AreaName':'Continent', 'RegName':'Region'}, inplace=True)
df_can['Total'] = df_can.sum(axis=1)

#slicing
years = list(map(str, range(1980, 2014)))

df_can.sort_values(by='Total', ascending=False, axis = 0, inplace = True)
df_top5 = df_can.head(5)
df_top5 = df_top5[years].transpose()

#Plot
df_top5.index = df_top5.index.map(int)
df_top5.plot(kind = 'line')

plt.title('Immigration Trend of Top 5 Countries')
plt.ylabel('Number of Immigrants')
plt.xlabel('Years')

plt.show()
```
# Week 2 - Basic and Specialized Visualization Tools
# Week 3 - Advanced Visualizations and Geospatial Data
