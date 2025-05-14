import pandas as pd
import numpy as np
from dask.optimization import inline
from sklearn.linear_model import LinearRegression
import matplotlib as plt
import RegressionLib as rl


#want to find if shots taken have any correlation to goals
df = pd.read_csv("game_teams_stats.csv")

df.info()
#52609 in my table

df.dropna(inplace=True)

df.info()
#wanted to get rid of null data in table
#29554 in my table
#56% of my table left

df.drop(columns=['won','settled_in','HoA','head_coach','startRinkSide'], inplace=True)
df.describe()

#print (df.query("game_id == 2015020693"))
#wanted to make sure 0 was not being drop from the table
Scored = df.query("shots > 0 and goals >= 0")

X = Scored[["shots", "pim", "takeaways","powerPlayOpportunities"]]
y = Scored["goals"]
reg = LinearRegression().fit(X,y)


print("skikit learn score")
print(reg.score(X,y))

#X = Scored[["shots",  "takeaways"]]
#y = Scored["goals"]



print("my Library")
reg = rl.LinearRegression().fit(X,y)
print(reg.score(X,y))



print (Scored[["shots", "goals", "takeaways","powerPlayOpportunities"]].corr())




