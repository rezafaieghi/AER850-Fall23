import pandas as pd
# from pandas.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit

#import data
df = pd.read_csv("data\\housing.csv")
df.dropna()

#stratified sampling
df["income_cat"] = pd.cut(df["median_income"],bins=[0., 1.5, 3.0, 4.5, 6., np.inf],labels=[1, 2, 3, 4, 5])
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(df, df["income_cat"]):
    strat_train_set = df.loc[train_index]
    strat_test_set = df.loc[test_index]
for a in (strat_train_set, strat_test_set):
    a.drop("income_cat", axis=1)

#anything from here, unless for testing, should be on only the train dataset
#to avoid data snooping bias.

#creating scatter plots using panda
attributes = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]
pd.plotting.scatter_matrix(df[attributes], figsize=(12, 8))

#looking at correlations
corr_matrix = strat_train_set.corr(numeric_only=True)
plt.figure()
# plt.matshow(corr_matrix)
sns.heatmap(np.abs(corr_matrix)); #this generates a better looking correlations
                                  #matrix compared to plt.matshow()

selected_variables = ['longitude', 'housing_median_age', 'total_rooms',
                     'median_income','ocean_proximity']

strat_train_set_selected = strat_train_set[selected_variables]
