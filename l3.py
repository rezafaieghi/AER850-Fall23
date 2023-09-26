import pandas as pd
# from pandas.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler

#import data
df = pd.read_csv("data\\housing.csv")

print(df.isna().any(axis=0).sum())  #how many rows have missing values?
print(df.isna().any(axis=1).sum()) #how many columns have missing values?
df = df.dropna()
df = df.reset_index(drop=True) 

my_encoder = OneHotEncoder(sparse_output=False)
my_encoder.fit(df[['ocean_proximity']])
encoded_data = my_encoder.transform(df[['ocean_proximity']])
category_names = my_encoder.get_feature_names_out()
encoded_data_df = pd.DataFrame(encoded_data, columns=category_names)
df = pd.concat([df, encoded_data_df], axis = 1)
df = df.drop(columns = 'ocean_proximity')


#stratified sampling
df["income_cat"] = pd.cut(df["median_income"],bins=[0., 1.5, 3.0, 4.5, 6., np.inf],labels=[1, 2, 3, 4, 5])
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(df, df["income_cat"]):
    strat_train_set = df.loc[train_index].reset_index(drop=True)
    strat_test_set = df.loc[test_index].reset_index(drop=True)
strat_train_set = strat_train_set.drop(columns=["income_cat"], axis = 1)
strat_test_set = strat_test_set.drop(columns=["income_cat"], axis = 1)

#anything from here, unless for testing, should be on only the train dataset
#to avoid data snooping bias.
train_y = strat_train_set['median_house_value']
df_X = strat_train_set.drop(columns = ["median_house_value"])

my_scaler = StandardScaler()
my_scaler.fit(df_X.iloc[:,0:-5])
scaled_data = my_scaler.transform(df_X.iloc[:,0:-5])
scaled_data_df = pd.DataFrame(scaled_data, columns=df_X.columns[0:-5])
train_X = scaled_data_df.join(df_X.iloc[:,-5:])

corr_matrix = (train_X.iloc[:,0:-5]).corr()
sns.heatmap(np.abs(corr_matrix))
corr1 = np.corrcoef(train_X['longitude'], train_y)
print(corr1[0,1])
corr2 = np.corrcoef(train_X['latitude'], train_y)
print(corr2[0,1])
corr3 = np.corrcoef(train_X['total_rooms'], train_y)
print(corr3[0,1])
corr4 = np.corrcoef(train_X['total_bedrooms'], train_y)
print(corr4[0,1])
corr5 = np.corrcoef(train_X['population'], train_y)
print(corr5[0,1])
corr6 = np.corrcoef(train_X['households'], train_y)
print(corr6[0,1])
train_X = train_X.drop(['longitude'], axis=1)
train_X = train_X.drop(['total_bedrooms'], axis=1)
train_X = train_X.drop(['population'], axis=1)
train_X = train_X.drop(['households'], axis=1)