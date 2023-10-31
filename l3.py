import pandas as pd
# from pandas.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler

#import data
df = pd.read_csv("data\\housing.csv")

# print(df.isna().any(axis=0).sum())  #how many columns have missing values?
# print(df.isna().any(axis=1).sum()) #how many rows have missing values?
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

train_X["rooms_per_household"] = train_X["total_rooms"]/train_X["households"]
train_X["bedrooms_per_room"] = train_X["total_bedrooms"]/train_X["total_rooms"]
train_X["population_per_household"]=train_X["population"]/train_X["households"]

columns_list = train_X.columns.tolist()
new_order = ['longitude',
  'latitude',
  'housing_median_age',
  'total_rooms',
  'total_bedrooms',
  'population',
  'households',
  'median_income',
   'rooms_per_household',
   'bedrooms_per_room',
   'population_per_household',
  'ocean_proximity_<1H OCEAN',
  'ocean_proximity_INLAND',
  'ocean_proximity_ISLAND',
  'ocean_proximity_NEAR BAY',
  'ocean_proximity_NEAR OCEAN']
train_X = train_X[new_order]

corr_matrix = (train_X.iloc[:,0:-5]).corr()
sns.heatmap(np.abs(corr_matrix))
corr1 = np.corrcoef(train_X['longitude'], train_y)
print("longitude correlation wiht y is: ", corr1[0,1])
corr2 = np.corrcoef(train_X['latitude'], train_y)
print("latitude correlation wiht y is: ",corr2[0,1])
corr3 = np.corrcoef(train_X['total_rooms'], train_y)
print("total rooms correlation wiht y is: ",corr3[0,1])
corr4 = np.corrcoef(train_X['total_bedrooms'], train_y)
print("total bedrooms correlation wiht y is: ",corr4[0,1])
corr5 = np.corrcoef(train_X['population'], train_y)
print("population correlation wiht y is: ", corr5[0,1])
corr6 = np.corrcoef(train_X['households'], train_y)
print("households correlation wiht y is: ", corr6[0,1])
# train_X = train_X.drop(['longitude'], axis=1)
# train_X = train_X.drop(['total_bedrooms'], axis=1)
# train_X = train_X.drop(['population'], axis=1)
# train_X = train_X.drop(['households'], axis=1)

plt.figure()
corr_matrix = train_X.corr()
sns.heatmap(np.abs(corr_matrix))

from sklearn.linear_model import LinearRegression
# model1 = LinearRegression()
# model1.fit(train_X, train_y)

# some_data = train_X.iloc[:10]
# some_data.columns = train_X.columns
# some_house_values = train_y.iloc[:10]

# for i in range(10):
#     some_predictions = model1.predict(some_data.iloc[i].values.reshape(1, -1))
#     some_actual_values = some_house_values.iloc[i]
#     print("Predictions:", some_predictions)
#     print("Actual values:", some_actual_values)

# model1_prediction = model1.predict(train_X)
from sklearn.metrics import mean_absolute_error
# model1_train_mae = mean_absolute_error(model1_prediction, train_y)
# print("Model 1 training MAE is: ", round(model1_train_mae,2))


from sklearn.ensemble import RandomForestRegressor
# model2 = RandomForestRegressor(n_estimators=30, random_state=42)
# model2.fit(train_X, train_y)
# model2_predictions = model2.predict(train_X)
# model2_train_mae = mean_absolute_error(model2_predictions, train_y)
# print("Model 2 training MAE is: ", round(model2_train_mae,2))


# for i in range(10):
#     some_predictions1 = model1.predict(some_data.iloc[i].values.reshape(1, -1))
#     some_predictions2 = model2.predict(some_data.iloc[i].values.reshape(1, -1))
#     some_actual_values = some_house_values.iloc[i]
    # print("Predictions Model 1:", some_predictions1)
    # print("Predictions Model 2:", some_predictions2)
    # print("Actual values:", some_actual_values)



# cross validation
from sklearn.model_selection import cross_val_score
model1 = LinearRegression()
model2 = RandomForestRegressor(n_estimators=30, random_state=42)

# Perform k-fold cross-validation for Model 1
scores_model1 = cross_val_score(model1, train_X, train_y, cv=5, scoring='neg_mean_absolute_error')
mae_model1 = -scores_model1.mean()
print("Model 1 Mean Absolute Error (CV):", round(mae_model1, 2))

# Perform k-fold cross-validation for Model 2
scores_model2 = cross_val_score(model2, train_X, train_y, cv=5, scoring='neg_mean_absolute_error')
mae_model2 = -scores_model2.mean()
print("Model 2 Mean Absolute Error (CV):", round(mae_model2, 2))




#GridSearchCV
from sklearn.model_selection import GridSearchCV
param_grid = {
    'n_estimators': [10, 30, 50],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}
model2 = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(model2, param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_search.fit(train_X, train_y)
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)
best_model2 = grid_search.best_estimator_



test_y = strat_test_set['median_house_value']
df_test_X = strat_test_set.drop(columns = ["median_house_value"])
scaled_data_test = my_scaler.transform(df_test_X.iloc[:,0:-5])
scaled_data_test_df = pd.DataFrame(scaled_data_test, columns=df_test_X.columns[0:-5])
test_X = scaled_data_test_df.join(df_test_X.iloc[:,-5:])
test_X["rooms_per_household"] = test_X["total_rooms"]/test_X["households"]
test_X["bedrooms_per_room"] = test_X["total_bedrooms"]/test_X["total_rooms"]
test_X["population_per_household"]=test_X["population"]/test_X["households"]
test_X = test_X[new_order]
test_X = test_X.drop(['longitude'], axis=1)
test_X = test_X.drop(['total_bedrooms'], axis=1)
test_X = test_X.drop(['population'], axis=1)
test_X = test_X.drop(['households'], axis=1)

# model1_test_predictions = model1.predict(test_X)
model2_test_predictions = best_model2.predict(test_X)
# model1_test_mae = mean_absolute_error(model1_test_predictions, test_y)
model2_test_mae = mean_absolute_error(model2_test_predictions, test_y)
# print("Model 1 MAE is: ", round(model1_test_mae,2))
print("Model 2 MAE is: ", round(model2_test_mae,2))





