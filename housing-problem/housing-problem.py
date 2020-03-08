import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost
import pickle

df = pd.read_csv("train.csv")
# print(df.isnull().sum())
print(df.columns[df.isnull().any()].tolist())
# identify the null values using a map
sns.heatmap(df.isnull(), yticklabels=False, cbar=False)
# plt.show()

nulls = pd.DataFrame(df.isnull().sum().sort_values(ascending=False)[:25])
nulls.columns = ["Null Count"]
# print(nulls)

# find the non numerical values and get a gist of it
categoricals = df.select_dtypes(exclude = [np.number])
print("Categorical Values ----->")
print(categoricals.describe())


print("shape of the data is ---> after import", df.shape)

# we will remove/modify null values; daa wragnling
df["LotFrontage"] = df["LotFrontage"].fillna(df["LotFrontage"].mean())
df.drop("Alley", axis=1, inplace=True)
mode_list = ["BsmtCond", "BsmtQual", "FireplaceQu", "GarageType",
"GarageFinish", "GarageQual", "GarageCond", "MasVnrType", "MasVnrArea",
"BsmtExposure", "BsmtFinType2"]
df.drop(['GarageYrBlt'],axis=1,inplace=True)
df.drop(['PoolQC','Fence','MiscFeature'],axis=1,inplace=True)
# drop id as it is not required
df.drop(['Id'],axis=1,inplace=True)

# print(df.head(20))

for col in mode_list:
  df[col] = df[col].fillna(df[col].mode()[0])

# lastly drop all columns with
df.dropna(inplace=True)

sns.heatmap(df.isnull(), yticklabels=False, cbar=False)
# plt.show()  # this will plot the above sns graph
print("shape of the data is ---> after cleaning", df.shape)

def category_onehot_multcols(columns):
  df_final=final_df
  i=0

  for field in columns:
    # print(field)
    df1 = pd.get_dummies(final_df[field],drop_first=True)
    final_df.drop([field],axis=1,inplace=True)

    if i==0:
      df_final=df1.copy()
    else:
      df_final=pd.concat([df_final,df1],axis=1)
    i=i+1

  df_final=pd.concat([final_df,df_final],axis=1)

  return df_final


main_df = df.copy()
test_df=pd.read_csv('formulatedtest.csv')
# print("test df shape ==> ", test_df.shape)

# axis = 0 means columns and 1 is for rows
final_df = pd.concat([df, test_df], axis = 0)
# print(final_df['SalePrice'])

columns=['MSZoning','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood',
         'Condition2','BldgType','Condition1','HouseStyle','SaleType','SaleCondition','ExterCond',
         'ExterQual','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',
         'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','Heating','HeatingQC',
         'CentralAir','Electrical','KitchenQual','Functional',
         'FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive']

final_df=category_onehot_multcols(columns)
print("final_df.shape ====> ", final_df.shape)
# print("final_df.columns.duplicated() ====> ", final_df.columns.duplicated())
final_df =final_df.loc[:,~final_df.columns.duplicated()]
print("final_df.shape after removing duplicates====> ", final_df.shape)

df_train = final_df.iloc[:1422, :]
df_test = final_df.iloc[1422:, :]

# inplace will actually remove the data from the or the dataframe
df_test.drop(["SalePrice"], axis=1, inplace=True)


xtrain = df_train.drop(["SalePrice"], axis=1)
ytrain = df_train["SalePrice"]
# print("y train  =>", final_df.head(20))


#  create a classifier
classifier = xgboost.XGBRegressor()
a = classifier.fit(xtrain, ytrain)

# #  create a classifier pickle file to store it.. so that no need to test again
filename = "classifier-pickle.pkl"
# pickle.dump(classifier, open(filename, 'wb'))



### load classifier and run the predictions belor
# # load the model from disk
# classifier = pickle.load(open(filename, 'rb'))
# result = classifier.fit(xtrain, ytrain)

# y_pred = classifier.predict(df_test)
# # print(result)

# print("=============== predictions ===========")
# print(y_pred)
# print("=============== predictions above ===========")

# ## Create Sample Submission file and Submit using ANN
# pred=pd.DataFrame(y_pred)
# sub_df=pd.read_csv('sample_submission.csv')
# datasets=pd.concat([sub_df['Id'],pred],axis=1)
# datasets.columns=['Id','SalePrice']
# datasets.to_csv('sample_submission.csv',index=False)