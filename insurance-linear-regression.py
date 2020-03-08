import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

insurance = pd.read_csv("./csv_files/insurance.csv")
print("\n")

print("================ Head ================")
print(insurance.head())
print("\n")

print("================ Shape ================")
print(insurance.shape)
print("\n")

print("================ Desc ================")
print(insurance.describe())
print("\n")

print("================ Nulls ================")
nulls = pd.DataFrame(insurance.isnull().sum().sort_values(ascending=False)[:25])
nulls.columns = ["Null Count"]
print(nulls)
print("\n")
# sns.heatmap(insurance.isnull(), yticklabels=False, cbar=False)
# plt.show()

print("================ Categoricals ================")
categoricals = insurance.select_dtypes(exclude = [np.number])
print(categoricals)
print("\n")

print("================ Dtypes ================")
print(insurance.dtypes)
print("\n")

print("================ Unique Regions ================")
print(insurance["region"].value_counts())
print("\n")


# changing the values of the categorical values
onehot_columns = ["sex", "smoker", "region"]
for column in onehot_columns:
  df1 = pd.get_dummies(insurance[column], drop_first = True) # prefix=column will give like sex_male
  insurance.drop([column], axis=1, inplace=True)
  insurance = pd.concat([insurance, df1], axis=1)


print("================ After Onehot ================")
print(insurance.head(20))
print("\n")

print("================ Shape After Onehot ================")
print(insurance.shape)
print("\n")

print("================ Splitting Data ================")

dependent_field = "charges"
X = insurance.drop([dependent_field], axis = 1)
y = insurance[dependent_field]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)

print("================ X variables ================")
print(X)
print("\n")

print("================ Y variables ================")
print(y)
print("\n")

print("================ X_train ================")
print(X_train)
print("\n")

print("================ y_train ================")
print(y_train)
print("\n")


regressor = LinearRegression()
regressor.fit(X_train, y_train) #training the algorithm

print("================ Intercept ================")
print(regressor.intercept_)
print("\n")

print("================ Coeff ================")
print(regressor.coef_)
print("\n")

y_pred = regressor.predict(X_test)
print("================ Y Predic ================")
# print(y_pred)
print("\n")

print("================ Actual vs Predicted ================")
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(df)

##### visualisation
df1 = df.head(25)
df1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

### plot the straight line
# plt.scatter(X_test["age"], y_test,  color='gray')
# plt.plot(X_test["age"], y_pred, color='red', linewidth=2)
# plt.show()