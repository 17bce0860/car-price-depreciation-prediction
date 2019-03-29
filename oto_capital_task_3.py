#importing libraries

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

#uploading csv file on google colab
from google.colab import files
upploaded = files.upload()

#creating dataframe and getting info

data = pd.read_csv("cars.csv")
data.head()
data.info()

#removing commas from kms attribute and converting to int
def remove_comma(k):
    z = [x for x in k if x!=',']
    j = int(''.join(z))
    return j

data['Kms'].astype(str,inplace=True)
data['Kms'] = data['Kms'].apply(remove_comma)
data['Model'].nunique()
data['Variant'].nunique()
data.drop(['MMV','Model','Variant'],axis=1,inplace=True)

#getting non-numeric data attributes

l3 = [x for x in data.columns if x not in data._get_numeric_data().columns]
data1 = data[l3]
data1.head()

#creating dummy variables for categorical data
data2 = pd.get_dummies(data1)

#getting numeric data and concatinating with dummy variable of categorical data

data._get_numeric_data()
data4 = pd.concat([data2,data._get_numeric_data()],axis=1)

#creatig train and test data

X = data4.iloc[:,0:len(data4.columns)-1]
y = data4.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


#creating linear regression object and fitting the data
lr = LinearRegression()
lr.fit(X_train,y_train)
predictions = lr.predict(X_test)

#checking predicitions by scatterplot and root mean square values

plt.scatter(predictions,y_test)
mean_squared_error(y_test,predictions)


#creating gardient boosting regressor object and fitting the data

rand = GradientBoostingRegressor()
rand.fit(X_train,y_train)
predictions1 = rand.predict(X_test)

#checking predicitions by scatterplot and root mean square values

plt.scatter(y_test,predictions1)
mean_squared_error(y_test,predictions1)

