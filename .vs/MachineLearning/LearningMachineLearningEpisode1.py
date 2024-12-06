# Importing libraries that we will need

import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model
import matplotlib as plt
import pickle

# Creating a dataframe with pandas Or In simple way loading the data that we have

DataFrame=pd.read_csv("E:\\Work\\Python\\MachineLearning\\datasets\\Student Proformance Dataset\\student-mat.csv",sep=";")
DataFrame.head()

# Filtering out Unnecessary data(Columns) BY Just using those columns which we need.
# This data we will use for training pupose.

# This will be our information from which we will predict
x=DataFrame[['age','Medu','Fedu','traveltime','studytime','failures','famrel','G1','G2']]
x.head()
# This will be the output or That column or data which we have to predict.

y=DataFrame[['G3']]
y.head()
# Above we have worked with dataframes but now we need arrays so we will convert
# the df into arrays using numpy.array

x=np.array(x)
y=np.array(y)
# Spliting the data into two parts total four parts where we will have x_train , x_test, y_train, y_test respectively

x_train,x_test,y_train,y_test= sklearn.model_selection.train_test_split(x, y, test_size=0.3)


# Generating model or initialising model we can use any model we want
Predictor=linear_model.LinearRegression()

# Now we are training the model on the train data set.
Predictor.fit(x_train,y_train)

# Now we are checking how it performed on the test dataset
print(Predictor.score(x_test,y_test))

# We can get tremendous amount and variety of data ranging from coefficients(Weights), max coeff ( max weightage) , Intercept( till which point values dont vary)
# And much more
print("the coefficients or weightage factors for the attributes selected are: \n",Predictor.coef_)
print('the max wieghtage for a attribute is as following:',Predictor.coef_.max())
print("The Intercept or the base value 'where the values dont vary and remains a constant' ",Predictor.intercept_)

# Here we are trying to analyse by ourself that what is the trend and how close is the actual value to the predicted value.

# Here we stor the predicted values for iterating and printing them,
another_predict=Predictor.predict(x_test)

# here we are print the Predicted values, information (on basic of what we predicted) and Actual values.
for i in range(len(another_predict)):
  print(another_predict[i],x_test[i],"now the true value",y_test[i])



print(Predictor.predict([[17,1,1,1,2,0,5,5,5]]))



with open("StudentModelLinearRegression.pickle","wb") as f:
    pickle.dump(Predictor,f)

pickle_in = open("StudentModelLinearRegression.pickle","rb")

Predictor = pickle.load(pickle_in)