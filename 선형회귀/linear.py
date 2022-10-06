import numpy as np
 
#Defining the class
class LinearRegression:
    def __init__(self, x , y):
        self.data = x
        self.label = y
        self.m = 0
        self.b = 0
        self.n = len(x)
         
    def fit(self , epochs , lr):
         
        #Implementing Gradient Descent
        for i in range(epochs):
            y_pred = self.m * self.data + self.b
             
            #Calculating derivatives w.r.t Parameters
            D_m = (-2/self.n)*sum(self.data * (self.label - y_pred))
            D_b = (-1/self.n)*sum(self.label-y_pred)
             
            #Updating Parameters
            self.m = self.m - lr * D_m
            self.c = self.b - lr * D_b
             
    def predict(self , inp):
        y_pred = self.m * inp + self.b 
        return y_pred

#importing Matplotlib for plotting
import matplotlib.pyplot as plt
 
#Loding the data
df = pd.read_csv('data_LinearRegression.csv')
 
#Preparing the data
x = np.array(df.iloc[:,0])
y = np.array(df.iloc[:,1])
 
#Creating the class object
regressor = LinearRegression(x,y)
 
#Training the model with .fit method
regressor.fit(1000 , 0.0001) # epochs-1000 , learning_rate - 0.0001
 
#Prediciting the values
y_pred = regressor.predict(x)
 
#Plotting the results
plt.figure(figsize = (10,6))
plt.scatter(x,y , color = 'green')
plt.plot(x , y_pred , color = 'k' , lw = 3)
plt.xlabel('x' , size = 20)
plt.ylabel('y', size = 20)
plt.show()