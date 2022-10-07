import numpy as np
from numpy import random
import math
#Likelihood 가능도(우도) PDF에서의 y값을 가능도로 본다. 정규분포를 따르는 PDF를 생각해보자. 0부터 100까지의 수직선에 있는 변수 중 50을 뽑을 확률은 원래는 0이지만 Likelihood에서는 0.4이다.
#PDF Probability Density Function 확률 밀도함수
#MLE Maximum Liklihood Estimator 최대 가능도 추정
#STD Standard Deviation 표준편차 (분산을 제곱근 한 것)
#Defining the class
class train_test_split():
    def __init__(self,data,target,test_size):
        self._data = data
        self._target = target
        self._test_size = test_size

    def train_test_split(self, data ,target ,test_size):
    #data.reset_index(drop=True, inplace=True) #데이터를 일단 섞어주기
        random.shuffle(data)
        random.shuffle(target)
        x_train = data.iloc[:round(len(data)*(1-test_size)),:]#0~0.7
        y_train = target.iloc[:round(len(target)*(1-test_size)),]
        x_test = data.iloc[round(len(data)*(1-test_size)):,:]#0.7~1
        y_test = target.iloc[round(len(target)*(1-test_size)):,]#0.7~1
        print("x_train: {} , x_test: {} ,y_train: {},y_test: {} ".format(x_train.shape, x_test.shape, y_train.shape, y_test.shape))

        return x_train , x_test , y_train , y_test 

class MLE():
    def __init__(self, samples, m, std, learning_rate, epochs, verbose=False):
        """
        samples: MLE를 얻기위한 표본
        m: mean 평균
        std: 표준편차
        learning_rate: 가중치(weights) 업데이트를 위한 alpha 값
        epochs: training epochs
        verbose: status 출력을 할것인지 여부
        """
        self._samples = samples
        self._m = m
        self._std = std
        self._learning_rate = learning_rate
        self._epochs = epochs
        self._verbose = verbose


    def likelihood(self, x, M):
        """
        확률 밀도함수 PDF는 정규분포 Normal distribution 입니다.
        PDF의 y 값은 likelihood와 같습니다.
        Likelihood = L(확률분포|관측값)
        x: 
        :return: likelihood of input x (likelihood of input x is same as y of pdf)
        """
       
        result = (1/np.sqrt(2*math.pi)*np.power(self._std, 2))*np.exp(-np.power(x-M,2)/(2*np.power(self._std, 2)))
    
        return np.prod(result)

    def fit(self):
        """
        훈련 평가
        우도를 최소화 하는 M은 경사 하강법으로 구할 수 있습니다.
        M은 샘플에 대한 MLE를 뜻합니다.
        """

        # M을 초기화 합니다.
        self._estimator = np.random.normal(self._m, self._std, 1)

        # epochs 횟수만큼 훈련을 진행합니다.
        self._training_process = []
        for epoch in range(self._epochs):
            likelihood = np.prod(self.likelihood(self._samples, self._m))
            prediction = np.prod(self.likelihood(self._samples, self._estimator))
            cost = self.cost(likelihood, prediction)
            self._training_process.append((epoch, cost))
            self.update(self._samples, self._estimator)

            # print status
            if self._verbose == True and ((epoch + 1) % 10 == 0):
                print("Iteration: %d ; cost = %.4f" % (epoch + 1, cost))


    def cost(self, likelihood, prediction):
        """
        비용함수
        likelihood: 모집단의 우도
        :param prediction: 표본의 우도
        return: 최적화된 파라미터 값을 반환합니다.
        """
        return math.sqrt(likelihood - prediction)


    def update(self, x, M):
        """
        경사 하강법을 업데이트 합니다.
        경사는 추정치 입니다.
        x: 표본
        M: 추정치
        """
        gradient = np.sum(np.exp(-(np.power(x - M, 2) / (2*math.pow(self._std, 2)))))
        if self._m > self._estimator:
            self._estimator += self._learning_rate * gradient
        else:
            self._estimator -= self._learning_rate * gradient


    def get_mle(self):
        """
        parameter getter
        return: estimator of MLE
        """
        return self._estimator

class LinearRegression:
    def __init__(self, x , y):
        self.data = x
        self.label = y
        self.m = 0
        self.b = 0
        self.n = len(x)
    
    def compute_cost(self, x, y, theta):
 
        predictions = x.dot(theta)
        errors = np.subtract(predictions, y) 
        sqrErrors = np.square(errors) 
        J = 1 / (2 * self.m) * np.sum(sqrErrors)

        return J
        
    
    def gradient_descent(self, x, y, theta, alpha, iterations):
        cost_history = np.zeros(iterations) #반복하는 횟수만큼 cost를 저장할 0이 들어있는 행렬 만들기

        for i in range(iterations): #반복 시작
            predictions = x.dot(theta) #예측값은 X에 세타를 곱해준  벡터 내적
            errors = np.subtract(predictions, y) #에러는 예측값과 y를 빼준 값
            sum_delta = (alpha / self.m) * x.transpose().dot(errors)#세타의 수정값 sum_delta 공식 그대로 넣은것 
            theta = theta - sum_delta #수정된 세타의 값 

            cost_history[i] = self.compute_cost(x, y, theta) #cost를 계산한 값에 관한 기록을 cost history에 저장하자.   

        return theta, cost_history

   


class LogisticRegression:
    def __init__(self,x,y):      
        self.intercept = np.ones((x.shape[0], 1))  #y절편
        self.x = np.concatenate((self.intercept, x), axis=1)#x절편 concatenate는 배열 합치기
        self.weight = np.zeros(self.x.shape[1])#가중치
        self.y = y#y값 
         
    #Sigmoid method
    def sigmoid(self, x, weight):
        z = np.dot(x, weight)
        return 1 / (1 + np.exp(-z))
     
    #method to calculate the Loss 코스트 함수와 비슷한 개념이라 보며 된다.
    def loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean() 
     
    #Method for calculating the gradients
    def gradient_descent(self, X, h, y):
        return np.dot(X.T, (h - y)) / y.shape[0]#X.T Same as self.transpose()
 
     
    def fit(self, lr , iterations):
        for i in range(iterations):
            sigma = self.sigmoid(self.x, self.weight)
             
            loss = self.loss(sigma,self.y)
 
            dW = self.gradient_descent(self.x , sigma, self.y)
             
            #Updating the weights
            self.weight -= lr * dW
 
        return print('fitted successfully to data')
     
    #Method to predict the class label.
    def predict(self, x_new , treshold):
        x_new = np.concatenate((self.intercept, x_new), axis=1)
        result = self.sigmoid(x_new, self.weight)
        result = result >= treshold
        y_pred = np.zeros(result.shape[0])
        for i in range(len(y_pred)):
            if result[i] == True: 
                y_pred[i] = 1
            else:
                continue
                 
        return y_pred


# if __name__ == "__main__":

#     # samples for MLE
#     samples = np.array([64, 64.5, 65, 65.5, 66])

#     # assumptions about the population
#     mean = np.array([65.0])
#     std = 5

#     # get MLE
#     estimator = MLE(samples, mean, std, learning_rate=0.1, epochs=30, verbose=True)
#     estimator.fit()
#     result = estimator.get_mle()
#     print(result)