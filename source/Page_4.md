
# Page 04
----
## Q1：


#### The solution of (a):


```python
import numpy as np
v = np.arange(-5,5,0.1)
u = np.arange(-5,5.1,0.1)
print("Let's see the last several numbers of two arrays:")
print("The last 5 numbers of v:",v[95:101])
print("The last 6 numbers of v:",u[95:102],'\n')

print("(1) Let's see the lenghth and type of each array:")
print("The lenghth of 'v' is", len(v),"and",type(v))
print("The lenghth of 'u' is", len(u),"and",type(u),'\n')

print("(2) Let's check the type of the numbers in each array:")
print("The type of numbers in 'v' is",type(v[0]))
print("The type of numbers in 'u' is",type(u[0]),'\n')

print("(3) Let's compare the inside numbers between two arrays:")
compare = [True if u[i]==v[i] else False for i in range(len(v))]
print('The number of "true" in list "compare" is',sum(compare),'which is equal to the length of array "v",',len(v),'.')
print('So, it means the first 100 numbers in array "u" are all equal to all the numbers in array "v".\n')

print('Therefore, the only difference between "u" and "v" is that "u" has one more number(5.) than "v".')
```

    Let's see the last several numbers of two arrays:
    The last 5 numbers of v: [4.5 4.6 4.7 4.8 4.9]
    The last 6 numbers of v: [4.5 4.6 4.7 4.8 4.9 5. ] 
    
    (1) Let's see the lenghth and type of each array:
    The lenghth of 'v' is 100 and <class 'numpy.ndarray'>
    The lenghth of 'u' is 101 and <class 'numpy.ndarray'> 
    
    (2) Let's check the type of the numbers in each array:
    The type of numbers in 'v' is <class 'numpy.float64'>
    The type of numbers in 'u' is <class 'numpy.float64'> 
    
    (3) Let's compare the inside numbers between two arrays:
    The number of "true" in list "compare" is 100 which is equal to the length of array "v", 100 .
    So, it means the first 100 numbers in array "u" are all equal to all the numbers in array "v".
    
    Therefore, the only difference between "u" and "v" is that "u" has one more number(5.) than "v".
    

#### The solution of (b):


```python
import matplotlib.pyplot as plt
x = np.arange(-5,5,0.1) #using np.arange to create a list of x
y = [np.exp(i) for i in x] #using x to compute y by function "np.exp()"
plt.plot(x,y)
```




    [<matplotlib.lines.Line2D at 0x1dd5d7c4fd0>]




![png](output_4_1.png)


## Q2:

#### The solution of (a):


```python
print('The normal command is like this:')
A = np.array([[2,0,2],
             [2,2,3],
             [5,3,9]])
print(A)
print('\n')
print('I found another method which is more convenient by using "reshape()":')
A = np.array([2,0,2,2,2,3,5,3,9]).reshape([3,3])
print(A)
```

    The normal command is like this:
    [[2 0 2]
     [2 2 3]
     [5 3 9]]
    
    
    I found another method which is more convenient by using "reshape()":
    [[2 0 2]
     [2 2 3]
     [5 3 9]]
    

#### The solution of (b):


```python
print("The result of A*A is ")
print(A*A)
print('The result of "A*A" is the dot product of two matrices.')
print('Specifically, it is a multiplication of the corresponding elements of two matrices, so both matrices must be of the same size')
```

    The result of A*A is 
    [[ 4  0  4]
     [ 4  4  9]
     [25  9 81]]
    The result of "A*A" is the dot product of two matrix.
    Specifically, it is a multiplication of the corresponding elements of two matrices, so both matrices must be of the same size
    

#### The solution of (c):


```python
print('The result of A@A is ')
print(A@A)
print('The result of "A@A" is the multiplication of two matrices.')
print('When the number of columns of the first matrix is equal to the number of rows of the second matrix,\
the two matrices can be multiplied together.')
```

    The result of A@A is 
    [[ 14   6  22]
     [ 23  13  37]
     [ 61  33 100]]
    The result of "A@A" is the multiplication of two matrices.
    When the number of columns of the first matrix is equal to the number of rows of the second matrix,the two matrices can be multiplied together.
    

#### The solution of (d):


```python
print('The result of A**2 is ')
print(A**2)
print('The result of "A**2" is the dot product with itself which is equivalent to "A*A"')
```

    The result of A**2 is 
    [[ 4  0  4]
     [ 4  4  9]
     [25  9 81]]
    The result of "A**2" is the dot product with itself which is equivalent to "A*A"
    

## Q3:

#### The solution of (a):

The k-nearest neighbors method with k=1 will first calculate the distance between the point to be predicted and each point in the dataset, and then use the label of the closest point as the prediction result. The commonly used distance is the Euclidean distance.

We can see that there are four nodes in the dataset and two nodes needed to be predicted. Let's calculate the distance between the first node (0,0) and all the notes in the dataset:

dis1 = $\sqrt{(0-1)^2 + (0-2)^2}$  = $\sqrt{5}$ 

dis2 = $\sqrt{(0-3)^2 + (0-1)^2}$  = $\sqrt{10}$

dis3 = $\sqrt{(0-(-4))^2 + (0-(-2))^2}$  = $2\sqrt{5}$

dis4 = $\sqrt{(0-(-3))^2 + (0-(-4))^2}$  = $5$

'dis1' is the smallest, so the node (1,2) in the dataset is the nearest node with node (0,0) which we want to predict. So, the predictive Class Label of node (0,0) is "1" which is the same with node (1,2)'s label.

To the node (-2,-3), there is a similiar step:

dis1_ = $\sqrt{(-2-1)^2 + (-3-2)^2}$  = $\sqrt{34}$ 

dis2_ = $\sqrt{(-2-3)^2 + (-3-1)^2}$  = $\sqrt{41}$

dis3_ = $\sqrt{(-2-(-4))^2 + (-3-(-2))^2}$  = $\sqrt{5}$

dis4_ = $\sqrt{(-2-(-3))^2 + (-3-(-4))^2}$  = $\sqrt{2}$

'dis4_' is the smallest, so the node (-3,-4) in the dataset is the nearest node with node (-2,-3) which we want to predict. So, the predictive Class Label of node (-2,-3) is "0" which is the same with node (-3,-4)'s label.

#### The solution of (b):

When K = 2, the calculation is similar. The difference is that we need to find the top two closest points in the data set to the point to be predicted.

Let's calculate the distance between the first node (0,0) and all the notes in the dataset:

dis1 = $\sqrt{(0-1)^2 + (0-2)^2}$  = $\sqrt{5}$ 

dis2 = $\sqrt{(0-3)^2 + (0-1)^2}$  = $\sqrt{10}$

dis3 = $\sqrt{(0-(-4))^2 + (0-(-2))^2}$  = $2\sqrt{5}$

dis4 = $\sqrt{(0-(-3))^2 + (0-(-4))^2}$  = $5$

'node1' and 'node2' are the top two closest points in the dataset, whose labels are both "1". So, the predictive Class Label of node (0,0) is "1". 


To the node (-2,-3), there is a similiar step:

dis1_ = $\sqrt{(-2-1)^2 + (-3-2)^2}$  = $\sqrt{34}$ 

dis2_ = $\sqrt{(-2-3)^2 + (-3-1)^2}$  = $\sqrt{41}$

dis3_ = $\sqrt{(-2-(-4))^2 + (-3-(-2))^2}$  = $\sqrt{5}$

dis4_ = $\sqrt{(-2-(-3))^2 + (-3-(-4))^2}$  = $\sqrt{2}$

'node3' and 'node4' are the top two closest points in the dataset, whose labels are both "0". So, the predictive Class Label of node (-2,-3) is "0". 

## Q4:


```python
import warnings
warnings.filterwarnings("ignore")
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
import numpy as np
 
X = np.array([40,50,100,200]).reshape([-1,1])  #the dataset and labels
y = np.array([8,10,22,50]).reshape([-1,1])

X_pred = np.array([[80]])  #the node we want to predict

plt.plot(X,y,'bd') # Let's see the dataset:
```




    [<matplotlib.lines.Line2D at 0x14fb87cb7b8>]




![png](output_20_1.png)


#### (a) when K = 1:


```python
knn_reg = KNeighborsRegressor(n_neighbors = 1)

knn_reg.fit(X,y)
y_pred = knn_reg.predict(X_pred)
print('The prediction of X_pred is:',y_pred)

plt.plot(X,y,'bd')
plt.plot(X_pred,y_pred,'ro')
```

    The prediction of X_pred is: [[22.]]
    




    [<matplotlib.lines.Line2D at 0x14fb888f978>]




![png](output_22_2.png)


When K=1, the prediction result is 22, which means that the price should be set to 22. From the figure above, we can also see that the point to be predicted (the red point) is closest to the right blue point, so the KNN algorithm takes the nearest neighbor's label as the prediction result.

#### (b) when K = 2:


```python
knn_reg = KNeighborsRegressor(n_neighbors = 2)

knn_reg.fit(X,y)
y_pred = knn_reg.predict(X_pred)
print('The prediction of X_pred is:',y_pred)

plt.plot(X,y,'bd')
plt.plot(X_pred,y_pred,'ro')
```

    The prediction of X_pred is: [[16.]]
    




    [<matplotlib.lines.Line2D at 0x14fb88dcc88>]




![png](output_25_2.png)


When K=2, the prediction result is 16, which means that the price should be set to 16. From the figure above, we can also see that the point to be predicted (the red point) is located between the second and third blue dots, which are the top two closest nodes to the red node. So, The KNN algorithm takes the average of the labels of the two blue points as the prediction result.

#### My Summary:

I will take 16 as the final prediction. Because the method to be applied in this question is regression, but when K=1, the KNN regression algorithm is essentially the same as the classification algorithm when K=1, which is not what we want. Therefore, the KNN regression algorithm at K=2 will be more accurate.
