import numpy as np
import matplotlib.pyplot as plt
# np.random.seed(0)

def make_theta(s):
    # Returns a list of randomly initialized parameter matrices for each layer of NN.   
    L=len(s)-1
    eps=1
    theta=[0]*(L)
    for l in range(1,L):
        theta[l]=np.random.random_sample((s[l+1],s[l]+1))*2*eps-eps
    return theta

def predict_y(X,theta,L,m):
    # returns y_predicted, calculated using forward propagation.
    z=[0]*(L+1)
    a=[0]*(L+1)
    a[1]=X
    for l in range(2,L+1):
        z[l]=theta[l-1]@a[l-1]
        a[l]=g(z[l])
        if(l!=L):
            a[l]=np.vstack((np.ones([1,m]),a[l]))
    return a[L]

def g(z):
    #  returns the sigmoid function.
    return 1/(1+np.exp(-z))

def gdash(z):
    #  returns derivative of sigmoid function
    return np.exp(-z)/((1+np.exp(-z)**2))

def check(yp,y):
    # returns accuracy of prediction.
    c1=0
    count=0
    for i in range(len(yp)):
        if yp[i][0]>=0.5:
            yp[i][0]=1
            c1+=1
        else:
            yp[i][0]=0
        if y[i][0]==yp[i][0]:
            count+=1
    # print(c1)
    return count

def cost(yp,y,theta,m,lam):
    # returns the cost, incorporating regularization.
    c1=np.sum(y*np.log(yp)+(1-y)*np.log(1-yp))
    c2=0
    for th in theta:
        c2+=np.sum(th**2)
    return lam*c2/(2*m)-c1/m 

fil=open("file2.txt")
A=np.loadtxt(fil)
m=800                                                           # no. of training examles
n=A.shape[1]-1                                                  # no, of features.
X=(np.hstack((np.ones([m,1]),A[0:m,0:n]))).T                    # training dta matrix
y=A[0:m,n:n+1].T                                                # trining data output

mv=A.shape[0]-m
Xval=(np.hstack((np.ones([mv,1]),A[m:m+mv,0:n]))).T             # validation data
yval=A[m:m+mv,n:n+1].T

fscal=np.array([1,1,1,1,1,500,1])
fscal=fscal.reshape(len(fscal),1)                               # implementing feature scaling 
Xval/=fscal
X/=fscal
s=np.array([0,n,5,5,1])                                          # no. of neurons in each layers(neural architecture)
L=len(s)-1                                                       # no. of layers

theta=make_theta(s)
z=[0]*(L+1)
a=[0]*(L+1)                                                    # initialization of different parameters required in backpropagation algorithm
delta=[0]*(L+1)                                                # D is the derivative of cost function wrt theta.
capdel=[0]*L
D=[0]*L

al=0.01  
epochs=8000                                                    # initialization of parameters of gradient descent
lamda=1                                                        # lamda is regularization constant
j=[]

for i in range(epochs):
    a[1]=X
    for l in range(2,L+1):
        z[l]=theta[l-1]@a[l-1]
        a[l]=g(z[l])                                           # the bacpropagation algorithm.
        if(l!=L):
            a[l]=np.vstack((np.ones([1,m]),a[l]))
    delta[L]=a[L]-y
    for l in range(L-1,1,-1):
        delta[l]=(theta[l][:,1:]).T@delta[l+1]*gdash(z[l])
    for l in range(1,L):
        capdel[l]=delta[l+1]@a[l].T
        D[l]=(capdel[l]+lamda*(theta[l]-theta[l][:,0:1]*np.hstack((np.ones([s[l+1],1]),np.zeros([s[l+1],s[l]])))))/m
        theta[l]-=al*D[l]                                      # gradient descent algorithm

    j.append(cost(a[L],y,theta,m,lamda))

# acc=check(a[L].T,y.T)
# print(f"accuracy on training data is {acc}/{m} i.e {acc*100/m}%")
plt.plot(range(epochs),j)                                               # plotting the cost function
# plt.show()

yp=predict_y(Xval,theta,L,mv)
acc=check(yp.T,yval.T)
print(f"accuracy on validation data is {acc}/{mv} i.e {acc*100/mv}%")  # finding accuracy of prediction on validation data
