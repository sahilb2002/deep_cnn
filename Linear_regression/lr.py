import numpy as np
import matplotlib.pyplot as plt
fil=open("data.txt")
def arrange(A):
    m=A.shape[0]
    for i in range(m-1):
        for j in range(i+1,m):
            if A[j][0]<A[i][0]:
                t=A[i][0]
                A[i][0]=A[j][0]
                A[j][0]=t
                t=A[i][1]
                A[i][1]=A[j][1]
                A[j][1]=t
    return A

def rsq(yp,y):
    m=len(y)
    ymean=np.sum(y)/m
    sst=np.sum((y-ymean)**2)
    ssr=np.sum((yp-y)**2)
    return 1-ssr/sst

# linear regression
A=np.loadtxt(fil,delimiter=",")
A=arrange(A)
m=A.shape[0]
X=np.hstack((np.ones([m,1]),A[:,0:1]))
y=A[:,1:2]
theta=np.linalg.inv(X.T@X)@X.T@y
yp=X@theta
print(rsq(yp,y)*100)
plt.scatter(X[:,1],y)
# plt.plot(X[:,1],yp)

# polynomial regression
n=4
X=np.hstack((X,X[:,1:2]**2,X[:,1:2]**3,X[:,1:2]**4))
I=np.identity(n+1)
I[0][0]=0
lamda=0
theta=np.linalg.inv(X.T@X+lamda*I)@X.T@y
yp=X@theta
print(rsq(yp,y)*100)
plt.plot(X[:,1],yp)
plt.savefig("plot.png")
plt.show()