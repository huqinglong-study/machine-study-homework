#!/usr/bin/env python
# coding: utf-8

# In[12]:


#HMM三选一作业第一题
#胡庆隆
#学号：201700171018



import numpy as np

class HMM():    
    def __init__(self,A,B,Pi,Ot):
        self.A=A
        self.B=B
        self.Pi=Pi
        self.N=np.shape(A)[0]
        self.M=np.shape(B)[1]
        self.O = Ot
        self.T=len(self.O)
        self.alpha = np.zeros((self.T,self.N))
        self.beita=np.zeros((self.T,self.N))
    
    def forward(self):     #前向算法
        
        for i in range(self.N):
            self.alpha[0][i]=self.Pi[i]*self.B[i][self.O[0]]
           
        
        for t in range(1,self.T):
            for i in range(self.N): 
                sumj=0
                for j in range(self.N):
                    sumj += self.alpha[t-1][j]*self.A[j][i]
                self.alpha[t][i]= sumj*self.B[i][self.O[t]]
    
    def backward(self):     #后向算法
        self.beita=np.zeros((self.T,self.N))
        for i in range(self.N):
            self.beita[self.T-1][i]=1.0
       
        for t in range(self.T-2,-1,-1):    
            for i in range(self.N):              
                for j in range(self.N):
                    self.beita[t][i] += self.A[i][j]*self.B[j][self.O[t+1]]*self.beita[t+1][j]
                    
        
        
    def qi_probability(self,t,i):  #给定A，B，Pi和观测O，在t时刻处于状态qi的概率
        
        t=t-1         #矩阵的位置为原数字减1
        i=i-1
        mutlip=self.alpha[t][i]*self.beita[t][i]
        sumab=0
        for j in range(self.N):
            sumab += self.alpha[t][j]*self.beita[t][j]
        return mutlip/sumab    

A=[[0.5,0.1,0.4],[0.3,0.5,0.2],[0.2,0.2,0.6]]   #定义A，B，Pi，和O
B=[[0.5,0.5],[0.4,0.6],[0.7,0.3]]
Pi=[0.2,0.3,0.5]


Ot=[0,1,0,0,1,0,1,1]

hmm=HMM(A,B,Pi,Ot)                #输入A，B，Pi，和O
hmm.forward()                     #运行前向产生alpha
hmm.backward()                    #运行后向产生beta

answer=hmm.qi_probability(4,3)    #输入t，i
print(answer)                     #输出结果
        


# In[ ]:





# In[ ]:





# In[ ]:




