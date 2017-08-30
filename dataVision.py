import numpy as np 
from sklearn.datasets import make_classification
x,y=make_classification(1000,n_features=20,n_informative=2,n_redundant=2,n_classes=2,random_state=0)

from pandas import DataFrame

df=DataFrame(np.hstack((x,y[:,None])),columns=range(20)+["class"])

import matplotlib.pyplot as plt 
import seaborn as sns
_=sns.pairplot(df[:50],vars=[8,11,12,14,19],hue="class",size=1.5)
#http://www.bida.org.cn/index.php?qa=7




#plt.figure(figsize=(12,10))
#_=sns.corrplot(df,annot=False)
#plt.show() 
from sklearn.svm import LinearSVC
from sklearn.learning_curve import learning_curve

def plot_learning_curve(estimator,title,x,y,ylim=None,cv=None,train_size=np.linspace(.1,1.0,5)):
    plt.figure()
    train_size,train_scores,test_scores=learning_curve(estimator,x,y,cv=5,n_jobs=1,train_sizes=train_size)
    train_scores_mean=np.mean(train_scores,axis=1)
    train_scores_std=np.std(train_scores,axis=1)
    test_scores_mean=np.mean(test_scores,axis=1)
    test_scores_std=np.std(test_scores,axis=1)
    plt.fill_between(train_size,train_scores_mean - train_scores_std,train_scores_mean+train_scores_std,alpha=0.1,color="r")
    plt.fill_between(train_size,test_scores_mean - test_scores_std,test_scores_mean+test_scores_std,alpha=0.1,color="r")
    plt.plot(train_size,train_scores_mean,'o-',color='r',label="training score")
    plt.plot(train_size,test_scores_mean,'o-',color='g',label="cross_validation score")
    plt.xlabel("trainning example")
    plt.ylabel("score")
    plt.grid("on")
    if ylim:
        plt.ylim(ylim)
    plt.title(title)
    plt.show()

plot_learning_curve(LinearSVC(C=10.0),"LinearSVC",x,y,ylim=(0.8,1.1),train_size=np.linspace(.1,1,5))