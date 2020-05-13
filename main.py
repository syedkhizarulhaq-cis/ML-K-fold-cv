import numpy as np
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2, f_classif
#from sklearn.model_selection import train_test_split
from sklearn.cluster import DBSCAN, SpectralClustering, AgglomerativeClustering
from sklearn import metrics
#from sklearn.cluster import AgglomerativeClustering
from classifierBank import Classifiers
from sklearn.model_selection import cross_val_predict
from sklearn.svm import SVR
from sklearn.model_selection import KFold
#from sklearn import metrics
#import numpy as np
#from classifierBank import Classifiers
import csv
from sklearn.model_selection import StratifiedKFold



#def my_metrics(y_true, y_pred):
    

path='./Dataset/new_df3.csv'
df=pd.read_csv(path)#, index_col='ProtocolName')
#df=df.dropna()
label=df['ProtocolName']
#data= df.select_dtypes(exclude=['object'])
#datas=data
data=df.drop(columns=['ProtocolName'])

x=np.array(data)
y=np.array(label)


classifier= Classifiers()
cv = KFold(n_splits=5)
cv=StratifiedKFold(n_splits=5,shuffle=True)         
for i in classifier.keys():
    acc=[]
    prMacro=[]
    reMacro=[]
    f1Macro=[]
    
    prMicro=[]
    reMicro=[]
    f1Micro=[]
    
    prWgh=[]
    reWgh=[]
    f1Wgh=[]
    
    clf = classifier[i]
        
    print("Classifier :", i)
    
    for train_index, test_index in cv.split(x,y):
#        X_train, X_test, y_train, y_test = x.iloc[train_index],x.iloc[test_index], y.iloc[train_index], y.iloc[test_index]
        X_train, X_test, y_train, y_test = x[train_index],x[test_index],y[train_index],y[test_index]
         
        clf.fit(X_train,y_train)
        
        y_pred = clf.predict(X_train)
#        
        y_test=y_train
#        
        
        acc.append(metrics.accuracy_score(y_test, y_pred))
        prMicro.append(metrics.precision_score(y_test, y_pred, average='micro'))
        reMicro.append(metrics.recall_score(y_test, y_pred, average='micro'))
        f1Micro.append(metrics.f1_score(y_test, y_pred, average='micro'))
        prMacro.append(metrics.precision_score(y_test, y_pred, average='macro'))
        reMacro.append(metrics.recall_score(y_test, y_pred, average='macro'))
        f1Macro.append(metrics.f1_score(y_test, y_pred, average='macro'))
        prWgh.append(metrics.precision_score(y_test, y_pred, average='weighted'))
        reWgh.append(metrics.recall_score(y_test, y_pred, average='weighted'))
        f1Wgh.append(metrics.f1_score(y_test, y_pred, average='weighted'))        
#        y_pred = clf.predict(X_tra)
#        print(metrics.accuracy_score(y_train, y_pred))
        
    r=[
       clf,
       
       np.mean(acc),
       np.std(acc),
       
       np.mean(prMicro),
       np.std(prMicro),
       
       np.mean(reMicro),
       np.std(reMicro),
       
       np.mean(f1Micro),
       np.std(f1Micro),
       
       np.mean(prMacro),
       np.std(prMacro),
       
       np.mean(reMacro),
       np.std(reMacro),
       
       np.mean(f1Macro),
       np.std(f1Macro),

       np.mean(prWgh),
       np.std(prWgh),
       
       np.mean(reWgh),
       np.std(reWgh),
       
       np.mean(f1Wgh),
       np.std(f1Wgh),
                ]
    
    with open('./Result/1_stratKfoldTrain.csv', 'a+', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(r)


#####################################################################

#cols=data.columns
#std=data.std()
#for i in cols:
#    s=std[i]
#    if s==0:
#        print(i)
#        data=df.drop(columns=i) 
#
#1/0
#Xtr, Xts, ytr, yts = train_test_split(data, label, test_size=0.02)
##z,pval=f_classif(Xts,yts)
#

####################################################################
#
#Xtr, Xts, ytr, yts = train_test_split(data, label, test_size=0.2)
#c = DBSCAN(eps=10,min_samples=5)

#c=SpectralClustering(n_clusters=12,assign_labels="discretize")
#c=AgglomerativeClustering(n_clusters=12)
#c.fit(Xts)
#
#print(metrics.accuracy_score(yts,c.labels_))
#SpectralClustering()
#





































################### PCA ###############################
#data=Xts
#from sklearn.decomposition import PCA
#pca = PCA(n_components=data.shape[1])
#pca.fit(data) #training PCA
#projected = pca.transform(data) #projecting the data onto Principal components
##print(digits.data.shape)
##print(projected.shape)
#plt.plot(pca.explained_variance_); plt.grid();
#plt.xlabel('Explained Variance')
#plt.figure()
#plt.plot(np.arange(len(pca.explained_variance_ratio_))+1,np.cumsum(pca.explained_variance_ratio_),'o-') #plot the scree graph
#plt.axis([1,len(pca.explained_variance_ratio_),0,1])
#plt.xlabel('number of components')
#plt.ylabel('cumulative explained variance');
#plt.title('Scree Graph')
#plt.grid()
#plt.show()

#################################################