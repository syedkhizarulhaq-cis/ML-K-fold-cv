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
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt
from pycm import ConfusionMatrix
#from sklearn import metrics
#import numpy as np
#from classifierBank import Classifiers
import csv



1/0
def saveFig(fig,FileName):
    fig.savefig(FileName+'.png',quality=95)
    fig.savefig(FileName+'.jpg',quality=95)
    fig.savefig(FileName+'.eps',quality=95)




#'APPLE_ITUNES':1,'SPOTIFY':1  -> Music
#'NETFLIX':2,'YOUTUBE':2 -> Video Streaming
#'SKYPE':3,'TWITTER':3,'INSTAGRAM':3,'WHATSAPP':3 -> Social
#'MICROSOFT':4,'OFFICE_365':4,'MS_ONE_DRIVE':4,'WINDOWS_UPDATE':4,'MSN':4 -> Microsoft Related Services (Microsoft)
#'APPLE':5,'APPLE_ICLOUD':5 -> Apple Related Services (Apple)
#'WIKIPEDIA':6 -> WIKIPEDIA
#'DNS':7 -> DNS
#'EBAY':8,'AMAZON':8 -> E-Commerce
#'SSL':9,'HTTP':9,'HTTP_CONNECT':9,'HTTP_PROXY':9 -> Web Protocols
#'GMAIL':10,'YAHOO':10 -> Mail
#'DROPBOX':11,'IP_ICMP':11,'CONTENT_FLASH':11,'CLOUDFLARE':11 -> Dropbox and Other
#'GOOGLE':12 -> GOOGLE
length=11
height=0.625*length
savingPath='./Result/FinalResult/'
plt.close('all')
mapping={1: 'Music', 2: 'Video', 3:'Social', 4:'Microsoft',
         5: 'Apple', 6:'Wikipedia', 7: 'DNS', 8:'E-commerce', 9:'Web',
         10:'Emails', 11:'Misc.', 12:'Google'}
    
labelNames=list(mapping.values())

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
#cv = KFold(n_splits=5)  

strfmt=['-','--','-.',':']
counter=-1
#clfList=[7,14,20,21]
clfList=[83,76,37]
Xtr, Xts, ytr, yts = train_test_split(x, y, test_size=0.2)


y_test=yts
# plt.figure(figsize=(length, height))
# # 
for i in clfList:#classifier.keys():
    
    clf = classifier[i]

#    clff = OneVsRestClassifier(clf)
    clf.fit(Xtr, ytr)
    y_pred=clf.predict(Xts)
#     yScores=clf.predict_proba(Xts)
# #    yScores=clff.predict_proba(Xts)


    print("Classifier :", i)
# #    print('Accuracy : ',metrics.accuracy_score(yts, clf.predict(Xts)))

    r=[
clf,
 metrics.accuracy_score(y_test, y_pred),
 metrics.precision_score(y_test, y_pred, average='micro'),
 metrics.recall_score(y_test, y_pred, average='micro'),
 metrics.f1_score(y_test, y_pred, average='micro'),
 metrics.precision_score(y_test, y_pred, average='macro'),
 metrics.recall_score(y_test, y_pred, average='macro'),
 metrics.f1_score(y_test, y_pred, average='macro'),
 metrics.precision_score(y_test, y_pred, average='weighted'),
 metrics.recall_score(y_test, y_pred, average='weighted'),
 metrics.f1_score(y_test, y_pred, average='weighted')         

   ]

#    1/0
    with open(savingPath+'SummarizedResult.csv', 'a+', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(r)
    
    
    
#     classes=np.unique(y)
# #    1/0
    
#     ll=pd.get_dummies(y_pred)
#     ll=np.array(ll)
#     ll=yScores
#     # counter=-1
#     for i in range(len(classes)):
#         # counter=counter+1
#         zz=np.zeros([len(yts),1])
#         print('For Class ',i)
        
        
#         ypred=yScores[:,i]
#         yAct=np.where(yts==classes[i])
#         zz[yAct]=1.0
#         zz=zz.astype(int)
        
#         # print('For Class {0} count is {1}'.format(i,sum(zz)))
        

#         a,b,c=metrics.roc_curve(zz.T[0],ll[:,i])#y_pred)
#         scrr=metrics.roc_auc_score(zz.T[0],ll[:,i])#y_test,ll[:,counter])
#         plt.plot(a,b,label=labelNames[i]+' Score: {0:.2f}'.format(scrr),ls=strfmt[(i)%len(strfmt)],linewidth=2)#,marker='.')
    
#     plt.legend()
#     plt.grid(True)    
    


#        print(metrics.accuracy_score(yts, clf.predict(Xts)))
    
    
#        print(metrics.accuracy_score(zz, ypred))
        # print(metrics.roc_curve(zz,ypred))
        
#1/0

    
    
    
    # import seaborn as sns
    # plt.figure()
    # sns.heatmap(metrics.confusion_matrix(y_test,y_pred),xticklabels=labelNames,yticklabels=labelNames, cmap='RdYlGn',annot=True)
    #
    # fig1=plt.figure(figsize=(length, height))
    fig1=plt.figure(figsize=(length, height))
    ax1=fig1.add_subplot(1,1,1)
    metrics.plot_confusion_matrix(clf,Xts,yts,
                                    normalize='pred',
                                  display_labels=labelNames,
                                  xticks_rotation=40,
                                  include_values=True,
                                    # values_format='d',
                                    values_format='.2f',
                                  cmap='RdPu',
                                  ax=ax1)

    fig2=plt.figure(figsize=(length, height))
    ax2=fig2.add_subplot(1,1,1)
    metrics.plot_confusion_matrix(clf,Xts,yts,
                                    # normalize='pred',
                                  display_labels=labelNames,
                                  xticks_rotation=40,
                                  include_values=True,
                                    values_format='d',
                                    # values_format='.2f',
                                  cmap='RdPu',
                                  ax=ax2)






    fname=savingPath+type(clf).__name__+'_idx_{0}'.format(i)
    saveFig(fig1,fname+'predNormalized')
    saveFig(fig2,fname+'Direct')
    cm=ConfusionMatrix(yts,y_pred)
    cm.save_csv(fname)
    
    with open(fname+'_overallStat.csv', 'w+', newline="") as csv_file:  
        writer = csv.writer(csv_file)
        for key, value in cm.overall_stat.items():
            writer.writerow([key, value])
    
    
    
    # plt.figure()
    # plt.hist(yts,bins=len(classes))
    
#        thisClass=np.where(yts==i)[0]
#            
#    
#    
#    1/0
#    
    
    
#    for train_index, test_index in cv.split(x):
#        X_train, X_test, y_train, y_test = x.iloc[train_index],x.iloc[test_index], y.iloc[train_index], y.iloc[test_index]
#        X_train, X_test, y_train, y_test = x[train_index],x[test_index],y[train_index],y[test_index]
         
#        clf.fit(X_train,y_train)
#        
#        y_pred = clf.predict(X_test)
##        
##        y_test=y_train
##        
#        
#        acc.append(metrics.accuracy_score(y_test, y_pred))
#        prMicro.append(metrics.precision_score(y_test, y_pred, average='micro'))
#        reMicro.append(metrics.recall_score(y_test, y_pred, average='micro'))
#        f1Micro.append(metrics.f1_score(y_test, y_pred, average='micro'))
#        prMacro.append(metrics.precision_score(y_test, y_pred, average='macro'))
#        reMacro.append(metrics.recall_score(y_test, y_pred, average='macro'))
#        f1Macro.append(metrics.f1_score(y_test, y_pred, average='macro'))
#        prWgh.append(metrics.precision_score(y_test, y_pred, average='weighted'))
#        reWgh.append(metrics.recall_score(y_test, y_pred, average='weighted'))
#        f1Wgh.append(metrics.f1_score(y_test, y_pred, average='weighted'))        
#        y_pred = clf.predict(X_tra)
#        print(metrics.accuracy_score(y_train, y_pred))
#
#    r=[
#       clf,
#       
#       np.mean(acc),
#       np.std(acc),
#       
#       np.mean(prMicro),
#       np.std(prMicro),
#       
#       np.mean(reMicro),
#       np.std(reMicro),
#       
#       np.mean(f1Micro),
#       np.std(f1Micro),
#       
#       np.mean(prMacro),
#       np.std(prMacro),
#       
#       np.mean(reMacro),
#       np.std(reMacro),
#       
#       np.mean(f1Macro),
#       np.std(f1Macro),
#
#       np.mean(prWgh),
#       np.std(prWgh),
#       
#       np.mean(reWgh),
#       np.std(reWgh),
#       
#       np.mean(f1Wgh),
#       np.std(f1Wgh),
#                ]
#    1/0
#    with open('./Result/Results.csv', 'a+', newline='') as file:
#        writer = csv.writer(file)
#        writer.writerow(r)


#    
#    
#    
#    
#    acc=[]
#    prMacro=[]
#    reMacro=[]
#    f1Macro=[]
#    
#    prMicro=[]
#    reMicro=[]
#    f1Micro=[]
#    
#    prWgh=[]
#    reWgh=[]
#    f1Wgh=[]