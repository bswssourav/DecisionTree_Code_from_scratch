
import pandas as pd
import xlrd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from inspect import getmembers
import graphviz
from sklearn.tree import export_graphviz

rootNode=""
root=" "
level=0
rootNode1=""
level1=0

def countPossitiveNegativeClassAt(recvData):
    #print(recvData)
    countPossi=0
    countNege=0
    for i in range(0, len(recvData)):
        if((recvData[i]=="yes")):
            countPossi=countPossi+1
        if(recvData[i]=="no"):
            countNege=countNege+1
    return [countPossi, countNege]


def entrophyClass(cnpossi,cnnege,recv):
    length=len(recv)
    etclass=0
    if(cnpossi/length==0):
        etclass=etclass+0
    else:
        etclass=-((cnpossi/length)* (np.log2(cnpossi/length)))+(-((cnnege/length)*(np.log2(cnnege/length))))
    return etclass





def entropy_each_attribute(df,att):
  Class = df.keys()[-1]

  Vtarget = df[Class].unique()
  #print(Vtarget)

  Vattribute = df[att].unique()
 # print(Vattribute)
  entropywt = 0
  for i in Vattribute:
      entropy = 0
      countat1=0

      for k in range(0,len(df)):
              if(i==df[att][k]):
                  countat1=countat1+1
      #print(countat1)
      #print(i)



      for j in Vtarget:
           countat2 = 0
           for k in range(0, len(df)):
              if (i == df[att][k] and j==df[Class][k]):
                  countat2 = countat2 + 1
           #print(countat2)
           #print(j)
           nuem = countat2
           deno = countat1
           frac = nuem/float(deno)
           if(frac==0):
               entropy += 0
           else:
               entropy = entropy+(-frac*np.log2(frac))
      frac2 = countat1/float(len(df))
      entropywt += frac2*entropy
  return entropywt







def entroNode(df):
    Classen = df.keys()[-1]
    entropy = 0
    values = df[Classen].unique()
    for i in values:
        frac = df[Classen].value_counts()[i] / len(df[Classen])
        entropy =entropy +( -frac * np.log2(frac))


    return entropy


def findBestSplit(df):
    InfoGain = []
    for key in df.keys()[:-1]:
        entropywt = entropy_each_attribute(df, key)
        fet=entroNode(df) - entropywt
        InfoGain.append(fet)
    #print("entrophywt== ", InfoGain)
    return df.keys()[:-1][np.argmax(InfoGain)]


def findRootNode(entroClass,df):
    InfoGain = []
    for key in df.keys()[:-1]:
        entropywt = entropy_each_attribute(df, key)
        InfoGain.append(entroClass - entropywt)
    # return df.keys()[:-1][np.argmax(IG)]
    print(InfoGain)

    print("InfoGain Of root node ======== ", np.max(InfoGain))
    print("And the selected root node::  ",df.keys()[:-1][np.argmax(InfoGain)])


def print_node(n):
    print("Current node of the tree::::::::::",n)



def BuildInfoGainDecissionTree(entroClass,df,max_depth,iter,Decissiontree=None):
    Class = df.keys()[-1]
    global level
    global rootNode
    attriVal=0
    #root=0
    #node=0
    """if(iter==0):
        root=findRootNode(entroClass, df)
        print("root== ",root)
        attriVal = np.unique(df[root])
        print(attriVal)"""


    node=findBestSplit(df)
    print_node(node)
    #print('node==',node)
    if(iter == 0):
        rootNode = node
        #print("mmm",root)
        iter=1
    attriVal=np.unique(df[node])
    if Decissiontree is None:
        Decissiontree = {}


        Decissiontree[node] = {}
    for i in attriVal:
        #print(node ,'=',i)
        if(level>max_depth):
            break
        subTable= df[df[node] == i].reset_index(drop=True)


        clValue, counts = np.unique(subTable['Label'], return_counts=True)

        if(len(counts)==1):
            if(rootNode==node):
                print(node, " = ", i, " : ", clValue[0])
                Decissiontree[node][i]=clValue[0]

            else:
                if(level>0):
                    for j in range(0,level):
                        print("   ",end=" ")
                        if(j+1==level):
                            print( "  |" ,node, " = ",i," : ",clValue[0])
                            Decissiontree[node][i] = clValue[0]
                else:
                    print("   |", node, " = ", i, " : ", clValue[0])
                    Decissiontree[node][i] = clValue[0]
        else:
            if(rootNode!=node):
                print("   |" ,node, " = ",i,)
                level=level+1
                Decissiontree[node][i] = node
            else:
                print(node, '=', i)
                Decissiontree[node][i] = node
                level = level + 1

            Decissiontree[node][i]=BuildInfoGainDecissionTree(entroClass, subTable,max_depth, iter)
    return  Decissiontree
            #print("checkooooo")



def predictTheTestData(decissionTree,testData):
    predict=0
    for i in decissionTree.keys():

        val = testData[i]
        decissionTree = decissionTree[i][val]
        predict = 0

        if type(decissionTree) is dict:
            predict = predictTheTestData(decissionTree,testData )
        else:
            predict = decissionTree
            break

    return predict



def DecissionTreeInfoGainMain():

    df=pd.read_excel("C:\\Users\\Sourav Biswas\\Desktop\\trainDataPart2.xlsx","Sheet1")
    #df = pd.DataFrame(data,columns=['price','maintanance','capacity','airbag','profitable'])

    print(df)
    [cnpossi,cnnege]=countPossitiveNegativeClassAt(df.Label)
    entroClass=entrophyClass(cnpossi,cnnege,df.Label)
    print(entroClass)
    key=df.keys()
    iter=0
    findRootNode(entroClass, df)
    print("......Information Gain Decission Tree Printing............")
    m_depth = input("Give the max depth value: ")
    max_depth = int(m_depth)
    decissionTree=BuildInfoGainDecissionTree(entroClass,df,max_depth,iter)
    #pprint.pprint(decissionTree)
    #trainData Predict-----------------------------

    trainData = pd.read_excel("C:\\Users\\Sourav Biswas\\Desktop\\trainDataPart2.xlsx", "Sheet1")
    trainData1 = trainData.iloc[:, :-1]
    print(".............Test Data prediction using Entrophy Gain Decission Tree...........")
    trainCorrect = 0

    for i in range(0, len(trainData)):
        trainRow = trainData1.iloc[i]
        # testRow=testRow.iloc[:,-1]
        ori = trainData.iloc[i][3566]
        predict = predictTheTestData(decissionTree, trainRow)
        if (predict == ori):
            trainCorrect = trainCorrect + 1
        print("predict for :: ", trainRow, " is :: ", predict)
    testAccuracy = trainCorrect/ len(trainData)
    print("Test Accuracy using Information Gain:==", testAccuracy)


    #test data Predict-------------------------
    testData=pd.read_excel("C:\\Users\\Sourav Biswas\\Desktop\\TestDataPart2.xlsx","Sheet1")
    testData1=testData.iloc[:, :-1]
    print(".............Test Data prediction using Entrophy Gain Decission Tree...........")
    testCorrect=0

    for i in range (0,len(testData)):
        testRow=testData1.iloc[i]
        #testRow=testRow.iloc[:,-1]
        ori=testData.iloc[i][3566]
        predict=predictTheTestData(decissionTree,testRow)
        if(predict==ori):
            testCorrect=testCorrect+1
        print("predict for :: ",testRow," is profitable :: ",predict)
    testAccuracy=testCorrect/len(testData)
    print("Test Accuracy using Information Gain:==",testAccuracy)


def DecissionTreeSklearn():
    testData = pd.read_excel("C:\\Users\\Sourav Biswas\\Desktop\\TestDataPart2.xlsx", "Sheet1")
    trainData = pd.read_excel("C:\\Users\\Sourav Biswas\\Desktop\\trainDataPart2.xlsx", "Sheet1")
    dataInput=trainData.drop('Label',axis='columns')
    dataInput_test = testData.drop('Label', axis='columns')
    testtarget=testData['Label']
    targetData=trainData['Label']
   # print(dataInput)
    print(targetData)






    dtmodel_infoGain = tree.DecisionTreeClassifier(criterion='entropy')

    print("fit value----", dtmodel_infoGain.fit(dataInput, targetData))
    print("model score====", dtmodel_infoGain.score(dataInput, targetData))
    #print("kkkkkkkkkkkkkkk",dtmodel_infoGain.tree_.init_error[1])
    test_predict=dtmodel_infoGain.predict(dataInput_test)
    test_accuracy=accuracy_score(test_predict, testtarget)
    print("Test Accuracy Score using Sklearn===",test_accuracy)




DecissionTreeInfoGainMain()
DecissionTreeSklearn()
plt.xlabel('Max_depth')
plt.ylabel('Training_accuracy and test_accuracy')
plt.title("Graph for training accuracy and Testing Accuracy")
trainAccuracy=[87.84,92.17,95.94,98.56,100,100]
testAccuracy=[82.17,82.07,81.47,82.60,81.20,80.88]
x=[5,7,10,15,20,30]
plt.plot(x,trainAccuracy, color='blue',marker='*',label='Train_accuracy curve')
plt.plot(x,testAccuracy, color='red',marker='.',label='Test_accuracy curve')
plt.legend()
plt.show()