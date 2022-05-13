
import pandas as pd
import xlrd
import numpy as np

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
    etclass=-((cnpossi/length)* (np.log2(cnpossi/length)))+(-((cnnege/length)*(np.log2(cnnege/length))))
    return etclass
def giniofClass(cnpossi,cnnege,recv):
    length=len(recv)
    gnclass=1-((cnpossi/length)*(cnpossi/length))+(-((cnnege/length)*(cnnege/length)))
    return gnclass



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


def UniqueValueofTarget(df,Class):

    Vtarget=df[Class].unique()
    return  Vtarget


def gini_each_attribute(df,att):
  Class = df.keys()[-1]




  Vtarget =UniqueValueofTarget(df,Class)
  #print(Vtarget)

  Vattribute = df[att].unique()
  #UniqueValueofAttribute(Vattribute)
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
               entropy = entropy+(-(frac)*(frac))
      frac2 = countat1/float(len(df))
      ginifi = 1 +entropy
      entropywt += frac2*ginifi

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
def findBestSplitGini(df):
    GiniIndex= []
    for key in df.keys()[:-1]:
        entropywt = gini_each_attribute(df, key)
        GiniIndex.append(entropywt)
    #print("entrophywt== ", InfoGain)
    return df.keys()[:-1][np.argmin(GiniIndex)]
def findRootNode(entroClass,df):
    InfoGain = []
    for key in df.keys()[:-1]:
        entropywt = entropy_each_attribute(df, key)
        InfoGain.append(entroClass - entropywt)
    # return df.keys()[:-1][np.argmax(IG)]
    print(InfoGain)

    print("InfoGain Of root node ======== ", np.max(InfoGain))
    print("And the selected root node::  ",df.keys()[:-1][np.argmax(InfoGain)])

def findGiniRootNode(giniClass, df):
        giniIndex = []
        for key in df.keys()[:-1]:
            entropywt = gini_each_attribute(df, key)
            giniIndex.append(entropywt)
        # return df.keys()[:-1][np.argmax(IG)]
        print(giniIndex)

        print("GiniIndex Of root node ======== ", np.min(giniIndex))
        print("And the selected root node::  ", df.keys()[:-1][np.argmin(giniIndex)])

    #return df.keys()[:-1][np.argmax(InfoGain)]
def print_node(n):
    print("Current node of the tree::::::::::",n)
def BuildInfoGainDecissionTree(entroClass,df,iter,Decissiontree=None):
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

        subTable= df[df[node] == i].reset_index(drop=True)


        clValue, counts = np.unique(subTable['profitable'], return_counts=True)

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

            Decissiontree[node][i]=BuildInfoGainDecissionTree(entroClass, subTable, iter)
    return  Decissiontree
            #print("checkooooo")

def BuildGiniIndexDecissionTree(entroClass,df,iter,Decissiontree=None):
    Class = df.keys()[-1]
    global level1
    global rootNode1
    attriVal=0
    #root=0
    #node=0
    #print("llllllllll")
    """if(iter==0):
        root=findRootNode(entroClass, df)
        print("root== ",root)
        attriVal = np.unique(df[root])
        print(attriVal)"""


    node = findBestSplitGini(df)
    #print('node==',node)
    if(iter == 0):
        rootNode1 = node
        #print("mmm",root)
        iter=1
    attriVal=np.unique(df[node])
    if Decissiontree is None:
        Decissiontree = {}


        Decissiontree[node] = {}
    for i in attriVal:
        #print(node ,'=',i)

        subTable= df[df[node] == i].reset_index(drop=True)


        clValue, counts = np.unique(subTable['profitable'], return_counts=True)

        if(len(counts)==1):
            if(rootNode1==node):
                print(node, "  ", i, " : ", clValue[0])
                Decissiontree[node][i]=clValue[0]

            else:
                if(level1>0):
                    for j in range(0,level1):
                        print("   ",end=" ")
                        if(j+1==level1):
                            print( "   |" ,node, " = ",i," : ",clValue[0])
                            Decissiontree[node][i] = clValue[0]
                else:
                    print("   |", node, " = ", i, " : ", clValue[0])
                    Decissiontree[node][i] = clValue[0]
        else:
            if(rootNode1!=node):
                print("   |" ,node, " = ",i)
                level1 = level1 + 1
                Decissiontree[node][i] = node
            else:
                print(node, '=', i)
                Decissiontree[node][i] = node

            Decissiontree[node][i]=BuildGiniIndexDecissionTree(entroClass, subTable, iter)
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

def predictTheTestDataGini(decissionTree,testData):
    prediction1=0
    for nodes in decissionTree.keys():

        val = testData[nodes]
        decissionTree = decissionTree[nodes][val]
        prediction1 = 0

        if type(decissionTree) is dict:
            prediction1 = predictTheTestData(decissionTree,testData )
        else:
            prediction1 = decissionTree
            break;

    return prediction1

def DecissionTreeInfoGainMain():

    df=pd.read_excel("C:\\Users\\Sourav Biswas\\Desktop\\DataPart1.xlsx","Traning")
    #df = pd.DataFrame(data,columns=['price','maintanance','capacity','airbag','profitable'])

    print(df)
    [cnpossi,cnnege]=countPossitiveNegativeClassAt(df.profitable)
    entroClass=entrophyClass(cnpossi,cnnege,df.profitable)
    print(entroClass)
    key=df.keys()
    iter=0
    findRootNode(entroClass, df)
    print("......Information Gain Decission Tree Printing............")
    decissionTree=BuildInfoGainDecissionTree(entroClass,df,iter)
    #pprint.pprint(decissionTree)
    testData=pd.read_excel("C:\\Users\\Sourav Biswas\\Desktop\\DataPart1.xlsx","Test")
    testData1=testData.iloc[:, :-1]
    print(".............Test Data prediction using Entrophy Gain Decission Tree...........")
    testCorrect=0
    for i in range (0,len(testData)):
        testRow=testData1.iloc[i]
        #testRow=testRow.iloc[:,-1]
        ori=testData.iloc[i][4]
        predict=predictTheTestData(decissionTree,testRow)
        if(predict==ori):
            testCorrect=testCorrect+1
        print("predict for :: ",testRow," is profitable :: ",predict)
    testAccuracy=(testCorrect/len(testData))*100
    print("Test Accuracy using Information Gain:==",testAccuracy,"%")


def DecissionTreeGiniMain():

    df=pd.read_excel("C:\\Users\\Sourav Biswas\\Desktop\\DataPart1.xlsx","Traning")
    #df = pd.DataFrame(data,columns=['price','maintanance','capacity','airbag','profitable'])

    print(df)
    [cnpossi,cnnege]=countPossitiveNegativeClassAt(df.profitable)
    giniClass=giniofClass(cnpossi,cnnege,df.profitable)
    #print(entroClass)
    key=df.keys()
    iter=0
    findGiniRootNode(giniClass, df)
    print("......Gini Index Decission Tree Printing............")
    decissionTree=BuildGiniIndexDecissionTree(giniClass,df,iter)
    #pprint.pprint(decissionTree)
    testData=pd.read_excel("C:\\Users\\Sourav Biswas\\Desktop\\DataPart1.xlsx","Test")
    testData1=testData.iloc[:, :-1]
    print(".............Test Data prediction using Gini Index Decission Tree...........")
    testCorrect=0
    for i in range (0,len(testData)):
        testRow=testData1.iloc[i]
        #testRow=testRow.iloc[:,-1]
        ori=testData.iloc[i][4]
        #print("oriii",ori)
        predict=predictTheTestDataGini(decissionTree,testRow)
        if(predict==ori):
            testCorrect=testCorrect+1;
        print("predict for :: ",testRow," is profitable :: ",predict)
    TestAccuracy=(testCorrect/len(testData))*100
    print("Test Accuracy using Gini===", TestAccuracy,"%")

def DecissionTreeSklearn():
    df = pd.read_excel("C:\\Users\\Sourav Biswas\\Desktop\\DataPart1.xlsx", "Traning")
    df_test = pd.read_excel("C:\\Users\\Sourav Biswas\\Desktop\\DataPart1.xlsx", "Test")
    dataInput=df.drop('profitable',axis='columns')
    dataInput_test = df_test.drop('profitable', axis='columns')
    testtarget=df_test['profitable']
    targetData=df['profitable']
   # print(dataInput)
    print(targetData)
    le_price = LabelEncoder()
    le_maintenance = LabelEncoder()
    le_capacity = LabelEncoder()
    le_airbug=LabelEncoder()
    le_profitable = LabelEncoder()
    dataInput['price_n'] = le_price.fit_transform(dataInput['price'])
    dataInput['mentanance_n'] = le_maintenance.fit_transform(dataInput['maintenance'])
    dataInput['capacity_n'] = le_capacity.fit_transform(dataInput['capacity'])
    dataInput['airbug_n'] = le_airbug.fit_transform(dataInput['airbag'])
    targetData_n=le_profitable.fit_transform(targetData)

    dataInput_test['price_n'] = le_price.fit_transform(dataInput_test['price'])
    dataInput_test['mentanance_n'] = le_maintenance.fit_transform(dataInput_test['maintenance'])
    dataInput_test['capacity_n'] = le_capacity.fit_transform(dataInput_test['capacity'])
    dataInput_test['airbug_n'] = le_airbug.fit_transform(dataInput_test['airbag'])
    testtarget_n=[1,1]
    print("test target",testtarget_n)
    print(dataInput)
    print("targetData===",targetData_n)

    dataInputs_n = dataInput.drop(['price', 'maintenance', 'capacity','airbag'], axis='columns')
    print(dataInputs_n)
    dataInputs_test_n = dataInput_test.drop(['price', 'maintenance', 'capacity', 'airbag'], axis='columns')
    print(dataInputs_test_n)

    dtmodel_gini=tree.DecisionTreeClassifier()
    dtmodel_infoGain = tree.DecisionTreeClassifier(criterion='entropy')
    print("fit value----",dtmodel_gini.fit(dataInputs_n,targetData_n))
    print("model score====",dtmodel_gini.score(dataInputs_n,targetData_n))
    print("fit value----", dtmodel_infoGain.fit(dataInputs_n, targetData_n))
    print("model score====", dtmodel_infoGain.score(dataInputs_n, targetData_n))
    #print("kkkkkkkkkkkkkkk",dtmodel_infoGain.tree_.init_error[1])
    test_pred=[]
    if(dtmodel_gini.predict([[2, 0, 2, 0]])==1):
        test_pred.append(1)
    if(dtmodel_gini.predict([[1, 1, 1, 0]])==1):
        test_pred.append(1)
    print(test_pred)
    test_accuracy=accuracy_score(test_pred, testtarget_n)
    print("Test accuracy Score using Sklearn===",test_accuracy*100,"%")
    print(getmembers(dtmodel_infoGain.tree_))
    print(getmembers(dtmodel_gini.tree_))

    #print(graph)
DecissionTreeInfoGainMain()
DecissionTreeGiniMain()
DecissionTreeSklearn()
print("Lable  generated by implemented model: ", level)