# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 10:38:56 2018
泰坦尼克号生还情况分析：船舱等级、性别、年龄、登船口岸
是否有cabin等条件与生还的关系分析
@author: Hou dongjie
"""
import pandas as pd #数据分析
import numpy as np #科学计算
import matplotlib.pyplot as plt #可视化
from pandas import Series,DataFrame
"""
0 step:数据读取
"""
data_train = pd.read_csv('train.csv')
print(data_train) #数据读取
#d = data_train.describe() #统计描述
data_train.info() #a rough information

"""
1 step:数据可视化,乘客各个属性分布
"""
fig = plt.figure() #create a new figure canvas
fig.set(alpha=0.4) #设定figue颜色参数
plt.subplot2grid((1,2),(0,0)) #在一张figure里分画几个小figure
data_train.Survived.value_counts().plot(kind='bar',width=0.35)
#plt.bar((0,1),data_train.Survived.value_counts(),width=0.35)
plt.title(u'获救情况(1-获救)',fontproperties='SimHei') #标题
plt.ylabel(r'人数',fontproperties='SimHei')

plt.subplot2grid((1,2),(0,1))
data_train.Pclass.value_counts().plot(kind="bar")
#plt.ylabel(u"人数",fontproperties='SimHei')
plt.title(u"乘客等级分布",fontproperties='SimHei')
plt.savefig("fig1.tiff",dpi=300)

fig = plt.figure()
#plt.plot()
plt.scatter(data_train.Survived,data_train.Age)
plt.ylabel(r'年龄',fontproperties='SimHei')
plt.grid(b=True, which='major', axis='y') 
plt.title(u"按年龄看获救分布 (1为获救)",fontproperties='SimHei')
plt.savefig('fig2.tiff',dpi=300)

fig = plt.figure()
fig.set(alpha=0.6)
plt.subplot2grid((1,3),(0,0), colspan=2) #占用两个位置
data_train.Age[data_train.Pclass == 1].plot(kind='kde')
data_train.Age[data_train.Pclass == 2].plot(kind='kde')
data_train.Age[data_train.Pclass == 3].plot(kind='kde') 
plt.xlabel(u"年龄",fontproperties='SimHei') 
plt.ylabel(u"密度",fontproperties='SimHei') 
plt.title(u"各等级的乘客年龄分布",fontproperties='SimHei') 
plt.legend((u'头等舱', u'2等舱',u'3等舱'),loc='best',prop={'family':'SimHei','size':15}) # sets our legend for our graph. 

plt.subplot2grid((1,3),(0,2))
data_train.Embarked.value_counts().plot(kind='bar')
plt.title(u"各登船口岸上船人数",fontproperties='SimHei')
plt.ylabel(u"人数",fontproperties='SimHei') 
plt.savefig('fig3.tiff',dpi=300)
plt.savefig('Tanic',dpi=300) 
plt.show()

"""
2 step:属性与获救结果的相关统计
"""
#乘客舱位等级vs获救情况
fig = plt.figure()
fig.set(alpha=0.2)

Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()

df = pd.DataFrame({u'Survived':Survived_1,u'No Survived':Survived_0}) #DataFrame 数据结构
df.plot(kind='bar',stacked=True)
plt.title(r'等级VS获救情况',fontproperties='SimHei')
plt.xlabel(u"乘客等级",fontproperties='SimHei') 
plt.ylabel(u"人数",fontproperties='SimHei') 
plt.savefig('fig4.tiff',dpi=300)
plt.show()


#性别VS获救情况
fig = plt.figure()
fig.set(alpha=0.2)

Survived_m = data_train.Survived[data_train.Sex == 'male'].value_counts()
Survived_f = data_train.Survived[data_train.Sex == 'female'].value_counts()
df = pd.DataFrame({r'male':Survived_m,r'female':Survived_f})
df.plot(kind='bar',stacked=True)
plt.title(u"性别VS获救情况",fontproperties='SimHei')
plt.xlabel(u"性别",fontproperties='SimHei') 
plt.ylabel(u"人数",fontproperties='SimHei')
plt.savefig('fig5.tiff',dpi=300)
plt.show()


#舱位级别下，性别获救情况
"""

"""
fig=plt.figure()
fig.set(alpha=0.65) # 设置图像透明度，无所谓
plt.title(u"舱等级和性别VS获救情况",fontproperties='SimHei')

ax1=fig.add_subplot(121)
data_train.Survived[data_train.Sex == 'female'][data_train.Pclass != 3].value_counts().plot(kind='bar', label="female highclass", color='#FA2479')
ax1.set_xticklabels([u"获救", u"未获救"], rotation=0,fontproperties='SimHei')
ax1.legend([u"女性/高级舱"], loc='best',prop={'family':'SimHei','size':15})

ax2=fig.add_subplot(122, sharey=ax1)
data_train.Survived[data_train.Sex == 'female'][data_train.Pclass == 3].value_counts().plot(kind='bar', label='female, low class', color='pink')
ax2.set_xticklabels([u"未获救", u"获救"],rotation=0,fontproperties='SimHei')
plt.legend([u"女性/低级舱"], loc='best',prop={'family':'SimHei','size':15})
plt.savefig('fig6.tiff',dpi=300)

fig=plt.figure()
fig.set(alpha=0.65) # 设置图像透明度，无所谓
plt.title(u"舱等级和性别VS获救情况",fontproperties='SimHei')

ax3=fig.add_subplot(121, sharey=ax1)
data_train.Survived[data_train.Sex == 'male'][data_train.Pclass != 3].value_counts().plot(kind='bar', label='male, high class',color='lightblue')
ax3.set_xticklabels([u"未获救", u"获救"],rotation = 0,fontproperties='SimHei')
plt.legend([u"男性/高级舱"], loc='best',prop={'family':'SimHei','size':15})

ax4=fig.add_subplot(122, sharey=ax1)
data_train.Survived[data_train.Sex == 'male'][data_train.Pclass == 3].value_counts().plot(kind='bar', label='male low class', color='steelblue')
ax4.set_xticklabels([u"未获救", u"获救"],rotation = 0,fontproperties='SimHei')
plt.legend([u"男性/低级舱"], loc='best',prop={'family':'SimHei','size':15})
plt.savefig('fig7.tiff',dpi=300)

#登船港口VS获救情况
fig = plt.figure()
fig.set(alpha=0.2)  # 设定图表颜色alpha参数

Survived_0 = data_train.Embarked[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Embarked[data_train.Survived == 1].value_counts()
df=pd.DataFrame({u'Survived':Survived_1, u'No Survived':Survived_0})
df.plot(kind='bar', stacked=True)
plt.title(u"登船港口VS获救情况",fontproperties='SimHei')
plt.xlabel(u"登船港口",fontproperties='SimHei') 
plt.ylabel(u"人数",fontproperties='SimHei') 
plt.savefig('fig8.tiff',dpi=300)
plt.show()


# 是否有CabinVS获救情况
fig = plt.figure()
fig.set(alpha=0.2)  # 设定图表颜色alpha参数

Survived_cabin = data_train.Survived[pd.notnull(data_train.Cabin)].value_counts()
Survived_nocabin = data_train.Survived[pd.isnull(data_train.Cabin)].value_counts()
df=pd.DataFrame({u'yes':Survived_cabin, u'no':Survived_nocabin}).transpose()
df.plot(kind='bar', stacked=True)
plt.title(u"按Cabin有无看获救情况",fontproperties='SimHei')
plt.xlabel(u"Cabin有无",fontproperties='SimHei') 
plt.ylabel(u"人数",fontproperties='SimHei')
plt.savefig('fig9.tiff',dpi=300)
plt.show()


"""
3 step:特征工程，数据预处理
"""
#使用scikit-learn中的RandomForest拟合补充确实的年龄数据
from sklearn.ensemble import RandomForestRegressor
def set_missing_ages(df):
    #将已经有的数值特征选取出来，放进RandomForestRegressor中
    age_df = df[['Age','Fare','Parch','SibSp','Pclass']]
    #乘客分成已知年龄和位置年龄两部分
    known_age = age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()
    
    # make y 目标年龄
    y = known_age[:,0]
    # make X 特征属性值
    X = known_age[:,1:]
    #建立model,then fit
    rfr = RandomForestRegressor(random_state=0,n_estimators=2000,n_jobs=-1)
    rfr.fit(X,y)
    
    #用训练的模型进行位置年龄预测
    predictedAges = rfr.predict(unknown_age[:,1::])
    #用预测的结果补充确实年龄数据
    df.loc[(df.Age.isnull()),'Age'] = predictedAges
    return df,rfr
def set_Cabin_type(df):
    df.loc[(df.Cabin.notnull()),'Cabin'] = 'yes'
    df.loc[(df.Cabin.isnull()),'Cabin'] = 'No'
    return df
data_train,rfr = set_missing_ages(data_train)
data_train = set_Cabin_type(data_train)
print(data_train)

"""
逻辑回归建模时，需要输入的特征为0、1二元数值，通常需要对类目类型二值化、因子化
以Cabin为例，原本一个属性维度，因为其取值为【yes,no】，而将其平展开为‘Cabin_yes'
'Cabin_no'两个属性，原本cabin为yes,在Cabin_yes下取值为1；原本在Cabin下为0，则在
Cabin_no下为1。其他情况为0.使用pandas的get_dummies实现，拼接在原来的data_train上
"""
dummies_Cabin = pd.get_dummies(data_train['Cabin'],prefix='Cabin')
dummies_Embarked = pd.get_dummies(data_train['Embarked'],prefix='Embarked')
dummies_Sex = pd.get_dummies(data_train['Sex'], prefix= 'Sex')
dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix= 'Pclass')

df = pd.concat([data_train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
print(df)

"""
细看看Age和Fare数值浮动大，将对逻辑回归或者梯度下降算法造成速度慢或者不收敛，
要通过sklearn的preprocessing模块进行scaling,也就是特征化到[-1,1]之间,置于表后
"""
import sklearn.preprocessing as preprocessing
scaler = preprocessing.StandardScaler()
age_scale_param = scaler.fit(df['Age'].values.reshape(-1,1))
df['Age_scaled'] = scaler.fit_transform(df['Age'].values.reshape(-1,1),age_scale_param)
fare_scale_param = scaler.fit(df['Fare'].values.reshape(-1,1))
df['Fare_scaled'] = scaler.fit_transform(df['Fare'].values.reshape(-1,1), fare_scale_param)
print(df)

"""
4. 万事俱备，开始建模
我们把需要的feature字段取出来，转成numpy格式，
使用scikit-learn中的LogisticRegression建模
""" 
from sklearn import linear_model
#用正则取取出我们要的属性值
train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pcalss_.*')
train_np = train_df.as_matrix()

#y即第0列：Survival结果
y = train_np[:,0]
#X即第一列及其以后：其特征属性值
X = train_np[:,1:]

#fit到logisticRegression之中
clf = linear_model.LogisticRegression(C=1.0,penalty='l1',tol=1e-6)
clf.fit(X,y)
print(clf)  #训练出一个模型

"""
对test_data做预处理
"""
data_test = pd.read_csv('test.csv')
data_test.loc[ (data_test.Fare.isnull()), 'Fare' ] = 0
# 接着我们对test_data做和train_data中一致的特征变换
# 首先用同样的RandomForestRegressor模型填上丢失的年龄
tmp_df = data_test[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]
null_age = tmp_df[data_test.Age.isnull()].as_matrix()
# 根据特征属性X预测年龄并补上
X = null_age[:, 1:]
predictedAges = rfr.predict(X)
data_test.loc[ (data_test.Age.isnull()), 'Age' ] = predictedAges

data_test = set_Cabin_type(data_test)
dummies_Cabin = pd.get_dummies(data_test['Cabin'], prefix= 'Cabin')
dummies_Embarked = pd.get_dummies(data_test['Embarked'], prefix= 'Embarked')
dummies_Sex = pd.get_dummies(data_test['Sex'], prefix= 'Sex')
dummies_Pclass = pd.get_dummies(data_test['Pclass'], prefix= 'Pclass')


df_test = pd.concat([data_test, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
#df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)


age_scale_param1 = scaler.fit(df_test['Age'].values.reshape(-1,1))
df_test['Age_scaled'] = scaler.fit_transform(df_test['Age'].values.reshape(-1,1), age_scale_param1)
fare_scale_param1 = scaler.fit(df_test['Fare'].values.reshape(-1,1))
df_test['Fare_scaled'] = scaler.fit_transform(df_test['Fare'].values.reshape(-1,1), fare_scale_param1)

print("df_test:",df_test)

"""
做初步预测，make a submission
"""
test1 = df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
test_np = test1.as_matrix()
X1 = test_np[:,0:] ###??????????????特征列14》》》》11
predictions = clf.predict(X1)
result = pd.DataFrame({'PassengerId':data_test['PassengerId'].as_matrix(), 'Survived':predictions.astype(np.int32)})
result.to_csv("logistic_regression_predictions.csv", index=False)
              
pd1=pd.read_csv("logistic_regression_predictions.csv")
print(pd1)




