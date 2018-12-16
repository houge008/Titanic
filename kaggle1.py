# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 10:38:56 2018
泰坦尼克号生还情况分析：船舱等级、性别、年龄、登船口岸
是否有cabin等条件与生还的关系分析
@author: Hou dongjie
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

