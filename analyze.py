import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy import interpolate
import apyori


rcParams['axes.unicode_minus'] = False

'''
    brief 将读取csv得到的dataframe合并
    param df1 左dataframe
    param df2 右dataframe
    return 合并后的dataframe
'''
def mergeDataFrame(df1 , df2) : 
    df1 = pd.merge(df1 , df2 , left_index = True , right_index = True)
    return df1

'''
    brief 做Apriori分析，得到关联规则
    param data 准备好的交易资料
    param support 最小支持度
    param cinfidence 最小置信度
    return 关联规则list
'''
def getRule(data , support , confidence) : 
    return list(apyori.apriori(transactions = data , min_support = support , min_confidence = confidence))

'''
    brief 在一堆关联规则中获取跟股市有关的规则
    param rule 所有关联规则list
    return 跟股市有关的关联规则list
'''

def getStockRule(rule) : 
    stockRule = []
    for i in rule : 
        for j in i[2] : 
            '''
            if "items_add=frozenset({'Stock_V1'})" in str(j) or "items_add=frozenset({'Stock_V2'})" in str(j) or "items_add=frozenset({'Stock_V3'})"in str(j) or "items_add=frozenset({'Stock_V4'})" in str(j) : 
                rules = str(j) + " , Support = " + str(i[1])
                stockRule.append(rules)
            '''
            if "items_add=frozenset({'Stock_V3'})"in str(j) or "items_add=frozenset({'Stock_V4'})" in str(j) : 
                rules = str(j) + " , Support = " + str(i[1])
                stockRule.append(rules)
    return stockRule


'''
    brief 打印出关联规则，不过因为规则太多，我只打印每个部分的规则长度，可以把注解去掉来看看规则
    param stockRule 跟股市有关的规则list
    parm modelName model的名称
'''
def printStockRule(stockRule , modelName) : 
    #counter = 0
    print("model{} has {} rules".format(modelName , len(stockRule)))
    '''
    for i in stockRule : 
        print("rule" + str(counter) + " : ")
        print(i)
        counter += 1
    '''

'''
    brief 将资料转换成Apriori可以接受的形式
    param data 资料都接好的dataframe（每一column都直接是一笔交易资料，滞后的处理在getTrainData中处理掉了）
    return 交易资料list
'''
def transformData(data) : 
    trainData = []
    for i in range(len(data)) : 
        temp = []
        for j in range(data.columns.size) : 
            temp.append(data.iloc[i , j])
        trainData.append(temp)
    return trainData

'''
    brief 从初始的同比增长率dataframe中截取各个年份段需要的部分，并做成新的dataframe
    param ratio_df 初始的同比增长dataframe
    param shift 滞后的月数int
    return 三个年份段的dataframe
'''
def getTrainData(ratio_df , shift) : 

    stockLst = ratio_df['Clsidx']

    trainData1 = pd.DataFrame(columns = ratio_df.columns)
    trainData2 = pd.DataFrame(columns = ratio_df.columns)
    trainData3 = pd.DataFrame(columns = ratio_df.columns)

    for i in range(96) : 
        trainData1.loc[i] = ratio_df.loc[i]
    for i in range(60 , 156) : 
        trainData2.loc[i - 60] = ratio_df.loc[i]
    for i in range(120 , 216 - shift) : 
        trainData3.loc[i - 120] = ratio_df.loc[i]
       
    for i in range(96) : 
        trainData1['Clsidx'].loc[i] = stockLst.loc[i + shift]
    for i in range(60 , 156) : 
        trainData2['Clsidx'].loc[i - 60] = stockLst.loc[i + shift]
    for i in range(120 , 216 - shift) : 
        trainData3['Clsidx'].loc[i - 120] = stockLst.loc[i + shift]
    
    return trainData1 , trainData2 , trainData3
    
def takeSupport(rule) : 
    return float(str(rule).split("=")[-1])


base_dir = "data/"

#df1 : 居民消费价格指数(上年同期＝100)_当期
df1 = pd.read_csv(base_dir + "CPI2.csv" , encoding = "utf-8")

#df2 : 对美元汇率
df2 = pd.read_csv(base_dir + "Exchange_Rate.csv" , encoding = "gbk")

#df3 : 固定资产投资完成额
df3 = pd.read_csv(base_dir + "FAI.csv" , encoding = "utf-8")

#df4 : 广义货币(M2)余额(亿元)  银行间同业拆借利率(7天)(%)
df4 = pd.read_csv(base_dir + "M2_loan.csv" , encoding = "gbk")

#df6 : 消费者信心指数
df6 = pd.read_csv(base_dir + "CCI2.csv" , encoding = "utf-8")

#df7 : 股價指數
df7 = pd.read_csv(base_dir + "SH_Index2.csv" , encoding = "utf-8" )

#df8 : 规模以上工业增加值_当期同比实际增速
df8 = pd.read_csv(base_dir + "IAV(ratio).csv" , encoding = "utf-8" )

#df9 : 社会消费品零售总额
df9 = pd.read_csv(base_dir + "consume_product.csv" , encoding = "utf-8" )

#df10 : 国房景气指数
df10 = pd.read_csv(base_dir + "houseIndex.csv" , encoding = "utf-8" )

#df11 : 进出口差额(美元)
df11 = pd.read_csv(base_dir + "exportData.csv" , encoding = "utf-8" )

#df12 : 实际利用外资金额(美元)_外商直接投资_当期
df12 = pd.read_csv(base_dir + "FDI.csv" , encoding = "utf-8" )



df1 = df1.drop(range(228 , 231))
df2 = df2.drop(range(228 , 245))
df4 = df4.drop(range(228 , 240))
df6 = df6.drop(range(228 , 231))
df7 = df7.drop(['Indexcd'] , axis = 1)
df7 = df7.drop(range(228 , 456))
df9 = df9.drop(range(228 , 247))
df11 = df11.drop(range(228 , 243))
df12 = df12.drop(range(228 , 230))



#國防景氣指數的缺失值先向上填充
df10 = df10.fillna(method = 'bfill')



#将<固定资产投资完成额>的累进数据转换成当月数据
temp = []
count = list(range(228))
count.reverse()
for i in count : 
    if i % 12 == 0 : 
        temp.append(df3['固定资产投资完成额'][i])
    else : 
        temp.append(df3['固定资产投资完成额'][i] - df3['固定资产投资完成额'][i - 1])
temp.reverse()
df3['FAI'] = temp
df3 = df3.drop(['固定资产投资完成额'] , axis = 1)



#因为已经对数据处理好了（都是1999/01 ~ 2017/12），所以将月份的column丢掉
df1 = df1.drop(['指标'] , axis = 1)
df2 = df2.drop(['月度'] , axis = 1)
df3 = df3.drop(['指标'] , axis = 1)
df4 = df4.drop(['月度'] , axis = 1)
df6 = df6.drop(['指标'] , axis = 1)
df7 = df7.drop(['Month'] , axis = 1)
df8 = df8.drop(['指标'] , axis = 1)
df9 = df9.drop(['指标'] , axis = 1)
df10 = df10.drop(['指标'] , axis = 1)
df11 = df11.drop(['指标'] , axis = 1)
df12 = df12.drop(['指标'] , axis = 1)




#以index合并各个资料成一个dataframe
df = pd.merge(df1 , df3 , left_index = True , right_index = True)
df = mergeDataFrame(df , df4)
df = mergeDataFrame(df , df6)
df = mergeDataFrame(df , df9)
df = mergeDataFrame(df , df11)
#df = mergeDataFrame(df , df10)
#df = mergeDataFrame(df , df12)
df = mergeDataFrame(df , df7)




#计算资料的同比增长率
ratio_df = pd.DataFrame(columns = df.columns)
for i in range(12 , 228) : 
    temp = []
    for j in range(df.columns.size) : 
        temp.append((df.iloc[i , j] - df.iloc[i - 12 , j]) / df.iloc[i - 12 , j] * 100)
    ratio_df.loc[i - 12] = temp

ratio_df = mergeDataFrame(ratio_df , df8)




'''
#做各指标同比增长率的散点图
fig , axes = plt.subplots(3 , 1 , figsize = (100 , 25))

ax0 = axes[0]
ax1 = axes[1]
ax2 = axes[2]

ax0.scatter(range(216) , ratio_df['居民消费价格指数(上年同期＝100)_当期'] , c = 'r' , s = 3)
ax0.set_title("居民消费价格指数(上年同期＝100)_当期成长率")

ax1.scatter(range(216) , ratio_df['FAI'] , c = 'r' , s = 3)
ax1.set_title("FAI成長率")

ax2.scatter(range(216) , ratio_df['广义货币(M2)余额(亿元)'] , c = 'r' , s = 3)
ax2.set_title("广义货币(M2)余额(亿元)成长率")

plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)
plt.show()


fig , axes = plt.subplots(3 , 1 , figsize = (100 , 25))

ax0 = axes[0]
ax1 = axes[1]
ax2 = axes[2]

ax0.scatter(range(216) , ratio_df['银行间同业拆借利率(7天)(%)'] , c = 'r' , s = 3)
ax0.set_title("银行间同业拆借利率(7天)(%)成长率")

ax1.scatter(range(216) , ratio_df['进出口差额(美元)'] , c = 'r' , s = 3)
ax1.set_title("进出口差额(美元)成长率")

ax2.scatter(range(216) , ratio_df['消费者信心指数'] , c = 'r' , s = 3)
ax2.set_title("消费者信心指数")

plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)
plt.show()


fig , axes = plt.subplots(3 , 1 , figsize = (100 , 25))

ax0 = axes[0]
ax1 = axes[1]
ax2 = axes[2]

ax0.scatter(range(216) , ratio_df['社会消费品零售总额'] , c = 'r' , s = 3)
ax0.set_title("社会消费品零售总额成长率")

ax1.scatter(range(216) , ratio_df['规模以上工业增加值_当期同比实际增速'] , c = 'r' , s = 3)
ax1.set_title("规模以上工业增加值_当期同比实际增速")

ax2.scatter(range(216) , ratio_df['Clsidx'] , c = 'r' , s = 3)
ax2.set_title("股市成长率")

plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)
plt.show()


fig , axes = plt.subplots(2 , 1 , figsize = (100 , 25))

ax0 = axes[0]
ax1 = axes[1]

ax0.scatter(range(216) , ratio_df['国房景气指数'] , c = 'r' , s = 3)
ax0.set_title("国房景气指数成长率")

ax1.scatter(range(216) , ratio_df['实际利用外资金额(美元)_外商直接投资_当期'] , c = 'r' , s = 3)
ax1.set_title("实际利用外资金额(美元)_外商直接投资_当期成长率")

plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)
plt.show()
'''




#将成长率离散并标签化
for i in range(216) : 
    if ratio_df['居民消费价格指数(上年同期＝100)_当期'][i] < -1.28 : 
        ratio_df['居民消费价格指数(上年同期＝100)_当期'][i] = 'CPI_V1'
    elif ratio_df['居民消费价格指数(上年同期＝100)_当期'][i] >= -1.28 and ratio_df['居民消费价格指数(上年同期＝100)_当期'][i] < 0.2 : 
        ratio_df['居民消费价格指数(上年同期＝100)_当期'][i] = 'CPI_V2'
    elif ratio_df['居民消费价格指数(上年同期＝100)_当期'][i] >= 0.2 and ratio_df['居民消费价格指数(上年同期＝100)_当期'][i] < 1.93 : 
        ratio_df['居民消费价格指数(上年同期＝100)_当期'][i] = 'CPI_V3'
    else : 
        ratio_df['居民消费价格指数(上年同期＝100)_当期'][i] = 'CPI_V4'

 
    if False : 
        if ratio_df['对美元汇率'][i] < -5 : 
            ratio_df['对美元汇率'][i] = 'ER_V1'
        elif ratio_df['对美元汇率'][i] >= -5 and ratio_df['对美元汇率'][i] < 0 : 
            ratio_df['对美元汇率'][i] = 'ER_V2'
        elif ratio_df['对美元汇率'][i] >= 0 and ratio_df['对美元汇率'][i] < 5 : 
            ratio_df['对美元汇率'][i] = 'ER_V3'
        else : 
            ratio_df['对美元汇率'][i] = 'ER_V4'
  
    if ratio_df['进出口差额(美元)'][i] < -31.79 : 
        ratio_df['进出口差额(美元)'][i] = 'NX_V1'
    elif ratio_df['进出口差额(美元)'][i] >= -31.79 and ratio_df['进出口差额(美元)'][i] < 7.08 : 
        ratio_df['进出口差额(美元)'][i] = 'NX_V2'
    elif ratio_df['进出口差额(美元)'][i] >= 7.08 and ratio_df['进出口差额(美元)'][i] < 62.01 : 
        ratio_df['进出口差额(美元)'][i] = 'NX_V3'
    else : 
        ratio_df['进出口差额(美元)'][i] = 'NX_V4'


    if ratio_df['FAI'][i] < 12.57 : 
        ratio_df['FAI'][i] = 'FAI_V1'
    elif ratio_df['FAI'][i] >= 12.57 and ratio_df['FAI'][i] < 21.75 : 
        ratio_df['FAI'][i] = 'FAI_V2'
    elif ratio_df['FAI'][i] >= 21.75 and ratio_df['FAI'][i] < 28.81 : 
        ratio_df['FAI'][i] = 'FAI_V3'
    else : 
        ratio_df['FAI'][i] = 'FAI_V4'
    
    if ratio_df['广义货币(M2)余额(亿元)'][i] < 13.34 : 
        ratio_df['广义货币(M2)余额(亿元)'][i] = 'M2_V1'
    elif ratio_df['广义货币(M2)余额(亿元)'][i] >= 13.34 and ratio_df['广义货币(M2)余额(亿元)'][i] < 15.83 : 
        ratio_df['广义货币(M2)余额(亿元)'][i] = 'M2_V2'
    elif ratio_df['广义货币(M2)余额(亿元)'][i] >= 15.83 and ratio_df['广义货币(M2)余额(亿元)'][i] < 18.13 : 
        ratio_df['广义货币(M2)余额(亿元)'][i] = 'M2_V3'
    else : 
        ratio_df['广义货币(M2)余额(亿元)'][i] = 'M2_V4'

    if ratio_df['银行间同业拆借利率(7天)(%)'][i] < -18.37 : 
        ratio_df['银行间同业拆借利率(7天)(%)'][i] = 'I_V1'
    elif ratio_df['银行间同业拆借利率(7天)(%)'][i] >= -18.37 and ratio_df['银行间同业拆借利率(7天)(%)'][i] < 0.33 : 
        ratio_df['银行间同业拆借利率(7天)(%)'][i] = 'I_V2'
    elif ratio_df['银行间同业拆借利率(7天)(%)'][i] >= 0.33 and ratio_df['银行间同业拆借利率(7天)(%)'][i] < 29.33 : 
        ratio_df['银行间同业拆借利率(7天)(%)'][i] = 'I_V3'
    else : 
        ratio_df['银行间同业拆借利率(7天)(%)'][i] = 'I_V4'
    
    
    if ratio_df['消费者信心指数'][i] < -2.26 : 
        ratio_df['消费者信心指数'][i] = 'CCI_V1'
    elif ratio_df['消费者信心指数'][i] >= -2.26 and ratio_df['消费者信心指数'][i] < 0.35 : 
        ratio_df['消费者信心指数'][i] = 'CCI_V2'
    elif ratio_df['消费者信心指数'][i] >= 0.35 and ratio_df['消费者信心指数'][i] < 3.24 : 
        ratio_df['消费者信心指数'][i] = 'CCI_V3'
    else : 
        ratio_df['消费者信心指数'][i] = 'CCI_V4'


    if ratio_df['社会消费品零售总额'][i] < 10.67 : 
        ratio_df['社会消费品零售总额'][i] = 'TRS_V1'
    elif ratio_df['社会消费品零售总额'][i] >= 10.67 and ratio_df['社会消费品零售总额'][i] < 13.46 : 
        ratio_df['社会消费品零售总额'][i] = 'TRS_V2'
    elif ratio_df['社会消费品零售总额'][i] >= 13.46 and ratio_df['社会消费品零售总额'][i] < 18.09 : 
        ratio_df['社会消费品零售总额'][i] = 'TRS_V3'
    else : 
        ratio_df['社会消费品零售总额'][i] = 'TRS_V4'

    if ratio_df['规模以上工业增加值_当期同比实际增速'][i] < 8.1 : 
        ratio_df['规模以上工业增加值_当期同比实际增速'][i] = 'IAV_V1'
    elif ratio_df['规模以上工业增加值_当期同比实际增速'][i] >= 8.1 and ratio_df['规模以上工业增加值_当期同比实际增速'][i] < 12.25 : 
        ratio_df['规模以上工业增加值_当期同比实际增速'][i] = 'IAV_V2'
    elif ratio_df['规模以上工业增加值_当期同比实际增速'][i] >= 12.25 and ratio_df['规模以上工业增加值_当期同比实际增速'][i] < 16 : 
        ratio_df['规模以上工业增加值_当期同比实际增速'][i] = 'IAV_V3'
    else : 
        ratio_df['规模以上工业增加值_当期同比实际增速'][i] = 'IAV_V4'

    if ratio_df['Clsidx'][i] < -14.86 : 
        ratio_df['Clsidx'][i] = 'Stock_V1'
    elif ratio_df['Clsidx'][i] >= -14.86 and ratio_df['Clsidx'][i] < -1.12 : 
        ratio_df['Clsidx'][i] = 'Stock_V2'
    elif ratio_df['Clsidx'][i] >= -1.12 and ratio_df['Clsidx'][i] < 21.28 : 
        ratio_df['Clsidx'][i] = 'Stock_V3'
    else : 
        ratio_df['Clsidx'][i] = 'Stock_V4'


    if False : 
        if ratio_df['国房景气指数'][i] < -2.54 : 
            ratio_df['国房景气指数'][i] = 'NRE_V1'
        elif ratio_df['国房景气指数'][i] >= -2.54 and ratio_df['国房景气指数'][i] < 0.42 : 
            ratio_df['国房景气指数'][i] = 'NRE_V2'
        elif ratio_df['国房景气指数'][i] >= 0.42 and ratio_df['国房景气指数'][i] < 2.28 : 
            ratio_df['国房景气指数'][i] = 'NRE_V3'
        else : 
            ratio_df['国房景气指数'][i] = 'NRE_V4'


        if ratio_df['实际利用外资金额(美元)_外商直接投资_当期'][i] < -4.58 : 
            ratio_df['实际利用外资金额(美元)_外商直接投资_当期'][i] = 'FDI_V1'
        elif ratio_df['实际利用外资金额(美元)_外商直接投资_当期'][i] >= -4.58 and ratio_df['实际利用外资金额(美元)_外商直接投资_当期'][i] < 4.22 : 
            ratio_df['实际利用外资金额(美元)_外商直接投资_当期'][i] = 'FDI_V2'
        elif ratio_df['实际利用外资金额(美元)_外商直接投资_当期'][i] >= 4.22 and ratio_df['实际利用外资金额(美元)_外商直接投资_当期'][i] < 20.63 : 
            ratio_df['实际利用外资金额(美元)_外商直接投资_当期'][i] = 'FDI_V3'
        else : 
            ratio_df['实际利用外资金额(美元)_外商直接投资_当期'][i] = 'FDI_V4'


#依据年份段将资料丢进三个list（每个list内有s0 ~ s7的dataframe）
p1_lst = []
p2_lst = []
p3_lst = []

for i in range(12) : 
    p1_df , p2_df , p3_df = getTrainData(ratio_df , i)
    p1_lst.append(p1_df)
    p2_lst.append(p2_df)
    p3_lst.append(p3_df)



#Apriori关联分析
p1_temp = []
p2_temp = []
p3_temp = []
for i in range(12) : 
    p1_temp.append(transformData(p1_lst[i]))
    p2_temp.append(transformData(p2_lst[i]))
    p3_temp.append(transformData(p3_lst[i]))
p1_lst = p1_temp
p2_lst = p2_temp
p3_lst = p3_temp


p1_result = []
p2_result = []
p3_result = []

for i in range(12) : 
    p1_result.append(getRule(p1_lst[i] , 0.05 , 0.9))
    p2_result.append(getRule(p2_lst[i] , 0.05 , 0.9))
    p3_result.append(getRule(p3_lst[i] , 0.05 , 0.9))

p1_rule = []
p2_rule = []
p3_rule = []

#提取需要的规则 （ 取跟stock相关的规则 ）

for i in range(12) : 
    p1_rule.append(getStockRule(p1_result[i]))
    p2_rule.append(getStockRule(p2_result[i]))
    p3_rule.append(getStockRule(p3_result[i]))

#打印出规则
for i in range(12) : 
    modelName = " p1_s" + str(i)
    printStockRule(p1_rule[i] , modelName)
print()

for i in range(12) : 
    modelName = " p2_s" + str(i)
    printStockRule(p2_rule[i] , modelName)
print()

for i in range(12) : 
    modelName = " p3_s" + str(i)
    printStockRule(p3_rule[i] , modelName)

for i in p1_rule : 
    i.sort(key = takeSupport , reverse = True)

for i in p2_rule : 
    i.sort(key = takeSupport , reverse = True)

for i in p3_rule : 
    i.sort(key = takeSupport , reverse = True)


for i in p1_rule[7] : 
    print(i)
print()

for i in p2_rule[5] : 
    print(i)
print()

for i in p3_rule[0] : 
    print(i)
print()


