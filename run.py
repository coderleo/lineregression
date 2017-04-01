#coding=utf-8
import re
import numpy as np
from sklearn import preprocessing,linear_model
import matplotlib.pyplot as plt
import pandas as pd
def open_file(file_name):
    file_data = []
    with open(file_name) as f:
        for index,line in enumerate(f):
            #items = line.split('//+')
            if index ==0:
                continue
            items = re.split(' +',line.replace('\r\n','').replace('NA','0'))
            file_data.append([items[3],items[4],items[5],items[6],items[7],items[8]])
            #2017-01-23 iphone6s      27
    #print file_data
    data = np.array(file_data,dtype=np.int64)
    #print data

    return data
    #x_scaled = c(data)
    #return x_scaled
def open_file_df(file_name):
    file_data = []
    columns = []
    indexs = []
    with open(file_name) as f:
        for index,line in enumerate(f):
            item = re.split(' +',line.replace('\n','').replace('\r','').replace('<NA>','iphone6s'))
            if index ==0:
                columns = item[1:]
                continue
            indexs.append(item[0])
            file_data.append(item[1:])
            
   
    data = pd.DataFrame(file_data,columns=columns,index=indexs)
    #print data.columns
    #print type()
    #data = data.fillna(0)
    data[data=='NA'] = np.nan
    #print data
    #a = data.head(2).values.tolist()
    #print  data.head(2).loc[:,2:3]
    for index,col in enumerate(data.columns):
        if index<2:
            continue
        data.loc[:,col] = data.loc[:,col].astype(np.float)
    
    data = data.interpolate()
    data = data.dropna()
    #print data.isnull()
    #print data
    return data
def show_plt(data):
    #print data[:,0:-1]
   #estoreNum
    regr = linear_model.LinearRegression()
    
    X = []
     
    Y = []#data['sales']
    for x,y in zip(data['newsNum'],data['sales']):
        X.append([float(x)])
        Y.append(float(y))
    regr.fit(X,Y)
    print regr.score(X,Y)
    #regr.u
    
    plt.scatter(X,Y,color = 'blue')
   # print regr.predict(X)
    plt.plot(X,regr.predict(X),linewidth=4,color='red')
   
    plt.xticks(())
    plt.yticks(())
    plt.show()
   
if __name__ == '__main__':
    
    data = open_file_df('data.txt')
    #print data
    show_plt(data)
    '''
    df = data.head(5)
    print df
    import statsmodels as sm
    from statsmodels.formula.api import ols
    from statsmodels.graphics.regressionplots import plot_regress_exog
    sales_model = ols('sales~bdindexNum',data=df).fit()
    s = sales_model.summary()
    print s 
    '''