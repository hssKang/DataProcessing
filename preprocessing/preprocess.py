import pandas as pd
import numpy as np
import sys
from sklearn import preprocessing

# Feature Reduction
def feature_reduction(data):
    threshold = len(data) * 0.5
    data.dropna(thresh=threshold, axis=1, inplace=True)             # drop if more than 50% null occurs
    
    return data

# Drop outlier with whisker
def outlier(data):
    features=data.select_dtypes(include=['int']).columns.tolist()+data.select_dtypes(include=['float']).columns.tolist()
    for feature in features:
        # Q1 Q3 IQR obtained for range data
        Q1 = data[feature].quantile(0.25)
        Q3 = data[feature].quantile(0.75)
        IQR = Q3 - Q1

        # Calculate the scope of whisker
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Consider data outside whisker as outlier and remove it
        data = data[(data[feature] >= lower_bound) & (data[feature] <= upper_bound)]
    
    return data

def find_value_over_80_percent(data):
    value_over_80_percent = []
    for col in data.columns:
        # 열의 각 값에 대한 빈도수를 계산 (정규화하여 비율로 표현)
        value_counts = data[col].value_counts(normalize=True)
        # 80% 이상의 값을 찾음
        major_values = value_counts[value_counts >= 0.7]
        if not major_values.empty:
            # 80% 이상의 값을 리스트에 추가
            value_over_80_percent.append(major_values.index[0])
        else:
            # 해당하는 값이 없으면 None 추가
            value_over_80_percent.append(None)
    return value_over_80_percent

# Missing data fill & drop
def missing_data(data):
    selected_data = find_value_over_80_percent(data)
    for i in range(len(data.columns)):
        if selected_data[i]!=None:
            data[data.columns[i]]=data[data.columns[i]].fillna(selected_data[i])
    data.dropna(axis=0, inplace=True)                               # Drop other na
    
    return data

# Robust Normalization
def data_normalization(data, model="standard"):
    if model=="minmax":
        m=preprocessing.MinMaxScaler()
    elif model=="robust":
        m=preprocessing.RobustScaler()
    else:
        m=preprocessing.StandardScaler()
        
    features=data.select_dtypes(include=['int']).columns.tolist()        
    for i in features:
        data[i]=m.fit_transform(np.array(data[i]).reshape(-1,1)).squeeze()
        
    features=data.select_dtypes(include=['float']).columns.tolist()        
    for i in features:
        data[i]=m.fit_transform(np.array(data[i]).reshape(-1,1)).squeeze()
    
    return data

# For easy handling, replace numerical data
def to_numeric_data(data):
    # Object class to 0, 1, 2...
    for feature in data.select_dtypes(include=['object']).columns.tolist():
        for i in range(len(data[feature].value_counts().index)):
            data[feature]=data[feature].replace(data[feature].value_counts().index[i], i)
    
    return data

def main(file, model):
    data=pd.read_csv(file)
    # DataFrame setting
    pd.set_option('future.no_silent_downcasting', True)
    pd.options.mode.copy_on_write = True
    
    ori_len=len(data)
    
    data=feature_reduction(data)
    data=outlier(data)
    data=missing_data(data)
    data=data_normalization(data,model)
    
    data=to_numeric_data(data)
    
    data.to_csv("result.csv",index=False)
    new_len=len(data)
    
    print("Save result : \n\noriginal data = ",ori_len,"\nnew data = ",new_len,"\ntotal null = ", data.isnull().sum().sum())



if __name__ == '__main__':
    file = sys.argv[1]
    model=sys.argv[2]
    main(file, model)