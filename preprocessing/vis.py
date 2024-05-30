import pandas as pd
import sys
import matplotlib.pyplot as plt

def data_exploration_vis(file):
    data=pd.read_csv(file)
    features_obj=data.select_dtypes(include=['object']).columns.tolist()
    features_float=data.select_dtypes(include=['float']).columns.tolist()
    features_int=data.select_dtypes(include=['int']).columns.tolist()
    

    for i in range(len(features_obj)):
        plt.figure(i+1)
        plt.title(features_obj[i])
        plt.pie(data[features_obj[i]].value_counts(), labels=data[features_obj[i]].value_counts().index)
        
    plt.show()
    
    for i in range(len(features_float)):
        plt.figure(i+1)
        plt.title(features_float[i])
        plt.boxplot(data[features_float[i]].dropna())

    plt.show()
    
    for i in range(len(features_int)):
        plt.figure(i+1)
        plt.title(features_int[i])
        plt.hist(data[features_int[i]].dropna())

    plt.show()
    
if __name__ == '__main__':
    file = sys.argv[1]
    data_exploration_vis(file)