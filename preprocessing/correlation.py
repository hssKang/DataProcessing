import pandas as pd
import sys
import matplotlib.pyplot as plt
import seaborn as sns

def correlation(file):
    data=pd.read_csv(file)
    pd.set_option('future.no_silent_downcasting', True)
    pd.options.mode.copy_on_write = True
    # Import dataframe to create correlation map (because the existing dataframe has not been preprocessed)
    data=to_numeric_data(data)
    # Visualization
    sns.heatmap(data.corr(), annot=True, cmap='YlOrRd')
    plt.show()
    
def to_numeric_data(data):
    # Object class to 0, 1, 2...
    for feature in data.select_dtypes(include=['object']).columns.tolist():
        for i in range(len(data[feature].value_counts().index)):
            data[feature]=data[feature].replace(data[feature].value_counts().index[i], i)
    
    return data
    
    
if __name__ == '__main__':
    file = sys.argv[1]
    correlation(file)