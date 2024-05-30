import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler

# Read csv
data=pd.read_csv("heart_raw data.csv")
# DataFrame setting
pd.set_option('future.no_silent_downcasting', True)
pd.options.mode.copy_on_write = True

# Data Exploration
def data_exploration_info(data):
    print("---------------------------------------------------------------------------head---------------------------------------------------------------------------")
    print(data.head())
    print("\n----------------------------------------------------------------------Number of Na----------------------------------------------------------------------")
    print(data.isna().sum())
    print("\n-----------------------------------------------------------------------Infomation-----------------------------------------------------------------------")
    print(data.info())
    print("\n------------------------------------------------------------------------Describe------------------------------------------------------------------------")
    print(data.iloc[:,0:-1].describe())     # Excluded the target
    
# Data Exploration with visualization
def data_exploration_vis(data):
    # histogram Visualization (Numerical data without outlier)
    plt.figure(1)
    plt.title("Age")
    plt.hist(data['age'], color="#960018")
    
    
    # Pie histogram Visualization (No Numerical data)
    plt.figure(2)
    plt.subplot(331)
    plt.title("Sex")
    plt.pie(data['sex'].value_counts(),labels=data['sex'].value_counts().index)
    
    plt.subplot(332)
    plt.title("Chest Pain")
    plt.pie(data["cp"].value_counts(),labels=data['cp'].value_counts().index)
    
    plt.subplot(333)
    plt.title("Fasting blood sugar")
    plt.pie(data["fbs"].value_counts(),labels=data['fbs'].value_counts().index)
    
    plt.subplot(334)
    plt.title("ECG observation at resting condition")
    plt.pie(data["restecg"].value_counts(),labels=data['restecg'].value_counts().index)
    
    plt.subplot(335)
    plt.title("Exercise induced angina")
    plt.pie(data["exang"].value_counts(),labels=data['exang'].value_counts().index)
    
    plt.subplot(336)
    plt.title("The slope of the peak exercise ST segment")
    plt.pie(data["slope"].value_counts(),labels=data['slope'].value_counts().index)
    
    plt.subplot(337)
    plt.title("Thalassemia")
    plt.pie(data["thal"].value_counts(),labels=data['thal'].value_counts().index)
    
    plt.subplot(338)
    plt.title("Number of major vessels (0-3) colored by flourosopy")
    plt.pie(data['ca'].value_counts(), labels=data['ca'].value_counts().index)
    
    plt.show()
    
    # Boxplot Visualization (Numerical data with outlier)
    plt.subplot(221)
    plt.title("Resting blood pressure")
    plt.boxplot(data['trestbps'].dropna())
    
    plt.subplot(222)
    plt.title("Cholesterol measure")
    plt.boxplot(data['chol'].dropna())
    
    plt.subplot(223)
    plt.title("Maximum heart rate achieved")
    plt.boxplot(data['thalch'].dropna())
    
    plt.subplot(224)
    plt.title("ST depression induced by exercise relative to rest")
    plt.boxplot(data['oldpeak'].dropna())

    plt.show()
    
# Correlation map
def data_correlation():
    # Import dataframe to create correlation map (because the existing dataframe has not been preprocessed)
    temp=pd.read_csv("heart_raw data.csv")
    to_numeric_data(temp)
    # Visualization
    sns.heatmap(temp.corr(), annot=True, cmap='YlOrRd')
    plt.show()
    
# Wrong data replacing
def wrong_data(data):
    data['trestbps']=data['trestbps'].replace(0, np.nan)            # trestbps 0 -> null
    data['chol']=data['chol'].replace(0, np.nan)                    # col 0 -> null
    
    return data
    
# Missing data fill & drop
def missing_data(data):
    data['fbs']=data['fbs'].fillna(True)                            # replace with the most common value(True) of "fbs"
    data['slope']=data['slope'].fillna('flat')                      # replace with the most common value(flat) of "slope"
    data.dropna(axis=0, inplace=True)                               # Drop other na
    
    return data

# Feature Reduction
def feature_reduction(data):
    data.drop(columns='id', inplace=True)                           # Drop id because it is meaningless data
    threshold = len(data) * 0.4
    data.dropna(thresh=threshold, axis=1, inplace=True)             # drop if more than 60% null occurs
    
    return data

# Drop outlier with whisker
def outlier(data, features):
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

# Robust Normalization
def data_normalization(data, columns):
    # Visualization with Before data
    _,(ax1,ax2)=plt.subplots(ncols=2)
    ax1.set_title("Before Scaling")
    for i in range(len(columns)):
        sns.kdeplot(data[columns[i]], ax=ax1)

    # Robust Normalization
    for i in columns:
        robust=RobustScaler()
        data[i]=robust.fit_transform(np.array(data[i]).reshape(-1,1)).squeeze()
        
    # Visualization with After data
    ax2.set_title("After Scaling")
    for i in range(len(columns)):
        sns.kdeplot(data[columns[i]], ax=ax2)
        
    plt.show()    
    
    return data

# For easy handling, replace numerical data
def to_numeric_data(data):
    # Object class to 0, 1, 2...
    for feature in data.select_dtypes(include=['object']).columns.tolist():
        for i in range(len(data[feature].value_counts().index)):
            data[feature]=data[feature].replace(data[feature].value_counts().index[i], i)
    
    return data

# Function call
data_exploration_info(data)
data_exploration_vis(data)
data_correlation()

feature_reduction(data)
wrong_data(data)
missing_data(data)

outlier(data, ['trestbps','chol','thalch','oldpeak'])
data_normalization(data, ['age','trestbps','chol','thalch','oldpeak'])

to_numeric_data(data)

data.to_csv("result.csv", index=False)