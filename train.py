import numpy as np
import pandas as pd

def simple_regression(X, y):
    pass
    
if __name__ == '__main__':
    df = pd.read_csv('data/train.csv')
    
    for col in df.columns:
        if(df[col].isnull().any()):
            del df[col]
    
    print(df.head())