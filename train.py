import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE

les = dict()
ohes = dict()

def preprocess_train(df):
    drop_cols = ['Id']
    df = df.drop(drop_cols, 1)
    
    cat_columns = []
    
    for col in df.columns:
        if df[col].dtype.name == 'object':
            cat_columns.append(col)
            
    for col in cat_columns:
        les[col] = LabelEncoder()
        df[col] = les[col].fit_transform(df[col])

    for col in cat_columns:
        df[col].fillna(np.nanmean(df[col].values), inplace = True)
    
    encoded = []
    
    for col in cat_columns:
        ohes[col] = OneHotEncoder()
        encoded_data = ohes[col].fit_transform(df[col].values.reshape(-1, 1)).toarray()
        n_columns = encoded_data.shape[1]
        columns = [col + '_%d' % i for i in range(n_columns)]
        encoded.append(pd.DataFrame(data = encoded_data, columns = columns))
    
    df_encoded = pd.concat(encoded, axis = 1)
    df_encoded.index = df.index
    
    df.drop(cat_columns, 1, inplace = True)
    df = pd.concat([df, df_encoded], axis = 1)
    
    return df
    
def preprocess(df):
    drop_cols = ['Id']
    df = df.drop(drop_cols, 1)
    
    cat_columns = []
    
    for col in df.columns:
        if df[col].dtype.name == 'object':
            cat_columns.append(col)
            
    for col in cat_columns:
        df[col] = les[col].transform(df[col])

    for col in df.columns:
        df[col].fillna(np.nanmean(df[col].values), inplace = True)
    
    encoded = []
    
    for col in cat_columns:
        encoded_data = ohes[col].transform(df[col].values.reshape(-1, 1)).toarray()
        n_columns = encoded_data.shape[1]
        columns = [col + '_%d' % i for i in range(n_columns)]
        encoded.append(pd.DataFrame(data = encoded_data, columns = columns))
        
    df_encoded = pd.concat(encoded, axis = 1)
    df_encoded.index = df.index
    
    df.drop(cat_columns, 1, inplace = True)
    df = pd.concat([df, df_encoded], axis = 1)
    
    return df
    
if __name__ == '__main__':
    np.random.seed(0)
    train_data = pd.read_csv('data/train.csv')
    test_data = pd.read_csv('data/test.csv')
    
    train_test_data = pd.concat([train_data, test_data])
    preprocess_train(train_test_data)
    
    train = preprocess(train_data)
    test = preprocess(test_data)
    
    X = train.drop('SalePrice', 1).values
    y = train['SalePrice'].values
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    model = RandomForestRegressor(n_estimators = 100, random_state = np.random.randint(1000))
    model.fit(X, y)
    
    X = test.values
    X = scaler.transform(X)
    y = model.predict(X)
    
    predictions = pd.DataFrame(dict({
        'SalePrice': y,
        'Id': test_data['Id']
    }))
    
    predictions.to_csv('data/prediction.csv', index = False)
    
    
    