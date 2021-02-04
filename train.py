from sklearn.svm import SVR
import argparse
import os
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.model_selection import train_test_split
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory


ds = TabularDatasetFactory.from_delimited_files("http://archive.ics.uci.edu/ml/machine-learning-databases/forest-fires/forestfires.csv")

def log1(x):
    return math.log(x+1)

def clean_data(data):
    months = ['dec','jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov']
    seasons = ['winter','winter','winter','spring','spring','spring','summer','summer','summer','autumn','autumn','autumn']
    days = ['sat','sun','mon','tue','wed','thu','fri']
    wd_we = [0,0,1,1,1,1,1]

    dataset = data.to_pandas_dataframe().dropna()
    dataset = dataset.replace(months,seasons)
    dataset = dataset.replace(days,wd_we)
    seasons = pd.get_dummies(dataset['month'])
    dataset = pd.concat([dataset, seasons], axis = 1)
    dataset = dataset.drop(['month'],axis=1)
    dataset['area'] = dataset['area'].apply(log1)
    
    scaler = StandardScaler()
    features = ['FFMC','DMC','DC','ISI','temp','RH','wind','rain'] 
    dataset[features] = scaler.fit_transform(dataset[features])
    
    x_df = dataset.drop(['area'],axis=1)
    y_df = dataset['area']
    return x_df, y_df


x, y = clean_data(ds)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)

run = Run.get_context()

def score_model(model, X, Y):
    y_pred = model.predict(X)
    y_range = max(Y)-min(Y)
    return mean_absolute_error(np.exp(y_pred),np.exp(Y))/y_range
    

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()
    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--gamma', type=str, default='scale', help="kernel coefficient")
    parser.add_argument('--epsilon', type=float, default=0.1, help="no penalty is associated in the training loss function with points predicted within a distance epsilon from the actual value.")

    args = parser.parse_args()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("gamma:", args.gamma)
    run.log("epsilon:", np.float(args.epsilon))

    model = SVR(C=args.C, gamma=args.gamma, epsilon=args.epsilon).fit(x_train, y_train)
    
    os.makedirs('outputs', exist_ok=True)
    joblib.dump(model, 'outputs/model.joblib')

    nmae = score_model(model, x_test, y_test)
    run.log("Normalized Mean Absolute Error", np.float(nmae))

if __name__ == '__main__':
    main()