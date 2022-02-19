from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json

###################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
model_path = os.path.join(config['output_model_path']) 


#################Function for training the model
def train_model():
    cur_path = os.getcwd()
    X = np.empty
    y = np.empty
    for f in os.listdir(dataset_csv_path):
        if(f.endswith('csv')):
            X = np.loadtxt(cur_path + "/" + dataset_csv_path + "/" + f, delimiter=',', skiprows=1, usecols=[1,2,3])
            y = np.loadtxt(cur_path + "/" + dataset_csv_path + "/" + f, delimiter=',', skiprows=1, usecols=[4])

    #use this logistic regression for training
    classif = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                    intercept_scaling=1, l1_ratio=None, max_iter=100,
                    multi_class='multinomial', n_jobs=None, penalty='l2',
                    random_state=0, solver='newton-cg', tol=0.0001, verbose=0,
                    warm_start=False)
    
    #fit the logistic regression to your data
    fitted_model = classif.fit(X, y)

    #write the trained model to your workspace in a file called trainedmodel.pkl
    with open(cur_path + "/" + model_path + "/" + 'trainedmodel.pkl', 'wb') as files:
        pickle.dump(fitted_model, files)

if __name__ == '__main__':
    train_model()