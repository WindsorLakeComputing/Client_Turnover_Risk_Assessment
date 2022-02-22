from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json

#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

test_data_path = os.path.join(config['test_data_path'])
model_path = os.path.join(config['output_model_path'])

#################Function for model scoring
def score_model():
    cur_path = os.getcwd()
    model = LogisticRegression()
    # load saved model
    with open(cur_path + "/" + model_path + "/" + 'trainedmodel.pkl', 'rb') as f:
        model = pickle.load(f)
    X = np.empty
    y = np.empty
    for f in os.listdir(test_data_path):
        if(f.endswith('csv')):
            X = np.loadtxt(cur_path + "/" + test_data_path + "/" + f, delimiter=',', skiprows=1, usecols=[1,2,3])
            y = np.loadtxt(cur_path + "/" + test_data_path + "/" + f, delimiter=',', skiprows=1, usecols=[4])

    predictions = model.predict(X)

    latest = open(cur_path + "/" + model_path + "/" + "latestscore.txt", "w")
    latest.write('F1 is: ' + str(metrics.f1_score(y, predictions)) + "\n")
    latest.close()
    return('F1 is: ' + str(metrics.f1_score(y, predictions)))

if __name__ == '__main__':
    score_model()