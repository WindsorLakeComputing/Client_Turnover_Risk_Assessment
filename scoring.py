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
def score_model(directory_to_score=test_data_path,file_to_score=False):
    cur_path = os.getcwd()
    model = LogisticRegression()
    # load saved model
    with open(cur_path + "/" + model_path + "/" + 'trainedmodel.pkl', 'rb') as f:
        model = pickle.load(f)
    X = np.empty
    y = np.empty
    if(file_to_score):
        X = np.loadtxt(file_to_score, delimiter=',', skiprows=1, usecols=[1, 2, 3])
        y = np.loadtxt(file_to_score, delimiter=',', skiprows=1, usecols=[4])

    else:
        for f in os.listdir(directory_to_score):
            print("the f is " + f)
            if(f.endswith('csv')):
                X = np.loadtxt(cur_path + "/" + directory_to_score + "/" + f, delimiter=',', skiprows=1, usecols=[1,2,3])
                y = np.loadtxt(cur_path + "/" + directory_to_score + "/" + f, delimiter=',', skiprows=1, usecols=[4])
    predictions = model.predict(X)
    return str(metrics.f1_score(y, predictions))

def print_predictions(predictions):
    latest = open(cur_path + "/" + model_path + "/" + "latestscore.txt", "w")
    latest.write('F1 is: ' + str(metrics.f1_score(y, predictions)) + "\n")
    latest.close()
    return('F1 is: ' + str(metrics.f1_score(y, predictions)))

if __name__ == '__main__':
    score_model()