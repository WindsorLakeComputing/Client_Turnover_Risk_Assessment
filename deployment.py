from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json



##################Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path']) 
model_path = os.path.join(config['output_model_path'])
output_folder_path = config['output_folder_path']

####################function for deployment
def store_model_into_pickle():
    cur_path = os.getcwd()
    old_loc_model = cur_path + "/" + model_path + "/" + 'trainedmodel.pkl'
    new_loc_model = cur_path + "/" + prod_deployment_path + "/" + 'trainedmodel.pkl'

    old_loc_model_score = cur_path + "/" + model_path + "/" + "latestscore.txt"
    new_loc_model_score = cur_path + "/" + prod_deployment_path + "/" + "latestscore.txt"

    old_ingested_files = cur_path + "/" + output_folder_path + "/" + "ingestedfiles.txt"
    new_ingested_files = cur_path + "/" + prod_deployment_path + "/" + "ingestedfiles.txt"
    # copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory
    os.rename(old_loc_model,new_loc_model)
    os.rename(old_loc_model_score, new_loc_model_score)
    os.rename(old_ingested_files, new_ingested_files)
        
if __name__ == '__main__':
    store_model_into_pickle()

