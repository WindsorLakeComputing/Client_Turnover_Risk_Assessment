
import pandas as pd
import re
import numpy as np
import pickle
import timeit
import os
import subprocess
import json
from ingestion import merge_multiple_dataframe
from training import train_model
from sklearn.linear_model import LogisticRegression

##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])

##################Function to get model predictions
def model_predictions(location=os.getcwd() + "/" + test_data_path):
    #read the deployed model and a test dataset, calculate predictions
    print("Inside model_prediction ... the location is " + location)
    cur_path = os.getcwd()
    model = LogisticRegression()
    # load saved model
    with open(cur_path + "/" + prod_deployment_path + "/" + 'trainedmodel.pkl', 'rb') as f:
        model = pickle.load(f)
    X = np.empty
    y = np.empty
    for f in os.listdir(location):
        if(f.endswith('csv')):
            X = np.loadtxt(location + "/" + f, delimiter=',', skiprows=1, usecols=[1,2,3])
            y = np.loadtxt(location + "/" + f, delimiter=',', skiprows=1, usecols=[4])

    predictions = model.predict(X)
    return predictions, y
##################Function to get summary statistics
def dataframe_summary():
    #calculate summary statistics here
    files = []
    client_activity = pd.DataFrame()
    cur_path = os.getcwd()
    stats = {}
    for f in os.listdir(dataset_csv_path):
        if (f.endswith('csv')):
            client_activity = client_activity.append(pd.read_csv(cur_path + "/" + dataset_csv_path + "/" + f))
            files.append(f)
    #print("The mean is " + str(client_activity["lastmonth_activity"].mean()))
    headers = list(client_activity.columns.values.tolist())
    for h in headers:
        if(h == "corporation"):
            continue
        stat = {}
        stat[h + "-mean"] = client_activity[h].mean()
        stat[h + "-median"] = client_activity[h].median()
        stat[h + "-std"] = client_activity[h].std()
        stats[h] = stat
    return stats

def missing_data():
    missing_data = []
    client_activity = pd.DataFrame()
    cur_path = os.getcwd()
    for f in os.listdir(dataset_csv_path):
        if (f.endswith('csv')):
            if (f == "finaldata.csv"):
                continue
            client_activity = client_activity.append(pd.read_csv(cur_path + "/" + dataset_csv_path + "/" + f))
            headers = list(client_activity.columns.values.tolist())
            for h in headers:
                missing_data.append((h,(client_activity.isnull()[h].sum() / (client_activity.isnull()[h].sum() + client_activity.count()[h])) * 100))
    return missing_data

##################Function to get timings
def execution_time():
    # calculate timing of training.py and ingestion.py
    times = []
    times.append(("ingestion.py", timeit.timeit(stmt=merge_multiple_dataframe,
                        number=1)))

    times.append(("training.py", timeit.timeit(stmt=train_model,
                        number=1)))
    print(times)


    return times
##################Function to check dependencies
def outdated_packages_list():
    req_file = open(os.getcwd() + "/" + 'requirements.txt', 'r')
    Lines = req_file.readlines()
    packages = []
    for line in Lines:
        installed = subprocess.check_output("pip index versions {package} | sed -nr 's/INSTALLED: ([0-9]+)/\\1/p'".format(
            package=re.split('==', line)[0]), shell=True)
        latest = subprocess.check_output("pip index versions {package} | sed -nr 's/LATEST:    ([0-9]+)/\\1/p'".format(
            package=re.split('==', line)[0]), shell=True)
        packages.append((re.split('==', line)[0],installed.decode('utf-8').strip(),latest.decode('utf-8').strip()))

    return packages

if __name__ == '__main__':
    model_predictions()
    #dataframe_summary()
    #missing_data()
    execution_time()
    #outdated_packages_list()





    
