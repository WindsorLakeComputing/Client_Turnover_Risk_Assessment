import os
import json
import training
import scoring
import deployment
#import diagnostics
#import reporting
from ingestion import merge_multiple_dataframe

###################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f)

input_folder_path = os.path.join(config['input_folder_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])

##################Check and read new data
#first, read ingestedfiles.txt
ingested_fs = []
for f in os.listdir(prod_deployment_path):
    if (f == "ingestedfiles.txt"):
        ing_file = open(os.getcwd() + "/" + prod_deployment_path + "/" + f, 'r')
        lines = ing_file.readlines()
        for line in lines:
            ingested_fs.append(line.strip())

print(ingested_fs)

new_fs = []
for f in os.listdir(input_folder_path):
    new_file = open(os.getcwd() + "/" + input_folder_path + "/" + f, 'r')
    lines = new_file.readlines()
    new_fs.append(f)
print(new_fs)

unseen_data = []


for n_f in new_fs:
    found = False
    for i_f in ingested_fs:
        if n_f == i_f:
            found = True
    if found is False:
        unseen_data.append(os.getcwd() + "/" + input_folder_path + "/" + n_f)
    found = False
print("unseen_data")
print(unseen_data)
if (unseen_data):
    merge_multiple_dataframe(unseen_data)
#[v for v in lst1 if not set(v) & set(lst2)]
#second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt



##################Deciding whether to proceed, part 1
#if you found new data, you should proceed. otherwise, do end the process here
#if(unseen_data):


##################Checking for model drift
#check whether the score from the deployed model is different from the score from the model that uses the newest ingested data


##################Deciding whether to proceed, part 2
#if you found model drift, you should proceed. otherwise, do end the process here



##################Re-deployment
#if you found evidence for model drift, re-run the deployment.py script

##################Diagnostics and reporting
#run diagnostics.py and reporting.py for the re-deployed model







