import pandas as pd
import numpy as np
import os
import json
from datetime import datetime


#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']



#############Function for data ingestion
#check for datasets, compile them together, and write to an output file
def merge_multiple_dataframe(input_files=[], output_folder_path=output_folder_path):
    cur_path = os.getcwd()
    client_activity = pd.DataFrame()
    files = []
    if(len(input_files) > 0):
        for f in input_files:
            client_activity = client_activity.append(pd.read_csv(f))
            files.append(f)
    else:
        for f in os.listdir(input_folder_path):
            client_activity = client_activity.append(pd.read_csv(cur_path + "/" + input_folder_path + "/" + f))
            files.append(f)

    ingested_files = open(cur_path + "/" + output_folder_path + "/" + "ingestedfiles.txt", "a+")
    for element in files:
        ingested_files.write(element + "\n")
    ingested_files.close()
    client_activity = client_activity.drop_duplicates()
    client_activity.reset_index(drop=True, inplace=True)
    client_activity.to_csv(cur_path + "/" + output_folder_path + "/" + "finaldata.csv", index = False)

if __name__ == '__main__':
    merge_multiple_dataframe()
