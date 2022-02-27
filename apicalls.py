import requests
import json
import os

#Specify a URL that resolves to your workspace
URL = "http://127.0.0.1/"

with open('config.json','r') as f:
    config = json.load(f)

model_path = os.path.join(config['output_model_path'])

#Call each API endpoint and store the responses
pth = {'path': '/home/kelbenj/Udacity/MachineLearningDevOps/Client_Turnover_Risk_Assessment/testdata'}
response1 = requests.post('http://0.0.0.0:8000/prediction', data=pth).content.decode('utf8')
response2 = requests.get('http://0.0.0.0:8000/scoring').content.decode('utf8')
response3 = requests.get('http://0.0.0.0:8000/summarystats').content.decode('utf8')
response4 = requests.get('http://0.0.0.0:8000/diagnostics').content.decode('utf8')

#combine all API responses
responses = "\n".join([response1 + response2 + response3 + response4])

print(responses)


with open(os.path.join(model_path, "api_returns.txt"), "a+") as f:
    f.write(responses)