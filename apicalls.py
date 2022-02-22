import requests

#Specify a URL that resolves to your workspace
URL = "http://127.0.0.1/"



#Call each API endpoint and store the responses
pth = {'path': '/home/kelbenj/Udacity/MachineLearningDevOps/Client_Turnover_Risk_Assessment/testdata'}
response1 = requests.post('http://0.0.0.0:8000/prediction', data=pth).text
response2 = requests.get('http://0.0.0.0:8000/scoring').text
response3 = requests.get('http://0.0.0.0:8000/summarystats').text
response4 = requests.get('http://0.0.0.0:8000/diagnostics').text

#combine all API responses
responses = response1 + response2 + response3 + response4

print(responses)




