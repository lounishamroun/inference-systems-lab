import requests
import json


payload={"prompt": "My favorite italian food is"}
json_payload=json.dumps(payload)
x = requests.post('http://127.0.0.1:8000/generate',json=json_payload)




