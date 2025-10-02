# load json file

import json
with open("jsons/results.json", "r") as f:
    data = json.load(f)
print(data)
