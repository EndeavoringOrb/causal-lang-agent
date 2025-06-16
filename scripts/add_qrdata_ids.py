import json

data = json.load(open("QRData/QRData/benchmark/QRData.json"))
for idx, item in enumerate(data):
    item["id"] = idx

# Save to a new file
with open("QRData/QRData/benchmark/QRData_ids.json", "w") as f:
    json.dump(data, f, indent=2)
