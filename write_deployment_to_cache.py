import json
import argparse
import os.path

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str)
parser.add_argument("--name", type=str)
parser.add_argument("--port", type=str)
args = parser.parse_args()

if os.path.exists("service.json"):
    with open("service.json", "r") as f:
        service = json.load(f)
else:
    service = {}

if args.port in service:
    service[args.port] = {
        "model": args.model,
        "name": args.name,
    }

with open("service.json", "w") as f:
    json.dump(service, f, indent=4)
