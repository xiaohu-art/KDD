import yaml
import json
import torch

def str2int(json_data):
    
    new_dict = {}
    for key, value in json_data.items():
        new_dict[int(key)] = value
    return new_dict

def init_env():
    with open("./data/datasets/config.yml") as f:
        config = yaml.safe_load(f)
    
    with open('./data/datasets/power_10kv.json') as json_file:
        power_10 = json.load(json_file)
    
    power_10 = str2int(power_10)
   
    with open('./data/datasets/all_dict.json') as json_file:
        topology = json.load(json_file)
    
    for key in topology:
        topology[key] = str2int(topology[key])
    
    return config, power_10, topology