import yaml
from addict import Dict

def load_config(path):
    with open(path) as file:
        config_dict = yaml.safe_load(file)
    return Dict(config_dict)
