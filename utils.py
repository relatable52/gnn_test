import yaml

def load_config(config_path):
    with open(config_path) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return cfg