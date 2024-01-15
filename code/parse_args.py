import yaml
from ENUMS import SUBJECTS, MODES, TRANSFORMERS, EXPERIMENTS

def load_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)

    if not config:
        raise ValueError(f"No params are present in the config.yaml")
    # Check if required parameters are present
    required_parameters = ['SUBJECTS', 'MODE', 'TRANSFORMER', 'EXPERIMENT']  # Add your required parameters here
    
    for param in required_parameters:
        if param not in config:
            raise ValueError(f"Required parameter '{param}' is missing in the YAML file.")

    # Check if the values of the parameters match the predefined enums
    if not isinstance(config['SUBJECTS'], list):
        raise ValueError("SUBJECT must be an array.")

    for subjectID in config['SUBJECTS']:
        if subjectID < 1 or subjectID > 109:
            raise ValueError(f"Invalid SUBJECT value: {subjectID}. Valid values are {SUBJECTS}")
    if not config['SUBJECTS']:
        config['SUBJECTS'] = list(range(1, 110))
    else:
        config['SUBJECTS'] = list(set(config['SUBJECTS']))
        config['SUBJECTS'].sort()

    if config['MODE'] not in MODES:
        raise ValueError(f"Invalid MODE value: {config['MODE']}. Valid values are {MODES}")

    if config['TRANSFORMER'] not in TRANSFORMERS:
        raise ValueError(f"Invalid TRANSFORMER value: {config['TRANSFORMER']}. Valid values are {TRANSFORMERS}")

    if config['EXPERIMENT'] not in EXPERIMENTS.keys():
        raise ValueError(f"Invalid EXPERIMENT value: {config['EXPERIMENT']}. Valid values are {EXPERIMENTS}")

    return config
