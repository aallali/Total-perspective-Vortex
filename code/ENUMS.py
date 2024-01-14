SUBJECTS = [1,2,3,4,5,6]
MODES = ["train", "predict", "all"]
TRANSFORMERS = ["FAST_CSP", "CSP", "SPoC"]

EXPERIMENTS = {
    "hands_vs_feet__action": ['do/hands', 'do/feet'],
    "hands_vs_feet__imagery": ['imagine/hands', 'imagine/feet'],
    "imagery_vs_action__hands": ['do/hands', 'imagine/hands'],
    "imagery_vs_action__feets": ['do/feet', 'imagine/feet'],
}

EXPERIMENTS_IDS = {
    'action': [5, 9, 13],
    'imagery': [6, 10, 14]
}
