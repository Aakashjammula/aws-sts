# app/core/utils.py
from config.settings import config

def printer(text, level):
    if config['log_level'] in ('info', 'debug') and level == 'info':
        print(text, flush=True)
    elif config['log_level'] == 'debug' and level == 'debug':
        print(text, flush=True)
