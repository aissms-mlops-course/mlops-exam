from distutils.command.config import config
import os
import requests
import shutil
import random
import argparse
import pandas as pd 
import numpy as np
import yaml

def get_data(config_file):
    config = read_params(config_file)
    return config

def read_params(config_file):
    with open(config_file) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    a = get_data(config_file=parsed_args.config)
