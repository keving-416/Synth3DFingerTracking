import yaml
import logging

from logging import config

# Configure Logger
def configure():
  with open("/home/mahanthg/external_5tb_hdd/kevin/MediaPipe/logging/config.yaml", 'rt') as f:
    config_data = yaml.safe_load(f.read())
    config.dictConfig(config_data)
