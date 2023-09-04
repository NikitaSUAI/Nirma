from src.tasks import *

import logging
import os
import pathlib
import sys
import yaml


logger = logging.getLogger()


class BasePipeline:

    def __init__(self, config):
        self.task_list = []
        for task_name in config.get('pipeline_tasks', []):
            logger.info(f"create task {task_name}")
            task_conf_path = config['pipeline_tasks'][task_name].get("config_path", None)
            task_conf_path = pathlib.Path(os.getcwd()).parent/task_conf_path
            logger.info(f"from config path {task_conf_path}")
            if task_conf_path:
                task_cfg = yaml.load(open(task_conf_path).read(), Loader=yaml.Loader)
                task_classname = task_cfg.get("classname", None)
                task = getattr(sys.modules[__name__], task_classname).init_from_config(task_cfg)
                self.task_list.append(task)

    def process(self, data):
        for task in self.task_list:
            data = task.process(data)
        return data

    def check_components(self):
        pass

    @staticmethod
    def init_from_config(config):
        cfg = yaml.load(open(config).read(), Loader=yaml.Loader)
        return BasePipeline(cfg)
