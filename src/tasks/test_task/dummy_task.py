from src.base.base_task import BaseTask
import logging


logger = logging.getLogger()


class DummyTask(BaseTask):

    def __init__(self, config_params):
        super().__init__()
        logger.info(f"create class instance witch main param {config_params.get('main_param', 'nothing')}")
        self.params = config_params

    def process(self, data):
        logger.info(f"process data")
        data['dummy'] = 'some work'
        return data

    def init_from_config(config):
        logger.info(f"init DummyTask class instance")
        return DummyTask(config_params=config)