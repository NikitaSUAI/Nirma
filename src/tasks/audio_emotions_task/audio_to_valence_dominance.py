from src.base.base_task import BaseTask
from src.tasks.audio_emotions_task.modules.valence_dominance_model import ValenceDominanceModel
import logging
import torch
import gc


logger = logging.getLogger()


class AudioToValenceDominance(BaseTask):

    def __init__(self, config_params):
        super().__init__()
        self.config_params = config_params
        self.low_resources = config_params.get("low_resources", False)
        if not self.low_resources:
            self.model = ValenceDominanceModel.init_from_config(config_params)

    def process(self, data):
        logger.info(f"start valence and dominance audio prediction")
        if not self.low_resources:
            data["valence_dominance"] = self.model.process(data['wave'])
        else:
            logger.info(f"working in low resource mode")
            model = ValenceDominanceModel.init_from_config(self.config_params)
            data["valence_dominance"] = model.process(data['wave'])
            gc.collect()
            torch.cuda.empty_cache()
            del model
            logger.info(f"free resources")
        logger.info(f"valence and dominance audio prediction complete")
        return data

    @staticmethod
    def init_from_config(config):
        logger.info(f"init ValenceDominanceModel class instance")
        return AudioToValenceDominance(config_params=config)
