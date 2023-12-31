from src.base.base_task import BaseTask
from src.tasks.audio_emotions_task.modules.emotion_model import AudioEmotionModel
import logging
import torch
import gc


logger = logging.getLogger()


class LLM(BaseTask):

    def __init__(self, config_params):
        super().__init__()
        self.config_params = config_params
        self.low_resources = config_params.get("low_resources", False)
        if not self.low_resources:
            self.model = AudioEmotionModel.init_from_config(config_params)

    def process(self, data):
        logger.info(f"start audio emotion prediction")
        if not self.low_resources:
            data["emotion"] = self.model.process(data['wave'])
        else:
            logger.info(f"working in low resource mode")
            model = AudioEmotionModel.init_from_config(self.config_params)
            data = model.process(data)
            gc.collect()
            torch.cuda.empty_cache()
            del model
            logger.info(f"free resources")
        logger.info(f"audio emotion prediction complete")
        return data

    def init_from_config(config):
        logger.info(f"init AudioToEmotion class instance")
        return AudioToEmotion(config_params=config)




