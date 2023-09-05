from transformers import HubertForSequenceClassification, Wav2Vec2FeatureExtractor
import torch
import logging
import os
import pathlib


logger = logging.getLogger()


class AudioEmotionModel:

    def __init__(self, config):
        feature_extractor_path = config.get("feature_extractor_weights_path")
        model_weight_path = config.get("model_weight_path")
        self.device = config["device"]
        try:
            logger.info(f"load audio emotion feature extractor weight from {model_weight_path}")
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(feature_extractor_path)
            logger.info(f"loading success")
        except Exception as e:
            logger.error(f"Error by load feature_extractor_wights {e}")
            logger.info(f"load extractor weights from repo")
            feature_extractor_path = "facebook/hubert-large-ls960-ft"
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(feature_extractor_path,
                                                                              token=config.get("access_token", None))
        try:
            logger.info(f"load audio emotion model weight from {model_weight_path}")
            self.model = HubertForSequenceClassification.from_pretrained(model_weight_path)
            logger.info(f"loading success")
        except Exception as e:
            logger.error(f"Error by load model_wights {e}")
            logger.info(f"load model weights from repo")
            model_weight_path = "xbgoose/hubert-speech-emotion-recognition-russian-dusha-finetuned"
            self.model = HubertForSequenceClassification.from_pretrained(model_weight_path,
                                                                         token=config.get("access_token", None))

        self.num2emotion = {0: 'neutral', 1: 'angry', 2: 'positive', 3: 'sad', 4: 'other'}

    def process(self, input_wave):
        waveform = torch.Tensor(input_wave).to(self.device)
        inputs = self.feature_extractor(
            waveform.unsqueeze(0),
            sampling_rate=self.feature_extractor.sampling_rate,
            return_tensors="pt",
            padding=True,
            max_length=16000 * 10,
            truncation=True
        )

        logits = self.model(inputs['input_values'][0]).logits
        predictions = torch.argmax(logits, dim=-1)
        predicted_emotion = self.num2emotion[predictions.numpy()[0]]

        return predicted_emotion

    @staticmethod
    def init_from_config(config):
        feature_extractor_path = config.get("feature_extractor_weights_path")
        config["feature_extractor_weights_path"] = pathlib.Path(os.getcwd()).parent / feature_extractor_path
        model_weight_path = config.get("model_weight_path")
        config["model_weight_path"] = pathlib.Path(os.getcwd()).parent / model_weight_path
        return AudioEmotionModel(config)
