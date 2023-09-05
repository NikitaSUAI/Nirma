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
        self.config = config
        self.sampling_rate = config.get("sampling_rate", 16000)
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
            self.model = HubertForSequenceClassification.from_pretrained(model_weight_path)

        self.num2emotion = {0: 'нейтральный', 1: 'злой', 2: 'радостный', 3: 'грустный', 4: 'неизвестно'}

    def _sec_to_samples(self, sec):
        return int(sec * self.sampling_rate)

    def process(self, data):
        waveform = data.get(self.config.get("wave_key", "wave"), [])
        if len(waveform) == 0:
            data[self.config.get("output_key", "segments_valence_dominance")] = []
            return data

        max_segment_len = self.config.get("max_segment_len", 10) * self.sampling_rate
        segments_emotions = []
        wave_roi = []
        segments = data.get(self.config.get("segment_key", "segments"), [])
        if not segments:
            wave_roi.append({"start": self._sec_to_samples(0), "end": self._sec_to_samples(len(waveform))})
        else:
            for seg in segments:
                wave_roi.append({"start": self._sec_to_samples(seg["start"]), "end": self._sec_to_samples(seg["end"])})

        waveform = torch.Tensor(waveform).to(self.device)

        for roi in wave_roi:
            wave_segment = waveform[roi['start']: roi['end']][-max_segment_len:]
            inputs = self.feature_extractor(
                wave_segment.unsqueeze(0),
                sampling_rate=self.feature_extractor.sampling_rate,
                return_tensors="pt",
                padding=True,
                max_length=16000 * 10,
                truncation=True
            )

            logits = self.model(inputs['input_values'][0]).logits
            predictions = torch.argmax(logits, dim=-1)
            predicted_emotion = self.num2emotion[predictions.numpy()[0]]
            segments_emotions.append(predicted_emotion)

        data[self.config.get("output_key", "segments_emotions")] = segments_emotions
        return data

    @staticmethod
    def init_from_config(config):
        feature_extractor_path = config.get("feature_extractor_weights_path")
        config["feature_extractor_weights_path"] = pathlib.Path(os.getcwd()) / feature_extractor_path
        model_weight_path = config.get("model_weight_path")
        config["model_weight_path"] = pathlib.Path(os.getcwd()) / model_weight_path
        return AudioEmotionModel(config)
