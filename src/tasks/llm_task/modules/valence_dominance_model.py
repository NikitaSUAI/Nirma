import torch
import logging
import os
import pathlib


logger = logging.getLogger()


class ValenceDominanceModel:

    def __init__(self, config):
        feature_extractor_path = config.get("feature_extractor_weights_path")
        model_weight_path = config.get("model_weight_path")
        self.device = config["device"]
        self.sampling_rate = config.get("sampling_rate", 16000)
        self.config = config
        try:
            logger.info(f"load valence dominance feature extractor weight from {model_weight_path}")
            self.feature_extractor = Wav2Vec2Processor.from_pretrained(feature_extractor_path)
            logger.info(f"loading success")
        except Exception as e:
            logger.error(f"Error by load feature_extractor_wights {e}")
            logger.info(f"load extractor weights from repo")
            feature_extractor_path = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
            self.feature_extractor = Wav2Vec2Processor.from_pretrained(feature_extractor_path,
                                                                       token=config.get("access_token", None))
        try:
            logger.info(f"load valence dominance model weight from {model_weight_path}")
            self.model = EmotionModel.from_pretrained(model_weight_path)
            logger.info(f"loading success")
        except Exception as e:
            logger.error(f"Error by load model_wights {e}")
            logger.info(f"load model weights from repo")
            model_weight_path = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
            self.model = EmotionModel.from_pretrained(model_weight_path)

    def _sec_to_samples(self, sec):
        return int(sec * self.sampling_rate)

    def process(self, data):
        waveform = data.get(self.config.get("wave_key", "wave"), [])
        if len(waveform) == 0:
            data[self.config.get("output_key", "segments_valence_dominance")] = []
            return data

        max_segment_len = self.config.get("max_segment_len", 10) * self.sampling_rate
        segments_valence_dominance = []
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
            features = self.feature_extractor(wave_segment, sampling_rate=self.sampling_rate)
            features = features['input_values'][0]
            features = features.reshape(1, -1)
            features = torch.from_numpy(features).to(self.device)
            with torch.no_grad():
                logits = self.model(features)[1]
            logits = logits.detach().cpu().numpy()
            logits = logits[0][-2:]
            segments_valence_dominance.append(logits)

        data[self.config.get("output_key", "segments_valence_dominance")] = segments_valence_dominance
        return data

    @staticmethod
    def init_from_config(config):
        feature_extractor_path = config.get("feature_extractor_weights_path")
        config["feature_extractor_weights_path"] = pathlib.Path(os.getcwd()) / feature_extractor_path
        model_weight_path = config.get("model_weight_path")
        config["model_weight_path"] = pathlib.Path(os.getcwd()) / model_weight_path
        return ValenceDominanceModel(config)