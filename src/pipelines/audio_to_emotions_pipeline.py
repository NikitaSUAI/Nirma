from src.base.pipeline_constructor import BasePipeline
import logging
import sys
import librosa


logger = logging.getLogger()


class TestPipeline(BasePipeline):

    def __init__(self, cfg):
        super().__init__(cfg)


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    test_wave, sr = librosa.load("test_wave.wav", sr=16000)
    cfg_path = "src/pipeline_cofigs/audio_to_emotion_pipeline.yaml"
    pipeline = TestPipeline.init_from_config(cfg_path)
    result = pipeline.process({"wave": test_wave})
    logging.info(f"reslut_emotion {result['emotion']}")
