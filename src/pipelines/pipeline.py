from src.base.pipeline_constructor import BasePipeline
import logging
import sys
import json

logger = logging.getLogger()


class TestPipeline(BasePipeline):

    def __init__(self, cfg):
        super().__init__(cfg)

def main():
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    cfg_path = "src/pipelines/pipeline_cofigs/test_pipeline.yaml"
    pipeline = TestPipeline.init_from_config(cfg_path)
    res = pipeline.process({"audio": "test_wave.wav"})
    for idx, (emo, vd) in enumerate(zip(res["segments_emotions"], res["segments_valence_dominance"])):
        res["segments"][idx]['emotion'] = emo
        res["segments"][idx]['valence'] = str(vd[0])
        res["segments"][idx]['dominance'] = str(vd[1])
    with open("test.json", "w") as f:
        json.dump(res["segments"], fp=f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()
