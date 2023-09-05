from src.base.pipeline_constructor import BasePipeline
import logging
import sys
import json
import argparse

logger = logging.getLogger()


class TestPipeline(BasePipeline):

    def __init__(self, cfg):
        super().__init__(cfg)


def main():
    parser = argparse.ArgumentParser(
                    prog='Nirma pipeline')
    parser.add_argument('-w', '--wav_file')
    parser.add_argument('-o', '--output', default="test.json")
    parser.add_argument('-c', '--cfg_path', default="src/pipelines/pipeline_cofigs/test_pipeline.yaml")
    args = parser.parse_args()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    
    pipeline = TestPipeline.init_from_config(args.cfg_path)
    res = pipeline.process({"audio": args.wav_file})
    for idx, (emo, vd) in enumerate(zip(res["segments_emotions"], res["segments_valence_dominance"])):
        res["segments"][idx]['emotion'] = emo
        res["segments"][idx]['valence'] = str(vd[0])
        res["segments"][idx]['dominance'] = str(vd[1])
    with open(args.output, "w") as f:
        json.dump(res["segments"], fp=f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()
