from src.base.pipeline_constructor import BasePipeline
import logging
import sys
import json
import click

logger = logging.getLogger()


class TestPipeline(BasePipeline):

    def __init__(self, cfg):
        super().__init__(cfg)
@click.command()
@click.option("--wav_file", "-w")
@click.option("--output", "-o", default="test.json")
@click.option("--conf", "-c", default="src/pipelines/pipeline_cofigs/test_pipeline.yaml")
def main(wav_file, output, conf):
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    cfg_path = conf
    pipeline = TestPipeline.init_from_config(cfg_path)
    res = pipeline.process({"audio": wav_file})
    for idx, (emo, vd) in enumerate(zip(res["segments_emotions"], res["segments_valence_dominance"])):
        res["segments"][idx]['emotion'] = emo
        res["segments"][idx]['valence'] = str(vd[0])
        res["segments"][idx]['dominance'] = str(vd[1])
    with open(output, "w") as f:
        json.dump(res["segments"], fp=f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()
