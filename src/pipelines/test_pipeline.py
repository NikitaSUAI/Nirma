from src.base.pipeline_constructor import BasePipeline
import logging
import sys


logger = logging.getLogger()


class TestPipeline(BasePipeline):

    def __init__(self, cfg):
        super().__init__(cfg)


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    cfg_path = "pipeline_cofigs/test_pipeline.yaml"
    pipeline = TestPipeline.init_from_config(cfg_path)
    pipeline.process({})
