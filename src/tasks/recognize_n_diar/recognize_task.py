from src.base.base_task import BaseTask
import logging
import whisperx
from pathlib import Path
from typing import Dict, Union
import torch
import gc

logger = logging.getLogger()


class RecognizeTask(BaseTask):

    def __init__(self, config_params:Dict[str, str]):
        """Recognize task.
        Used whisperX. 
        Recognize with Whisper and align text with wav2vec phoneme model.

        Alignment model load when you call 'process'.
        
        Args:
            config_params (Dict): Dict with params.
            Params:
                * device - "cuda" or "cpu". Default:"cpu".
                * batch_size - inference batch size. Default:1. 
                * precision - "float16"/"float32"/"int8". Default:"float16".
                * whisper_model - Shape of whisper model. Default:"tiny".
                * cache_dir - Directory with models. Default:"./cache/asr_models".
                * low_resources - If True - remove model after inference. Default:False.
                * use_alignment - Default:True
        """
        super().__init__()
        logger.info(f"create class instance witch main param {config_params.get('main_param', 'nothing')}")
        self.device = config_params.get("device", "cpu")
        self.batch_size = config_params.get("batch_size", 1)
        self.precision = config_params.get("precision", "float16")
        self.whisper_model = config_params.get("whisper_model", "tiny")
        self.cache_dir = Path(config_params.get("cache_dir", "./cache/asr_models"))
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.low_resources = config_params.get("low_resources", False)
        self.use_alignment = config_params.get("use_alignment", True)
        
        
        if not self.low_resources:
            self.asr_model = whisperx.load_model(self.whisper_model, 
                                        self.device,
                                        compute_type=self.precision,
                                        download_root=self.cache_dir,
                                        )


    def process(self, data:Dict[str, Union[str, torch.Tensor]]) -> Dict:
        """Recognize wav file.

        Args:
            data (Dict): Dict with:
                * audio : str or torch.Tensor with wav file.

        Returns:
            Dict: Dict with result:
            * audio: torch.Tensor - wav file with shape [Time] in FLOAT32 (from -1 to 1)
            * segments: List[Dict[str, str]]:
                [
                    {
                        "start"     : float(begin time),
                        "end"       : float(end time),
                        "text"      : <sentence>,
                        * "words"   : List[Dict[str, Union[str, float]]]:  # - optional 
                        {
                            "start" : float(begin time), 
                            "end"   : float(end time),
                            "word"  : <word>,
                            "score" : float(from 0 to 1),
                        }
                    }
                ] 
        """
        if isinstance(data["audio"], str):
            data["audio"] = whisperx.load_audio(data["audio"])
            
        data["audio"] = data["audio"].squeeze()
        data["audio"] -= data["audio"].mean()
        data["audio"] /= data["audio"].abs().max()
        
        logger.info(f"start speech recognition")
        if not self.low_resources:
            result = self.asr_model.transcribe(data["audio"],
                                               batch_size=self.batch_size)
        else:
            asr_model = whisperx.load_model(self.whisper_model, 
                                        self.device,
                                        compute_type=self.precision,
                                        download_root=self.cache_dir,
                                        )
            data.update(asr_model.transcribe(data["audio"],
                        batch_size=self.batch_size))
            gc.collect(); torch.cuda.empty_cache(); del asr_model
        
        if not self.use_alignment:
            return data 

        logger.info(f"start forced alignment")
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], 
                                                      device=self.device)
        data.update(whisperx.align(result["segments"], model_a, metadata,
                                data["audio"], self.device, 
                                return_char_alignments=False))
        gc.collect(); torch.cuda.empty_cache(); del model_a
        
        logger.info(f"task complete")
        return data
    
    @staticmethod
    def init_from_config(config:Dict[str, str]):
        logger.info(f"init RecognizeTask class instance")
        return RecognizeTask(config_params=config)