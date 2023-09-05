from src.base.base_task import BaseTask
import logging
import whisperx
from pathlib import Path
from typing import Dict, Union
import torch
import gc
import numpy as np

logger = logging.getLogger()


class DiarizeTask(BaseTask):

    def __init__(self, config_params:Dict[str, str]):
        """Diarization task.
        Used PyAnnote Audio for diarization and align speakers to text 
        
        Args:
            config_params (Dict): Dict with params.
            Params:
                * device - "cuda" or "cpu". Default:"cpu". 
                * cache_dir - Directory with models. Default:"./cache/asr_models".
                * low_resources - If True - remove model after inference. Default:False.
                * access_token - access token for hugging-face 
        """
        super().__init__()
        logger.info(f"create class instance witch main param {config_params.get('main_param', 'nothing')}")
        self.device = config_params.get("device", "cpu")
        self.cache_dir = Path(config_params.get("cache_dir", "./cache/asr_models"))
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.low_resources = config_params.get("low_resources", False)
        self.access_token = config_params.get("access_token", None)
        
        if not self.low_resources:
            self.diarize_model = whisperx.DiarizationPipeline(use_auth_token=self.access_token,
                                                              device=self.device)


    def process(self, data:Dict[str, Union[str, torch.Tensor]]) -> Dict:
        """Recognize wav file.

        Args:
            data (Dict): Dict with:
                * audio     : str or torch.Tensor with wav file.
                * segments  : List[Dict[str, str]]
                [
                    {
                        "start"     : float(begin time),
                        "end"       : float(end time),
                        "text"      : <sentence>,
                        * "words"   : List[Dict[str, Union[str, float]]]: 
                        {
                            "start" : float(begin time), 
                            "end"   : float(end time),
                            "word"  : <word>,
                            "score" : float(from 0 to 1),
                        }
                    }
                ]

        Returns:
            Dict: Dict with result:
            * audio : path to audio file
            * segments: List[Dict[str, str]]
                [
                    {
                        "start"     : float(begin time),
                        "end"       : float(end time),
                        "text"      : <sentence>,
                        "speaker"   : str(<speaker_id>),
                        * "words"   : List[Dict[str, Union[str, float]]]:  # - optional 
                        {
                            "start"     : float(begin time), 
                            "end"       : float(end time),
                            "word"      : <word>,
                            "score"     : float(from 0 to 1), # score from alignment
                            "speaker"   : str(<speaker_id>),
                        }
                    }
                ] 
        """
        if isinstance(data["audio"], str):
            logger.info(f"load wav from {data['audio']}")
            audio = whisperx.load_audio(data["audio"])
        elif isinstance(data["audio"], torch.Tensor):
            audio = data["audio"].numpy()
        else:
            audio = data["audio"]

        audio = audio.squeeze()

        if np.abs(audio).max() > 1:
            logger.info(
                f"""Convert audio from INT16 to FLOAT32. 
                Whisper requires an audio recording in FLOAT32 format."""
            )
            audio = audio / (2**15)

        data['audio'] = audio

        logger.info(f"start diarization")
        if not self.low_resources:
            diarize_segments = self.diarize_model(audio)
            
            data.update(whisperx.assign_word_speakers(diarize_segments, data))
        else:
            diarize_model = whisperx.DiarizationPipeline(use_auth_token=self.access_token,
                                                              device=self.device)
            diarize_segments = diarize_model(audio)
            
            data.update(whisperx.assign_word_speakers(diarize_segments, data))
            gc.collect(); torch.cuda.empty_cache(); del diarize_model
        
        logger.info(f"diarization complete")
        return data
    
    @staticmethod
    def init_from_config(config:Dict[str, str]):
        logger.info(f"init RecognizeTask class instance")
        return DiarizeTask(config_params=config)