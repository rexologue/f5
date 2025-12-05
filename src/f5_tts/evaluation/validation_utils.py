from pathlib import Path
from typing import List, Dict, Optional

from jiwer import cer

from .gigaam import load_model as load_gigaam_model
from .gigaam.onnx_utils import load_onnx, infer_onnx

from .nisqa.NISQA_model import nisqaModel
from .wespeaker.wespeaker.cli.speaker import Speaker

# Defines default paths relative to this file
BASE_PATH = Path(__file__).parent
MODELS_DIR = BASE_PATH / "models"  # Suggested structure: keep models in one place

class CER:
    """
    Wrapper for GigaAM ASR (ONNX).
    """
    def __init__(self, model_dir: Optional[Path] = None) -> None:
        self.model_dir = model_dir or (BASE_PATH / "gigaam" / "model")
        
        if not self.model_dir.exists():
            print(f"Loading/Converting GigaAM model to {self.model_dir}...")
            model = load_gigaam_model(
                "v3_ctc",
                download_root=str(self.model_dir.parent)
            )
            model.to_onnx(dir_path=str(self.model_dir))
            
        self.sessions, self.model_cfg = load_onnx(str(self.model_dir), "v3_ctc") 

    def transcribe(self, wav_path: Path) -> str:
        try:
            text = infer_onnx(
                str(wav_path), 
                self.model_cfg, # type: ignore
                self.sessions
            ) 
            return (text or "").strip() 
        except Exception as e:
            print(f"Error transcribing {wav_path}: {e}")
            return ""
    
    def calculate_cer(self, base_text: str, gen_wav: Path) -> float:
        hypothesis = self.transcribe(gen_wav)
        if not hypothesis:
            return 1.0 # Penalize empty transcription
        return cer(base_text, hypothesis)
    
    def calculate_cer_list(self, base_texts: List[str], samples_list: List[Path]) -> float:
        if not base_texts: 
            return 0.0
            
        total_cer = 0.0
        for t, s in zip(base_texts, samples_list):
            t = t.lower().replace("+", "").strip()
            total_cer += self.calculate_cer(t, s)

        return total_cer / len(base_texts)

class COSSIM:
    """
    Wrapper for WeSpeaker.
    """
    def __init__(self, ref_wav: Path, device: str = 'cpu', model_dir: Optional[Path] = None):
        model_path = model_dir or (BASE_PATH / "wespeaker")
        self.model = self._load_model(model_path, device)
        
        # Pre-calculate ref embedding to save time
        self.ref_embed = self.model.extract_embedding(str(ref_wav))

    def _load_model(self, model_dir: Path, device: str) -> Speaker:
        avg = model_dir / "avg_model.pt"
        cfg = model_dir / "config.yaml"

        if not avg.exists() or not cfg.exists():
            raise FileNotFoundError(f"Missing avg_model.pt or config.yaml in {model_dir}")

        model = Speaker(str(model_dir))
        model.set_device(device)
        return model
    
    def __call__(self, samples_list: List[Path]) -> float:
        if not samples_list:
            return 0.0

        cs = 0.0
        for sample in samples_list:
            embed = self.model.extract_embedding(str(sample))
            cs += self.model.cosine_similarity(self.ref_embed, embed)

        return cs / len(samples_list)

class MOS:
    """
    Wrapper for NISQA to ensure model is loaded only once.
    """
    def __init__(self, model_path: Optional[Path] = None):
        # Default path setup
        weights_file = model_path or (BASE_PATH / "nisqa" / "nisqa_mos_only.tar")
        
        self.args = {
            "pretrained_model": str(weights_file),
            "mode": "predict_dir",
            "ms_channel": 1,
            "data_dir": None # Will be set dynamically
        }
        # Load model once during init
        self.nisqa = nisqaModel(self.args)

    def predict(self, samples_list: List[Path]) -> float:
        if not samples_list:
            return 0.0

        # NISQA works best on directories. We use the directory of the first file.
        # Assumption: All samples are in the same directory.
        target_dir = samples_list[0].parent
        
        # Update args to point to the current directory
        self.nisqa.args['data_dir'] = str(target_dir)
        self.nisqa._loadDatasetsFolder()
        
        # Predict returns a DataFrame
        df_results = self.nisqa.predict()

        return df_results["mos_pred"].mean()

class SpeechEval:
    def __init__(self, ref_wav: Path, device: str = "cuda"):
        # Initialize all models once
        print("Initializing Evaluation Models...")
        self.cer_model = CER()
        self.cossim_model = COSSIM(ref_wav, device=device)
        self.mos_model = MOS() 
        print("Models initialized.")

    def __call__(
        self,
        base_texts: List[str],
        samples_list: List[Path]
    ) -> Dict[str, float]:
        
        if not samples_list:
            return {"mos": 0.0, "cossim": 0.0, "cer": 0.0}

        # Calculate metrics
        _mos = self.mos_model.predict(samples_list)
        _cossim = self.cossim_model(samples_list)
        _cer = self.cer_model.calculate_cer_list(base_texts, samples_list)

        return {
            "mos": round(_mos, 4), 
            "cossim": round(_cossim, 4), 
            "cer": round(_cer, 4)
        }