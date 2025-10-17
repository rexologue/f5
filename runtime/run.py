from pathlib import Path
from f5tts_trt import F5TTS

f5tts = F5TTS(
    config_path=Path("/home/user5/f5_trt/trt_f5_engine"),           # директория с config.json и rank*.engine
    vocab_file=Path("/home/user5/f5_model/vocab.txt"),
    ckpt_file=Path("/home/user5/f5_model/espeech_tts_rlv2.pt"),
    vocoder_trt_engine_path=Path("/home/user5/f5_trt/vocoder_engine.plan"),    
    device="cuda:5",
)

wav, sr, spec = f5tts.infer(
    ref_file="/home/user5/ref.wav",
    ref_text="то есть вы представьте себе планка два и семь, где-то какие-то сучки, на маленьком кусочке это просто будет не видно. ну вы же сами понимаете это.",
    gen_text="Без сучка и задоринки, говорили они - кто бы мог подумать!",
    nfe_step=16,
    file_wave="/home/user5/trt.wav",
    seed=None,
)