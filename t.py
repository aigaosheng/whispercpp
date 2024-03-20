# import ffmpeg
import numpy as np
from torchaudio import load
import ffmpeg

# from whispercpp import Whisper
# from whispercpp import api, utils
# utils.MODELS_URL
import whispercpp
# from whispercpp import api, utils
whispercpp.utils.MODELS_URL.update({"large-v3": "https://huggingface.co/datasets/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin"})
# utils.MODELS_URL
# params = (  # noqa # type: ignore
#             api.Params.from_enum(api.SAMPLING_GREEDY)
#             .with_print_progress(False)
#             .with_print_realtime(False)
#             .build()
#         )
# params.language="zh"
# params
# w = Whisper.from_params("large-v3", params, basedir = "/home/gs/work/audiolm-pytorch/whisper/whisper.cpp/models")
# whispercpp.api.N_MEL = 128
w = whispercpp.Whisper.from_pretrained("large-v3", basedir = "/home/gs/work/audiolm-pytorch/whisper/whisper.cpp/models", no_state = False)
w.params.language="zh"
# whispercpp.api.N_MEL = 128


sample_rate = 16000
try:
    y, _ = (
        ffmpeg.input("/home/gs/Documents/zhaozhongxiang.wav", threads=0)
        .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sample_rate)
        .run(
            cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True
        )
    )
except ffmpeg.Error as e:
    raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e
arr = np.frombuffer(y, np.int16).flatten().astype(np.float32) / 32768.0

print(w.transcribe(arr))
