# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from vocos import Vocos
from vocos.feature_extractors import EncodecFeatures

from conv_stft import STFT

opset_version = 17


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--vocoder-path",
        type=Path,
        help="Path to Vocoder dir",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default="./vocos_vocoder.onnx",
        help="Output path",
    )
    return parser.parse_args()


class ISTFTHead(nn.Module):
    def __init__(self, n_fft: int, hop_length: int):
        super().__init__()
        self.out = None
        self.stft = STFT(fft_len=n_fft, win_hop=hop_length, win_len=n_fft)

    def forward(self, x: torch.Tensor):
        x = self.out(x).transpose(1, 2)
        mag, p = x.chunk(2, dim=1)
        mag = torch.exp(mag)
        mag = torch.clip(mag, max=1e2)
        real = mag * torch.cos(p)
        imag = mag * torch.sin(p)
        audio = self.stft.inverse(input1=real, input2=imag, input_type="realimag")
        return audio


class VocosVocoder(nn.Module):
    def __init__(self, vocos_vocoder):
        super(VocosVocoder, self).__init__()
        self.vocos_vocoder = vocos_vocoder
        istft_head_out = self.vocos_vocoder.head.out
        n_fft = self.vocos_vocoder.head.istft.n_fft
        hop_length = self.vocos_vocoder.head.istft.hop_length
        istft_head_for_export = ISTFTHead(n_fft, hop_length)
        istft_head_for_export.out = istft_head_out
        self.vocos_vocoder.head = istft_head_for_export

    def forward(self, mel):
        waveform = self.vocos_vocoder.decode(mel)
        return waveform


def export_VocosVocoder(vocos_vocoder, output_path, verbose):
    vocos_vocoder = VocosVocoder(vocos_vocoder).cuda()
    vocos_vocoder.eval()

    dummy_batch_size = 8
    dummy_input_length = 500

    dummy_mel = torch.randn(dummy_batch_size, 100, dummy_input_length).cuda()

    with torch.no_grad():
        dummy_waveform = vocos_vocoder(mel=dummy_mel)
        print(dummy_waveform.shape)

    dummy_input = dummy_mel

    torch.onnx.export(
        vocos_vocoder,
        dummy_input,
        str(output_path),
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["mel"],
        output_names=["waveform"],
        dynamic_axes={
            "mel": {0: "batch_size", 2: "input_length"},
            "waveform": {0: "batch_size", 1: "output_length"},
        },
        verbose=verbose,
    )

    print("Exported to {}".format(output_path))


def load_vocoder(
        local_path: Path, 
        device="cpu", 
    ) -> Vocos:

    config_path = local_path / "config.yaml"
    model_path = local_path / "pytorch_model.bin"

    vocoder = Vocos.from_hparams(str(config_path))
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)

    if isinstance(vocoder.feature_extractor, EncodecFeatures):
        encodec_parameters = {
            "feature_extractor.encodec." + key: value
            for key, value in vocoder.feature_extractor.encodec.state_dict().items()
        }

        state_dict.update(encodec_parameters)

    vocoder.load_state_dict(state_dict)
    vocoder = vocoder.eval().to(device)
    
    return vocoder


if __name__ == "__main__":
    args = get_args()
    vocoder = load_vocoder(local_path=args.vocoder_path)
    export_VocosVocoder(vocoder, args.output_path, verbose=False)
