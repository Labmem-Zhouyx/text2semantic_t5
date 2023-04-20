from pathlib import Path
from einops import rearrange, pack, unpack
import joblib
import fairseq
from accelerate import Accelerator
import os
import torch
import torchaudio
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import sys
from torchaudio.functional import resample
import numpy as np


def exists(val):
    return val is not None


def convert_audio(wav: torch.Tensor, sr: int, target_sr: int, target_channels: int):
    assert wav.shape[0] in [1, 2], "Audio must be mono or stereo."
    if target_channels == 1:
        wav = wav.mean(0, keepdim=True)
    elif target_channels == 2:
        *shape, _, length = wav.shape
        wav = wav.expand(*shape, target_channels, length)
    elif wav.shape[0] == 1:
        wav = wav.expand(target_channels, -1)
    wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
    return wav

def round_down_nearest_multiple(num, divisor):
    return num // divisor * divisor

def curtail_to_multiple(t, mult):
    data_len = t.shape[-1]
    return t[..., :round_down_nearest_multiple(data_len, mult)]

class HubertWithKmeans(nn.Module):
    """
    checkpoint and kmeans can be downloaded at https://github.com/facebookresearch/fairseq/tree/main/examples/hubert
    or you can train your own
    """

    def __init__(
            self,
            checkpoint_path,
            kmeans_path,
            target_sample_hz=16000,
            seq_len_multiple_of=None
    ):
        super().__init__()
        self.target_sample_hz = target_sample_hz
        self.seq_len_multiple_of = seq_len_multiple_of

        model_path = Path(checkpoint_path)
        kmeans_path = Path(kmeans_path)

        assert model_path.exists(), f'path {checkpoint_path} does not exist'
        assert kmeans_path.exists(), f'path {kmeans_path} does not exist'

        checkpoint = torch.load(checkpoint_path)
        load_model_input = {checkpoint_path: checkpoint}
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(load_model_input)

        self.model = model[0].eval()

        kmeans = joblib.load(kmeans_path)
        self.kmeans = kmeans

    @property
    def groups(self):
        return 1

    @property
    def codebook_size(self):
        return self.kmeans.n_clusters

    @torch.no_grad()
    def forward(
            self,
            wav_input,
            flatten=True,
            input_sample_hz=None
    ):
        device = wav_input.device

        if exists(input_sample_hz):
            wav_input = resample(wav_input, input_sample_hz, self.target_sample_hz)

        if exists(self.seq_len_multiple_of):
            wav_input = curtail_to_multiple(wav_input, self.seq_len_multiple_of)

        embed, _ = self.model.extract_features(source=wav_input,
                                               padding_mask=None,
                                               mask=False,
                                               output_layer=9)
        embed, packed_shape = pack([embed], '* d')
        sys.stdout = open(os.devnull, 'w')
        codebook_indices = self.kmeans.predict(embed.cpu().detach().numpy())
        sys.stdout = sys.__stdout__
        codebook_indices = torch.from_numpy(codebook_indices).to(device).long()

        if flatten:
            return codebook_indices

        codebook_indices, = unpack(codebook_indices, packed_shape, '*')
        return codebook_indices




def load_model(checkpoint_path,kmeans_path,device):
    wav2vec = HubertWithKmeans(
        checkpoint_path=checkpoint_path,
        kmeans_path=kmeans_path,
        target_sample_hz=16000,
        seq_len_multiple_of=320
    )
    wav2vec.eval()
    return wav2vec.to(device)


# 默认batch_size为1
class EncodecDataset(Dataset):
    def __init__(self, folder, suffix='.wav', sample_rate=24000, channels=1):
        super().__init__()
        self.paths = []
        self.suffix = suffix
        self.sample_rate = sample_rate
        self.channels = channels

        folder = Path(folder)
        self.paths = [*folder.rglob(f'*{self.suffix}')]

        assert len(self.paths) > 0

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        file = self.paths[item]

        data, sr = torchaudio.load(file)
        if data.shape[-1] == 0:
            print(file)
            return None

        data = convert_audio(data, sr, self.sample_rate, self.channels)
        out_file = Path(file).with_suffix('.npy')

        output = (data, out_file)
        return output


def collecte_fn(samples):
    assert len(samples) == 1
    if samples[0] is None:
        return None
    input = torch.cat([sample[0] for sample in samples], dim=0)
    out_file = [sample[1] for sample in samples]
    return (input, out_file)


if __name__ == '__main__':
    accelerate = Accelerator()
    checkpoint_path = './pretrained_models/hubert_base_ls960.pt'
    kmeans_path = './pretrained_models/hubert_base_ls960_L9_km500.bin'
    input_folder = '/mnt/bd/zhouyixuan-volume-1/data/LibriTTS/raw_data'
    output_folder = '/mnt/bd/zhouyixuan-volume-1/data/LibriTTS/semantic_data'
    suffix = '.wav'
    device = 'cuda'
    sampling_rate = 16000
    channels = 1
    num_workers = 16

    semantic_model = load_model(checkpoint_path, kmeans_path, device)
    ds = EncodecDataset(input_folder, suffix, sampling_rate, channels)
    dl = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collecte_fn, num_workers=num_workers)
    (semantic_model, dl) = accelerate.prepare(semantic_model, dl)

    for batch in tqdm(dl):
        if batch is None:
            continue
        with torch.no_grad():
            input, out_file = batch
            out_file = out_file[0]

            if input.shape[1] > sampling_rate * 60:
                input = input[:, :sampling_rate * 60]
            if input.shape[1] < 1600:
                continue

            try:
                input = input.to(device)
                semantic_token = semantic_model(input, input_sample_hz=sampling_rate)
                semantic_token = F.pad(semantic_token, (0, 1), value=semantic_token[-1])
            except:
                print(input.shape)
                continue

            # 存semantic
            out_file = Path(str(out_file).replace(input_folder, output_folder))
            if not out_file.parent.exists():
                os.makedirs(out_file.parent, exist_ok=True)
            semantic_token = torch.unique_consecutive(semantic_token).cpu().numpy()
            np.save(out_file, semantic_token)