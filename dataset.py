import os
import torch
import numpy as np
from torch.utils.data import Dataset
from collections import OrderedDict


class T5Dataset(Dataset):

    def __init__(self, path, eos_id=701, hp=None):
        self.path = path
        self.dir_path = os.path.abspath(os.path.dirname(path))
        self.hp = hp
        self.metas = self.get_metadata(path)
        self.eos_id = eos_id

    def get_metadata(self, path):
        with open(path, 'r') as f:
            metas = [l.strip() for l in f]
        return metas

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, idx):
        # text_id [T,] (phonemes + punctuations)
        # semantic_id [T',] (HuBERT: 0-499)

        x = self.metas[idx].split('|')
        basename_file = x[0] + ".npy"

        # load semantic id
        semantic_id = np.load(os.path.join(self.dir_path, "semantic_data", basename_file))
        # load text id
        text_id = np.load(os.path.join(self.dir_path, "transcript_data", basename_file))

        return text_id, semantic_id

    def collate_fn(self, batches):
        results = []
        for text_id, semantic_id in batches:
            # text and semantic offset
            text_id = text_id + 1 # padding
            semantic_id = semantic_id + 1 + 200 # padding + len(text)

            # add EOS
            text_id = np.asarray(list(text_id) + [self.eos_id])
            semantic_id = np.asarray(list(semantic_id) + [self.eos_id])

            results.append([text_id, semantic_id])

        # length padding
        text_ids = []
        text_id_lens = []
        semantic_ids = []
        semantic_id_lens = []

        max_text_len = max(text_id.shape[0] for text_id, _ in results)
        max_semantic_len = max(semantic_id.shape[0] for _, semantic_id in results)
        for text_id, semantic_id in results:
            text_id_lens.append(text_id.shape[0])
            text_id = np.pad(text_id,
                         (0, max_text_len - text_id.shape[0]),
                         mode='constant',
                         constant_values=0)
            semantic_id_lens.append(semantic_id.shape[0])
            semantic_id = np.pad(semantic_id,
                         (0, max_semantic_len - semantic_id.shape[0]),
                         mode='constant',
                         constant_values=0)
          
            text_ids.append(text_id)
            semantic_ids.append(semantic_id)

        # to numpy
        text_ids = np.asarray(text_ids)
        text_id_lens = np.asarray(text_id_lens)
        semantic_ids = np.asarray(semantic_ids)
        semantic_id_lens = np.asarray(semantic_id_lens)
        # to torch
        text_ids = torch.from_numpy(text_ids)
        text_id_lens = torch.from_numpy(text_id_lens)
        semantic_ids = torch.from_numpy(semantic_ids)
        semantic_id_lens = torch.from_numpy(semantic_id_lens)

        return text_ids, text_id_lens, semantic_ids, semantic_id_lens


if __name__ == '__main__':
    dataset = T5Dataset(path='/mnt/bd/zhouyixuan-volume-1/data/LibriTTS/val_metadata.txt')
    batches = [dataset.__getitem__(i) for i in range(3)]
    x = dataset.collate_fn(batches)
