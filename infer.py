import os
import torch
import random
import numpy as np
from pathlib import Path
import yaml
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import T5Config, T5ForConditionalGeneration

from prepare_data import read_lexicon, preprocess_english
from text import _clean_text


class T5Dataset(Dataset):

    def __init__(self, path, config, eos_id=701):
        self.path = path
        self.dir_path = os.path.abspath(os.path.dirname(path))
        self.metas = self.get_metadata(path)
        self.eos_id = eos_id
        self.cleaners = config["preprocessing"]["text"]["text_cleaners"]
        self.lexicon = read_lexicon(config["path"]["lexicon_path"])
        
    def get_metadata(self, path):
        metas = []
        with open(path) as f:
            for line in f.readlines():
                metas.append(line.strip("\n").split('|'))
        return metas

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, idx):
        # text_id [T,] (phonemes + punctuations)
        x = self.metas[idx]
        basename = x[0] + ".npy"
        text = x[1]
        text = _clean_text(text, self.cleaners)
        phones, text_id = preprocess_english(text, self.lexicon, self.cleaners)

        return basename, text_id

    def collate_fn(self, batches):
        results = []
        for basename, text_id in batches:
            # text offset
            text_id = text_id + 1 # padding

            # add EOS
            text_id = np.asarray(list(text_id) + [self.eos_id])

            results.append([basename, text_id])

        basenames = []
        # length padding
        text_ids = []
        text_id_lens = []

        max_text_len = max(text_id.shape[0] for _, text_id in results)
        for basename, text_id in results:
            text_id_lens.append(text_id.shape[0])
            text_id = np.pad(text_id,
                         (0, max_text_len - text_id.shape[0]),
                         mode='constant',
                         constant_values=0)        
            text_ids.append(text_id)
            basenames.append(basename)

        # to numpy
        text_ids = np.asarray(text_ids)
        text_id_lens = np.asarray(text_id_lens)
        # to torch
        text_ids = torch.from_numpy(text_ids)
        text_id_lens = torch.from_numpy(text_id_lens)

        return text_ids, text_id_lens, basenames


@torch.no_grad()
def main(args, device):
    device = args.device
    os.makedirs(args.out_dir, exist_ok=True)
    # prepare models
    model = prepare_models(args, device)
 
    #### prepare dataset and inference !!!!
    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    testset = T5Dataset(args.text_file, config=config)

    test_loader = DataLoader(testset,
                              num_workers=8,
                              shuffle=False,
                              sampler=None,
                              batch_size=1,
                              collate_fn=testset.collate_fn,
                              worker_init_fn=seed_worker,
                              pin_memory=True,
                              drop_last=True)
    for i, loaded_data in enumerate(test_loader):
        # import pdb; pdb.set_trace()
        text_ids, text_id_lens, basenames = to_device(loaded_data, device=device)
        out_seqs = model.generate(
            input_ids=text_ids,
            max_length=2048, 
            num_beams=5, 
            no_repeat_ngram_size=2)
        # move offset and eos
        out_seqs = out_seqs[:, 1:-1] - 201      
        out_file = os.path.join(args.out_dir, basenames[0])
        print(out_file, out_seqs)
        np.save(out_file, out_seqs.cpu().numpy())

def prepare_models(args, device):
    # t5 config
    t5_config = T5Config(
        vocab_size=1+200+500+1, # padding=0, text id, semantic id, EOS
        d_model=512, #  Size of the encoder layers and the pooler layer
        d_kv=64,
        d_fft=2048,
        num_layers=6,
        num_heads=8,
        relative_attention_num_buckets=32,
        relative_attention_max_distance=128,
        dropout_rate = 0.1,
        layer_norm_epsilon = 1e-06,
        initializer_factor = 1.0,
        feed_forward_proj = 'relu',
        is_encoder_decoder = True,
        use_cache = True,
        pad_token_id = 0,
        decoder_start_token_id=0,
        eos_token_id = 701,
    )
    model = T5ForConditionalGeneration(config=t5_config).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    return model


def to_device(tensors, device):
    tensors_to_device = []
    for tensor in tensors:
        if isinstance(tensor, torch.Tensor):
            tensors_to_device.append(tensor.to(device))
        else:
            tensors_to_device.append(tensor)
    return tensors_to_device


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--config", 
                        type=str, 
                        help="path to preprocess.yaml")
    parser.add_argument('--ckpt',
                        type=str,
                        required=True,
                        help='model ckpt path')
    parser.add_argument('--text_file',
                        type=str,
                        required=True,
                        help='text file')
    parser.add_argument('--out_dir',
                        type=str,
                        required=True,
                        help='dir of output semantic tokens numpy file')
    parser.add_argument('--device',
                        type=str,
                        default='cpu',
                        help='Inference device, \"cpu\" or \"cuda\"')
    args = parser.parse_args()

    assert 'cpu' in args.device or 'cuda' in args.device, "device must be \"cpu\" or \"cuda\""
    
    main(args, device=args.device)
