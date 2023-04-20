import argparse
import yaml
import os
import re

import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm
from g2p_en import G2p
g2p = G2p()

from text import _clean_text, text_to_sequence, symbols


def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon


def preprocess_english(text, lexicon, cleaners):
    
    phones = []
    words = re.split(r"([,;.\-\?\!\s+])", text)
    for w in words:
        if w.lower() in lexicon:
            phones += lexicon[w.lower()]
        elif w in symbols:
            phones += w
        else:
            phones += list(filter(lambda p: p != " ", g2p(w)))
            # continue 
    phones = "{" + "}{".join(phones) + "}"
    phones = phones.replace("}{", " ")
    # print("Raw Text Sequence: {}".format(text))
    # print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(
        text_to_sequence(
            phones, cleaners
        )
    )
    return phones, np.array(sequence)


def process_subset(config, subset_list):
    sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
    max_wav_value = config["preprocessing"]["audio"]["max_wav_value"]
    cleaners = config["preprocessing"]["text"]["text_cleaners"]
    lexicon = read_lexicon(config["path"]["lexicon_path"])

    in_dir = config["path"]["corpus_path"]
    rawdata_dir = config["path"]["raw_path"]
    transcript_dir = config["path"]["transcript_path"]
    corpus = config["dataset"]
    metadata = []
    os.makedirs(rawdata_dir, exist_ok=True)
    os.makedirs(transcript_dir, exist_ok=True)

    for dset in subset_list:
        for speaker in tqdm(os.listdir(os.path.join(in_dir, dset)), desc=f"{corpus}/{dset}"):
            for chapter in os.listdir(os.path.join(in_dir, dset, speaker)):
                for file_name in os.listdir(os.path.join(in_dir, dset, speaker, chapter)):
                    if file_name[-4:] != ".wav":
                        continue
                    base_name = file_name[:-4]
                    text_path = os.path.join(
                        in_dir, dset, speaker, chapter, "{}.normalized.txt".format(base_name)
                    )
                    wav_path = os.path.join(
                        in_dir, dset, speaker, chapter, "{}.wav".format(base_name)
                    )
                    with open(text_path) as f:
                        text = f.readline().strip("\n")
                    text = _clean_text(text, cleaners)
                    wav, _ = librosa.load(wav_path, sampling_rate)
                    wav = wav / max(abs(wav)) * max_wav_value
                    wavfile.write(
                        os.path.join(rawdata_dir, "{}.wav".format(base_name)),
                        sampling_rate,
                        wav.astype(np.int16),
                    )
                    with open(
                        os.path.join(rawdata_dir, "{}.lab".format(base_name)),
                        "w",
                    ) as f1:
                        f1.write(text)

                    phones, text_id = preprocess_english(text, lexicon, cleaners)
                    np.save(os.path.join(transcript_dir, "{}.npy".format(base_name)), text_id)
                    # metadata: basename|rawtext|speaker|dset
                    metadata.append("|".join([base_name, text, phones, speaker, dset]))
    return metadata


def prepare_libritts(config):
    train_set = config["subsets"].get("train", None)
    val_set = config["subsets"].get("val", None)
    test_set = config["subsets"].get("test", None)

    train_metadata = process_subset(config, train_set)
    with open(os.path.join("/mnt/bd/zhouyixuan-volume-1/data/LibriTTS/train_metadata.txt"), "w",) as f1:
        f1.write("\n".join(train_metadata))
    val_metadata = process_subset(config, val_set)
    with open(os.path.join("/mnt/bd/zhouyixuan-volume-1/data/LibriTTS/val_metadata.txt"), "w",) as f1:
        f1.write("\n".join(val_metadata))
    test_metadata = process_subset(config, test_set)
    with open(os.path.join("/mnt/bd/zhouyixuan-volume-1/data/LibriTTS/test_metadata.txt"), "w",) as f1:
        f1.write("\n".join(test_metadata))
    

if __name__ == "__main__":
   
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="path to preprocess.yaml")
    args = parser.parse_args()

    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    prepare_libritts(config)
    