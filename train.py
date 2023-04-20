import os
import torch
import numpy as np
import random
import shutil
import math
import torch.nn.functional as F

from tqdm import tqdm

from transformers import T5Config, T5ForConditionalGeneration
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel
from torch.distributed import init_process_group
from torch.utils.data.distributed import DistributedSampler

from dataset import T5Dataset


def to_device(tensors, device):
    tensors_to_device = []
    for tensor in tensors:
        if isinstance(tensor, torch.Tensor):
            tensors_to_device.append(tensor.to(device))
        else:
            tensors_to_device.append(tensor)
    return tensors_to_device


def tensorboard_logging(writer, logging_dict, step):
    for key in logging_dict:
        writer.add_scalar(key, logging_dict[key], step)


def calculate_model_params(model):
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print("Total params: {}, size of saving: {}M".format(
        pytorch_total_params,
        pytorch_total_params*4/1024/1024))
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total trainable params: {}, size of saving: {}M".format(
        pytorch_total_params,
        pytorch_total_params*4/1024/1024))
    return


def change_lr(opt, lr):
    for g in opt.param_groups:
        g['lr'] = lr
    return


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    return


def sequence_mask(seq_lens, max_len=None, device='cpu'):
    b = seq_lens.shape[0]
    if max_len is None:
        max_len = seq_lens.max()
    mask = torch.arange(max_len).unsqueeze(0).to(device) # [1, t]
    mask = mask < (seq_lens.unsqueeze(1)) # [1, t] + [b, 1] = [b, t]
    mask = mask.float()
    return mask


def compute_loss(logits, target, mask, compute_acc=False):
    logits = logits.contiguous()
    target = target.contiguous()
    mask = mask.contiguous()

    # logits_flat: (batch * max_len, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # log_probs_flat: (batch * max_len, num_classes)
    log_probs_flat = F.log_softmax(logits_flat)
    # target_flat: (batch * max_len, 1)
    target_flat = target.view(-1, 1)
    # losses_flat: (batch * max_len, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    # losses: (batch, max_len)
    losses = losses_flat.view(*target.size()) * mask
    # mask: (batch, max_len)
    loss = losses.sum() / mask.sum()

    if compute_acc:
        with torch.no_grad():
            pred_true = torch.eq(log_probs_flat.argmax(-1), target_flat.squeeze(1)).view(*target.size()) * mask
            acc = pred_true.sum().float() / mask.sum()
        return loss, acc

    return loss


def train(rank, args, hp):
    save_path = args.save_path
    meta_path = args.meta_path
    num_gpus = args.num_gpus

    # reproduction setting
    random.seed(1996 + rank)
    np.random.seed(1997 + rank)
    torch.manual_seed(1998 + rank)
    torch.cuda.set_device(rank)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # logging and path settings
    device = 'cuda:{}'.format(rank)
    if rank == 0:
        save_checkpoint_path = os.path.join(save_path, 'checkpoints')
        logging_path = os.path.join(save_path, 'logs')
        # create paths
        os.makedirs(save_checkpoint_path, exist_ok=True)
        os.makedirs(logging_path, exist_ok=True)
        # save config
        shutil.copyfile(args.config, os.path.join(save_path, 'config.yaml'))
        # tensorboard writer
        writer = SummaryWriter(logging_path)

    train_step = 0
    train_epoch = 0
    save_step = 0

    # DDP training setting
    if num_gpus > 1:
        init_process_group(backend=hp.dist_backend,
                           init_method=hp.dist_url,
                           world_size=num_gpus,
                           rank=rank)

    # prepare dataset
    trainset = T5Dataset(meta_path, hp=hp)
    sampler = DistributedSampler(trainset,
                                 num_replicas=num_gpus,
                                 rank=rank,
                                 shuffle=True,
                                 seed=2000,
                                 drop_last=True) if num_gpus > 1 else None
    shuffle = False if num_gpus > 1 else True
    train_loader = DataLoader(trainset,
                              num_workers=7,
                              shuffle=shuffle,
                              sampler=sampler,
                              batch_size=hp.batch_size,
                              collate_fn=trainset.collate_fn,
                              worker_init_fn=seed_worker,
                              pin_memory=True,
                              drop_last=True)

    # t5 config - base
    t5_config = T5Config(
        vocab_size=1+200+500+1, # padding=0, text id, semantic id, EOS
        d_model=1024, #  Size of the encoder layers and the pooler layer
        d_kv=64,
        d_fft=2816,
        num_layers=12,
        num_heads=16,
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
    # t5 config - large
    # t5_config = T5Config(
    #     vocab_size=1+200+500+1, # padding=0, text id, semantic id, EOS
    #     d_model=1024, #  Size of the encoder layers and the pooler layer
    #     d_kv=64,
    #     d_fft=2816,
    #     num_layers=12,
    #     num_heads=16,
    #     relative_attention_num_buckets=32,
    #     relative_attention_max_distance=128,
    #     dropout_rate = 0.1,
    #     layer_norm_epsilon = 1e-06,
    #     initializer_factor = 1.0,
    #     feed_forward_proj = 'relu',
    #     is_encoder_decoder = True,
    #     use_cache = True,
    #     pad_token_id = 0,
    #     decoder_start_token_id=0,
    #     eos_token_id = 701,
    # )

    # prepare model
    model = T5ForConditionalGeneration(config=t5_config).to(device)
    calculate_model_params(model)

    # restore model
    restore_path = args.restore_path
    if restore_path is not None and os.path.exists(restore_path):
        restore_ckpt = torch.load(restore_path, map_location=device)
        model.load_state_dict(restore_ckpt['model'], strict=True)
        train_step = restore_ckpt['train_step']
    else:
        restore_ckpt = None

    if num_gpus > 1:
        torch.distributed.barrier()

    # multi-cards training
    if num_gpus > 1:
        model = DistributedDataParallel(model, device_ids=[rank], output_device=rank)

    # prepare optimizer
    opt = torch.optim.AdamW(model.parameters(), hp.learning_rate, betas=[0.9, 0.999], eps=1e-8)

    # restore optimizer
    if restore_ckpt is not None:
        opt.load_state_dict(restore_ckpt['opt'])
    # TODO: learning rate warmup

    if num_gpus > 1:
        torch.distributed.barrier()

    ### TRAIN LOOP !!! ###
    train_epoch = train_step // (len(trainset) // max(1, num_gpus))
   
    while train_step <= hp.max_steps:
        if sampler is not None:
            sampler.set_epoch(train_epoch)
        # epoch lr schedule
        # current_lr = max(hp.learning_rate * (0.9 ** train_epoch), 2e-6)
        # change_lr(opt, current_lr)
        for i, loaded_data in enumerate(tqdm(train_loader)):
            
            text_ids, text_id_lens, semantic_ids, semantic_id_lens = to_device(loaded_data, device=device)
            b, t = text_ids.shape
            attention_mask = sequence_mask(text_id_lens, max_len=None, device=device)
            t5_lm_outputs = model(
                input_ids=text_ids,
                attention_mask=attention_mask,
                labels=semantic_ids,  
            )
            # import pdb; pdb.set_trace()
            # calculate loss
            logits = t5_lm_outputs['logits'] # [b, t, n_token]
            loss_mask = sequence_mask(semantic_id_lens, max_len=semantic_id_lens.max(), device=device)
            loss, acc = compute_loss(logits, semantic_ids, mask=loss_mask, compute_acc=True)
            # update
            # if train_step < 20_000:
            #     current_lr = hp.learning_rate * min(1.0, train_step / 20_000)
            #     change_lr(opt, current_lr)
            current_lr = 1 / math.sqrt(max(train_step, 10000))
            opt.zero_grad()
            loss.backward()
            opt.step()

            # step logging
            train_step += 1
            if rank == 0:
                save_step += 1
                logging_dict = {
                    "Train/loss": loss.item(),
                    "Train/acc": acc.item(),
                }
                print("Epoch: {}, step: {}k, loss: {:.4f}, acc: {:4f}, lr: {}\n".format(train_epoch, train_step // 1000, loss.item(), acc.item(), current_lr))
                tensorboard_logging(writer, logging_dict, train_step)
                # TODO: evaluation

            log_step = 10_000
            # step saving
            if train_step == 1 or train_step % log_step == 0:
                if rank == 0:
                    save_dict = {
                        "model": model.module.state_dict() if num_gpus > 1 else model.state_dict(),
                        "train_step": train_step,
                    }
                    if train_step % 50_000 == 0:
                        save_dict.update({"opt": opt.state_dict()})
                    torch.save(save_dict, os.path.join(save_checkpoint_path, "{}k_ckpt.pyt".format(train_step // 1000)))
                if num_gpus > 1:
                    torch.distributed.barrier()

        # epoch logging
        train_epoch += 1
        if num_gpus > 1:
            torch.distributed.barrier()

        if rank == 0:
            # epoch saving
            if save_step > log_step:
                # prevent saving checkpoint too frequently
                save_dict = {
                    "model": model.module.state_dict() if num_gpus > 1 else model.state_dict(),
                    "opt": opt.state_dict(),
                    "train_step": train_step,
                }
                torch.save(save_dict, os.path.join(save_checkpoint_path, "latest_ckpt.pyt"))
                save_step = save_step % log_step

        if num_gpus > 1:
            torch.distributed.barrier()


if __name__ == "__main__":
    import argparse
    from utils import get_config_from_file

    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config', type=str, required=True, help='Config yaml')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save checkpoints')
    parser.add_argument('--meta_path', type=str, required=True, help='Path of metadata')

    parser.add_argument('--restore_path', default=None, type=str, help='restore checkpoints')

    parser.add_argument('--rank', default=0, type=int, help='rank')
    parser.add_argument('--num_gpus', default=0, type=int, help='rank')

    args = parser.parse_args()
    hp = get_config_from_file(args.config).hparams

    train(rank=args.rank, args=args, hp=hp)
