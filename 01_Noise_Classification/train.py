import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# ──────────────────────────────────────────
# Label mapping (10 classes)
# high_pass_filter + low_pass_filter → band_pass_filter
# ──────────────────────────────────────────
LABEL_LIST = [
    'clean',            # 0
    'background_noise', # 1
    'background_music', # 2
    'gaussian_noise',   # 3
    'band_pass_filter', # 4
    'echo',             # 5
    'pitch_shift',      # 6
    'time_stretch',     # 7
    'reverberation',    # 8
    'auto_tune',        # 9
]
LABEL2IDX = {l: i for i, l in enumerate(LABEL_LIST)}
IDX2LABEL  = {i: l for l, i in LABEL2IDX.items()}
NUM_CLASSES = len(LABEL_LIST)


# ──────────────────────────────────────────
# Model / Dataset factory
# ──────────────────────────────────────────
def build_model(args, device):
    if args.model == 'ssast':
        from model.ssast_model import ASTModel as SSASTModel
        model = SSASTModel(
            label_dim=NUM_CLASSES,
            fshape=args.fshape,
            tshape=args.tshape,
            fstride=args.fstride,
            tstride=args.tstride,
            input_fdim=128,
            input_tdim=args.target_length,
            model_size=args.model_size,
            pretrain_stage=False,
            load_pretrained_mdl_path=args.pretrained_mdl_path,
        )
    elif args.model == 'ast':
        from model.ast_model import ASTModel
        model = ASTModel(
            label_dim=NUM_CLASSES,
            fstride=args.fstride,
            tstride=args.tstride,
            input_fdim=128,
            input_tdim=args.target_length,
            imagenet_pretrain=args.imagenet_pretrain,
            audioset_pretrain=args.audioset_pretrain,
            audioset_pretrain_path=args.audioset_pretrain_path,
            model_size=args.model_size,
        )
    elif args.model == 'fusion':
        from model.multi_feature_fusion import FusionNet
        model = FusionNet(
            num_classes=NUM_CLASSES,
            branch_output_dim=1024,
            spec_shape=(1, args.input_height, args.input_width),
            mfcc_shape=(1, 13, args.input_width),
            f0_len=args.f0_len,
        )
    elif args.model == 'htsat':
        from model.htsat_model import HTSATModel
        model = HTSATModel(
            num_classes=NUM_CLASSES,
            load_pretrained_path=args.pretrained_mdl_path,
        )
    elif args.model == 'hubert':
        from model.hubert_model import HubertClassifier
        model = HubertClassifier(
            num_classes=NUM_CLASSES,
            pretrained_path=args.pretrained_mdl_path,
            pooling=args.pooling,
            freeze_feature_extractor=args.freeze_feature_extractor,
        )
    elif args.model == 'wav2vec2':
        from model.wav2vec2_model import Wav2Vec2Classifier
        model = Wav2Vec2Classifier(
            num_classes=NUM_CLASSES,
            pretrained_path=args.pretrained_mdl_path,
            pooling=args.pooling,
            freeze_feature_extractor=args.freeze_feature_extractor,
        )
    elif args.model == 'cnn8rnn':
        from model.cnn8rnn_model import CNN8RNNClassifier
        model = CNN8RNNClassifier(
            num_classes=NUM_CLASSES,
            pretrained_path=args.pretrained_mdl_path,
        )
    elif args.model == 'cnnlstm':
        from model.cnnlstm import CNNLSTMClassifier
        model = CNNLSTMClassifier(num_classes=NUM_CLASSES)
    elif args.model == 'cnnlstm_2ff':
        from model.cnnlstm_2ff import create_cnnlstm_2ff
        model = create_cnnlstm_2ff(
            num_classes=NUM_CLASSES,
            branch_output_dim=getattr(args, 'branch_output_dim', 512),
            fusion_hidden_dim=getattr(args, 'fusion_hidden_dim', 512),
            dropout=getattr(args, 'dropout', 0.5)
        )
    elif args.model == 'cnnlstm_3ff':
        from model.cnnlstm_3ff import create_cnnlstm_3ff
        model = create_cnnlstm_3ff(
            num_classes=NUM_CLASSES,
            branch_output_dim=getattr(args, 'branch_output_dim', 512),
            fusion_hidden_dim=getattr(args, 'fusion_hidden_dim', 512),
            dropout=getattr(args, 'dropout', 0.5)
        )
    elif args.model == 'cnnlstm_3ff_interaction':
        from model.cnnlstm_3ff_interaction import create_cnnlstm_3ff_interaction
        model = create_cnnlstm_3ff_interaction(
            num_classes=NUM_CLASSES,
            branch_output_dim=getattr(args, 'branch_output_dim', 512),
            interaction_hidden_dim=getattr(args, 'interaction_hidden_dim', 1024),
            dropout=getattr(args, 'dropout', 0.5)
        )
    elif args.model == 'cnnlstm_3ff_weight':
        from model.cnnlstm_3ff_weight import create_cnnlstm_3ff_weight
        model = create_cnnlstm_3ff_weight(
            num_classes=NUM_CLASSES,
            branch_output_dim=getattr(args, 'branch_output_dim', 512),
            dropout=getattr(args, 'dropout', 0.5)
        )
    elif args.model == 'dass':
        from model.dass_model import DASSClassifier
        model = DASSClassifier(
            num_classes=NUM_CLASSES,
            pretrained_path=args.pretrained_mdl_path,
            model_size=args.model_size,
            imagenet_pretrain=args.imagenet_pretrain,
        )
    elif args.model == 'clap':
        from model.clap_model import CLAPClassifier
        model = CLAPClassifier(
            num_classes=NUM_CLASSES,
            pretrained_path=args.pretrained_mdl_path,
            amodel=getattr(args, 'amodel', 'HTSAT-tiny'),
            freeze_encoder=getattr(args, 'freeze_encoder', False),
        )
    else:
        raise ValueError(f'Unknown model: {args.model}')

    return model.to(device)


def build_loaders(args):
    if args.model in ('ast', 'ssast', 'dass'):
        from datautils.dataset_ast import ASTDataset
        train_ds = ASTDataset(
            args.train_protocol, split='train',
            target_length=args.target_length,
            freqm=args.freqm, timem=args.timem,
            mean=args.norm_mean, std=args.norm_std,
            is_train=True,
        )
        dev_ds = ASTDataset(
            args.train_protocol, split='dev',
            target_length=args.target_length,
            freqm=0, timem=0,
            mean=args.norm_mean, std=args.norm_std,
            is_train=False,
        )
    elif args.model == 'cnnlstm':
        from datautils.dataset_cnnlstm import CNNLSTMDataset
        train_ds = CNNLSTMDataset(
            args.train_protocol, split='train',
            clip_duration=args.clip_duration, is_train=True,
        )
        dev_ds = CNNLSTMDataset(
            args.train_protocol, split='dev',
            clip_duration=args.clip_duration, is_train=False,
        )
    elif args.model == 'cnnlstm_2ff':
        from datautils.dataset_cnnlstm_2ff import CNNLSTM_2FF_Dataset, collate_2ff
        f0_method = getattr(args, 'f0_method', 'crepe')
        train_ds = CNNLSTM_2FF_Dataset(
            args.train_protocol, split='train',
            clip_duration=args.clip_duration, is_train=True,
            f0_method=f0_method
        )
        dev_ds = CNNLSTM_2FF_Dataset(
            args.train_protocol, split='dev',
            clip_duration=args.clip_duration, is_train=False,
            f0_method=f0_method
        )
    elif args.model in ('cnnlstm_3ff', 'cnnlstm_3ff_interaction', 'cnnlstm_3ff_weight'):
        from datautils.dataset_cnnlstm_3ff import CNNLSTM_3FF_Dataset, collate_3ff
        f0_method = getattr(args, 'f0_method', 'crepe')
        train_ds = CNNLSTM_3FF_Dataset(
            args.train_protocol, split='train',
            clip_duration=args.clip_duration, is_train=True,
            f0_method=f0_method
        )
        dev_ds = CNNLSTM_3FF_Dataset(
            args.train_protocol, split='dev',
            clip_duration=args.clip_duration, is_train=False,
            f0_method=f0_method
        )
    elif args.model == 'clap':
        from datautils.dataset_clap import CLAPDataset
        train_ds = CLAPDataset(
            args.train_protocol, split='train',
            clip_duration=args.clip_duration, is_train=True,
        )
        dev_ds = CLAPDataset(
            args.train_protocol, split='dev',
            clip_duration=args.clip_duration, is_train=False,
        )
    elif args.model in ('htsat', 'hubert', 'wav2vec2', 'cnn8rnn'):
        ds_cls = None
        if args.model in ('htsat', 'cnn8rnn'):
            from datautils.dataset_htsat import HTSATDataset as ds_cls
        else:
            from datautils.dataset_hubert import HubertDataset as ds_cls
        train_ds = ds_cls(
            args.train_protocol, split='train',
            clip_duration=args.clip_duration,
            is_train=True,
        )
        dev_ds = ds_cls(
            args.train_protocol, split='dev',
            clip_duration=args.clip_duration,
            is_train=False,
        )
    elif args.model == 'fusion':
        from datautils.data_multi_fusion import MultiFeatureDataset, gen_list
        d_meta, train_ids = gen_list(args.protocol_file, is_train=True)
        _, dev_ids = gen_list(args.protocol_file, is_dev=True)
        train_ds = MultiFeatureDataset(train_ids, d_meta, is_train=True)
        dev_ds   = MultiFeatureDataset(dev_ids,   d_meta, is_train=False)
    else:
        raise ValueError(f'Unknown model: {args.model}')

    # Use custom collate_fn for 2FF and 3FF models
    collate_fn = None
    if args.model == 'cnnlstm_2ff':
        collate_fn = collate_2ff
    elif args.model in ('cnnlstm_3ff', 'cnnlstm_3ff_interaction', 'cnnlstm_3ff_weight'):
        collate_fn = collate_3ff

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=collate_fn
    )
    dev_loader = DataLoader(
        dev_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=collate_fn
    )
    return train_loader, dev_loader


def build_eval_loader(args):
    if args.model in ('ast', 'ssast', 'dass'):
        from datautils.dataset_ast import ASTDataset
        ds = ASTDataset(
            args.eval_protocol, split=None,
            target_length=args.target_length,
            freqm=0, timem=0,
            mean=args.norm_mean, std=args.norm_std,
            is_train=False,
        )
    elif args.model == 'cnnlstm':
        from datautils.dataset_cnnlstm import CNNLSTMDataset
        ds = CNNLSTMDataset(
            args.eval_protocol, split=None,
            clip_duration=args.clip_duration, is_train=False,
        )
    elif args.model == 'cnnlstm_2ff':
        from datautils.dataset_cnnlstm_2ff import CNNLSTM_2FF_Dataset, collate_2ff
        f0_method = getattr(args, 'f0_method', 'crepe')
        ds = CNNLSTM_2FF_Dataset(
            args.eval_protocol, split=None,
            clip_duration=args.clip_duration, is_train=False,
            f0_method=f0_method
        )
    elif args.model in ('cnnlstm_3ff', 'cnnlstm_3ff_interaction', 'cnnlstm_3ff_weight'):
        from datautils.dataset_cnnlstm_3ff import CNNLSTM_3FF_Dataset, collate_3ff
        f0_method = getattr(args, 'f0_method', 'crepe')
        ds = CNNLSTM_3FF_Dataset(
            args.eval_protocol, split=None,
            clip_duration=args.clip_duration, is_train=False,
            f0_method=f0_method
        )
    elif args.model in ('htsat', 'hubert', 'wav2vec2', 'cnn8rnn'):
        if args.model in ('htsat', 'cnn8rnn'):
            from datautils.dataset_htsat import HTSATDataset as ds_cls
        else:
            from datautils.dataset_hubert import HubertDataset as ds_cls
        ds = ds_cls(
            args.eval_protocol, split=None,
            clip_duration=args.clip_duration,
            is_train=False,
        )
    elif args.model == 'fusion':
        from datautils.data_multi_fusion import MultiFeatureDataset, gen_list
        eval_ids = gen_list(args.protocol_file, is_eval=True)
        ds = MultiFeatureDataset(eval_ids, labels=None, is_eval=True)
    else:
        raise ValueError(f'Unknown model: {args.model}')

    return DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                      num_workers=args.num_workers, pin_memory=True)


# ──────────────────────────────────────────
# Training utilities
# ──────────────────────────────────────────
class EarlyStop:
    def __init__(self, patience=5, save_path='out/best.pth'):
        self.patience  = patience
        self.best_loss = float('inf')
        self.counter   = 0
        self.early_stop = False
        self.save_path  = save_path

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter   = 0
            os.makedirs(os.path.dirname(self.save_path) or '.', exist_ok=True)
            torch.save(model.state_dict(), self.save_path)
            print(f'  [EarlyStop] best model saved  loss={val_loss:.4f}')
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def _forward(model, batch, device, model_type):
    """Unified forward pass for different model input signatures."""
    if model_type == 'ssast':
        x, label = batch
        x, label = x.to(device), label.to(device)
        logits = model(x, task='ft_avgtok')
    elif model_type == 'ast':
        x, label = batch
        x, label = x.to(device), label.to(device)
        logits = model(x)
    elif model_type in ('dass', 'cnnlstm', 'clap'):
        x, label = batch
        x, label = x.to(device), label.to(device)
        logits = model(x)
    elif model_type in ('htsat', 'hubert', 'wav2vec2', 'cnn8rnn'):
        x, label = batch
        x, label = x.to(device), label.to(device)
        logits = model(x)  # x: (B, num_samples)
    elif model_type == 'cnnlstm_2ff':
        spec, f0, label = batch
        logits = model(spec.to(device), f0.to(device))
        label  = label.to(device)
    elif model_type in ('cnnlstm_3ff', 'cnnlstm_3ff_interaction', 'cnnlstm_3ff_weight'):
        spec, mfcc, f0, label = batch
        logits = model(spec.to(device), mfcc.to(device), f0.to(device))
        label  = label.to(device)
    elif model_type == 'fusion':
        spec, mfcc, f0, label = batch
        logits = model(spec.to(device), mfcc.to(device), f0.to(device))
        label  = label.to(device)
    return logits, label


def train_epoch(model, loader, optimizer, criterion, device, args, scaler=None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch in tqdm(loader, desc='train', ncols=100):
        optimizer.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits, label = _forward(model, batch, device, args.model)
                loss = criterion(logits, label)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, label = _forward(model, batch, device, args.model)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * label.size(0)
        correct    += (logits.argmax(1) == label).sum().item()
        total      += label.size(0)

    return total_loss / total, correct / total * 100


def evaluate(model, loader, criterion, device, args):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for batch in tqdm(loader, desc='eval ', ncols=100):
            logits, label = _forward(model, batch, device, args.model)
            loss = criterion(logits, label)
            total_loss += loss.item() * label.size(0)
            correct    += (logits.argmax(1) == label).sum().item()
            total      += label.size(0)

    return total_loss / total, correct / total * 100


def produce_evaluation_file(model, loader, device, args, save_path):
    model.eval()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    dataset   = loader.dataset
    file_idx  = 0

    class_cols = '\t'.join(LABEL_LIST)
    with torch.no_grad(), open(save_path, 'w') as fh:
        fh.write(f'file_path\ttrue_label\tpredicted_label\tscore\t{class_cols}\n')
        for batch in tqdm(loader, desc='infer', ncols=100):
            logits, labels = _forward(model, batch, device, args.model)
            probs = torch.softmax(logits, dim=1)
            scores, preds = probs.max(dim=1)

            for i, (pred, score, true_label) in enumerate(zip(
                    preds.cpu().numpy(),
                    scores.cpu().numpy(),
                    labels.cpu().numpy())):
                file_path, _ = dataset.samples[file_idx]
                file_idx += 1
                class_scores = '\t'.join(f'{p:.4f}' for p in probs[i].cpu().numpy())
                fh.write(
                    f'{file_path}\t'
                    f'{IDX2LABEL[int(true_label)]}\t'
                    f'{IDX2LABEL[int(pred)]}\t'
                    f'{float(score):.4f}\t'
                    f'{class_scores}\n'
                )

    # 정확도 계산 후 출력
    correct, total = 0, 0
    with open(save_path) as fh:
        next(fh)  # 헤더 skip
        for line in fh:
            parts = line.strip().split('\t')
            if len(parts) == 4 and parts[1] == parts[2]:
                correct += 1
            total += 1
    print(f'Evaluation results saved to {save_path}')
    print(f'Accuracy: {correct}/{total} = {correct/total*100:.2f}%')


# ──────────────────────────────────────────
# Training loop (n-best checkpoints + resume)
# ──────────────────────────────────────────
def train_loop(model, train_loader, dev_loader, optimizer, criterion,
               writer, args, device, scaler=None):
    save_dir = args.save_path.rstrip('/') + '/'
    os.makedirs(save_dir, exist_ok=True)
    state_path = os.path.join(save_dir, 'train_state.npz')

    N_BEST = 3
    bests        = np.full(N_BEST, float('inf'))
    best_loss    = float('inf')
    not_improving = 0
    start_epoch  = 0

    if args.resume and os.path.exists(state_path):
        st = np.load(state_path)
        bests         = st['bests']
        best_loss     = float(st['best_loss'])
        not_improving = int(st['not_improving'])
        start_epoch   = int(st['epoch'])
        print(f'[Resume] epoch={start_epoch}, best_loss={best_loss:.4f}')

    for epoch in range(start_epoch, args.num_epochs):
        print(f'\n===== Epoch {epoch + 1}/{args.num_epochs} =====')
        tr_loss, tr_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, args, scaler)
        dv_loss, dv_acc = evaluate(
            model, dev_loader, criterion, device, args)

        print(f'  train  loss={tr_loss:.4f}  acc={tr_acc:.2f}%')
        print(f'  dev    loss={dv_loss:.4f}  acc={dv_acc:.2f}%')

        writer.add_scalar('loss/train', tr_loss, epoch + 1)
        writer.add_scalar('loss/dev',   dv_loss, epoch + 1)
        writer.add_scalar('acc/train',  tr_acc,  epoch + 1)
        writer.add_scalar('acc/dev',    dv_acc,  epoch + 1)

        if dv_loss < best_loss:
            best_loss     = dv_loss
            not_improving = 0
            torch.save(model.state_dict(), os.path.join(save_dir, 'best.pth'))
            print('  [best model saved]')
        else:
            not_improving += 1

        for i in range(N_BEST):
            if bests[i] > dv_loss:
                for t in range(N_BEST - 1, i, -1):
                    src = os.path.join(save_dir, f'best_{t - 1}.pth')
                    dst = os.path.join(save_dir, f'best_{t}.pth')
                    if os.path.exists(src):
                        os.replace(src, dst)
                bests[i] = dv_loss
                torch.save(model.state_dict(), os.path.join(save_dir, f'best_{i}.pth'))
                break

        np.savez(state_path, bests=bests, best_loss=best_loss,
                 not_improving=not_improving, epoch=epoch + 1)

        if not_improving >= args.early_stop_patience:
            print(f'Early stopping at epoch {epoch + 1}')
            break

    print(f'\nTraining done. Best dev loss: {best_loss:.4f}')
    writer.close()


# ──────────────────────────────────────────
# Main
# ──────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Noise Classification Training')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML config file (e.g. config/ast.yaml). '
                             'CLI args override YAML values.')

    # ── Common ──
    parser.add_argument('--model', type=str, default='ssast', choices=['ssast', 'ast', 'htsat', 'hubert', 'wav2vec2', 'cnn8rnn', 'dass', 'cnnlstm', 'clap', 'fusion'])
    parser.add_argument('--is_train', action='store_true')
    parser.add_argument('--is_eval',  action='store_true')
    parser.add_argument('--batch_size',           type=int,   default=32)
    parser.add_argument('--num_epochs',           type=int,   default=30)
    parser.add_argument('--learning_rate',        type=float, default=1e-5)
    parser.add_argument('--early_stop_patience',  type=int,   default=5)
    parser.add_argument('--save_path',  type=str, default='out/ast/')
    parser.add_argument('--log_dir',    type=str, default='runs/ast')
    parser.add_argument('--num_workers',type=int, default=8)
    parser.add_argument('--resume',     action='store_true',
                        help='Resume training from train_state.npz in save_path')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Load model weights from checkpoint')
    parser.add_argument('--save_results', type=str, default='out/eval_result.txt')

    # ── SSAST ──
    parser.add_argument('--pretrained_mdl_path', type=str, default=None,
                        help='Path to SSAST pretrained .pth (DataParallel state dict)')
    parser.add_argument('--fshape', type=int, default=16, help='Patch size on frequency axis')
    parser.add_argument('--tshape', type=int, default=16, help='Patch size on time axis')

    # ── AST / SSAST shared ──
    parser.add_argument('--train_protocol', type=str,
                        default='protocols/train_protocol.txt',
                        help='Protocol file for train+dev (file_path subset label)')
    parser.add_argument('--eval_protocol', type=str,
                        default='protocols/eval_protocol.txt',
                        help='Protocol file for eval (file_path subset label)')
    parser.add_argument('--target_length',  type=int,   default=512)
    parser.add_argument('--fstride',        type=int,   default=10)
    parser.add_argument('--tstride',        type=int,   default=10)
    parser.add_argument('--freqm',          type=int,   default=48)
    parser.add_argument('--timem',          type=int,   default=192)
    parser.add_argument('--norm_mean',      type=float, default=-4.2677393)
    parser.add_argument('--norm_std',       type=float, default=4.5689974)
    parser.add_argument('--imagenet_pretrain', action='store_true', default=True)
    parser.add_argument('--no_imagenet_pretrain', dest='imagenet_pretrain',
                        action='store_false')
    parser.add_argument('--audioset_pretrain', action='store_true', default=False)
    parser.add_argument('--audioset_pretrain_path', type=str, default=None)
    parser.add_argument('--model_size', type=str, default='base384',
                        choices=['tiny224', 'small224', 'base224', 'base384', 'small', 'medium'])

    # ── HTSAT / HuBERT 공통 ──
    parser.add_argument('--clip_duration', type=float, default=10.0,
                        help='Audio clip length in seconds (default: 10.0)')

    # ── HuBERT ──
    parser.add_argument('--pooling', type=str, default='mean',
                        choices=['mean', 'first', 'attention'],
                        help='HuBERT hidden state pooling method')
    parser.add_argument('--freeze_feature_extractor', action='store_true', default=True,
                        help='Freeze HuBERT CNN feature extractor during fine-tuning')
    parser.add_argument('--no_freeze_feature_extractor', dest='freeze_feature_extractor',
                        action='store_false')

    # ── CLAP ──
    parser.add_argument('--amodel', type=str, default='HTSAT-tiny',
                        choices=['HTSAT-tiny', 'HTSAT-base'],
                        help='CLAP audio encoder architecture')
    parser.add_argument('--freeze_encoder', action='store_true', default=False,
                        help='Freeze CLAP encoder, train head only')

    # ── Fusion ──
    parser.add_argument('--protocol_file', type=str, default=None)
    parser.add_argument('--input_height',  type=int, default=128)
    parser.add_argument('--input_width',   type=int, default=126)
    parser.add_argument('--f0_len',        type=int, default=126)

    # ── YAML config: load first, CLI args override ──
    pre, _ = parser.parse_known_args()
    if pre.config:
        import yaml
        with open(pre.config) as f:
            cfg = yaml.safe_load(f)
        # convert null → None
        cfg = {k: (None if v == 'null' else v) for k, v in cfg.items()}
        parser.set_defaults(**cfg)

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    if torch.cuda.is_available():
        print(f'  GPU: {torch.cuda.get_device_name(0)} (x{torch.cuda.device_count()})')

    model = build_model(args, device)
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if args.is_eval:
        print(f'Parameters: {total:,} total (eval mode — no gradient update)')
    else:
        print(f'Parameters: {total:,} total, {trainable:,} trainable')

    if args.model_path:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f'Loaded weights: {args.model_path}')

    if args.is_eval:
        eval_loader = build_eval_loader(args)
        produce_evaluation_file(model, eval_loader, device, args, args.save_results)
        sys.exit(0)

    if args.is_train:
        train_loader, dev_loader = build_loaders(args)
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=args.learning_rate)
        writer    = SummaryWriter(log_dir=args.log_dir)
        scaler    = GradScaler() if (args.model == 'ast' and device.type == 'cuda') else None

        print(f'\nTraining {args.model.upper()}  '
              f'train={len(train_loader.dataset)}  '
              f'dev={len(dev_loader.dataset)}  '
              f'batch={args.batch_size}')

        train_loop(model, train_loader, dev_loader, optimizer, criterion,
                   writer, args, device, scaler)


if __name__ == '__main__':
    main()
