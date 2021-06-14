from model import Generator, Discriminator
from encoders import RNN_ENCODER
from utils import FullDataset, RandomDataset
from system_config import PATH
from model_config import PARAM_MODEL, PARAM_OPTIM, PARAM_TRAIN, PARAM_AUG
from wandb_config import API_KEY

from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from kornia.enhance.normalize import Normalize
import torch
import torch.nn as nn
import multiprocessing
import os

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import pytorch_lightning as pl
import wandb

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_vars(mode, load_name=None):
    assert mode in ['train', 'val', 'weights'], 'Unknown mode'
    system = PARAM_TRAIN['system']
    dataset = PARAM_TRAIN['dataset']

    path = PATH['datasets'][dataset][mode]

    if system == 'hpc':
        pretrained_path = os.path.join('pretrained', dataset)
        load_from_path = 'checkpoints'

    elif system == 'colab':
        pretrained_path = os.path.join('/content/drive/MyDrive/pretrained', dataset)
        load_from_path = os.path.join('/content/drive/MyDrive/', 'checkpoints')

    if mode in ['train', 'val']:
        img = path['data']['img']
        cap = path['data']['cap']

        idx2id_img = torch.load(os.path.join(pretrained_path, mode, path['mappings']['idx2id_img']))
        id2idx_cap = torch.load(os.path.join(pretrained_path, mode, path['mappings']['id2idx_cap']))
        nn_img = torch.load(os.path.join(pretrained_path, mode, path['faiss_nn']['nn_img']))

        word2idx_path = os.path.join(pretrained_path, PATH['datasets'][dataset]['word2idx'])

        return {'root': img,
                'caps_path': cap,
                'idx2id_img': idx2id_img,
                'id2idx_cap': id2idx_cap,
                'nn_list': nn_img,
                'word2idx_path': word2idx_path}

    rnn_path = os.path.join(pretrained_path, path['text_enc'])
    resnet_path = os.path.join(pretrained_path, path['resnet'])

    # load_name = path['learner']  # pl model
    if load_name is not None:
        load_name = os.path.join(load_from_path, dataset, load_name)
    save_dir = os.path.join(load_from_path, dataset)
    return rnn_path, resnet_path, save_dir, load_name


def stack_imgs_wo_aug(target_imgs, nn_imgs, restored_imgs, n_display, dset=None):
    wh = torch.zeros((n_display, 1, 3, 256, 256),  dtype=torch.float).type_as(target_imgs)
    target_imgs = target_imgs[:n_display].unsqueeze(1)
    nn_imgs = nn_imgs[:n_display].reshape(n_display, 5, 3, *target_imgs.shape[3:])
    restored_imgs = restored_imgs[:n_display].unsqueeze(1)
    return torch.cat((target_imgs, wh, nn_imgs, wh, restored_imgs), 1)


def stack_imgs_with_aug(target_imgs, nn_imgs, restored_imgs, n_display):
    transform = nn.Sequential(Normalize(mean=torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225])))
    wh = transform(torch.zeros((n_display, 1, 3, 224, 224),  dtype=torch.float)).type_as(target_imgs)
    target_imgs = transform(target_imgs[:n_display].unsqueeze(1))
    nn_imgs = nn_imgs[:n_display].reshape(n_display, 5, 3, *target_imgs.shape[3:])
    if dset == 'val':
        nn_imgs = transform(nn_imgs)
    restored_imgs = transform(restored_imgs[:n_display].unsqueeze(1))
    return torch.cat((target_imgs, wh, nn_imgs, wh, restored_imgs), 1)

class DFLearner(pl.LightningModule):
    def __init__(self, gen, discr, discr_lambds, text_enc, optim_params, nz, apply_aug):
        super().__init__()
        self.G = gen
        self.D = discr
        self.discr_lambds = discr_lambds
        self.text_enc = text_enc
        self.optim_params = optim_params
        self.noise_dim = nz

        self.n_display = 8
        self.g_step = 0
        self.d_step = 0
        if apply_aug:
            self.stack_imgs = stack_imgs_with_aug
        else:
            self.stack_imgs = stack_imgs_wo_aug
        self.sample_train = None
        self.sample_val = None

    def on_epoch_start(self):
        for p in self.text_enc.parameters():
            p.requires_grad = False
        self.text_enc.eval()

    def forward(self, cap, cap_len, nn_imgs):
        bs = len(cap)
        hidden = self.text_enc.init_hidden(bs)
        _, cap_emb = self.text_enc(cap, cap_len, hidden)
        cap_emb = cap_emb.detach()

        z = torch.randn(bs, self.noise_dim).type_as(cap_emb)
        return self.G(z, cap_emb, nn_imgs), cap_emb

    def training_step(self, batch, batch_idx, optimizer_idx):
        if batch_idx == 0:
            self.on_epoch_start()
        cap, cap_len, target_img, nn_imgs = batch
        gen_img, cap_emb = self(cap, cap_len, nn_imgs)

        # Update D
        if optimizer_idx == 0:
            loss_fn = nn.ReLU()
            # real images, matching features
            real_features = self.D(target_img)
            output = self.D.COND_DNET(real_features, cap_emb)
            loss_D_real = loss_fn(1.0 - output).mean()

            # real images, mismatching features
            output = self.D.COND_DNET(real_features[:(batch_size - 1)], cap_emb[1:batch_size])
            loss_D_mismatch = loss_fn(1.0 + output).mean()

            # gen images, matching features
            fake_features = self.D(gen_img.detach())
            loss_D_fake = self.D.COND_DNET(fake_features, cap_emb)
            loss_D_fake = loss_fn(1.0 + loss_D_fake).mean()

            # GP
            loss_GP = self.compute_grad_penalty(target_img, cap_emb)

            loss_D = self.discr_lambds['lambd_real'] * loss_D_real + \
                     self.discr_lambds['lambd_mismatch'] + loss_D_mismatch + \
                     self.discr_lambds['lambd_fake'] * loss_D_fake + \
                     self.discr_lambds['lambd_gp'] * loss_GP

            self.log('train_d_real_step_loss', loss_D_real),
            self.log('train_d_mismatch_step_loss', loss_D_mismatch),
            self.log('train_d_fake_step_loss', loss_D_fake),
            self.log('train_d_gp_step_loss', loss_GP),
            self.log('train_d_step_loss', loss_D),
            self.log('d_step', self.d_step)
            self.d_step += 1
            return {'loss': loss_D}

        # Update G
        elif optimizer_idx == 1:
            features = self.D(gen_img)
            output = self.D.COND_DNET(features, cap_emb)
            loss_G = -output.mean()
            self.log('train_g_step_loss', loss_G),
            self.log('g_step', self.g_step)

            if self.sample_train is None:
                self.sample_train = self.stack_imgs(target_img, nn_imgs, gen_img, self.n_display, 'train')
                self.sample_train = self.sample_train.reshape(-1, 3, 256, 256)
            self.g_step += 1
            return {'loss': loss_G}

    def training_epoch_end(self, outputs):
        avg_d_loss = torch.stack([x['loss'] for x in outputs[0]]).mean()
        avg_g_loss = torch.stack([x['loss'] for x in outputs[1]]).mean()
        grid_img = make_grid(self.sample_train, nrow=9)

        self.log('train_d_epoch_loss', avg_d_loss)
        self.log('train_g_epoch_loss', avg_g_loss)
        self.log('train imgs', [wandb.Image(grid_img)])
        self.log('epoch', self.current_epoch)
        self.sample_train = None

    def validation_step(self, batch, *args):
        if self.sample_val is None:
          cap, cap_len, target_img, nn_imgs = batch
          gen_img, cap_emb = self(cap, cap_len, nn_imgs)
          self.sample_val = self.stack_imgs(target_img, nn_imgs, gen_img, self.n_display, 'val')
          self.sample_val = self.sample_val.reshape(-1, 3, 256, 256)
        return {'data': batch}

    def validation_epoch_end(self, *args):
        grid_img = make_grid(self.sample_val.cpu(), nrow=9)
        self.log('epoch', self.current_epoch, sync_dist=True)
        self.log('val imgs', [wandb.Image(grid_img)], sync_dist=True)
        self.sample_val = None

    def configure_optimizers(self):
        optimizer_D = Adam(self.D.parameters(), lr=self.optim_params['lr_D'], betas=self.optim_params['betas'])
        optimizer_G = Adam(self.G.parameters(), lr=self.optim_params['lr_G'], betas=self.optim_params['betas'])
        return [optimizer_D, optimizer_G], []

    def compute_grad_penalty(self, img, cap_emb):
        interpolated = (img.data).requires_grad_()
        cap_inter = (cap_emb.data).requires_grad_()
        features = self.D(interpolated)
        out = self.D.COND_DNET(features, cap_inter)
        grads = torch.autograd.grad(outputs=out,
                                    inputs=(interpolated, cap_inter),
                                    grad_outputs=torch.ones(out.size()).type_as(img),
                                    retain_graph=True,
                                    create_graph=True,
                                    only_inputs=True)
        grad0 = grads[0].view(grads[0].size(0), -1)
        grad1 = grads[1].view(grads[1].size(0), -1)
        grad = torch.cat((grad0, grad1), dim=1)
        grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
        d_loss_gp = torch.mean(grad_l2norm ** self.discr_lambds['gp_p'])
        return d_loss_gp

if __name__ == '__main__':
    system = PARAM_TRAIN['system']
    apply_aug = PARAM_TRAIN['apply_aug']
    batch_size = PARAM_TRAIN['batch_size']
    dataset = PARAM_TRAIN['dataset']
    load_ckpt_dir = PARAM_TRAIN['load_ckpt_dir']
    epochs = PARAM_TRAIN['epochs']
    gpus = torch.cuda.device_count()
    cpus = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])

    train_dataset_params = get_vars('train')
    train_dataset = RandomDataset(apply_aug,
                                  PARAM_AUG,
                                  **PARAM_MODEL['dataset'],
                                  **train_dataset_params)
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              drop_last=True,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=cpus)

    val_dataset_params = get_vars('val')
    val_dataset = RandomDataset(False,
                                PARAM_AUG,
                                split='val',
                                len_dset=20,
                                **PARAM_MODEL['dataset'],
                                **val_dataset_params)
    val_loader = DataLoader(val_dataset,
                            batch_size=8,
                            drop_last=True,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=cpus)

    rnn_path, resnet_path, save_dir, load_dir = get_vars('weights')

    map_location = 'cuda:0' if gpus else 'cpu'
    text_encoder = RNN_ENCODER(ntoken=train_dataset.n_words, **PARAM_MODEL['text_enc']).to(device)
    text_encoder.load_state_dict(torch.load(rnn_path, map_location=map_location))

    gen = Generator(resnet_path, **PARAM_MODEL['gen']).to(device)

    discr = Discriminator(PARAM_MODEL['discr']['ndf'])

    # if load_ckpt_dir is not None:
    #     model = DFLearner.load_from_checkpoint(load_dir,
    #                                            gen=gen,
    #                                            discr=discr,
    #                                            discr_lambds=PARAM_MODEL['discr'],
    #                                            text_enc=text_encoder,
    #                                            optim_params=PARAM_OPTIM,
    #                                            nz=PARAM_GEN['nz'],
    #                                            apply_aug=apply_aug)
    #
    # else:
    #     gen_weights = torch.load(gen_path, map_location=map_location)['state_dict']
    #     for k, v in gen_weights.items():
    #         m, g_key = k.split('.', maxsplit=1)
    #         if m == 'G':
    #             gen.state_dict()[g_key].data.copy_(v.data)
    #
    #     discr_weights = torch.load(discr_path, map_location=map_location)['state_dict']
    #     for k, v in discr_weights.items():
    #         m, d_key = k.split('.', maxsplit=1)
    #         if m == 'D':
    #             discr.state_dict()[d_key].data.copy_(v.data)

    model = DFLearner(gen,
                      discr,
                      PARAM_MODEL['discr'],
                      text_encoder,
                      PARAM_OPTIM,
                      PARAM_MODEL['gen']['nz'],
                      apply_aug)


    os.environ["WANDB_API_KEY"] = API_KEY
    os.environ['WANDB_MODE'] = 'dryrun'
    wandb_logger = WandbLogger(project='DF GAN', name = 'Concat features from resnet')
    wandb_hyperparams = {**PARAM_MODEL, **PARAM_OPTIM}
    wandb_logger.log_hyperparams(wandb_hyperparams)

    accelerator = 'ddp' if gpus == 2 else None
    saving_ckpt = ModelCheckpoint(dirpath=save_dir,
                                  filename='{epoch}',
                                  verbose=True,
                                  monitor='train_d_epoch_loss',
                                  mode='min',
                                  save_top_k=-1)                                  

    trainer = pl.Trainer(max_epochs=epochs,
                         gpus=gpus,
                         accelerator=accelerator,
                         logger=wandb_logger,
                         checkpoint_callback=saving_ckpt)

    trainer.fit(model, train_loader, val_loader)