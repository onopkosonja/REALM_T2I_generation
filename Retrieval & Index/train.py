from bpe_voc import BpeVocabulary
from LearnDataloader import get_dataloader
from VSE_model import VSE
from Evaluation import eval_recall

from system_config import PATH
from model_config import PARAM_MODEL, PARAM_OPTIM, PARAM_TRAIN
from wandb_config import API_KEY

import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger


def get_learn_vars(system, mode, dataset, resnet_version=None, word_emb_size=None, load_name=None):
    valid_systems = ['hpc', 'colab', 'local']
    if system not in valid_systems:
        raise ValueError('System must be one of {}'.format(valid_systems))

    if system == 'hpc':
        pref_data = ''
        pref_model = ''
    elif system == 'colab':
        pref_data = '/content'
        pref_model = '/content/drive/MyDrive/vse'

    path = PATH['datasets'][dataset]['learn'][mode]
    img = os.path.join(pref_data, path['data']['img'])
    cap = os.path.join(pref_data, path['data']['cap'])

    if mode == 'val':
        return img, cap

    resnet_path = os.path.join(pref_model, PATH['cnn']['resnet_path'], 'resnet{}.pth'.format(resnet_version))

    bpe_path = PATH['language_model']['bpe']
    bpe_model = os.path.join(pref_model, bpe_path['model'])
    bpe_emb = os.path.join(pref_model, bpe_path['emb'].format(word_emb_size))
    bpe_saved_emb = os.path.join(pref_model, bpe_path['saved_emb'])

    if load_name is not None:
        load_name = os.path.join(pref_model, 'checkpoints', dataset, load_ckpt)

    save_dir = os.path.join(pref_model, 'checkpoints', dataset)
    return img, cap, resnet_path, bpe_model, bpe_emb, bpe_saved_emb, load_name, save_dir


class VSElearner(pl.LightningModule):
    def __init__(self, vse_params, optim_params):
        super().__init__()
        self.model = VSE(**vse_params)
        self.optim_params = optim_params
        self.val_check = True
        self.train_step = 0
        self.val_step = 0

    def forward(self, mode, imgs, caps, lengths):
        epoch = self.current_epoch
        return self.model(imgs, caps, lengths, epoch, mode)

    def training_step(self, batch, batch_idx):
        loss = self.forward('train', *batch)
        self.logger.log_metrics({'train_step_loss': loss, 'train_step': self.train_step})
        self.train_step += 1
        return {'loss': loss}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.logger.log_metrics({'train_loss': avg_loss, 'epoch': self.current_epoch})

    def validation_step(self, batch, batch_idx):
        if self.val_check:
            if batch_idx == 1:
                self.val_check = False
            return

        loss, img_embs, cap_embs = self.forward('val', *batch)
        self.val_step += 1
        return {'val_loss': loss, 'img_embs': img_embs, 'cap_embs': cap_embs}

    def validation_epoch_end(self, outputs):
        if outputs:
            avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

            img_embs = [x['img_embs'] for x in outputs]
            cap_embs = [x['cap_embs'] for x in outputs]
            imgs_search_recall, caption_search_recall = eval_recall(img_embs, cap_embs)
            rk_1_img_retr, rk_5_img_retr, rk_10_img_retr = imgs_search_recall
            rk_1_img_cap, rk_5_img_cap, rk_10_img_cap = caption_search_recall

            self.logger.log_metrics({'val_loss': avg_loss,
                                     'epoch': self.current_epoch,
                                     'img_recall@1': rk_1_img_retr,
                                     'img_recall@5': rk_5_img_retr,
                                     'img_recall@10': rk_10_img_retr,
                                     'cap_recall@1': rk_1_img_cap,
                                     'cap_recall@5': rk_5_img_cap,
                                     'cap_recall@10': rk_10_img_cap})
            return {'val_loss': avg_loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.optim_params['lr'])


if __name__ == '__main__':
    system = PARAM_TRAIN['system']
    dataset = PARAM_TRAIN['dataset']
    batch_size = PARAM_TRAIN['batch_size']
    epochs = PARAM_TRAIN['epochs']
    load_ckpt_dir = PARAM_TRAIN['load_ckpt_dir']
    output_emb_size = PARAM_MODEL['emb_size']
    resnet_version = PARAM_MODEL['img']['resnet_version']
    img_size = PARAM_MODEL['img']['size']
    crop_size = PARAM_MODEL['img']['crop_size']
    word_emb_size = PARAM_MODEL['cap']['word_emb_size']
    rnn_layers = PARAM_MODEL['cap']['rnn_layers']
    loss_margin = PARAM_MODEL['loss']['margin']
    sh_loss_epochs = PARAM_MODEL['loss']['sh_epochs']
    optim_params = PARAM_OPTIM

    train_root, train_path, resnet_path, bpe_model, bpe_emb, bpe_saved_emb, load_dir, save_dir = \
    get_learn_vars(system,
                'train',
                dataset,
                resnet_version,
                word_emb_size,
                load_ckpt_dir)

    val_root, val_path = get_learn_vars(system, 'val', dataset)

    voc = BpeVocabulary(emb_size=word_emb_size,
                        model_file=bpe_model,
                        emb_file=bpe_emb,
                        saved_embs=bpe_saved_emb)

    img_enc_params = {
        'model_name': 'resnet{}'.format(resnet_version),
        'model_path': resnet_path,
        'emb_size': output_emb_size
    }

    text_enc_params = {
        'pretrained_embs': voc.embs,
        'word_dim': word_emb_size,
        'emb_size': output_emb_size,
        'num_layers': rnn_layers,
        'padding_value': len(voc) - 1
    }

    vse_params = {
        'img_enc_params': img_enc_params,
        'text_enc_params': text_enc_params,
        'margin': loss_margin,
        'sh_epochs': sh_loss_epochs
    }

    train_loader = get_dataloader(dataset, 'train', voc, train_root, train_path, img_size, crop_size, batch_size)
    val_loader = get_dataloader(dataset, 'val', voc, val_root, val_path, img_size, crop_size, batch_size)

    model = VSElearner(vse_params, optim_params)
    if load_dir is not None:
        model = VSElearner(vse_params, optim_params).load_from_checkpoint(load_dir)
        
    os.environ["WANDB_API_KEY"] = API_KEY
    os.environ['WANDB_MODE'] = 'dryrun'
    wandb_logger = WandbLogger(name=dataset, project='retrieval')
    wandb_params = vse_params
    wandb_params['dataset'] = dataset
    wandb_logger.log_hyperparams(wandb_params)

    gpus = torch.cuda.device_count()
    accelerator = 'ddp' if gpus == 2 else None
    saving_ckpt = ModelCheckpoint(dirpath=save_dir, 
                                  filename='{epoch}-{val_loss:.3f}',
                                  save_top_k=3, 
                                  monitor='val_loss', 
                                  verbose=True)

    trainer = pl.Trainer(max_epochs=epochs,
                         gpus=gpus,
                         accelerator=accelerator,
                         logger=wandb_logger,
                         checkpoint_callback=saving_ckpt)

    trainer.fit(model, train_loader, val_loader)
