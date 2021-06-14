PARAM_MODEL = {
    'emb_size': 1024,
    'img': {
        'size': 256,
        'crop_size': 224,
        'resnet_version': 152
    },
    'cap': {
        'word_emb_size': 300,
        'rnn_layers': 1
    },
    'loss': {
        'margin': 0.2,
        'sh_epochs': 5
    }

}


PARAM_OPTIM = {
    'lr': 1e-3
}


PARAM_TRAIN = {
    'system': 'hpc',
    'dataset': 'coco', #open_images
    'batch_size': 128,
    'epochs': 300,
    'load_ckpt_dir': None,
}


PARAM_RETRIEVE = {
    'system': 'colab',
    'ckpt': 'epoch=72.ckpt',
    'batch_size': 200
}



