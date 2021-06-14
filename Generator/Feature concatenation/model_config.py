PARAM_MODEL = {
  'dataset':{
      'max_len': 18
  },
  'gen': {
      'ngf': 32,
      'nz': 100
  },
  'discr': {
      'ndf': 32,
      'lambd_real': 1.0,
      'lambd_mismatch': 0.5,
      'lambd_fake': 0.5,
      'lambd_gp': 2.0,
      'gp_p': 6
  },
  'text_enc': {
      'rnn_type': 'LSTM',
      'ninput': 300,
      'drop_prob': 0.5,
      'nhidden': 256,
      'nlayers': 1,
      'bidirectional': True
  }
}

PARAM_OPTIM = {'lr_G': 1e-4,
               'lr_D': 4e-4,
               'l2_coef': 0,
               'betas': (0.0, 0.9)
}

PARAM_AUG = {
    'color_jitter': ([0.5, 1.2], [0.5, 1.5]),  # brightness, contrast, saturation, hue
    'color_p': 0.5,
    'grayscale_p': 0.2,
    'gaus_blur': ((3, 3), (1.5, 1.5)),  # kernel size, sigma
    'gaus_p': 0.1,
    'resize_size': 270,
    'crop_size': 256,
}


PARAM_TRAIN = {'system': 'hpc',
               'dataset': 'coco',
               'apply_aug': False,
               'batch_size': 24,
               'epochs': 300,
               'load_ckpt_dir': None
}


# FAISS_PARAM = {'n_nearest': 5,
#                'system': 'hpc',
#                'batch_size': 128,
#                'saving_dir': 'nn_img'
# }
