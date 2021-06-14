PATH = {
    'cnn': {
        'resnet_path': 'resnet'
    },
    'language_model': {
        'bpe': {
            'model': 'bpe_data/en.wiki.bpe.vs10000.model',
            'emb': 'bpe_data/en.wiki.bpe.vs10000.d{}.w2v.bin',
            'saved_emb': 'bpe_data/bpe_embs.npy'
        }
    },
    'datasets': {
        'coco': {
            'learn': {
                'train': {
                    'data': {
                        'img': 'data/coco/train2014',
                        'cap': 'data/coco/annotations/captions_train2014.json'
                    },
                },
                'val': {
                    'data': {
                        'img': 'data/coco/val2014',
                        'cap': 'data/coco/annotations/captions_val2014.json'
                    }
                }
            },
            'retrieve': {
                'train': {
                    'img_emb': 'pretrained/coco/train/img_emb.pkl',
                    'id2idx_img': 'pretrained/coco/train/id2idx_img.pkl',
                    'idx2id_img': 'pretrained/coco/train/idx2id_img.pkl',
                    'cap_emb': 'pretrained/coco/train/cap_emb.pkl',
                    'id2idx_cap': 'pretrained/coco/train/id2idx_cap.pkl'
                },
                'val': {
                    'img_emb': 'pretrained/coco/val/img_emb.pkl',
                    'id2idx_img': 'pretrained/coco/val/id2idx_img.pkl',
                    'idx2id_img': 'pretrained/coco/val/idx2id_img.pkl',
                    'cap_emb': 'pretrained/coco/val/cap_emb.pkl',
                    'id2idx_cap': 'pretrained/coco/val/id2idx_cap.pkl'

                }
            }
        },
        'open_images': {
            'learn': {
                'train': {
                    'data': {
                        'img': 'data/oi/train',
                        'cap': 'data/oi/open_images_train_v6_captions.jsonl'
                    },
                },
                'val': {
                    'data': {
                        'img': 'data/oi/val',
                        'cap': 'data/oi/open_images_validation_captions.jsonl'
                    }
                }
            },
            'retrieve': {
                'train': {
                    'img_emb': 'pretrained/oi/train/img_emb.pkl',
                    'id2idx_img': 'pretrained/oi/train/id2idx_img.pkl',
                    'idx2id_img': 'pretrained/oi/train/idx2id_img.pkl',
                    'cap_emb': 'pretrained/oi/train/cap_emb.pkl',
                    'id2idx_cap': 'pretrained/oi/train/id2idx_cap'
                },
                'val': {
                    'img_emb': 'pretrained/oi/val/img_emb.pkl',
                    'id2idx_img': 'pretrained/oi/val/id2idx_img.pkl',
                    'idx2id_img': 'pretrained/oi/val/idx2id_img.pkl',
                    'cap_emb': 'pretrained/oi/val/cap_emb.pkl',
                    'id2idx_cap': 'pretrained/oi/val/id2idx_cap'

                }
            }
        }

    }
}



