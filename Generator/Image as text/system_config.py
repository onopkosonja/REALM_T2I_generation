PATH = {
    'datasets': {
        'coco': {
            'weights': {
                'gen': 'epoch=44.ckpt',
                'discr': 'epoch=44.ckpt',
                'text_enc': 'text_encoder100.pth',
                'inception': 'inception_v3_google-1a9a5a14.pth',
                'learner': None
                },
            'word2idx': 'captions.pickle',
            'train': {
                'data': {
                    'img': 'data/train2014',
                    'cap': 'data/annotations/captions_train2014.json'
                },
                'mappings': {
                    'id2idx_img': 'id2idx_img.pkl',
                    'idx2id_img': 'idx2id_img.pkl',
                    'id2idx_cap': 'id2idx_cap.pkl'
                },
                'faiss_nn': {
                    'nn_img': 'nn_img.pkl'
                }
            },
            'val':{
                'data': {
                    'img': 'data/val2014',
                    'cap': 'data/annotations/captions_val2014.json'
                },
                'mappings': {
                    'id2idx_img': 'id2idx_img.pkl',
                    'idx2id_img': 'idx2id_img.pkl',
                    'id2idx_cap': 'id2idx_cap.pkl'
                },
                'faiss_nn': {
                    'nn_img': 'nn_img.pkl'
                }

            }
        }
    }
}



