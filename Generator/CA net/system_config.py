PATH = {
    'datasets': {
        'coco': {
            'weights': {
                'text_enc': 'text_encoder100.pth',
                'learner': None,
                'resnet': 'resnet34-333f7ec4.pth'
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



