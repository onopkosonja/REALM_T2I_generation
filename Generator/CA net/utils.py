import numpy as np
import os
import torch
import pickle
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from torchvision import transforms
from nltk.tokenize import RegexpTokenizer
from PIL import Image


class CapTransform:
    def __init__(self, word2idx_path, max_len):
        with open(word2idx_path, 'rb') as f:
            self.word2idx = pickle.load(f)[3]
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.max_len = max_len

    def __call__(self, cap):
        cap = cap.replace('\ufffd\ufffd', ' ')
        words = self.tokenizer.tokenize(cap.lower())
        tokens = []
        for w in words:
            w = w.encode('ascii', 'ignore').decode('ascii')
            if len(w) > 0 and w in self.word2idx:
                tokens.append(self.word2idx[w])
        tokens = torch.LongTensor(tokens)

        num_tokens = len(tokens)
        # pad with 0s (i.e., '<end>')
        x = torch.zeros((self.max_len), dtype=torch.long)
        if num_tokens <= self.max_len:
            x_len = num_tokens
            x[:x_len] = tokens
        else:
            x_len = self.max_len
            ix = list(np.arange(self.max_len))  # 1, 2, 3,..., maxNum
            np.random.shuffle(ix)
            ix = ix[:self.max_len]
            ix = np.sort(ix)
            x[:] = tokens[ix]
        return x, x_len

    def __len__(self):
        return len(self.word2idx)


class FullDataset(Dataset):
    def __init__(self, apply_aug, aug_params, word2idx_path, max_len, root, caps_path, id2idx_cap, idx2id_img, nn_list):
        super().__init__()
        self.imgs_root = root
        self.coco = COCO(caps_path)
        self.ids = list(self.coco.anns.keys())

        self.id2idx_cap = id2idx_cap
        self.idx2id_img = idx2id_img

        self.nn_list = nn_list

        self.cap_transform = CapTransform(word2idx_path, max_len)
        self.n_words = len(self.cap_transform)

        color_jitter = aug_params['color_jitter']
        color_p = aug_params['color_p']
        grayscale_p = aug_params['grayscale_p']
        gaus_blur = aug_params['gaus_blur']
        gaus_p = aug_params['gaus_p']
        crop_size = aug_params['crop_size']
        resize_size = aug_params['resize_size']

        self.target_transform = transforms.Compose([
            transforms.Resize((crop_size, crop_size)),
            transforms.ToTensor()])

        if apply_aug:
            self.nn_transform = transforms.Compose([
                transforms.Resize(resize_size),
                transforms.RandomResizedCrop((crop_size, crop_size)),
                transforms.RandomApply([transforms.ColorJitter(*color_jitter)], p=color_p),
                transforms.RandomApply([transforms.Grayscale(num_output_channels=3)], p=grayscale_p),
                transforms.RandomHorizontalFlip(),
                # transforms.RandomApply([transforms.GaussianBlur(*gaus_blur)], p=gaus_p),
                transforms.ToTensor(),
                transforms.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225]))
            ])

        else:
            self.nn_transform = self.target_transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        ann_id = self.ids[index]

        # caption = a list of indices for a sentence
        cap = self.coco.anns[ann_id]['caption']
        cap, cap_len = self.cap_transform(cap)

        # target image
        img_id = self.coco.anns[ann_id]['image_id']
        img_path = self.coco.loadImgs(img_id)[0]['file_name']
        target_img = Image.open(os.path.join(self.imgs_root, img_path)).convert('RGB')
        target_img = self.target_transform(target_img)

        # nearest images
        cap_id = self.coco.anns[ann_id]['id']
        cap_idx = self.id2idx_cap[cap_id]
        nn_idxs = self.nn_list[cap_idx]
        nn_imgs = []
        for i in nn_idxs:
            nn_id = self.idx2id_img[i]
            nn_img = Image.open(os.path.join(self.imgs_root, nn_id)).convert('RGB')
            nn_imgs.append(self.target_transform(nn_img))
        nn_imgs = torch.cat(nn_imgs, 0)

        return cap, cap_len, target_img, nn_imgs

class RandomDataset(FullDataset):
    def __init__(self, apply_aug, aug_params, word2idx_path, max_len, root, caps_path, id2idx_cap, idx2id_img, nn_list,
                 split='train', len_dset=None):
        super().__init__(apply_aug, aug_params, word2idx_path, max_len, root, caps_path, id2idx_cap, idx2id_img, nn_list)
        self.ids = list(self.coco.imgs)
        self.split = split
        self.len_dset = len_dset

    def __len__(self):
        if self.split == 'train':
            return len(self.ids)
        elif self.split == 'val':
            return self.len_dset

    def __getitem__(self, index):
        img_id = self.ids[index]
        cap_ids = self.coco.getAnnIds(imgIds=img_id)
        cap_id = np.random.choice(cap_ids)

        # caption = a list of indices for a sentence
        cap = self.coco.loadAnns(ids=[cap_id])[0]
        cap = cap['caption']
        cap, cap_len = self.cap_transform(cap)

        # target image
        img_path = self.coco.loadImgs(img_id)[0]['file_name']
        target_img = Image.open(os.path.join(self.imgs_root, img_path)).convert('RGB')
        target_img = self.target_transform(target_img)

        # nearest images
        cap_idx = self.id2idx_cap[cap_id]
        nn_idxs = self.nn_list[cap_idx]
        nn_imgs = []
        for i in nn_idxs:
            nn_id = self.idx2id_img[i]
            nn_img = Image.open(os.path.join(self.imgs_root, nn_id)).convert('RGB')
            nn_imgs.append(self.nn_transform(nn_img))
        nn_imgs = torch.cat(nn_imgs, 0)
        return cap, cap_len, target_img, nn_imgs