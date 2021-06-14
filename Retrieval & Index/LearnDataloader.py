from torch.utils.data import DataLoader, Dataset
from pycocotools.coco import COCO
from PIL import Image
from torchvision.datasets import CocoCaptions
import torchvision.transforms as transforms
import os
import torch
import pandas as pd
import multiprocessing


def image_transform(img_size, crop_size):
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomResizedCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])


class CaptionTransform:
    def __init__(self, voc, mode):
        self.voc = voc
        self.mode = mode

    def __call__(self, cap):
        if self.mode == 'list':
            return [torch.LongTensor(self.voc(c)) for c in cap]
        return torch.LongTensor(self.voc(cap))


def collate_fn_for_custom(data):
    imgs, caps = zip(*data)
    imgs_tensor = torch.stack(imgs, 0)

    lengths = [len(c) for c in caps]
    max_len = max(lengths)
    caps_tensor = torch.zeros(size=(len(caps), max_len),  dtype=torch.long)
    for i, c in enumerate(caps):
        l = lengths[i]
        caps_tensor[i, :l] = c
    return imgs_tensor, caps_tensor, torch.LongTensor(lengths).cpu()


def collate_fn_coco_captions( data):
    imgs, queries = zip(*data)

    imgs_tensor = torch.stack(imgs, 0)
    imgs_tensor = torch.repeat_interleave(imgs_tensor, repeats=5, dim=0)

    lengths = [len(c) for q in queries for c in q[:5]]
    max_len = max(lengths)
    caps_tensor = torch.zeros(size=(5 * len(queries), max_len), dtype=torch.long)

    for index, query in enumerate(queries):
        for i, cap in enumerate(query[:5]):
            l = lengths[5 * index + i]
            caps_tensor[5 * index + i, :l] = cap
    return imgs_tensor, caps_tensor, torch.LongTensor(lengths).cpu()


class CocoDataset(Dataset):
    def __init__(self, img_root, cap_path, img_transform, cap_transfrom):
        super().__init__()
        self.imgs_root = img_root
        self.caps_path = COCO(cap_path)
        self.ids = list(self.caps_path.anns.keys())
        self.img_transform = img_transform
        self.cap_transfrom = cap_transfrom

    def __getitem__(self, index):
        ann_id = self.ids[index]
        path = self.caps_path
        img_id = path.anns[ann_id]['image_id']
        img_path = path.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.imgs_root, img_path)).convert('RGB')
        cap = path.anns[ann_id]['caption']

        img = self.img_transform(img)
        cap = self.cap_transfrom(cap)
        return img, cap

    def __len__(self):
        return len(self.ids)


class OIDataset(Dataset):
    def __init__(self, img_root, cap_path, img_transform, cap_transfrom):
        self.img_root = img_root
        self.img_transform = img_transform
        self.cap_transfrom = cap_transfrom

        self.anns = pd.read_json(path_or_buf=cap_path, lines=True)
        self.anns = self.anns[['caption', 'image_id']]
        self.anns.drop_duplicates(subset='image_id', inplace=True, ignore_index=True)
        self.ids = self.anns

    def __getitem__(self, index):
        item = self.anns.loc[index]
        cap = item.caption
        img = Image.open(os.path.join(self.img_root, item.image_id + '.jpg')).convert('RGB')

        img = self.img_transform(img)
        cap = self.cap_transfrom(cap)
        return img, cap

    def __len__(self):
        return len(self.ids)

def get_dataloader(dataset_name, mode, voc, img_root, cap_path, img_size, crop_size, batch_size):
    cap_mode = 'list' if dataset_name == 'coco' and mode == 'val' else None
    dataset_params = {
        'img_root': img_root,
        'cap_path': cap_path,
        'img_transform': image_transform(img_size, crop_size),
        'cap_transfrom': CaptionTransform(voc, cap_mode)
    }

    if dataset_name == 'coco' and mode == 'train':
        dataset = CocoDataset(**dataset_params)
        collate_fn = collate_fn_for_custom
    elif dataset_name == 'coco' and mode == 'val':
        dataset = CocoCaptions(root=dataset_params['img_root'],
                               annFile=dataset_params['cap_path'],
                               transform=dataset_params['img_transform'],
                               target_transform=dataset_params['cap_transfrom'])
        collate_fn = collate_fn_coco_captions
        batch_size = 200
    elif dataset_name == 'open_images':
        dataset = OIDataset(**dataset_params)
        collate_fn = collate_fn_for_custom

    if mode == 'train':
        shuffle = True
    else:
        shuffle = False

    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=multiprocessing.cpu_count(),
                            pin_memory=True,
                            collate_fn=collate_fn)
    return dataloader
