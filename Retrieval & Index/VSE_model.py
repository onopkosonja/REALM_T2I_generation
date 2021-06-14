import torch
from torch import nn
from torchvision import models
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.autograd import Variable
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def l2norm(x):
    norm = torch.norm(x, dim=1, keepdim=True)
    return x / norm


class ImageEncoder(nn.Module):
    """Pre-trained resnet-152. Fine-tuning. Image representation from penultimate FC layer(dim=2048) ->
    joint embedding space dim=1024. Normalization in joint embedding space.
    """

    def __init__(self, model_name, model_path, emb_size):
        super().__init__()
        self.cnn = self.load_cnn(model_name, model_path, emb_size)

    def load_cnn(self, model_name, model_path, embed_dim):
        net = models.__dict__[model_name]()
        net.load_state_dict(torch.load(model_path))
        net.fc = nn.Linear(net.fc.in_features, embed_dim)
        return net

    def forward(self, imgs):
        embeds = self.cnn(imgs)
        embeds = l2norm(embeds)
        return embeds


class CaptionEncoder(nn.Module):
    """ GRU based LM. Word embeddings dim=300. GRU hidden dim=1024.
    """

    def __init__(self, pretrained_embs, word_dim, emb_size, num_layers, padding_value):
        super().__init__()
        self.emb_size = emb_size
        self.padding_value = padding_value
        self.word_embedder = nn.Embedding.from_pretrained(pretrained_embs)
        self.rnn = nn.GRU(word_dim, emb_size, num_layers, batch_first=True)
        self.h0 = nn.Parameter(torch.randn(1, 1, emb_size).type(torch.FloatTensor), requires_grad=True)

    def forward(self, caps, lengths):
        lengths_cpu = torch.as_tensor(lengths, dtype=torch.int64, device='cpu')
        word_embs = self.word_embedder(caps)
        packed_embs = pack_padded_sequence(word_embs, lengths_cpu, batch_first=True, enforce_sorted=False)

        h0 = self.h0.repeat(1, len(caps), 1)
        out, _ = self.rnn(packed_embs, h0)

        cap_embs, _ = pad_packed_sequence(out, batch_first=True, padding_value=self.padding_value)

        # taking final embedding of input sentence aka embs with index = (lenghts - 1)
        # reshape so that output tensor has size = (batch_size, embed_dim)
        I = lengths.view(-1, 1, 1)
        I = Variable(I.expand(caps.size(0), 1, self.emb_size) - 1)
        out = torch.gather(cap_embs, 1, I).squeeze(1)

        return out


class ContrastiveLoss(nn.Module):
    """Triplet loss: either max of hinges(MH) or sum of hinges(SH) in batch. Similarity function is cosine distance.
    """

    def __init__(self, margin, mh_loss=False):
        super().__init__()
        self.margin = margin
        self.sim = self.cosine_dist
        self.mh_loss = mh_loss

    def cosine_dist(self, x1, x2):
        return torch.mm(x1, x2.T)

    def forward(self, imgs, caps):
        scores = self.sim(imgs, caps)
        d = scores.diag().view(imgs.size(0), 1)  # positive pair score
        d1 = d.expand_as(scores)  # img-caption matrix filled with pos (img, caption) score
        d2 = d.t().expand_as(scores)  # img-caption matrix filled with pos (caption, img) score

        # positive pair score + margin VS negative pairs score (non-negative)
        # -> compute loss for all possible pairs (img, caption) и (caption, img)
        cost_capt = (self.margin + scores - d1).clamp(min=0)  # caption retrieval
        cost_img = (self.margin + scores - d2).clamp(min=0)  # image retrieval

        # diagonal -> 0
        mask = torch.eye(scores.size(0), dtype=torch.bool).to(device)
        I = Variable(mask)
        cost_capt = cost_capt.masked_fill_(I, 0)
        cost_img = cost_img.masked_fill_(I, 0)

        if self.mh_loss:
            # find negative with max score in img-caption triplet loss matrices
            cost_capt = cost_capt.max(1)[0]
            cost_img = cost_img.max(0)[0]

        return (cost_capt + cost_img).sum()


class VSE(nn.Module):
    def __init__(self, img_enc_params, text_enc_params, margin, sh_epochs):
        super().__init__()
        self.img_enc = ImageEncoder(**img_enc_params)
        self.text_enc = CaptionEncoder(**text_enc_params)
        self.sh_epochs = sh_epochs
        self.sh_loss = ContrastiveLoss(margin)
        self.mh_loss = ContrastiveLoss(margin, mh_loss=True)

    def forward(self, imgs, caps, lengths, epoch, mode):
        img_emb = self.img_enc(imgs)
        cap_emb = self.text_enc(caps, lengths)
        if epoch <= self.sh_epochs:
            loss = self.sh_loss(img_emb, cap_emb)
        else:
            loss = self.mh_loss(img_emb, cap_emb)

        if mode == 'train':
            return loss
        return loss, img_emb, cap_emb
