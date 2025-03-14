# -*- coding: utf-8 -*-
# @Author  : Monster_Xia
# @Time    : 2023/4/30 15:04
# @Function:

import torch
import torch.nn as nn

class Inner_Encoder(nn.Module):
    def __init__(self, img_size=4, patch_size=2, in_channels=2, embedding_dim=128, num_heads=16, num_layers=3,
                 dropout=0.1):
        super(Inner_Encoder, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embedding_dim = embedding_dim
        self.patch_embed = nn.Conv2d(in_channels, embedding_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embedding_dim))
        self.dropout = nn.Dropout(dropout)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, dropout=dropout),
            num_layers=num_layers
        )
        self.linear = nn.Linear(self.num_patches, 1)

    def forward(self, x):
        x = self.patch_embed(x)  # (batch_size, embedding_dim, num_patches_h, num_patches_w)
        x = x.flatten(2).transpose(1, 2)  # (batch_size, num_patches, embedding_dim)
        x += self.pos_embed
        x = self.dropout(x)
        x = self.transformer_encoder(x)  # (batch_size, num_patches, embedding_dim)
        x = self.linear(x.transpose(1, 2))  # (batch_size, embedding_dim, 1)
        x = x.flatten(1)  # (batch_size, embedding_dim)

        return x


class ImageEncoder(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=2, embedding_dim=128, num_heads=16, num_layers=3,
                 dropout=0.1):
        super(ImageEncoder, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.sqrt_num_patches = img_size // patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embedding_dim = embedding_dim
        self.patch_embed = nn.Conv2d(in_channels, embedding_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embedding_dim))
        self.dropout = nn.Dropout(dropout)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, dropout=dropout),
            num_layers=num_layers
        )
        self.linear = nn.Linear(self.num_patches, 1)
        self.inner_encoder = Inner_Encoder(img_size=patch_size, patch_size=2, in_channels=in_channels,
                                           embedding_dim=embedding_dim, num_heads=num_heads, num_layers=num_layers,
                                           dropout=dropout)

    def forward(self, x):
        split_images = []
        for i in range(self.sqrt_num_patches):
            row_images = torch.chunk(x, self.sqrt_num_patches, dim=2)
            for j in range(self.sqrt_num_patches):
                split_images.append(row_images[j][:, :, :, i * self.patch_size:(i + 1) * self.patch_size])

        x = self.patch_embed(x)  # (batch_size, embedding_dim, num_patches_h, num_patches_w)
        x = x.flatten(2).transpose(1, 2)  # (batch_size, num_patches, embedding_dim)
        x += self.pos_embed  # (batch_size, num_patches, embedding_dim)

        for i, (batch_pic) in enumerate(split_images):
            inner_code = self.inner_encoder(batch_pic)
            x[:, i, :] += inner_code

        nn_input = x
        x = self.dropout(x)
        x = self.transformer_encoder(x)  # (batch_size, num_patches, embedding_dim)
        x = self.linear(x.transpose(1, 2))  # (batch_size, embedding_dim, 1)
        x = x.flatten(1)  # (batch_size, embedding_dim)

        return x, nn_input



class ImageDecoder(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=2, embedding_dim=128, num_heads=16, num_layers=3,
                 dropout=0.1):
        super(ImageDecoder, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        self.sqrt_num_patches = img_size // patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embedding_dim))
        self.unpatch_embed = nn.ConvTranspose2d(embedding_dim, in_channels, kernel_size=patch_size, stride=patch_size)
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=embedding_dim, nhead=num_heads, dropout=dropout),
            num_layers=num_layers
        )
        self.linear = nn.Linear(embedding_dim, embedding_dim * self.num_patches)

    def generate_square_subsequent_mask(self, size):
        mask = torch.triu(torch.ones(size, size), diagonal=1)
        mask = mask.float().masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(self, target, x):
        x = self.linear(x)
        x = torch.reshape(x, [-1, self.num_patches, self.embedding_dim])

        tgt_mask = self.generate_square_subsequent_mask(target.size(0))
        tgt_mask = tgt_mask.to(x.device)

        x = self.transformer_decoder(target, x, tgt_mask=tgt_mask)
        x -= self.pos_embed  # (batch_size, num_patches, embedding_dim)
        x = x.transpose(1, 2).reshape(-1, self.embedding_dim, self.sqrt_num_patches, self.sqrt_num_patches)
        x = self.unpatch_embed(x)

        return x


class ImageTransformer(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=2, embedding_dim=128, num_heads=4, num_layers=3,
                 dropout=0.1):
        super().__init__()
        self.encoder = ImageEncoder(img_size=img_size, patch_size=patch_size, in_channels=in_channels,
                                    embedding_dim=embedding_dim, num_heads=num_heads, num_layers=num_layers,
                                    dropout=dropout)
        self.decoder = ImageDecoder(img_size=img_size, patch_size=patch_size, in_channels=in_channels,
                                    embedding_dim=embedding_dim, num_heads=num_heads, num_layers=num_layers,
                                    dropout=dropout)

    def forward(self, x):
        coded, nn_input = self.encoder(x)
        x = self.decoder(nn_input, coded)
        return x