import torch
import torch.nn as nn
import torch.nn.functional as F


class NetVLAD(nn.Module):
    """
    A class to represent a NetVLAD architecture, as described in:
    'NetVLAD: CNN architecture of weakly supervised recognition' Arandjelovix et al
    A NetVLAD implementation based on the following sources:
    https://github.com/Nanne/pytorch-NetVlad - Pytorch NetVLAD without whitening and PCA and with different normalization
    https://github.com/Relja/netvlad/blob/master/relja_simplenn_tidy.m - Original implementation in Matlab
    https://github.com/uzh-rpg/netvlad_tf_open - Tensorflow NetVLAD (later used as a reference for HFNet)
    """

    def __init__(self):
        """
        NetVLAD constructor
        """
        super(NetVLAD, self).__init__()

        # vgg-16 encoder
        self._conv1_1 = nn.Conv2d(3, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self._relu1_1 = nn.ReLU()
        self._conv1_2 = nn.Conv2d(64, 64, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self._pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False)
        self._relu1_2 = nn.ReLU()
        self._conv2_1 = nn.Conv2d(64, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self._relu2_1 = nn.ReLU()
        self._conv2_2 = nn.Conv2d(128, 128, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self._pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False)
        self._relu2_2 = nn.ReLU()
        self._conv3_1 = nn.Conv2d(128, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self._relu3_1 = nn.ReLU()
        self._conv3_2 = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self._relu3_2 = nn.ReLU()
        self._conv3_3 = nn.Conv2d(256, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self._pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False)
        self._relu3_3 = nn.ReLU()
        self._conv4_1 = nn.Conv2d(256, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self._relu4_1 = nn.ReLU()
        self._conv4_2 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self._relu4_2 = nn.ReLU()
        self._conv4_3 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self._pool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False)
        self._relu4_3 = nn.ReLU()
        self._conv5_1 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self._relu5_1 = nn.ReLU()
        self._conv5_2 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))
        self._relu5_2 = nn.ReLU()
        self._conv5_3 = nn.Conv2d(512, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1))

        # VLAD layer
        self._num_clusters = 64
        self._dim = 512
        self._centroids = nn.Parameter(torch.rand(self._num_clusters, self._dim))
        self._vlad_conv = nn.Conv2d(self._dim, self._num_clusters, kernel_size=(1, 1), bias=False)
        self._centroids = nn.Parameter(torch.rand(self._num_clusters, self._dim))

        # Whitening and pca
        self._wpca = nn.Conv2d(32768, 4096, kernel_size=[1, 1])

    def vgg16_descriptor(self, x0, latent_return_layers=None):
        """
        Embed an input image with the VGG16 encoder
        :param x0: (torch.tensor) input image batch of size N
        :param latent_return_layers: list<str> list of names of latent layers, for which the latent representations
            should be returned
        :return: Nx 512-dimensional feature map
        """
        latent_return_repr = []
        x1 = self._conv1_1(x0)
        x2 = self._relu1_1(x1)
        x3 = self._conv1_2(x2)
        x4 = self._pool1(x3)
        x5 = self._relu1_2(x4)
        x6 = self._conv2_1(x5)
        x7 = self._relu2_1(x6)
        x8 = self._conv2_2(x7)
        x9 = self._pool2(x8)
        x10 = self._relu2_2(x9)
        x11 = self._conv3_1(x10)
        x12 = self._relu3_1(x11)
        x13 = self._conv3_2(x12)
        x14 = self._relu3_2(x13)
        x15 = self._conv3_3(x14)
        x16 = self._pool3(x15)
        x17 = self._relu3_3(x16)
        x18 = self._conv4_1(x17)
        x19 = self._relu4_1(x18)
        x20 = self._conv4_2(x19)
        x21 = self._relu4_2(x20)
        x22 = self._conv4_3(x21)
        x23 = self._pool4(x22)
        x24 = self._relu4_3(x23)
        x25 = self._conv5_1(x24)
        x26 = self._relu5_1(x25)
        x27 = self._conv5_2(x26)
        x28 = self._relu5_2(x27)
        x29 = self._conv5_3(x28)

        if latent_return_layers is not None:
            latent_map = {"conv3":x15, "conv4":x22, "conv5":x29}
            for layer_name in latent_return_layers:
                latent_return_repr.append((latent_map.get(layer_name)))
        return x29, latent_return_repr


    def vlad_descriptor(self, x):
        """
        Apply a VLAD layer on an embedded tensor, followed by dimensionality reduction, whitening and normalization
        :param x: embedded tensor
        :return: 32-K tenstor (VLAD descriptor before dimensionality reduction)
        """
        batch_size, orig_num_of_channels = x.shape[:2]

        # Normalize across the descriptors dimension
        x = F.normalize(x, p=2, dim=1)

        # Soft-assignment of descriptors to cluster centers
        # NxDxWxH map interpreted as NxDxK descriptors
        soft_assign = self._vlad_conv(x).view(batch_size, self._num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1)

        # VLAD core
        x_flatten = x.view(batch_size, orig_num_of_channels, -1)

        # Centroids are originally saved with minus sign, so the following operation translates to: x - C
        vlad = x_flatten.unsqueeze(0).permute(1, 0, 2, 3) + self._centroids.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)

        vlad *= soft_assign.unsqueeze(2)
        vlad = vlad.sum(dim=-1)

        # Intra-normalization (implemented as in matchconvnet)
        vlad = self.matconvnet_normalize(vlad, dim=2)
        vlad = vlad.permute(0,2,1)
        vlad = vlad.flatten(1)

        # L2 post normalization
        vlad = self.matconvnet_normalize(vlad, dim=1)

        return vlad # Shape: NX32k

    def reduced_vlad_descriptor(self, vlad_desc):
        reduced_vlad_desc = self._wpca(vlad_desc.reshape((*vlad_desc.shape, 1, 1)))
        reduced_vlad_desc = reduced_vlad_desc.view(reduced_vlad_desc.shape[0], reduced_vlad_desc.shape[1])
        reduced_vlad_desc = F.normalize(reduced_vlad_desc, p=2, dim=1)
        return reduced_vlad_desc

    def matconvnet_normalize(self, x, dim, eps=1e-12):
        denom = torch.sqrt(torch.sum(x**2, dim=dim, keepdim=True) + eps)
        return x/denom


    def forward(self, img, latent_return_layers=None):
        """
       Forward pass of the network
       :param img: (torch.tensor) input image batch of size N
       :param latent_return_layers: list<str> list of names of latent layers, for which the latent representations
            should be returned
       :return: a dictionary containing the following:
                    (1) Nx4096 VLAD descriptors
                    (2) Nx32K VLAD descriptors
                    (3) dictionary of latent representations whose keys are the strings given latent_return_layers
                        and its values are the corresponding latent representations
        """
        vgg_desc, latent_reprs = self.vgg16_descriptor(img, latent_return_layers=latent_return_layers)
        vlad_desc = self.vlad_descriptor(vgg_desc)
        reduced_vlad_desc = self.reduced_vlad_descriptor(vlad_desc)

        res = {'global_desc': reduced_vlad_desc,
               'raw_global_desc': vlad_desc,
               'latent_reprs': latent_reprs,
               'input_size': img.shape[2:]}
        return res




