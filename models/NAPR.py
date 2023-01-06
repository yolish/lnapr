import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision

import torch
import torch.nn as nn
import torch.nn.functional as F

# positional encoding from nerf
def positional_encoding(
    tensor, num_encoding_functions=6, include_input=True, log_sampling=True
) -> torch.Tensor:
    r"""Apply positional encoding to the input.
    Args:
        tensor (torch.Tensor): Input tensor to be positionally encoded.
        encoding_size (optional, int): Number of encoding functions used to compute
            a positional encoding (default: 6).
        include_input (optional, bool): whether or not to include the input in the
            positional encoding (default: True).
    Returns:
    (torch.Tensor): Positional encoding of the input tensor.
    """
    # TESTED
    # Trivially, the input tensor is added to the positional encoding.
    encoding = [tensor] if include_input else []
    frequency_bands = None
    if log_sampling:
        frequency_bands = 2.0 ** torch.linspace(
            0.0,
            num_encoding_functions - 1,
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )
    else:
        frequency_bands = torch.linspace(
            2.0 ** 0.0,
            2.0 ** (num_encoding_functions - 1),
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )

    for freq in frequency_bands:
        for func in [torch.sin, torch.cos]:
            encoding.append(func(tensor * freq))

    # Special case, for no positional encoding
    if len(encoding) == 1:
        return encoding[0]
    else:
        return torch.cat(encoding, dim=-1)


# PoseEncoder implementation from: https://github.com/yolish/camera-pose-auto-encoders/blob/main/models/pose_encoder.py
class PoseEncoder(nn.Module):

    def __init__(self, encoder_dim, apply_positional_encoding=True,
                 num_encoding_functions=6, shallow_mlp=False):

        super(PoseEncoder, self).__init__()
        self.apply_positional_encoding = apply_positional_encoding
        self.num_encoding_functions = num_encoding_functions
        self.include_input = True
        self.log_sampling = True
        x_dim = 3
        q_dim = 4
        if self.apply_positional_encoding:
            x_dim = x_dim + self.num_encoding_functions * x_dim * 2
            q_dim = q_dim + self.num_encoding_functions * q_dim * 2
        if shallow_mlp:
            self.x_encoder = nn.Sequential(nn.Linear(x_dim, 64), nn.ReLU(),
                                           nn.Linear(64,encoder_dim))
            self.q_encoder = nn.Sequential(nn.Linear(q_dim, 64), nn.ReLU(),
                                           nn.Linear(64,encoder_dim))
        else:
            self.x_encoder = nn.Sequential(nn.Linear(x_dim, 64), nn.ReLU(),
                                           nn.Linear(64,128),
                                           nn.ReLU(),
                                           nn.Linear(128,256),
                                           nn.ReLU(),
                                           nn.Linear(256, encoder_dim)
                                           )
            self.q_encoder = nn.Sequential(nn.Linear(q_dim, 64), nn.ReLU(),
                                           nn.Linear(64, 128),
                                           nn.ReLU(),
                                           nn.Linear(128, 256),
                                           nn.ReLU(),
                                           nn.Linear(256, encoder_dim)
                                       )



        self.x_dim = x_dim
        self.q_dim = q_dim
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, pose):
        if self.apply_positional_encoding:
            encoded_x = positional_encoding(pose[:, :3])
            encoded_q = positional_encoding(pose[:, 3:])
        else:
            encoded_x = pose[:, :3]
            encoded_q = pose[:, 3:]

        latent_x = self.x_encoder(encoded_x)
        latent_q = self.q_encoder(encoded_q)
        return latent_x, latent_q




def batch_dot(v1, v2):
    """
    Dot product along the dim=1
    :param v1: (torch.tensor) Nxd tensor
    :param v2: (torch.tensor) Nxd tensor
    :return: N x 1
    """
    out = torch.mul(v1, v2)
    out = torch.sum(out, dim=1, keepdim=True)
    return out

def qmult(quat_1, quat_2):
    """
    Perform quaternions multiplication
    :param quat_1: (torch.tensor) Nx4 tensor
    :param quat_2: (torch.tensor) Nx4 tensor
    :return: quaternion product
    """
    # Extracting real and virtual parts of the quaternions
    q1s, q1v = quat_1[:, :1], quat_1[:, 1:]
    q2s, q2v = quat_2[:, :1], quat_2[:, 1:]

    qs = q1s*q2s - batch_dot(q1v, q2v)
    qv = q1v.mul(q2s.expand_as(q1v)) + q2v.mul(q1s.expand_as(q2v)) + torch.cross(q1v, q2v, dim=1)
    q = torch.cat((qs, qv), dim=1)
    return q

def compute_abs_pose_torch(rel_pose, abs_pose_neighbor):
    abs_pose_query = torch.zeros_like(rel_pose)
    abs_pose_query[:, :3] = abs_pose_neighbor[:, :3] + rel_pose[:, :3]
    abs_pose_query[:, 3:] = qmult(abs_pose_neighbor[:, 3:], rel_pose[:, 3:])
    return abs_pose_query


class PoseRegressor(nn.Module):
    """ A simple MLP to regress a pose component"""

    def __init__(self, decoder_dim, output_dim, use_prior=False):
        """
        decoder_dim: (int) the input dimension
        output_dim: (int) the outpur dimension
        use_prior: (bool) whether to use prior information
        """
        super().__init__()
        ch = 1024
        self.fc_h = nn.Linear(decoder_dim, ch)
        self.use_prior = use_prior
        if self.use_prior:
            self.fc_h_prior = nn.Linear(decoder_dim * 2, ch)
        self.fc_o = nn.Linear(ch, output_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        """
        Forward pass
        """
        if self.use_prior:
            x = F.gelu(self.fc_h_prior(x))
        else:
            x = F.gelu(self.fc_h(x))

        return self.fc_o(x)

class NAPR(nn.Module):
    def __init__(self, config, backbone_path):
        super().__init__()
        self.avg_pooling_2d = nn.AdaptiveAvgPool2d(1)
        backbone_type = config.get("rpr_backbone_type")
        if backbone_type == "resnet50":
            backbone = torchvision.models.resnet50(pretrained=False)
            backbone.load_state_dict(torch.load(backbone_path))
            self.backbone = nn.Sequential(*(list(backbone.children())[:-2]))
            backbone_dim = 2048

        elif backbone_type == "resnet34":
            backbone = torchvision.models.resnet34(pretrained=False)
            backbone.load_state_dict(torch.load(backbone_path, map_location="cpu"))
            self.backbone = nn.Sequential(*(list(backbone.children())[:-2]))
            backbone_dim = 512

        elif backbone_type == "mobilenet":
            backbone = torchvision.models.mobilenet_v2(pretrained=False)
            backbone.load_state_dict(torch.load(backbone_path, map_location="cpu"))
            self.backbone = nn.Sequential(*list(backbone.children())[:-1])
            backbone_dim = 1280

        elif backbone_type == "efficientnet":
            # Efficient net
            self.backbone = torch.load(backbone_path)
            backbone_dim = 1280
        else:
            raise NotImplementedError("backbone type: {} not supported".format(backbone_type))
        self.backbone_type = backbone_type


        # Encoders
        self.rpr_encoder_dim = config.get("rpr_hidden_dim")
        rpr_num_heads = config.get("rpr_num_heads")
        rpr_dim_feedforward = config.get("rpr_dim_feedforward")
        rpr_dropout = config.get("rpr_dropout")
        rpr_activation = config.get("rpr_activation")
        self.num_neighbors = config.get("num_neighbors")

        # The learned pose token for delta_x and delta_q
        self.pose_token_embed_dx = nn.Parameter(torch.zeros((1, backbone_dim)), requires_grad=True)
        self.pose_token_embed_dq = nn.Parameter(torch.zeros((1, backbone_dim)), requires_grad=True)

        # The pose encoder
        self.pose_encoder = PoseEncoder(backbone_dim)

        self.proj = nn.Linear(backbone_dim, self.rpr_encoder_dim)
        self.proj_x = nn.Linear(backbone_dim, self.rpr_encoder_dim)
        self.proj_q = nn.Linear(backbone_dim, self.rpr_encoder_dim)

        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=self.rpr_encoder_dim,
                                                               nhead=rpr_num_heads,
                                                               dim_feedforward=rpr_dim_feedforward,
                                                               dropout=rpr_dropout,
                                                               activation=rpr_activation)

        self.rpr_transformer_encoder_x = nn.TransformerEncoder(transformer_encoder_layer,
                                                             num_layers=config.get("rpr_num_encoder_layers"),
                                                             norm=nn.LayerNorm(self.rpr_encoder_dim))
        self.rpr_transformer_encoder_q = nn.TransformerEncoder(transformer_encoder_layer,
                                                               num_layers=config.get("rpr_num_encoder_layers"),
                                                               norm=nn.LayerNorm(self.rpr_encoder_dim))

        self.ln = nn.LayerNorm(self.rpr_encoder_dim)
        self.rel_regressor_x = PoseRegressor(self.rpr_encoder_dim*2, 3)
        self.rel_regressor_q = PoseRegressor(self.rpr_encoder_dim*2, 4)

        # Initialize FC layers
        for m in list(self.modules()):
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)

    def forward_backbone(self, img):
        '''
        Returns a global latent encoding of the img
        :param img: N x Cin x H x W
        :return: z: N X Cout
        '''
        if self.backbone_type == "efficientnet":
            z = self.backbone.extract_features(img)
        else:
            z = self.backbone(img)
        z = self.avg_pooling_2d(z).flatten(start_dim=1)
        return z


    def forward(self, data, encode_refs=True):
        '''
        :param query: N x Cin x H x W
        :param refs: N x K x Cin x H x W
        :param ref_pose: N x 7
        :param encode_refs: boolean, whether to encode the ref images
        :return: p the pose of the query
        '''
        query = data.get('query')
        refs = data.get('knn')
        ref_pose = data.get('ref_pose')

        z_query = self.forward_backbone(query) # shape: N x Cout

        if encode_refs:
            n, k, h, w, c = refs.shape
            refs = refs.reshape(n*k, h, w, c)
            z_refs = self.forward_backbone(refs)
            z_refs = z_refs.reshape(n, k, -1)
        else:
            z_refs = refs

        # Prepare sequence (learned_token + neighbors)
        # todo consider giving different neighbors for x and q estimation
        z_refs = z_refs.transpose(0,1) # shape: K x  N x Cout
        bs = z_refs.shape[1]

        # Encode the reference pose and add to the learned tokens
        enc_x, enc_q = self.pose_encoder(ref_pose) # N x Cout, N x Cout
        pose_token_embed_dx = self.pose_token_embed_dx.unsqueeze(1).repeat(1, bs, 1) + enc_x.unsqueeze(0)
        pose_token_embed_dq = self.pose_token_embed_dq.unsqueeze(1).repeat(1, bs, 1) + enc_q.unsqueeze(0)
        seq_x = torch.cat((pose_token_embed_dx, z_query.unsqueeze(0), z_refs)) # shape: K+1 x  N x Cout
        seq_q = torch.cat((pose_token_embed_dq, z_query.unsqueeze(0), z_refs))  # shape: K+1 x  N x Cout

        # Project
        seq_x = self.proj_x(seq_x)
        seq_q = self.proj_q(seq_q)
        #seq_x = self.proj_x(seq_x)
        #seq_q = self.proj_q(seq_q)

        # Aggregate neighbors and take output at the learNable token position - giving a latent repr. of the env.
        z_x_scene = self.ln(self.rpr_transformer_encoder_x(seq_x))[0]
        z_q_scene = self.ln(self.rpr_transformer_encoder_q(seq_q))[0]

        # Concat the env. repr for x and q with the query latent # TODO consider outputting two latent for the query
        z_query = self.proj(z_query)
        z_x = torch.cat((z_x_scene, z_query), dim=1)
        z_q = torch.cat((z_q_scene, z_query), dim=1)

        # regress the deltas
        delta_x = self.rel_regressor_x(z_x)
        delta_q = self.rel_regressor_q(z_q)
        p = torch.cat((delta_x, delta_q), dim=1)
        pose_neigh = torch.zeros((bs, self.num_neighbors, 7)).to(ref_pose.device)
        for i in range(self.num_neighbors):
            z_ref = z_refs[i]
            z_ref = self.proj(z_ref)
            z_x = torch.cat((z_x_scene, z_ref), dim=1)
            z_q = torch.cat((z_q_scene, z_ref), dim=1)
            delta_x = self.rel_regressor_x(z_x)
            delta_q = self.rel_regressor_q(z_q)
            p_neigh = torch.cat((delta_x, delta_q), dim=1)
            pose_neigh[:, i, :] = p_neigh
        # compute the ref pose
        #x = ref_pose[:, :3] + delta_x
        #q = ref_pose[:, 3:] + delta_q
        #q = qmult(ref_pose[:, 3:], delta_q)

        #p = torch.cat((delta_x,delta_q), dim=1)

        return {"pose":p, "pose_neigh":pose_neigh }

'''
class NSRPR(nn.Module):
    def __init__(self, config):
        super().__init__()
        input_dim = config.get("input_dim")
        d_model = config.get("d_model")
        nhead = config.get("nhead")
        dim_feedforward = config.get("dim_feedforward")
        dropout = config.get("dropout")
        transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                                               dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        self.transformer_decoder = nn.TransformerDecoder(transformer_decoder_layer,
                                                               num_layers=config.get("num_decoder_layers"),
                                                               norm=nn.LayerNorm(d_model))
        self.ln = nn.LayerNorm(d_model)
        self.proj = nn.Linear(input_dim, d_model)
        self.cls1 = nn.Sequential(nn.Linear(d_model, 1), nn.LogSoftmax(dim=1))
        self.cls2 = nn.Sequential(nn.Linear(d_model, 1), nn.LogSoftmax(dim=1))
        self.rel_regressor_x = PoseRegressor(d_model, 3)
        self.rel_regressor_q = PoseRegressor(d_model, 4)
        # Initialize FC layers
        for m in list(self.modules()):
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
    def forward(self, query, knn):
        knn = self.proj(knn)
        query = self.proj(query)
        # apply first classifier on knn
        knn_distr_before = self.cls1(knn)
        # apply decoder
        out = self.ln(self.transformer_decoder(knn, query.unsqueeze(1)))
        # apply second classifier on decoders outputs
        knn_distr_after = self.cls2(out)
        # apply regressors
        returned_value = {}
        num_neighbors = knn.shape[1]
        for i in range(num_neighbors):
            rel_x = self.rel_regressor_x(out[:, i, :])
            rel_q = self.rel_regressor_q(out[:, i, :])
            returned_value["rel_pose_{}".format(i)] = torch.cat((rel_x, rel_q), dim=1)
        returned_value["knn_distr_before"] = knn_distr_before
        returned_value["knn_distr_after"] = knn_distr_after
        # return the relative poses and the log-softmax from the first and second classifier
        return returned_value
class NS2RPR(nn.Module):
    def __init__(self, config):
        super().__init__()
        input_dim = config.get("input_dim")
        d_model = config.get("d_model")
        nhead = config.get("nhead")
        dim_feedforward = config.get("dim_feedforward")
        dropout = config.get("dropout")
        transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                                               dropout=dropout, activation='gelu', batch_first=True)
        self.transformer_decoder_rot = nn.TransformerDecoder(transformer_decoder_layer,
                                                               num_layers=config.get("num_decoder_layers"),
                                                               norm=nn.LayerNorm(d_model))
        self.ln = nn.LayerNorm(d_model)
        self.proj = nn.Linear(input_dim, d_model)
        self.cls1 = nn.Sequential(nn.Linear(d_model, 1), nn.LogSoftmax(dim=1))
        self.cls2 = nn.Sequential(nn.Linear(d_model, 1), nn.LogSoftmax(dim=1))
        self.rel_regressor_x = PoseRegressor(d_model, 3)
        self.rel_regressor_q = PoseRegressor(d_model, 4)
        # Initialize FC layers
        for m in list(self.modules()):
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
    def forward(self, query, knn):
        knn = self.proj(knn)
        query = self.proj(query)
        # apply first classifier on knn
        knn_distr_before = self.cls1(knn)
        # apply decoder
        out = self.ln(self.transformer_decoder(knn, query.unsqueeze(1)))
        # apply second classifier on decoders outputs
        knn_distr_after = self.cls2(out)
        # apply regressors
        returned_value = {}
        num_neighbors = knn.shape[1]
        for i in range(num_neighbors):
            rel_x = self.rel_regressor_x(out[:, i, :])
            rel_q = self.rel_regressor_q(out[:, i, :])
            returned_value["rel_pose_{}".format(i)] = torch.cat((rel_x, rel_q), dim=1)
        returned_value["knn_distr_before"] = knn_distr_before
        returned_value["knn_distr_after"] = knn_distr_after
        # return the relative poses and the log-softmax from the first and second classifier
        return returned_value
'''