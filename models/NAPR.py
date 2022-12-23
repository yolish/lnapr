import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision

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

        self.proj = nn.Linear(backbone_dim, self.rpr_encoder_dim)

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
        self.rel_regressor_x = PoseRegressor(self.rpr_encoder_dim, 3)
        self.rel_regressor_q = PoseRegressor(self.rpr_encoder_dim, 4)

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


    def forward(self, query, refs, ref_pose, encode_refs=True):
        '''

        :param query: N x Cin x H x W
        :param refs: N x K x Cin x H x W
        :param ref_pose: N x 7
        :param encode_refs: boolean, whether to encode the ref images
        :return: p the pose of the query
        '''
        z_query = self.forward_backbone(query) # shape: N x Cout

        if encode_refs:
            n, k, h, w, c = refs.shape
            refs = refs.reshape(n*k, h, w, c)
            z_refs = self.forward_backbone(refs)
            z_refs = z_refs.reshape(n, k, -1)
        else:
            z_refs = refs

        # prepare sequence (query + neighbors)
        # todo consider giving different neighbors for x and q estimation
        z_query = z_query.unsqueeze(0)
        z_refs = z_refs.transpose(0,1) # shape: K x  N x Cout
        seq = torch.cat((z_query, z_refs)) # shape: K+1 x  N x Cout
        # project
        seq = self.proj(seq)

        # aggregate and take output at the query's position (biased towards it)
        z_x = self.ln(self.rpr_transformer_encoder_x(seq))[0]
        z_q = self.ln(self.rpr_transformer_encoder_q(seq))[0]

        # regress the deltas
        delta_x = self.rel_regressor_x(z_x)
        delta_q = self.rel_regressor_x(z_q)

        # compute the ref pose
        x = ref_pose[:, :3] + delta_x
        q = ref_pose[:, 3:] + delta_q

        p = torch.cat((x,q), dim=1)

        return {"pose":p}








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










