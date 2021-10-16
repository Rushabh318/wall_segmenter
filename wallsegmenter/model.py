import torch
import torch.nn as nn

from . import ppm, resnet


# Definining Segmentation Module
class SegmentationModule(nn.Module):
    def __init__(self, net_enc, net_dec, device):
        super(SegmentationModule, self).__init__()
        self.encoder = net_enc
        self.decoder = net_dec
        self.device = device

    def forward(self, feed_dict, seg_size=None):
        pred = self.decoder(
            self.encoder(feed_dict["img_data"].to(self.device)), seg_size=seg_size
        )
        return pred


# Function for building the encoder part of the Segmentation Module
def build_encoder(resnet50_weights_path, encoder_weights_path):
    orig_resnet = resnet.resnet50()
    orig_resnet.load_state_dict(
        torch.load(resnet50_weights_path, map_location=lambda storage, loc: storage),
        strict=False,
    )
    net_encoder = resnet.ResnetDilated(orig_resnet, dilate_scale=8)
    net_encoder.load_state_dict(
        torch.load(encoder_weights_path, map_location=lambda storage, loc: storage),
        strict=False,
    )
    return net_encoder


# Function for building the decoder part of the Segmentation Module
def build_decoder(
    decoder_weights_path,
    train_only_wall=True,
    fc_dim=2048,
    num_class=150,
    use_softmax=True,
):
    net_decoder = ppm.PPM(num_class=num_class, fc_dim=fc_dim, use_softmax=use_softmax)
    # net_decoder.apply(weights_init)

    # When flag "train_only_wall" is set to true, the last layer of decoder is set to have only 2 classes
    if train_only_wall:
        net_decoder.conv_last[4] = torch.nn.Conv2d(512, 2, kernel_size=1)
    net_decoder.load_state_dict(
        torch.load(decoder_weights_path, map_location=lambda storage, loc: storage),
        strict=False,
    )
    return net_decoder


def build_segmenter(
    resnet50_weights_path,
    encoder_weights_path,
    decoder_weights_path,
    device,
    train_only_wall=True,
    decoder_fc_dim=2048,
    decoder_num_class=150,
    decoder_use_softmax=True,
):
    encoder = build_encoder(resnet50_weights_path, encoder_weights_path)
    decoder = build_decoder(
        decoder_weights_path,
        train_only_wall=train_only_wall,
        fc_dim=decoder_fc_dim,
        num_class=decoder_num_class,
        use_softmax=decoder_use_softmax,
    )
    segmenter = SegmentationModule(encoder, decoder, device)
    segmenter.eval()
    segmenter.to(device)
    return segmenter
