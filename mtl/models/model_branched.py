import torch
import torch.nn.functional as F

from mtl.models.model_parts import Encoder, get_encoder_channel_counts, ASPP, DecoderDeeplabV3p


class ModelBranched(torch.nn.Module):
    def __init__(self, cfg, outputs_desc):
        super().__init__()
        self.outputs_desc = outputs_desc
        #1 channel for depth and n channels for n semantic segmentation classes
        ch_out = sum(outputs_desc.values())

        self.encoder = Encoder(
            cfg.model_encoder_name,
            pretrained=True,
            zero_init_residual=True,
            replace_stride_with_dilation=(False, False, True),
        )

        ch_out_encoder_bottleneck, ch_out_encoder_4x = get_encoder_channel_counts(cfg.model_encoder_name)

        #ASPP and decoder for semantic segmentation task
        self.aspp_semseg = ASPP(ch_out_encoder_bottleneck, 256)

        self.decoder_semseg = DecoderDeeplabV3p(256, ch_out_encoder_4x, ch_out-1)

        #ASPP and decoder for depth estimation task
        self.aspp_depth = ASPP(ch_out_encoder_bottleneck, 256)

        self.decoder_depth = DecoderDeeplabV3p(256, ch_out_encoder_4x, 1)

    def forward(self, x):
        input_resolution = (x.shape[2], x.shape[3])

        features = self.encoder(x)

        # Uncomment to see the scales of feature pyramid with their respective number of channels.
        print(", ".join([f"{k}:{v.shape[1]}" for k, v in features.items()]))
        lowest_scale = max(features.keys())

        features_lowest = features[lowest_scale]

        #Semantic Segmentation
        features_tasks_semseg = self.aspp_semseg(features_lowest)
        predictions_4x_semseg,_ = self.decoder_semseg(features_tasks_semseg, features[4])

        predictions_1x_semseg = F.interpolate(predictions_4x_semseg, size=input_resolution, mode='bilinear', align_corners=False)

        #Depth estimation
        features_tasks_depth = self.aspp_depth(features_lowest)
        predictions_4x_depth,_ = self.decoder_depth(features_tasks_depth, features[4])

        predictions_1x_depth = F.interpolate(predictions_4x_depth, size=input_resolution, mode='bilinear', align_corners=False)
        out = {}
        offset = 0

        for task, num_ch in self.outputs_desc.items():
            if task == 'semseg':
                out[task] = predictions_1x_semseg[:, offset:offset+num_ch, :, :]
                offset += num_ch
            elif task == 'depth':
                out[task] = predictions_1x_depth[:,:, :, :]
                offset += num_ch

            else:
                print("mod erreur, should be either MOD_SEMSEG or MOD_DEPTH")
        return out
