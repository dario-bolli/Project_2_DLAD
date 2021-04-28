import torch
import torch.nn.functional as F

from mtl.models.model_parts import Encoder, get_encoder_channel_counts, ASPP, DecoderDeeplabV3p,SelfAttention, Decoder


class ModelDistillation(torch.nn.Module):
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

        #Self-Attention
        self.attention_semseg = SelfAttention(256, 256)
        self.attention_depth = SelfAttention(256, 256)

        #Decoder 3 and 4
        self.decoder3 = Decoder(256,256,ch_out-1)
        self.decoder4 = Decoder(256,256,1)

    def forward(self, x):
        input_resolution = (x.shape[2], x.shape[3])

        features = self.encoder(x)

        # Uncomment to see the scales of feature pyramid with their respective number of channels.
        print(", ".join([f"{k}:{v.shape[1]}" for k, v in features.items()]))
        lowest_scale = max(features.keys())

        features_lowest = features[lowest_scale]

        #Semantic Segmentation
        features_tasks_semseg = self.aspp_semseg(features_lowest)
        predictions_4x_semseg,features_dec1 = self.decoder_semseg(features_tasks_semseg, features[4])

        predictions_1x_semseg = F.interpolate(predictions_4x_semseg, size=input_resolution, mode='bilinear', align_corners=False)

        print("features_dec1: ", features_dec1.shape)
        #Depth estimation
        features_tasks_depth = self.aspp_depth(features_lowest)
        predictions_4x_depth,features_dec2 = self.decoder_depth(features_tasks_depth, features[4])

        predictions_1x_depth = F.interpolate(predictions_4x_depth, size=input_resolution, mode='bilinear', align_corners=False)
        print("features_dec2: ", features_dec2.shape)
        #Self-Attention
        features_attention_semseg = self.attention_semseg(features_dec1)
        features_attention_depth = self.attention_depth(features_dec2)

        print("features_attention_semseg shape: ", features_attention_semseg.shape)
        print("features_attention_depth: ", features_attention_depth.shape)
        print("predictions_1x_semseg: ", predictions_1x_semseg.shape)
        print("predictions_1x_depth: ", predictions_1x_depth.shape)
        
        #Decoder 3 and 4
        out_semseg_4x,_ = self.decoder3(features_attention_depth, features_dec1)
        out_semseg_1x = F.interpolate(out_semseg_4x, size=input_resolution, mode='bilinear', align_corners=False)
        out_depth_4x,_ = self.decoder4(features_attention_semseg, features_dec2)
        out_depth_1x = F.interpolate(out_depth_4x, size=input_resolution, mode='bilinear', align_corners=False)
        print("out_semseg: ", out_semseg_1x.shape)
        print("out_depth: ", out_depth_1x.shape)

        out = {}
        offset = 0

        for task, num_ch in self.outputs_desc.items():
            if task == 'semseg':
                out[task] = out_semseg_1x[:, offset:offset+num_ch, :, :]
                offset += num_ch
            elif task == 'depth':
                out[task] = out_depth_1x[:,:, :, :]
                offset += num_ch

            else:
                print("mod erreur, should be either MOD_SEMSEG or MOD_DEPTH")
        return out
