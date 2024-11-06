import torch.nn as nn
import torch
import torch.nn.functional as F
import pdb 
import scipy
import numpy as np
import torch_dct as freq_transform

class SimpleResidualBlock(nn.Module):
    def __init__(self, input_channel_size, out_channel_size, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channel_size, out_channel_size, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel_size)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channel_size, out_channel_size, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel_size)
        if stride == 1:
            if (input_channel_size == out_channel_size):
                self.shortcut = nn.Identity()
            else:
                self.shortcut = nn.Conv2d(input_channel_size, out_channel_size, kernel_size=1, stride=stride)
        else:
            self.shortcut = nn.Sequential(nn.Conv2d(input_channel_size, out_channel_size, kernel_size=1, stride=stride),
                                        nn.Conv2d(out_channel_size, out_channel_size, kernel_size=1, stride=stride))
        self.relu2 = nn.ReLU()
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        shortcut = self.shortcut(x)
        out = self.relu2(out + shortcut)        
        return out

class ClassificationNetwork18(nn.Module):
    def __init__(self, feat_dim = 8):
        super().__init__()
        self.pos_dim = 3
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 1), 
            SimpleResidualBlock(64, 64),
            SimpleResidualBlock(64, 64),
            SimpleResidualBlock(64, 128, 2),
            SimpleResidualBlock(128, 128),
            SimpleResidualBlock(128, 256, 2),
            SimpleResidualBlock(256, 256),
            # SimpleResidualBlock(256, 512, 2),
            # SimpleResidualBlock(512, 512), # 1024
        )

        self.pos_branch = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.pos_dim)
        )

        # self.feature_branch = nn.Sequential(
        #     nn.Linear(512, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, feat_dim)
        # )
        self.pooling_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), # For each channel, collapses (averages) the entire feature map (height & width) to 1x1
            nn.Flatten(), 
        )
        self.feature_branch = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, feat_dim),
        )

    def forward(self, x, return_embedding=False):
        #x = x[:,3:, :, :]
        #pdb.set_trace()
        feature = self.encoder(x)
        feature = self.pooling_layer(feature)
        feature_output = self.feature_branch(feature)
        # pick task
        #return torch.sigmoid(feature_output)
        # pour task
        #return feature_output
        return torch.sigmoid(feature_output)

    def activation_vis(self, x, return_embedding=False):
        h, w = x.shape[2], x.shape[3]
        #x = x[:,3:, :, :]
        #print(f"x shape {x.shape}")
        feature = self.encoder(x)[0]
        feature = feature.cpu().detach().numpy()
        h_resize = int(h/feature.shape[2])
        resized_feature = scipy.ndimage.zoom(feature, (1,h_resize, h_resize), order=1)
        resized_feature = np.transpose(resized_feature, (1, 2, 0)).reshape(h*h, -1)
        # resized_feature = F.interpolate(feature, size=(h, w), mode='bilinear', align_corners=False) #nearest
        
        device = self.feature_branch[0].weight.device
        fc_weight = self.feature_branch[0].weight.detach().cpu().numpy()
        activation_map = np.dot(resized_feature, fc_weight.T).reshape(h,h,-1)
        #pdb.set_trace()
        return np.sum(activation_map, axis=2), activation_map
        #pdb.set_trace()
        # activation_map = torch.matmul(feature, self.feature_branch[0].weight.T)
        # print(f"activation_map {activation_map}")
        # activation_map_resized = F.interpolate(torch.unsqueeze(torch.unsqueeze(torch.tensor(activation_map), 0), 0), size=(h, w), mode='bilinear', align_corners=False)
        # activation_map_resized = activation_map_resized.cpu().squeeze().numpy()
        # return activation_map_resized


class ResNet18_freq(nn.Module):
    def __init__(self, feat_dim = 8):
        super().__init__()
        self.pos_dim = 3
        self.encoder = nn.Sequential(
            nn.Conv2d(8, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 1), 
            SimpleResidualBlock(64, 64),
            SimpleResidualBlock(64, 64),
            SimpleResidualBlock(64, 128, 2),
            SimpleResidualBlock(128, 128),
            SimpleResidualBlock(128, 256, 2),
            SimpleResidualBlock(256, 256),
            # SimpleResidualBlock(256, 512, 2),
            # SimpleResidualBlock(512, 512), # 1024
        )

        self.pos_branch = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.pos_dim)
        )

        # self.feature_branch = nn.Sequential(
        #     nn.Linear(512, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, feat_dim)
        # )
        self.pooling_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), # For each channel, collapses (averages) the entire feature map (height & width) to 1x1
            nn.Flatten(), 
        )
        self.feature_branch = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, feat_dim),
        )

    def forward(self, x, return_embedding=False):
        with torch.no_grad():
            x_freq = freq_transform.dct_2d(x)
        # Concatenate x and x_freq along the channels
        x = torch.cat((x, x_freq), dim=1)
        feature = self.encoder(x)
        feature = self.pooling_layer(feature)
        feature_output = self.feature_branch(feature)
        # pick task
        #return torch.sigmoid(feature_output)
        # pour task
        #return feature_output
        return torch.sigmoid(feature_output)

# class ResNet18_freq(nn.Module):
#     def __init__(self, feat_dim = 8):
#         super().__init__()
#         self.pos_dim = 3
#         self.pre_encoder = nn.Conv2d(8, 64, kernel_size=3, stride=1, padding=1, bias=False)
#         self.temp_conv_layer = nn.Conv2d(4, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         self.encoder = nn.Sequential(
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size = 3, stride = 1), 
#             SimpleResidualBlock(64, 64),
#             SimpleResidualBlock(64, 64),
#             SimpleResidualBlock(64, 128, 2),
#             SimpleResidualBlock(128, 128),
#             SimpleResidualBlock(128, 256, 2),
#             SimpleResidualBlock(256, 256),
#             # SimpleResidualBlock(256, 512, 2),
#             # SimpleResidualBlock(512, 512), # 1024
#         )

#         self.pos_branch = nn.Sequential(
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Linear(128, self.pos_dim)
#         )

#         # self.feature_branch = nn.Sequential(
#         #     nn.Linear(512, 128),
#         #     nn.ReLU(),
#         #     nn.Linear(128, feat_dim)
#         # )
#         self.pooling_layer = nn.Sequential(
#             nn.AdaptiveAvgPool2d((1, 1)), # For each channel, collapses (averages) the entire feature map (height & width) to 1x1
#             nn.Flatten(), 
#         )
#         self.feature_branch = nn.Sequential(
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Linear(128, feat_dim),
#         )

#     def forward(self, x, return_embedding=False):
#         with torch.no_grad():
#             x_freq = freq_transform.dct_2d(x)
#         # Concatenate x and x_freq along the channels
#         x = torch.cat((x, x_freq), dim=1)
#         x = self.pre_encoder(x)
#         feature = self.encoder(x)
#         feature = self.pooling_layer(feature)
#         feature_output = self.feature_branch(feature)
#         # pick task
#         #return torch.sigmoid(feature_output)
#         # pour task
#         #return feature_output
#         return torch.sigmoid(feature_output)

#     def activation_vis(self, x, return_embedding=False):
#         # with torch.no_grad():
#         #     x_freq = freq_transform.dct_2d(x)
#         # # Concatenate x and x_freq along the channels
#         #x = torch.cat((x, x), dim=1)
#         #pdb.set_trace()
        
#         with torch.no_grad():
#             self.temp_conv_layer.weight[:, :4, :, :] = self.pre_encoder.weight[:, :4, :, :]
        
#         h, w = x.shape[2], x.shape[3]
#         x = self.temp_conv_layer(x)
#         feature = self.encoder(x)[0]
#         feature = feature.cpu().detach().numpy()
#         h_resize = int(h/feature.shape[2])
#         resized_feature = scipy.ndimage.zoom(feature, (1,h_resize, h_resize), order=1)
#         resized_feature = np.transpose(resized_feature, (1, 2, 0)).reshape(h*h, -1)
#         # resized_feature = F.interpolate(feature, size=(h, w), mode='bilinear', align_corners=False) #nearest
        
#         device = self.feature_branch[0].weight.device
#         fc_weight = self.feature_branch[0].weight.detach().cpu().numpy()
#         activation_map = np.dot(resized_feature, fc_weight.T).reshape(h,h,-1)
#         #pdb.set_trace()
#         return np.sum(activation_map, axis=2), activation_map
#         #pdb.set_trace()
#         # activation_map = torch.matmul(feature, self.feature_branch[0].weight.T)
#         # print(f"activation_map {activation_map}")
#         # activation_map_resized = F.interpolate(torch.unsqueeze(torch.unsqueeze(torch.tensor(activation_map), 0), 0), size=(h, w), mode='bilinear', align_corners=False)
#         # activation_map_resized = activation_map_resized.cpu().squeeze().numpy()
#         # return activation_map_resized
