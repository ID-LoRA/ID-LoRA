"""
WavLM Large + ECAPA-TDNN Speaker Verification Model

This module implements a hybrid speaker verification model that combines:
- WavLM Large from HuggingFace Transformers as the feature extractor
- ECAPA-TDNN head for speaker embedding extraction

Based on Microsoft UniSpeech: https://github.com/microsoft/UniSpeech
Achieves 0.431% EER on VoxCeleb1-O (state-of-the-art)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from transformers import WavLMModel


class Res2Conv1dReluBn(nn.Module):
    """Res2Conv1d + BatchNorm1d + ReLU"""
    
    def __init__(self, channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, scale=4):
        super().__init__()
        assert channels % scale == 0
        self.scale = scale
        self.width = channels // scale
        self.nums = scale if scale == 1 else scale - 1
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(self.nums):
            self.convs.append(nn.Conv1d(self.width, self.width, kernel_size, stride, padding, dilation, bias=bias))
            self.bns.append(nn.BatchNorm1d(self.width))
    
    def forward(self, x):
        out = []
        spx = torch.split(x, self.width, 1)
        for i in range(self.nums):
            sp = spx[i] if i == 0 else sp + spx[i]
            sp = self.bns[i](F.relu(self.convs[i](sp)))
            out.append(sp)
        if self.scale != 1:
            out.append(spx[self.nums])
        return torch.cat(out, dim=1)


class Conv1dReluBn(nn.Module):
    """Conv1d + BatchNorm1d + ReLU"""
    
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)
        self.bn = nn.BatchNorm1d(out_channels)
    
    def forward(self, x):
        return self.bn(F.relu(self.conv(x)))


class SE_Connect(nn.Module):
    """Squeeze-and-Excitation connection for 1D"""
    
    def __init__(self, channels, se_bottleneck_dim=128):
        super().__init__()
        self.linear1 = nn.Linear(channels, se_bottleneck_dim)
        self.linear2 = nn.Linear(se_bottleneck_dim, channels)
    
    def forward(self, x):
        out = x.mean(dim=2)
        out = F.relu(self.linear1(out))
        out = torch.sigmoid(self.linear2(out))
        return x * out.unsqueeze(2)


class SE_Res2Block(nn.Module):
    """SE-Res2Block of ECAPA-TDNN"""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, scale, se_bottleneck_dim):
        super().__init__()
        self.Conv1dReluBn1 = Conv1dReluBn(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.Res2Conv1dReluBn = Res2Conv1dReluBn(out_channels, kernel_size, stride, padding, dilation, scale=scale)
        self.Conv1dReluBn2 = Conv1dReluBn(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.SE_Connect = SE_Connect(out_channels, se_bottleneck_dim)
        
        self.shortcut = None
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        residual = self.shortcut(x) if self.shortcut else x
        x = self.Conv1dReluBn1(x)
        x = self.Res2Conv1dReluBn(x)
        x = self.Conv1dReluBn2(x)
        x = self.SE_Connect(x)
        return x + residual


class AttentiveStatsPool(nn.Module):
    """Attentive weighted mean and standard deviation pooling"""
    
    def __init__(self, in_dim, attention_channels=128, global_context_att=False):
        super().__init__()
        self.global_context_att = global_context_att
        
        if global_context_att:
            self.linear1 = nn.Conv1d(in_dim * 3, attention_channels, kernel_size=1)
        else:
            self.linear1 = nn.Conv1d(in_dim, attention_channels, kernel_size=1)
        self.linear2 = nn.Conv1d(attention_channels, in_dim, kernel_size=1)
    
    def forward(self, x):
        if self.global_context_att:
            context_mean = x.mean(dim=-1, keepdim=True).expand_as(x)
            context_std = torch.sqrt(x.var(dim=-1, keepdim=True) + 1e-10).expand_as(x)
            x_in = torch.cat((x, context_mean, context_std), dim=1)
        else:
            x_in = x
        
        alpha = torch.tanh(self.linear1(x_in))
        alpha = torch.softmax(self.linear2(alpha), dim=2)
        mean = torch.sum(alpha * x, dim=2)
        residuals = torch.sum(alpha * (x ** 2), dim=2) - mean ** 2
        std = torch.sqrt(residuals.clamp(min=1e-9))
        return torch.cat([mean, std], dim=1)


class ECAPA_TDNN_Head(nn.Module):
    """ECAPA-TDNN head that takes WavLM features as input"""
    
    def __init__(self, feat_dim=1024, channels=512, emb_dim=256, global_context_att=False):
        super().__init__()
        
        self.instance_norm = nn.InstanceNorm1d(feat_dim)
        
        self.layer1 = Conv1dReluBn(feat_dim, channels, kernel_size=5, padding=2)
        self.layer2 = SE_Res2Block(channels, channels, kernel_size=3, stride=1, padding=2, dilation=2, scale=8, se_bottleneck_dim=128)
        self.layer3 = SE_Res2Block(channels, channels, kernel_size=3, stride=1, padding=3, dilation=3, scale=8, se_bottleneck_dim=128)
        self.layer4 = SE_Res2Block(channels, channels, kernel_size=3, stride=1, padding=4, dilation=4, scale=8, se_bottleneck_dim=128)
        
        self.conv = nn.Conv1d(channels * 3, 1536, kernel_size=1)
        self.pooling = AttentiveStatsPool(1536, attention_channels=128, global_context_att=global_context_att)
        self.bn = nn.BatchNorm1d(1536 * 2)
        self.linear = nn.Linear(1536 * 2, emb_dim)
    
    def forward(self, x):
        # x: [batch, time, feat_dim] from WavLM
        x = x.transpose(1, 2)  # -> [batch, feat_dim, time]
        x = self.instance_norm(x)
        
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        
        out = torch.cat([out2, out3, out4], dim=1)
        out = F.relu(self.conv(out))
        out = self.bn(self.pooling(out))
        out = self.linear(out)
        
        return out


class WavLMECAPATDNN(nn.Module):
    """WavLM Large + ECAPA-TDNN for speaker verification
    
    Uses HuggingFace's WavLM Large as feature extractor and
    ECAPA-TDNN head for speaker embedding extraction.
    """
    
    def __init__(self, checkpoint_path=None):
        super().__init__()
        
        # Load WavLM Large from HuggingFace
        self.wavlm = WavLMModel.from_pretrained("microsoft/wavlm-large")
        self.wavlm.eval()
        for param in self.wavlm.parameters():
            param.requires_grad = False
        
        # ECAPA-TDNN head
        # WavLM Large has 1024 hidden dim
        self.ecapa_head = ECAPA_TDNN_Head(feat_dim=1024, channels=512, emb_dim=256)
        
        # Learnable weight for layer aggregation (like in UniSpeech)
        self.feature_weight = nn.Parameter(torch.zeros(25))  # 24 layers + 1 for last hidden state
        
        # Load checkpoint if provided
        if checkpoint_path:
            self._load_ecapa_weights(checkpoint_path)
    
    def _load_ecapa_weights(self, checkpoint_path):
        """Load ECAPA-TDNN head weights from UniSpeech checkpoint"""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        model_state = state_dict['model']
        
        # Extract ECAPA head weights (exclude feature_extract which is WavLM)
        ecapa_state = {}
        for k, v in model_state.items():
            if not k.startswith('feature_extract.'):
                # Map the key names
                new_key = k
                if k.startswith('instance_norm.'):
                    new_key = k  # Keep as is
                elif k.startswith('layer'):
                    new_key = k  # Keep as is
                elif k == 'feature_weight':
                    self.feature_weight.data = v
                    continue
                else:
                    new_key = k
                
                ecapa_state[new_key] = v
        
        # Load ECAPA head weights
        missing, unexpected = self.ecapa_head.load_state_dict(ecapa_state, strict=False)
        if missing:
            print(f"Missing keys when loading ECAPA head: {missing}")
        if unexpected:
            print(f"Unexpected keys when loading ECAPA head: {unexpected}")
        
        print("Loaded ECAPA-TDNN head weights from UniSpeech checkpoint")
    
    def extract_features(self, waveform):
        """Extract WavLM features with learnable layer aggregation"""
        with torch.no_grad():
            outputs = self.wavlm(waveform, output_hidden_states=True)
        
        # Stack all hidden states: [num_layers, batch, time, hidden_dim]
        hidden_states = torch.stack(outputs.hidden_states, dim=0)
        
        # Apply learnable weights for layer aggregation
        weights = F.softmax(self.feature_weight, dim=0)
        weights = weights.view(-1, 1, 1, 1)
        
        # Weighted sum: [batch, time, hidden_dim]
        features = (weights * hidden_states).sum(dim=0)
        
        return features
    
    def forward(self, waveform):
        """Extract speaker embedding from waveform
        
        Args:
            waveform: [batch, time] tensor at 16kHz
            
        Returns:
            Speaker embedding: [batch, 256]
        """
        features = self.extract_features(waveform)
        embedding = self.ecapa_head(features)
        return embedding


def load_wavlm_ecapa_model(checkpoint_path=None, device='cpu'):
    """Load WavLM + ECAPA-TDNN model
    
    Args:
        checkpoint_path: Path to UniSpeech checkpoint (optional)
        device: Device to load model on
        
    Returns:
        Loaded model ready for inference
    """
    model = WavLMECAPATDNN(checkpoint_path=checkpoint_path)
    model = model.to(device)
    model.eval()
    return model
