import torch
import torch.nn as nn
import torch.nn.functional as F

class FusedFeatureModel(nn.Module):
    """
    A train-time model that receives pre-extracted features, applies gating, 
    and projects to the final embedding space.
    """
    def __init__(self, feature_dims, embedding_dim=128, use_gate=True, big_fusion_head=0, use_models=dict[str, bool]):
        super().__init__()
        
        self.feature_dims = feature_dims
        self.use_gate = use_gate
        self.total_input_dim = sum(self.feature_dims.values())
        self.use_models = use_models
        
        if self.use_gate:
            if big_fusion_head >= 2:
                gating_hidden_dim = self.total_input_dim // 2
            else:
                gating_hidden_dim = self.total_input_dim // 4

            self.gating_head = nn.Sequential(
                nn.Linear(self.total_input_dim, gating_hidden_dim),
                nn.GELU(), 
                nn.Linear(gating_hidden_dim, gating_hidden_dim // 2),
                nn.GELU(),
                nn.Linear(gating_hidden_dim // 2, self.total_input_dim),
                nn.Sigmoid()
            )

        if big_fusion_head == 3:
            self.fusion_head = nn.Sequential(
                nn.Linear(self.total_input_dim, 16384),
                nn.BatchNorm1d(16384),
                nn.GELU(),
                nn.Dropout(0.3),
                
                nn.Linear(16384, 12288),
                nn.BatchNorm1d(12288),
                nn.GELU(),
                nn.Dropout(0.3),
                
                nn.Linear(12288, 8192),
                nn.BatchNorm1d(8192),
                nn.GELU(),
                nn.Dropout(0.3),
                
                nn.Linear(8192, 6144),
                nn.BatchNorm1d(6144),
                nn.GELU(),
                nn.Dropout(0.3),
                
                nn.Linear(6144, 4096),
                nn.BatchNorm1d(4096),
                nn.GELU(),
                nn.Dropout(0.3),
                
                nn.Linear(4096, 2048),
                nn.BatchNorm1d(2048),
                nn.GELU(),
                nn.Dropout(0.3),
                
                nn.Linear(2048, embedding_dim)
            )
        elif big_fusion_head == 2:
            self.fusion_head = nn.Sequential(
                nn.Linear(self.total_input_dim, 8192),
                nn.BatchNorm1d(8192),
                nn.ReLU(),
                nn.Dropout(0.3),
                
                nn.Linear(8192, 6144),
                nn.BatchNorm1d(6144),
                nn.ReLU(),
                nn.Dropout(0.3),
                
                nn.Linear(6144, 4096),
                nn.BatchNorm1d(4096),
                nn.ReLU(),
                nn.Dropout(0.3),
                
                nn.Linear(4096, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Dropout(0.3),
                
                nn.Linear(2048, embedding_dim)
            )

        elif big_fusion_head == 1:
            self.fusion_head = nn.Sequential(
                nn.Linear(self.total_input_dim, 4096),
                nn.BatchNorm1d(4096),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(4096, 3072),
                nn.BatchNorm1d(3072),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(3072, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(2048, embedding_dim)
            )
        else:
            self.fusion_head = nn.Sequential(
                nn.Linear(self.total_input_dim, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, embedding_dim)
            )

    def forward(self, **feature_inputs):
        """
        Receives a dictionary of pre-extracted feature tensors (clip, segformer, etc.)
        and processes them through the fusion head.
        """
        feature_tensors = [
            feature_inputs[modality]
            for modality in self.feature_dims.keys()
            if modality in feature_inputs and self.use_models.get(modality, False)
        ]

        if not feature_tensors:
            raise ValueError("No features provided to the FeatureFusionHead.")

        unfused_features = torch.cat(feature_tensors, dim=1)
        
        if self.use_gate:
            gate = self.gating_head(unfused_features)
            gated_features = unfused_features * gate
        else:
            gated_features = unfused_features
        
        final_embedding = self.fusion_head(gated_features)
        
        return F.normalize(final_embedding, p=2, dim=1)