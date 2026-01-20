import torch.nn as nn
from pytorch_metric_learning import losses

class ProxyAnchorLoss(nn.Module):
    """
    A wrapper for pytorch-metric-learning's ProxyAnchorLoss.
    Manages the loss, which requires features (embeddings) and labels.
    """
    def __init__(self, num_classes, embedding_dim, margin=0.1, alpha=32.0):
        super().__init__()
        self.loss_func = losses.ProxyAnchorLoss(
            num_classes=num_classes, 
            embedding_size=embedding_dim, 
            margin=margin, 
            alpha=alpha
        )

    def forward(self, embeddings, labels):
        """
        Calculates the loss based on the embeddings and their corresponding labels.
        
        Args:
            embeddings (torch.Tensor): The final L2-normalized feature vectors 
                                       from FusedFeatureModel (B, embedding_dim).
            labels (torch.Tensor): The class labels for each sample (B,).
            
        Returns:
            torch.Tensor: The calculated Proxy Anchor Loss.
        """
        return self.loss_func(embeddings, labels)