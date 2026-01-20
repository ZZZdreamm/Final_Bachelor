import torch
import torch.nn as nn
from torchvision import models as tv_models
from transformers import AutoModelForDepthEstimation, CLIPConfig, CLIPVisionModel, SegformerModel, DPTModel
from transformers import CLIPImageProcessor, SegformerImageProcessor, DPTImageProcessor

CLIP_MODEL_ID = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
SEGFORMER_MODEL_ID = "nvidia/mit-b5"
DPT_MODEL_ID = "Intel/dpt-large"
MIDAS_MODEL_ID = "Intel/dpt-hybrid-midas"

clip_processor = CLIPImageProcessor.from_pretrained(CLIP_MODEL_ID)
segformer_processor = SegformerImageProcessor.from_pretrained(SEGFORMER_MODEL_ID)
dpt_processor = DPTImageProcessor.from_pretrained(DPT_MODEL_ID)
midas_processor = DPTImageProcessor.from_pretrained(MIDAS_MODEL_ID)

MODEL_PROCESSORS = {
    'clip': clip_processor,
    'segformer': segformer_processor,
    'dpt': dpt_processor,
    "midas": midas_processor
}

class FeatureExtractor(nn.Module):
    """
    Adapter of the original FusedFeatureModel, used *only* for feature extraction
    and to calculate the required dimensions. All backbones are initialized and frozen.
    """
    def __init__(self, use_models=None):
        super().__init__()
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if use_models is None:
            use_models = {'clip': True, 'segformer': True, 'midas': True, 'dpt': True}
        
        self.use_models = use_models
        self.feature_dims = {}
        
        
        if self.use_models.get('clip'):
            clip_config = CLIPConfig.from_pretrained(CLIP_MODEL_ID)
            self.clip_model = CLIPVisionModel.from_pretrained(
                CLIP_MODEL_ID,
                config=clip_config.vision_config,
            )
            self.clip_model.to(device)
            self._freeze_parameters(self.clip_model)
            self.feature_dims['clip'] = self.clip_model.config.hidden_size
        
        if self.use_models.get('segformer'):
            self.segformer_model = SegformerModel.from_pretrained(SEGFORMER_MODEL_ID)
            self.segformer_model.to(device)
            self._freeze_parameters(self.segformer_model)
            self.feature_dims['segformer'] = self.segformer_model.config.hidden_sizes[-1] 
        
        if self.use_models.get('midas'):
            self.midas_model = AutoModelForDepthEstimation.from_pretrained(MIDAS_MODEL_ID) 
            self.midas_model.to(device)
            self._freeze_parameters(self.midas_model)
            self.feature_dims['midas'] = 768 

        if self.use_models.get('dpt'):
            self.dpt_model = DPTModel.from_pretrained(DPT_MODEL_ID)
            self.dpt_model.to(device)
            self._freeze_parameters(self.dpt_model)
            self.feature_dims['dpt'] = self.dpt_model.config.hidden_size 
            
            

        if self.use_models.get('resnet'):
            self.resnet = tv_models.resnet152(weights=tv_models.ResNet152_Weights.DEFAULT).to(device)
            self.feature_dims['resnet'] = self.resnet.fc.in_features
            self.resnet.fc = nn.Identity()
            self._freeze_parameters(self.resnet)

        if self.use_models.get('mobilenet'):
            self.mobilenet = tv_models.mobilenet_v3_large(weights=tv_models.MobileNet_V3_Large_Weights.DEFAULT).to(device)
            self.feature_dims['mobilenet'] = self.mobilenet.classifier[3].in_features
            self.mobilenet.classifier[3] = nn.Identity()
            self._freeze_parameters(self.mobilenet)

        if self.use_models.get('efficientnet'):
            self.efficientnet = tv_models.efficientnet_v2_l(weights=tv_models.EfficientNet_V2_L_Weights.DEFAULT).to(device)
            self.feature_dims['efficientnet'] = self.efficientnet.classifier[1].in_features
            self.efficientnet.classifier[1] = nn.Identity()
            self._freeze_parameters(self.efficientnet)

        if self.use_models.get('vit'):
            self.vit = tv_models.vit_l_16(weights=tv_models.ViT_L_16_Weights.DEFAULT).to(device)
            self.feature_dims['vit'] = self.vit.heads.head.in_features
            self.vit.heads.head = nn.Identity()
            self._freeze_parameters(self.vit)
        
        self.total_input_dim = sum(self.feature_dims.values())
        if self.total_input_dim == 0:
            raise ValueError("No backbones are selected in `use_models`. At least one must be True.")

    def _freeze_parameters(self, model):
        for param in model.parameters():
            param.requires_grad = False
            
    def _ensure_batch_dim(self, input_tensor):
        """
        Adds a batch dimension (at position 0) if the tensor has only 3 dimensions (C, H, W).
        """
        if input_tensor is not None and input_tensor.dim() == 3:
            return input_tensor.unsqueeze(0)
        return input_tensor

    def extract_all_features(self, 
            clip_input=None, segformer_input=None, dpt_input=None, midas_input=None,
            resnet_input=None, mobilenet_input=None, efficientnet_input=None, vit_input=None):
        """
        Runs the feature extraction for all enabled backbones and returns a dictionary 
        of the feature vectors.
        """
        extracted_feature_dict = {}
        
        if self.use_models.get('clip'):
            if clip_input is None: raise ValueError("clip_input must be provided when clip is enabled.")
            clip_input = self._ensure_batch_dim(clip_input)
            clip_features = self.clip_model(clip_input).pooler_output
            extracted_feature_dict['clip'] = clip_features
        
        if self.use_models.get('segformer'):
            if segformer_input is None: raise ValueError("segformer_input must be provided when segformer is enabled.")
            segformer_input = self._ensure_batch_dim(segformer_input)
            segformer_output = self.segformer_model(segformer_input, output_hidden_states=True)
            segformer_features = segformer_output.last_hidden_state.mean(dim=[2, 3]) 
            extracted_feature_dict['segformer'] = segformer_features
            
        if self.use_models.get('dpt'):
            if dpt_input is None: raise ValueError("dpt_input must be provided when dpt is enabled.")
            dpt_input = self._ensure_batch_dim(dpt_input)
            dpt_output = self.dpt_model(dpt_input)
            dpt_features = dpt_output.pooler_output
            extracted_feature_dict['dpt'] = dpt_features
        
        if self.use_models.get('midas'):
            if midas_input is None: raise ValueError("midas_input must be provided when midas is enabled.")
            midas_input = self._ensure_batch_dim(midas_input)
            midas_output = self.midas_model(midas_input, output_hidden_states=True)
            midas_features = midas_output.hidden_states[-1][:, 1:, :].mean(dim=1)
            extracted_feature_dict['midas'] = midas_features
            
        
        if self.use_models.get('resnet'):
            if resnet_input is None: raise ValueError("resnet_input missing")
            extracted_feature_dict['resnet'] = self.resnet(self._ensure_batch_dim(resnet_input))

        if self.use_models.get('mobilenet'):
            if mobilenet_input is None: raise ValueError("mobilenet_input missing")
            extracted_feature_dict['mobilenet'] = self.mobilenet(self._ensure_batch_dim(mobilenet_input))

        if self.use_models.get('efficientnet'):
            if efficientnet_input is None: raise ValueError("efficientnet_input missing")
            extracted_feature_dict['efficientnet'] = self.efficientnet(self._ensure_batch_dim(efficientnet_input))

        if self.use_models.get('vit'):
            if vit_input is None: raise ValueError("vit_input missing")
            extracted_feature_dict['vit'] = self.vit(self._ensure_batch_dim(vit_input))
            
        return extracted_feature_dict
    
    def extract_feature_from_model(self, model_name: str, input_tensor: torch.Tensor):
        """
        Extracts features from a specific backbone model.
        """
        if model_name not in self.use_models or not self.use_models[model_name]:
            raise ValueError(f"Model '{model_name}' is not enabled in use_models.")
        
        input_tensor = self._ensure_batch_dim(input_tensor)
        
        if model_name == 'clip':
            return self.clip_model(input_tensor).pooler_output
        elif model_name == 'segformer':
            output = self.segformer_model(input_tensor, output_hidden_states=True)
            return output.last_hidden_state.mean(dim=[2, 3])
        elif model_name == 'dpt':
            output = self.dpt_model(input_tensor)
            return output.pooler_output
        elif model_name == 'midas':
            output = self.midas_model(input_tensor, output_hidden_states=True)
            return output.hidden_states[-1][:, 1:, :].mean(dim=1)
        elif model_name == 'resnet':
            return self.resnet(input_tensor)
        elif model_name == 'mobilenet':
            return self.mobilenet(input_tensor)
        elif model_name == 'efficientnet':
            return self.efficientnet(input_tensor)
        elif model_name == 'vit':
            return self.vit(input_tensor)
        else:
            raise ValueError(f"Unknown model name: {model_name}")
    