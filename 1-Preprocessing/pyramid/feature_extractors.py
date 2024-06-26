import os
import timm
import torch
from torchvision import models, transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
OPENAI_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_STD = [0.26862954, 0.26130258, 0.27577711]

MODEL2CONSTANTS = {
	"resnet50_trunc": {
		"mean": IMAGENET_MEAN,
		"std": IMAGENET_STD
	},
	"uni_v1":
	{
		"mean": IMAGENET_MEAN,
		"std": IMAGENET_STD
	},
	"conch_v1":
	{
		"mean": OPENAI_MEAN,
		"std": OPENAI_STD
	}
}

class Identity(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x


class TimmCNNEncoder(torch.nn.Module):
    def __init__(self, model_name: str = 'resnet50.tv_in1k', 
                 kwargs: dict = {'features_only': True, 'out_indices': (3,), 'pretrained': True, 'num_classes': 0}, 
                 pool: bool = True):
        super().__init__()
        assert kwargs.get('pretrained', False), 'only pretrained models are supported'
        self.model = timm.create_model(model_name, **kwargs)
        self.model_name = model_name
        if pool:
            self.pool = torch.nn.AdaptiveAvgPool2d(1)
        else:
            self.pool = None
    
    def forward(self, x):
        out = self.model(x)
        if isinstance(out, list):
            assert len(out) == 1
            out = out[0]
        if self.pool:
            out = self.pool(out).squeeze(-1).squeeze(-1)
        return out

def TorchvisionCNNEncoder(backbone: str = 'resnet50', finetuned: str = ''):
    model = getattr(models, backbone)(pretrained=True)
    if 'resnet' in backbone or 'resnext' in backbone:
        feature_dim = model.fc.in_features
        model.fc = Identity()
    elif 'dense' in backbone:
        feature_dim = model.classifier.in_features
        model.classifier = Identity()
    elif 'vit' in backbone:
        feature_dim = model.heads.head.in_features
        model.heads = Identity()
    if finetuned:
        model.load_state_dict(torch.load(finetuned))
        print(f'Loaded finetuned model from {finetuned}')
    return model, feature_dim

def has_CONCH():
    HAS_CONCH = False
    CONCH_CKPT_PATH = ''
    # check if CONCH_CKPT_PATH is set and conch is installed, catch exception if not
    try:
        from conch.open_clip_custom import create_model_from_pretrained
        # check if CONCH_CKPT_PATH is set
        if 'CONCH_CKPT_PATH' not in os.environ:
            raise ValueError('CONCH_CKPT_PATH not set')
        HAS_CONCH = True
        CONCH_CKPT_PATH = os.environ['CONCH_CKPT_PATH']
    except Exception as e:
        print(e)
        print('CONCH not installed or CONCH_CKPT_PATH not set')
    return HAS_CONCH, CONCH_CKPT_PATH

def has_UNI():
    HAS_UNI = False
    UNI_CKPT_PATH = ''
    # check if UNI_CKPT_PATH is set, catch exception if not
    try:
        # check if UNI_CKPT_PATH is set
        if 'UNI_CKPT_PATH' not in os.environ:
            raise ValueError('UNI_CKPT_PATH not set')
        HAS_UNI = True
        UNI_CKPT_PATH = os.environ['UNI_CKPT_PATH']
    except Exception as e:
        print(e)
    return HAS_UNI, UNI_CKPT_PATH

def get_encoder(model_name, target_img_size=512, finetuned=''):
    custom_models = ['uni_v1', 'conch_v1']
    if model_name not in custom_models:
        model, feature_dim = TorchvisionCNNEncoder(model_name, finetuned)
    elif model_name == 'uni_v1':
        feature_dim = 1024
        HAS_UNI, UNI_CKPT_PATH = has_UNI()
        assert HAS_UNI, 'UNI is not available'
        model = timm.create_model("vit_large_patch16_224",
                            init_values=1e-5, 
                            num_classes=0, 
                            dynamic_img_size=True)
        model.load_state_dict(torch.load(UNI_CKPT_PATH, map_location="cpu"), strict=True)
    elif model_name == 'conch_v1':
        feature_dim = 1024
        HAS_CONCH, CONCH_CKPT_PATH = has_CONCH()
        assert HAS_CONCH, 'CONCH is not available'
        from conch.open_clip_custom import create_model_from_pretrained
        model, _ = create_model_from_pretrained("conch_ViT-B-16", CONCH_CKPT_PATH)
        model.forward = partial(model.encode_image, proj_contrast=False, normalize=False)
    else:
        raise NotImplementedError('model {} not implemented'.format(model_name))
    
    constants = {
		"mean": IMAGENET_MEAN,
		"std": IMAGENET_STD
	}
    img_transforms = transforms.Compose([
        transforms.Resize(target_img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=constants['mean'], std=constants['std'])
    ])

    return model, feature_dim, img_transforms