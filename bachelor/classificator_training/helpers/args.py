from argparse import Namespace
import argparse
import re

def parse_and_get_args(load_model_name: str) -> Namespace:
    parser = argparse.ArgumentParser(
        description="Update model usage flags via command-line arguments. "
                    "Values must be passed as integers: 1 (True) or 0 (False)."
    )

    parser.add_argument('--clip', required=False, default=1, type=int, help="Set 'clip' status")
    parser.add_argument('--segformer', required=False, default=0, type=int, help="Set 'segformer' status")
    parser.add_argument('--midas', required=False, default=0, type=int, help="Set 'midas' status")
    parser.add_argument('--dpt', required=False, default=0, type=int, help="Set 'dpt' status")
    
    parser.add_argument('--resnet', required=False, default=0, type=int, help="Set 'resnet' status")
    parser.add_argument('--mobilenet', required=False, default=0, type=int, help="Set 'mobilenet' status")
    parser.add_argument('--efficientnet', required=False, default=0, type=int, help="Set 'efficientnet' status")
    parser.add_argument('--vit', required=False, default=0, type=int, help="Set 'vit' status")

    parser.add_argument('--gate', required=False, default=0, type=int, help="Set 'gate' status")
    parser.add_argument('--batch', required=False, default=64, type=int, help="Set batch size")
    parser.add_argument('--train_type', required=False, default="hardmining", type=str, help="Set train type")
    parser.add_argument('--big_fusion_head', required=False, default=1, type=int, help="Set fusion head type")
    parser.add_argument('--lr', required=False, default=0.00001, type=float, help="Set learning rate")
    parser.add_argument('--margin', required=False, default=0.2, type=float, help="Set margin")
    parser.add_argument('--alpha', required=False, default=32.0, type=float, help="Set alpha")
    
    args = parser.parse_args()
    args = _override_args_from_model_name(args, load_model_name)
    
    return args

    
def _override_args_from_model_name(args: Namespace, load_model_name: str) -> Namespace:
    if not load_model_name:
        return args

    pattern = re.compile(
        r".*?"
        r"(?:_clip(?P<clip>\d+))?"
        r"(?:_segformer(?P<segformer>\d+))?"   
        r"(?:_midas(?P<midas>\d+))?"           
        r"(?:_dpt(?P<dpt>\d+))?"               
        r"(?:_resnet(?P<resnet>\d+))?"         
        r"(?:_mobilenet(?P<mobilenet>\d+))?"   
        r"(?:_efficientnet(?P<efficientnet>\d+))?" 
        r"(?:_vit(?P<vit>\d+))?"               
        r"_gate(?P<gate>\d+)"
        r"_batch(?P<batch>\d+)"
        r"_traintype(?P<train_type>[^_]+)"
        r"_bigfusionhead(?P<big_fusion_head>[^_]+)"
        r"_lr(?P<lr>[\d\.\-e]+)"
        r"_margin(?P<margin>[\d\.\-e]+)"
        r"_alpha(?P<alpha>[\d\.\-e]+|$)"
    )

    match = pattern.search(load_model_name)

    if match:
        extracted_values = match.groupdict()
        
        int_keys = ['clip', 'segformer', 'midas', 'dpt', 'resnet', 'mobilenet', 'efficientnet', 'vit', 'gate', 'batch']
        float_keys = ['lr', 'margin', 'alpha'] 
        
        for key, value in extracted_values.items():
            if hasattr(args, key) and value is not None:
                try:
                    if key in int_keys:
                        setattr(args, key, int(value))
                    elif key in float_keys:
                        setattr(args, key, float(value))
                    elif key == 'big_fusion_head':
                        try:
                            setattr(args, key, int(value))
                        except ValueError:
                            setattr(args, key, value)
                    else:
                        setattr(args, key, value)
                except ValueError as e:
                    pass

    return args
