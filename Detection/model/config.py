cfg_mnet = {
    'name': 'mobilenet',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'image_size': 640,
    'return_layers': {'6': 1, '10': 2, '16': 3},
    'in_channels': [32, 64, 160],
    'out_channel': 64
}

cfg_re50 = {
    'name': 'resnet50',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'image_size': 840,
    'return_layers': {'layer2': 1, 'layer3': 2, 'layer4': 3},
    'in_channels': [512, 1024, 2048],
    'out_channel': 256
}