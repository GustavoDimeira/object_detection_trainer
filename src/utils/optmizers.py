optimizers = {
    "AdamW": {
        "optimizer": {
            'type': 'AdamW', 'lr': None, 'weight_decay': 0.0001, 'paramwise_cfg': {
                'custom_keys': {'backbone': {'lr_mult': 0.1}, 'sampling_offsets': {'lr_mult': 0.1}, 'reference_points': {'lr_mult': 0.1}}
            }
        },
        "optimizer_config": {'grad_clip': {'max_norm': 0.1, 'norm_type': 2}}
    },
    "SGD": {
        "optimizer": {'type': 'SGD', 'lr': None, 'momentum': 0.9, 'weight_decay': 0.0001},
        "optimizer_config": {'grad_clip': None}
    }
}
