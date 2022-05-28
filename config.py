config = {
    'path': '/root/Survival/',   # 项目路径
    'path_raw': './data/raw/',  # 原始数据路径
    'path_ok': './data/ok/',    # 处理后数据路径
    'path_processed': './data/ok/',  # 处理后数据路径
    'save_path': './models/',  # 模型保存路径
    'n_epochs': 6000,
    'optimizer': 'AdamW',
    'optim_hparas': {
        'lr': 1e-3,
    },
    'seed': 42,
    'select_all': False,  # Whether to use all features.
    'valid_ratio': 0.16,   # validation_size = train_size * valid_ratio
    'batch_size': 512,
    'early_stop': 1000
}
