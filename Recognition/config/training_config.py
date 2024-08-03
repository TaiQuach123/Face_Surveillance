class TrainingConfig(object):
    train_root = "data/recognition/CASIA-align-112x112"
    save_train_txt = "data/recognition/casia_text_file.txt"
    lfw_val_root = "data/recognition/lfw-aligned"
    lfw_val_txt = "data/recognition/lfw_test_pair.txt"
    classify = 'softmax'
    num_classes = 10575
    metric = 'subcenter_arc_margin'
    easy_margin = False
    loss = 'focal_loss'
    batch_size = 128
    warmup_iters = 0
    optim = 'sgd'