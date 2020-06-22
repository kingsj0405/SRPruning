# SR Pruning

## Summary

Network Pruning for Super Resolution

### Usage

You can see help with following command

```bash
kingsj0405@local:/SRPruning# python main.py --help
INFO: Showing help with the command 'main.py -- --help'.

NAME
    main.py

SYNOPSIS
    main.py COMMAND

COMMANDS
    COMMAND is one of the following:

     train

     pruning

     hello
python main.py --help
```

This is example run of `train`. You should edit `src/config.py` before this.

```bash
kingsj0405@local:/SRPruning# CUDA_VISIBLE_DEVICES=1 python main.py train
[INFO] Set configuration
[INFO] Make directory /app/NAS2_sejong/SRPruning/vdsr/config/
[INFO] Make directory /app/NAS2_sejong/SRPruning/vdsr/samples/v45
[INFO] Make directory /app/NAS2_sejong/SRPruning/vdsr/checkpoint/v45
[INFO] Make directory /app/NAS2_sejong/SRPruning/vdsr/summary/v45
[INFO] Save config to /app/NAS2_sejong/SRPruning/vdsr/config/v45.cfg
[INFO] Experiment v45 set up
[INFO] Set random seed
[INFO] Get training dataset and data_loader
[INFO] Prepare net, optimizer, loss for training
[INFO] Load checkpoint from /app/NAS2_sejong/SRPruning/vdsr/checkpoint/v22/SRPruning_epoch_0.pth
[INFO] Load pruning report from /app/NAS2_sejong/SRPruning/vdsr/pruning/p34/pruning-report.json
[INFO] Load mask index from /app/NAS2_sejong/SRPruning/vdsr/pruning/p34/channel_mask_1.pickle
[INFO] Start training loop
[INFO] Save checkpoint before training
  0%|                                                                          | 1/10000 [00:21<59:37:55, 21.47s/it]
  0%|                                                                                        | 0/13 [00:00<?, ?it/s]
```

Experiment version should be unique. If not, it will failed.

```bash
kingsj0405@local:/SRPruning# CUDA_VISIBLE_DEVICES=1 python main.py train
[INFO] Set configuration
[ERROR] Configuration /app/NAS2_sejong/SRPruning/vdsr/config/v45.cfg already exists
[ERROR] Stop Experiment v45
```

### Repository Structure

There are two modules(`model`, `pruning`) and small scripts(`config`, `dataset`, `loss`, `util`).

```bash
kingsj0405@local:/SRPruning# tree .
.
|-- README.md
|-- main.py
|-- model
|   |-- MSRResNet
|   |   |-- MSRResNetx4_model
|   |   |   `-- MSRResNetx4.pth
|   |   |-- README.md
|   |   |-- SRResNet.py
|   |   |-- __init__.py
|   |   |-- test_demo.py
|   |   `-- utils
|   |       |-- __pycache__
|   |       |   |-- utils_image.cpython-37.pyc
|   |       |   `-- utils_logger.cpython-37.pyc
|   |       |-- test.bmp
|   |       |-- utils_image.py
|   |       `-- utils_logger.py
|   |-- __init__.py
|   |-- layer.py
|   `-- vdsr.py
|-- pruning
|   |-- README.md
|   |-- __init__.py
|   `-- pruning.py
|-- requirements.txt
`-- src
    |-- config.py
    |-- dataset.py
    |-- loss.py
    `-- util.py

7 directories, 23 files
```