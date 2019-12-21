# [FOTS: Fast Oriented Text Spotting with a Unified Network](https://arxiv.org/abs/1801.01671) text detection branch reimplementation ([PyTorch](https://pytorch.org/))
## Train
1. Train with SynthText for 9 epochs
    ```sh
    time python3 train.py --train-folder SynthText/ --batch-size 21 --batches-before-train 2
    ```
    At this point the result was `Epoch 8: 100%|█████████████| 390/390 [08:20<00:00,  1.01it/s, Mean loss=0.49507]`.
2. Train with ICDAR15
    Change data set in train.py and run
    ```sh
    time python3 train.py --train-folder icdar15/ --batch-size 21 --batches-before-train 2 --continue-training
    ```
    It is expected that the provided `--train-folder` contains unzipped `ch4_training_images` and `ch4_training_localization_transcription_gt`.
    The result was `Epoch 600: 100%|█████████████| 48/48 [01:06<00:00,  1.04s/it, Mean loss=0.07742]`.
### Weight decay history:
Epoch   185: reducing learning rate of group 0 to 5.0000e-04.
Epoch   274: reducing learning rate of group 0 to 2.5000e-04.
Epoch   325: reducing learning rate of group 0 to 1.2500e-04.
Epoch   370: reducing learning rate of group 0 to 6.2500e-05.
Epoch   410: reducing learning rate of group 0 to 3.1250e-05.
Epoch   484: reducing learning rate of group 0 to 1.5625e-05.
Epoch   517: reducing learning rate of group 0 to 7.8125e-06.
Epoch   550: reducing learning rate of group 0 to 3.9063e-06.
## Test
```sh
python3 test.py --images-folder ch4_test_images/ --output-folder res/ --checkpoint epoch_600_checkpoint.pt && zip -jmq runs/u.zip res/* && python2 script.py -g=gt.zip -s=runs/u.zip

```
`ch4_training_images` and `ch4_training_localization_transcription_gt` are available in [Task 4.4: End to End (2015 edition)](http://rrc.cvc.uab.es/?ch=4&com=downloads). `script.py` and `ch4_test_images` can be found in [My Methods](https://rrc.cvc.uab.es/?ch=4&com=mymethods&task=1) (`Script: IoU` and `test set samples`).
It gives `Calculated!{"precision": 0.8700890518596124, "recall": 0.7997111218103033, "hmean": 0.8334169593577522, "AP": 0}`. The pretrained models are here: https://drive.google.com/open?id=1xaVshLRrMEkb9LA46IJAZhlapQr3vyY2
