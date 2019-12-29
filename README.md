# [FOTS: Fast Oriented Text Spotting with a Unified Network](https://arxiv.org/abs/1801.01671) text detection branch reimplementation ([PyTorch](https://pytorch.org/))

## Train
1. Train with SynthText for 9 epochs
    ```sh
    time python3 train.py --train-folder SynthText/ --batch-size 21 --batches-before-train 2
    ```
    At this point the result was `Epoch 8: 100%|█████████████| 390/390 [08:28<00:00,  1.00it/s, Mean loss=0.98050]`.
2. Train with ICDAR15

    Replace a data set in `data_set = datasets.SynthText(args.train_folder, datasets.transform)` with `datasets.ICDAR2015` in [`train.py`](./train.py) and run
    ```sh
    time python3 train.py --train-folder icdar15/ --continue-training --batch-size 21 --batches-before-train 2
    ```
    It is expected that the provided `--train-folder` contains unzipped `ch4_training_images` and `ch4_training_localization_transcription_gt`. To avoid saving model at each epoch, the line `if True:` in [`train.py`](./train.py) can be replaced with `if epoch > 60 and epoch % 6 == 0:`

    The result was `Epoch 582: 100%|█████████████| 48/48 [01:05<00:00,  1.04s/it, Mean loss=0.11290]`.

### Weight decay history:
Epoch   175: reducing learning rate of group 0 to 5.0000e-04.

Epoch   264: reducing learning rate of group 0 to 2.5000e-04.

Epoch   347: reducing learning rate of group 0 to 1.2500e-04.

Epoch   412: reducing learning rate of group 0 to 6.2500e-05.

Epoch   469: reducing learning rate of group 0 to 3.1250e-05.

Epoch   525: reducing learning rate of group 0 to 1.5625e-05.

Epoch   581: reducing learning rate of group 0 to 7.8125e-06.

## Test
```sh
python3 test.py --images-folder ch4_test_images/ --output-folder res/ --checkpoint epoch_582_checkpoint.pt && zip -jmq runs/u.zip res/* && python2 script.py -g=gt.zip -s=runs/u.zip
```
`ch4_training_images` and `ch4_training_localization_transcription_gt` are available in [Task 4.4: End to End (2015 edition)](http://rrc.cvc.uab.es/?ch=4&com=downloads). `script.py` and `ch4_test_images` can be found in [My Methods](https://rrc.cvc.uab.es/?ch=4&com=mymethods&task=1) (`Script: IoU` and `test set samples`).

It gives `Calculated!{"precision": 0.8694968553459119, "recall": 0.7987481945113144, "hmean": 0.8326223337515684, "AP": 0}`.

The pretrained models are here: https://drive.google.com/open?id=1xaVshLRrMEkb9LA46IJAZhlapQr3vyY2

[`test.py`](./test.py) has a commented code to visualize results.

## Difference with the paper
1. The model is different compared to what the paper describes. An explanation is in [`model.py`](./model.py).
2. The authors of FOTS could not train on clipped words because they also have a recognition branch. The whole word is required to be present on an image to be able to be recognized correctly. This reimplementation has only detection branch and that allows to train on crops of the words.
3. The paper suggest using some other data sets in addition. Training on SynthText is simplified in this reimplementation.
