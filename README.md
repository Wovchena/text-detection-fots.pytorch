# [FOTS: Fast Oriented Text Spotting with a Unified Network](https://arxiv.org/abs/1801.01671) text detection branch reimplementation ([PyTorch](https://pytorch.org/))
## Train
```sh
python3 train.py --train-folder icdar15/
```
It is expected that the provided `--train-folder` contains unzipped `ch4_training_images` and `ch4_training_localization_transcription_gt`.
## Test
```sh
python3 test.py --images-folder ch4_test_images/  --output-folder res/ --checkpoint best_checkpoint.pt
```
`ch4_training_images`, `ch4_training_localization_transcription_gt` and `ch4_test_images` are available in [Task 4.4: End to End (2015 edition)](http://rrc.cvc.uab.es/?ch=4&com=downloads).
## Results
The best result `"recall": 0.7746750120365913, "precision": 0.8085427135678392, "hmean": 0.7912466191295796` for ICDAR 2015 was reached at [commit](https://github.com/Wovchena/FOTSBasedTextDetection/pull/1/commits/90db90cb915d7d611a7da4380b2cca3add6b9d8b) with 0.95 confidence treshold. The pretrained model is here: https://drive.google.com/open?id=1xaVshLRrMEkb9LA46IJAZhlapQr3vyY2

See attention branch for experiments with attention
