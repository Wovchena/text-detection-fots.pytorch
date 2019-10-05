"""
Steps of this script can be reused (after accelerating it's performance (NMS and locality aware NMS mostly)) to evaluate
the model during training with different postprocessing thresholds. As for final testing, this script lacks multiscale
testing.
"""
import argparse
import os
import pathlib
import random

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import torch

from model import FOTSModel
from modules.parse_polys import parse_polys
import tqdm
import datasets
from icdar_eval import rrc_evaluation_funcs
from icdar_eval import script


evaluationParams = {
    'IOU_CONSTRAINT': 0.5,
    'AREA_PRECISION_CONSTRAINT': 0.5,
    'LTRB': False,  # LTRB:2points(left,top,right,bottom) or 4 points(x1,y1,x2,y2,x3,y3,x4,y4)
    'CONFIDENCES': True,  # Detections must include confidence value. AP will be calculated
    'PER_SAMPLE_RESULTS': True  # Generate per sample results and produce data for visualization
}


def infer(model, data_loader):
    training = model.training
    model.eval()
    res = [None] * len(dl)
    DEVICE = next(model.parameters()).device
    with torch.no_grad():
        for inputs, file_ids in tqdm.tqdm(data_loader, desc='Infer', ncols=80):
            logits, distances, angles = model(inputs.to(DEVICE))
            batch_size = logits.shape[0]
            confidences = torch.sigmoid(logits).cpu().split(batch_size, 0)
            distances = distances.cpu().split(batch_size, 0)
            angles = angles.cpu().split(batch_size, 0)
            for idx, file_id in enumerate(file_ids):
                res[file_id] = confidences[idx].squeeze().numpy(), distances[idx].squeeze().numpy(),\
                               angles[idx].squeeze().numpy()
    if training:
        model.train()
    return res


def save_polys(res_path, image_polys):
    with open('{}'.format(os.path.join(res_path, 'res_img_{}.txt'.format(file_id + 1))), 'w') as f:
        for id in range(image_polys.shape[0]):
            f.write('{}, {}, {}, {}, {}, {}, {}, {}\n'.format(
                int(image_polys[id, 0]), int(image_polys[id, 1]), int(image_polys[id, 2]), int(image_polys[id, 3]),
                int(image_polys[id, 4]), int(image_polys[id, 5]), int(image_polys[id, 6]), int(image_polys[id, 7])
            ))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-folder', type=pathlib.Path, required=True, help='path to the folder with test images')
    parser.add_argument('--checkpoint', type=pathlib.Path, required=True, help='path to the checkpoint to test')
    parser.add_argument('-g', '--ground-truth', type=pathlib.Path, required=True,
                        help='gt.zip file from Evaluation Scripts (https://rrc.cvc.uab.es/?ch=4&com=mymethods&task=1)')
    parser.add_argument('--output-folder', type=pathlib.Path, help='path to the output folder with result labels')
    args = parser.parse_args()

    conf_thresholds = []
    iou_thresholds = []
    hmeans = []
    if os.path.isfile('thresholds.csv'):
        with open('thresholds.csv', 'r') as thresholds_csv:
            for line in thresholds_csv:
                values = line.split(',')
                assert 3 == len(values)
                conf_thresholds.append(float(values[0]))
                iou_thresholds.append(float(values[1]))
                hmeans.append(float(values[2]))
        assert len(conf_thresholds) == len(iou_thresholds) and len(iou_thresholds) == len(hmeans)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        cmap = matplotlib.cm.get_cmap('rainbow')
        normalize = matplotlib.colors.Normalize(vmin=min(hmeans), vmax=max(hmeans))
        colors = [cmap(normalize(value)) for value in hmeans]
        ax.scatter(conf_thresholds, iou_thresholds, zs=hmeans, s=5, c=colors, depthshade=False)
        ax.set_xlabel('conf')
        ax.set_ylabel('IoU')
        plt.show()
        plt.close(fig)
    else:
        with open('thresholds.csv', 'w'):
            pass

    if args.output_folder and not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    gt = rrc_evaluation_funcs.load_zip_file(args.ground_truth, 'gt_img_([0-9]+).txt')
    gt_dict = {}
    for resFile in gt:  # this is string numbers
        gtFile = rrc_evaluation_funcs.decode_utf8(gt[resFile])
        pointsList_gt, _, transcriptionsList_gt \
            = rrc_evaluation_funcs.get_tl_line_values_from_file_contents(gtFile, False, False,True,False)
        gt_dict[resFile] = pointsList_gt, transcriptionsList_gt

    net = FOTSModel()
    checkpoint = torch.load(args.checkpoint)
    print('Epoch ', checkpoint['epoch'])
    net.load_state_dict(checkpoint['model_state_dict'])
    net = net.eval().cuda()

    OUT_WIDTH = 2240  # TODO: multiscale
    OUT_HEIGHT = round(OUT_WIDTH * 720 / 1280 / 32) * 32
    icdar15_test = datasets.Icdar15_test(str(args.images_folder), OUT_WIDTH)
    dl = torch.utils.data.DataLoader(icdar15_test, batch_size=1, num_workers=3)

    tensors = infer(net, dl)

    # random search for the best conf_threshold and iou_threshold.
    # the best known: conf_threshold, iou_threshold = 0.5074167540548656, 0.2333475569801364
    # giving hmean=0.8023314749113026
    while True:
        conf_threshold = random.uniform(0.5, 0.999)
        iou_threshold = random.uniform(0.1, 0.5)

        parallelograms = [None] * len(tensors)
        for file_id, (confidence, distance, angle) in enumerate(tqdm.tqdm(tensors,
                desc=f'conf_t={conf_threshold}, IoU_t={iou_threshold}', ncols=80)):
            polys = parse_polys(confidence, distance, angle, conf_threshold, iou_threshold)
            if len(polys):
                polys_without_probs = polys[:, :-1]
                polys_without_probs = polys_without_probs.reshape((len(polys), 4, 2))
                polys_without_probs *= [1280 / OUT_WIDTH, 720 / OUT_HEIGHT]
                polys_without_probs = polys_without_probs.reshape((len(polys), 8))
                if args.output_folder:
                    print('saved')
                    save_polys(str(args.output_folder), polys_without_probs)
            if (len(polys)):
                # assign const probabilities to each poly to be able to compute AP, as a temporary workaround
                # until parse_polys() will return probabilities while now it returns sum of contributed polys probs
                polys[:, -1] = [conf_threshold]
            parallelograms[file_id] = polys
        # TODO for threshold in conf_threshold..0.99
        predictions_dict = {}  # disct{str(number): tuple([boxes], [confs])}
        for polys_id, image_parallelograms_with_conf in enumerate(parallelograms):
            image_parallelograms = tuple(
                parallelogram[:-1] for parallelogram in image_parallelograms_with_conf
                if parallelogram[-1] >= conf_threshold)
            image_conf = tuple(
                parallelogram[-1] for parallelogram in image_parallelograms_with_conf
                if parallelogram[-1] >= conf_threshold)
            predictions_dict[str(polys_id + 1)] = (image_parallelograms, image_conf)
            # TODO: evaluate_method() doesn't round predictions_dict. Do it manually before call of evaluate_method()
            # TODO: BTW parse_polys() should take care of rescaling and this rounding
        metrics_values = script.evaluate_method(gt_dict, predictions_dict, evaluationParams)
        print(conf_threshold, iou_threshold, metrics_values['method'])

        with open('thresholds.csv', 'a') as thresholds_csv:
            thresholds_csv.write(f"{conf_threshold},{iou_threshold},{metrics_values['method']['hmean']}\n")
        conf_thresholds.append(conf_threshold)
        iou_thresholds.append(iou_threshold)
        hmeans.append(metrics_values['method']['hmean'])

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        cmap = matplotlib.cm.get_cmap('rainbow')
        normalize = matplotlib.colors.Normalize(vmin=min(hmeans), vmax=max(hmeans))
        colors = [cmap(normalize(value)) for value in hmeans]
        ax.scatter(conf_thresholds, iou_thresholds, zs=hmeans, s=5, c=colors, depthshade=False)
        ax.set_xlabel('conf')
        ax.set_ylabel('IoU')
        plt.savefig('thresholds.png')
        plt.close(fig)
