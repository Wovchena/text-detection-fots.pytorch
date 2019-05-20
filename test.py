import argparse
import os

import cv2
import numpy as np
import torch

from model import FOTSModel
from modules.parse_polys import parse_polys
import re


def test(net, images_folder, output_folder, vis_folder, suff):
    for image_name in os.listdir(images_folder):
        image = cv2.imread(os.path.join(images_folder, image_name), cv2.IMREAD_COLOR)
        # due to bad net arch sizes have to be mult of 32, so hardcode it
        WIDTH = 1280 // 2#832
        HEIGHT = 704 // 2#480
        scale_x = WIDTH / image.shape[1]  # 2240 # 1280
        scale_y = HEIGHT / image.shape[0]  # 1248 # 704
        scaled_image = cv2.resize(image, dsize=(0, 0), fx=scale_x, fy=scale_y, interpolation=cv2.INTER_CUBIC)
        const_scaled_image = scaled_image.copy()

        scaled_image = scaled_image[:, :, ::-1].astype(np.float32)
        orig_permuted_scaled_image = scaled_image.copy()
        scaled_image = (scaled_image / 255 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        image_tensor = torch.from_numpy(np.expand_dims(np.transpose(scaled_image, axes=(2, 0, 1)), axis=0)).float()

        confidence, distances, angle = net(image_tensor[:, :, :, :].cuda())
        confidence = confidence.squeeze().data.cpu().numpy()
        distances = distances.squeeze().data.cpu().numpy()
        angle = angle.squeeze().data.cpu().numpy()
        polys = parse_polys(confidence, distances, angle, 0.6, 0.3)#, img=orig_scaled_image)
        # with open('{}'.format(os.path.join(output_folder, 'res_{}.txt'.format(prefix))), 'w') as f:
        #     for id in range(polys.shape[0]):
        #         f.write('{}, {}, {}, {}, {}, {}, {}, {}\n'.format(
        #             int(polys[id, 0] / scale_x), int(polys[id, 1] / scale_y), int(polys[id, 2] / scale_x), int(polys[id, 3] / scale_y),
        #             int(polys[id, 4] / scale_x), int(polys[id, 5] / scale_y), int(polys[id, 6] / scale_x), int(polys[id, 7] / scale_y)
        #         ))

        for poly_id, poly_with_conf in enumerate(sorted(polys, key=lambda p: p[0])):
            poly = poly_with_conf[:8].reshape(4, 2)
            int_poly = poly.round().astype(int)
            conf = poly_with_conf[8]

            crop_corner_x, crop_corner_y = 0, 0
            within_poly_ones = np.zeros_like(confidence, dtype=float)
            # within_poly_ones[(poly / 2).round().astype(int)[0, 1], (poly / 2).round().astype(int)[0, 0]] = 1
            cv2.fillConvexPoly(within_poly_ones, (poly / 2).round().astype(int), 1)
            ref_confidence_roi = within_poly_ones * confidence
            threshold = np.sum(ref_confidence_roi) / np.count_nonzero(within_poly_ones) * 0.01

            croped_permuted_scaled_image = orig_permuted_scaled_image
            res_croped_permuted_scaled_image = croped_permuted_scaled_image.copy()
            cv2.polylines(res_croped_permuted_scaled_image, int_poly[None, :, :], isClosed=True, color=(255, 255, 255))
            mask = np.zeros((HEIGHT, WIDTH), dtype=float)
            for square_y in range(0, croped_permuted_scaled_image.shape[0] - 11, 11):
                for square_x in range(0, croped_permuted_scaled_image.shape[1] - 11, 11):
                    croped_permuted_scaled_image_with_square = croped_permuted_scaled_image.copy()
                    croped_permuted_scaled_image_with_square[square_y:square_y+11, square_x:square_x+11, :] = np.random.rand(11, 11, 3).astype(np.float32) * 255
                    inp = (croped_permuted_scaled_image_with_square / 255 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
                    image_tensor = torch.from_numpy(np.expand_dims(np.transpose(inp, axes=(2, 0, 1)), axis=0)).float()
                    tmp_confidence, _, _ = net(image_tensor.cuda())
                    tmp_confidence = tmp_confidence.squeeze().cpu().numpy()

                    within_poly_ones = np.zeros_like(tmp_confidence, dtype=float)
                    cv2.fillConvexPoly(within_poly_ones, ((poly) / 2).round().astype(int), 1)
                    res_confidence_roi = within_poly_ones * tmp_confidence
                    cv2.imshow('conf', within_poly_ones * tmp_confidence)
                    diff = np.sum(np.abs(ref_confidence_roi - res_confidence_roi)) / np.count_nonzero(within_poly_ones)
                    if diff > threshold:
                        mask[square_y:square_y+11, square_x:square_x+11] = diff
                        patch = res_croped_permuted_scaled_image[square_y:square_y+11, square_x:square_x+11, :].copy()
                        patch *= (1 - diff)
                        res_croped_permuted_scaled_image[square_y:square_y + 11, square_x:square_x + 11] = patch.astype(np.uint8)
                        #res_croped_permuted_scaled_image[square_y:square_y+11,
                        #square_x:square_x+11, :] = 0#\
                            # (res_croped_permuted_scaled_image[square_y:square_y+11, square_x:square_x+11, :].astype(np.float64)
                            #  * (1 - diff)).clip(0, 255).astype(np.uint8)
                        # cv2.imshow('croped_permuted_scaled_image_with_square',
                        #            res_croped_permuted_scaled_image.clip(0, 255).astype(np.uint8)[:, :, ::-1])
                        # cv2.waitKey(1)
                    # cv2.imshow('croped_permuted_scaled_image_with_square', res_croped_permuted_scaled_image.clip(0, 255).astype(np.uint8)[:, :, ::-1])
                    # cv2.waitKey()
            mask = (mask - mask.min()) / (mask.max() - mask.min())
            cv2.imshow('mask', mask)
            green_mask = np.zeros_like(croped_permuted_scaled_image)
            green_mask[:, :, 1] = (mask * 255).astype(np.uint8)
            green_image = (0.5 * green_mask + 0.5 * croped_permuted_scaled_image[:, :, ::-1]).astype(np.uint8)
            cv2.polylines(green_image, (int_poly - np.array([crop_corner_x, crop_corner_y]))[None, :, :], isClosed=True, color=(255, 255, 255))
            out_file_name = f"{vis_folder}/{image_name[:-4]}-P{poly_id}-{conf:.3}-{suff}.jpg"
            cv2.imwrite(out_file_name, green_image)
            print(out_file_name)
            cv2.imshow('saliency', green_image)
            cv2.waitKey(1)

            cv2.imshow('within_poly_ones', within_poly_ones)
            cv2.imshow('const_scaled_image', (within_poly_ones[:, :, None] * cv2.resize(const_scaled_image, dsize=(within_poly_ones.shape[1], within_poly_ones.shape[0]))).astype(np.uint8))
            cv2.waitKey(1)



        # visualize
        # reshaped_pred_polys = []
        # numpy_reshaped_pred_polys = np.array(reshaped_pred_polys)
        # for id in range(polys.shape[0]):
        #     reshaped_pred_polys.append(np.array([int(polys[id, 0] / scale_x), int(polys[id, 1] / scale_y), int(polys[id, 2] / scale_x), int(polys[id, 3] / scale_y),
        #                 int(polys[id, 4] / scale_x), int(polys[id, 5] / scale_y), int(polys[id, 6] / scale_x), int(polys[id, 7] / scale_y)]).reshape((4, 2)))
        #     numpy_reshaped_pred_polys = np.stack(reshaped_pred_polys)
        # strong_gt_quads = []
        # weak_gt_quads = []
        # lines = [line.rstrip('\n') for line in open(os.path.join(os.path.join(images_folder, '../Challenge4_Test_Task4_GT'), 'gt_' + image_name[:-4] + '.txt'),
        #                                             encoding='utf-8-sig')]
        # pattern = re.compile('^' + '(\\d+),' * 8 + '(.+)$')
        # for line in lines:
        #     matches = pattern.findall(line)[0]
        #     numbers = np.array(matches[:8], dtype=float)
        #     if '###' == matches[8]:
        #         weak_gt_quads.append(numbers.reshape((4, 2)))
        #     else:
        #         strong_gt_quads.append(numbers.reshape((4, 2)))
        # if len(strong_gt_quads):
        #     numpy_strong_gt_quads = np.stack(strong_gt_quads)
        #     cv2.polylines(image, numpy_strong_gt_quads.round().astype(int), True, (0, 0, 255))
        # if len(weak_gt_quads):
        #     numpy_weak_gt_quads = np.stack(weak_gt_quads)
        #     cv2.polylines(image, numpy_weak_gt_quads.round().astype(int), True, (0, 255, 255))
        # cv2.polylines(image, numpy_reshaped_pred_polys.round().astype(int), True, (255, 0, 0))
        # cv2.imshow('img', image)
        # print(image_name)
        # cv2.waitKey(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-folder', type=str, required=True, help='path to the folder with test images')
    parser.add_argument('--output-folder', type=str, default='fots_test_results',
                        help='path to the output folder with result labels')
    parser.add_argument('--checkpoint', type=str, required=True, help='path to the checkpoint to test')
    parser.add_argument('--vis-folder', type=str, required=True, help='out images')
    parser.add_argument('--suff', type=str, required=True, help='add to im name')
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)
    os.makedirs(args.vis_folder, exist_ok=True)

    net = FOTSModel()
    checkpoint = torch.load(args.checkpoint)
    net.load_state_dict(checkpoint['model_state_dict'])
    net = net.eval().cuda()

    with torch.no_grad():
        test(net, args.images_folder, args.output_folder, args.vis_folder, args.suff)
