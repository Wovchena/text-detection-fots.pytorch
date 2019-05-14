import argparse
import os

import cv2
import numpy as np
import torch


from model import FOTSModel
from modules.parse_polys import parse_polys
import re
import tqdm
from create_explainer import get_explainer

DEVICE = torch.device("cuda")

def test(net, images_folder, output_folder, vis_folder, vis_suffix):
    pattern = re.compile('^' + '(\\d+),' * 8 + '(.+)$')
    for image_name in tqdm.tqdm(os.listdir(images_folder), desc="Test", ncols=80):
        prefix = image_name[:image_name.rfind('.')]
        image = cv2.imread(os.path.join(images_folder, image_name), cv2.IMREAD_COLOR)
        # due to bad net arch sizes have to be mult of 32, so hardcode it
        scale_x = 2240 / image.shape[1]  # 2240 # 1280
        scale_y = 1248 / image.shape[0]  # 1248 # 704
        scaled_image = cv2.resize(image, dsize=(0, 0), fx=scale_x, fy=scale_y, interpolation=cv2.INTER_CUBIC)

        input_scaled_image = scaled_image[:, :, ::-1].astype(np.float32)
        input_scaled_image = (input_scaled_image / 255 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        image_tensor = torch.from_numpy(np.expand_dims(np.transpose(input_scaled_image, axes=(2, 0, 1)), axis=0)).float().to(DEVICE)

        explainer = get_explainer(net, "excitation_backprop")
        excitations_per_poly = explainer.explain(image_tensor)

        strong_gt_quads = []
        weak_gt_quads = []
        lines = [line.rstrip('\n') for line in open(os.path.join(os.path.join(images_folder, '../Challenge4_Test_Task4_GT'), 'gt_' + image_name[:-4] + '.txt'),
                                                    encoding='utf-8-sig')]
        for line in lines:
            matches = pattern.findall(line)[0]
            numbers = np.array(matches[:8], dtype=float)
            if '###' == matches[8]:
                weak_gt_quads.append(numbers.reshape((4, 2)))
            else:
                strong_gt_quads.append(numbers.reshape((4, 2)))

        for i, (excitation, poly) in enumerate(excitations_per_poly):
            local_scaled_image = scaled_image.copy()
            bbox_local_scaled_image = scaled_image.copy()
            # local_scaled_image.shape == HWC
            # excitation.shape == BCHW
            excitation = excitation[0, 0].cpu().numpy()
            excitation = cv2.resize(excitation, dsize=(local_scaled_image.shape[1], local_scaled_image.shape[0]), interpolation=cv2.INTER_NEAREST)
            # excitation = torch.nn.functional.interpolate(excitation, size=(local_scaled_image.shape[0], local_scaled_image.shape[1]), mode='bilinear', align_corners=False).squeeze().cpu().numpy()
            local_scaled_image = local_scaled_image.astype(float)
            local_scaled_image = local_scaled_image * excitation[:, :, None]
            local_scaled_image = local_scaled_image.round().clip(0, 255).astype(np.uint8)

            if len(strong_gt_quads):
                numpy_strong_gt_quads = np.stack(strong_gt_quads) * np.array([scale_x, scale_y])
                cv2.polylines(local_scaled_image, numpy_strong_gt_quads.round().astype(int), True, (0, 0, 255))
                cv2.polylines(bbox_local_scaled_image, numpy_strong_gt_quads.round().astype(int), True, (0, 0, 255))
            if len(weak_gt_quads):
                numpy_weak_gt_quads = np.stack(weak_gt_quads) * np.array([scale_x, scale_y])
                cv2.polylines(local_scaled_image, numpy_weak_gt_quads.round().astype(int), True, (0, 255, 255))
                cv2.polylines(bbox_local_scaled_image, numpy_weak_gt_quads.round().astype(int), True, (0, 255, 255))
                # dont draw on excitation ma cos it is grayscale and it will be hard to see difference between bbox types
            cv2.polylines(local_scaled_image, poly, True, (255, 0, 0))
            cv2.polylines(excitation, poly, True, 1)
            cv2.polylines(bbox_local_scaled_image, poly, True, (255, 0, 0))

            local_scaled_image = cv2.resize(local_scaled_image, dsize=(1280, 720), interpolation=cv2.INTER_CUBIC)
            excitation = cv2.resize(excitation, dsize=(1280, 720), interpolation=cv2.INTER_CUBIC)
            bbox_local_scaled_image = cv2.resize(bbox_local_scaled_image, dsize=(1280, 720), interpolation=cv2.INTER_CUBIC)

            if not os.path.exists(vis_folder):
                os.makedirs(vis_folder)
            cv2.imwrite(vis_folder + os.sep + image_name[:-4] + "-P" + str(i) + "-" + vis_suffix + "-Attn.jpg", local_scaled_image)
            cv2.imwrite(vis_folder + os.sep + image_name[:-4] + "-P" + str(i) + "-" + vis_suffix + "-Weights.jpg", (excitation * 255).round().clip(0, 255).astype(np.uint8))
            cv2.imwrite(vis_folder + os.sep + image_name[:-4] + "-P" + str(i) + "-" + vis_suffix + "-Det.jpg", bbox_local_scaled_image)

            # cv2.imshow('bp', local_scaled_image)
            # cv2.imshow('excitation', excitation)
            # cv2.imshow('bbox_local_scaled_image', bbox_local_scaled_image)
            # cv2.waitKey()

        # res_confidence, distances, angle = net(image_tensor)
        # confidence = res_confidence.squeeze().data.cpu().numpy()
        # distances = distances.squeeze().data.cpu().numpy()
        # angle = angle.squeeze().data.cpu().numpy()
        # polys = parse_polys(confidence, distances, angle, 0.5, 0.3)
        # with open('{}'.format(os.path.join(output_folder, 'res_{}.txt'.format(prefix))), 'w') as f:
        #     for id in range(polys.shape[0]):
        #         f.write('{}, {}, {}, {}, {}, {}, {}, {}\n'.format(
        #             int(polys[id, 0] / scale_x), int(polys[id, 1] / scale_y), int(polys[id, 2] / scale_x), int(polys[id, 3] / scale_y),
        #             int(polys[id, 4] / scale_x), int(polys[id, 5] / scale_y), int(polys[id, 6] / scale_x), int(polys[id, 7] / scale_y)
        #         ))

        # prob_outputs = res_confidence.data.clone()
        # prob_outputs.fill_(0.0)
        # print(torch.argmax(res_confidence))
        # prob_outputs.flatten().scatter_(0, torch.argmax(res_confidence), 1.0)
        # prob_inputs = eb.utils.excitation_backprop(net, image_tensor.to(DEVICE), prob_outputs, contrastive=False, target_layer=0)
        # print(prob_inputs)
        # highlighted_img = prob_inputs.squeeze()
        # highlighted_img = highlighted_img * torch.tensor([0.229, 0.224, 0.225])[:, None, None] + torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        # highlighted_img = highlighted_img.permute(1, 2, 0).numpy()
        # print(highlighted_img.dtype)
        # print(highlighted_img)
        # print(highlighted_img.dtype)
        # cv2.imshow('bp', highlighted_img)

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
        # cv2.imshow('detections', image)
        # print(image_name)
        # cv2.waitKey()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-folder', type=str, required=True, help='path to the folder with test images')
    parser.add_argument('--output-folder', type=str, default='fots_test_results',
                        help='path to the output folder with result labels')
    parser.add_argument('--checkpoint', type=str, required=True, help='path to the checkpoint to test')
    # parser.add_argument('--height-size', type=int, default=1260, help='height size to resize input image')
    parser.add_argument('--vis-folder', type=str, default='vis',
                        help='path to the output folder with images')
    parser.add_argument('--vis-suffix', type=str, required=True,
                        help='added to images names')
    args = parser.parse_args()

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    net = FOTSModel()
    checkpoint = torch.load(args.checkpoint, map_location=DEVICE)
    net.load_state_dict(checkpoint['model_state_dict'])
    net = net.to(DEVICE)

    test(net, args.images_folder, args.output_folder, args.vis_folder, args.vis_suffix)
