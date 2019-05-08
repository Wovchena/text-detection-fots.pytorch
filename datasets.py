import math
import os
import random
import re

import cv2
import numpy as np
import scipy.io
import torch
import torch.utils.data
import torchvision


def point_dist_to_line(p1, p2, p3):
    # compute the distance from p3 to p1-p2
    if not np.array_equal(p1, p2):
        return np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)
    else:
        return np.linalg.norm(p3 - p1)


def transform(im, quads, texts, normalizer, icdar):
    # upscale
    scale = 2560 / np.maximum(im.shape[0], im.shape[1])
    upscaled = cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    quads = quads * scale
    # rotate
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = upscaled.shape[:2]
    (cX, cY) = (w / 2, h / 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    angle = torch.empty(1).uniform_(-10, 10).item()
    M = cv2.getRotationMatrix2D((cX, cY), angle=angle, scale=1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))  # TODO replace with round and do it later
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    rotated = cv2.warpAffine(upscaled, M, (nW, nH))
    quads = cv2.transform(quads, M)
    # stretch
    strechK = torch.empty(1).uniform_(0.8, 1.2).item()
    stretched = cv2.resize(rotated, None, fx=1, fy=strechK, interpolation=cv2.INTER_CUBIC)
    quads[:, :, 1] = quads[:, :, 1] * strechK
    nH *= strechK
    nH = int(nH)
    # crop
    output_scale = 2
    quads /= output_scale
    smaller_bounds = []
    bigger_bounds = []
    for quad in quads:
        smaller_bound = np.amin(quad, axis=0, out=None, keepdims=False)
        smaller_bounds.append(smaller_bound)
        bigger_bound = np.amax(quad, axis=0, out=None, keepdims=False)
        bigger_bounds.append(bigger_bound)
    smaller_bounds_of_trainable_quads = np.array(smaller_bounds)[texts]
    crop_size = 640 // output_scale
    if len(smaller_bounds_of_trainable_quads):
        the_smallest_bound_x, the_smallest_bound_y = np.amin(smaller_bounds_of_trainable_quads, axis=0)
        the_biggest_bound_x, the_biggest_bound_y = np.amax(smaller_bounds_of_trainable_quads, axis=0)  # top left corner of corp region will be randomed, so there is max(smaller_bounds)
    else:
        the_smallest_bound_x, the_smallest_bound_y = 0, 0
        the_biggest_bound_x, the_biggest_bound_y = stretched.shape[1] // output_scale - crop_size, stretched.shape[0] // output_scale - crop_size
    the_smallest_crop_point_x = max(int(the_smallest_bound_x) - crop_size, 0)
    the_smallest_crop_point_y = max(int(the_smallest_bound_y) - crop_size, 0)
    the_biggest_crop_point_x = min(math.ceil(the_biggest_bound_x), stretched.shape[1] // output_scale - crop_size)
    the_biggest_crop_point_y = min(math.ceil(the_biggest_bound_y), stretched.shape[0] // output_scale - crop_size)
    if the_smallest_crop_point_x >= the_biggest_crop_point_x or the_smallest_crop_point_y >= the_biggest_crop_point_y:  # torch.randint requires this
        # print('cant crop')
        return icdar[torch.randint(low=0, high=len(icdar), size=(1,), dtype=torch.int16).item()]
    good_crop_point = None  # (150, 0)
    for _ in range(100):  # TODO it can be better to find intersections with rboxes or quads and remove if str(dirEntry.name) != 'img_636.jpg' and str(dirEntry.name) != 'img_624.jpg'
        crop_point = (torch.randint(low=the_smallest_crop_point_x, high=the_biggest_crop_point_x, size=(1,), dtype=torch.int16).item(),
                      torch.randint(low=the_smallest_crop_point_y, high=the_biggest_crop_point_y, size=(1,), dtype=torch.int16).item())
        covered_at_least_one_quad = False
        for quad_i in range(len(quads)):
            if texts[quad_i] and smaller_bounds[quad_i][0] >= crop_point[0] and smaller_bounds[quad_i][1] >= crop_point[1] \
                    and bigger_bounds[quad_i][0] <= crop_point[0] + crop_size and bigger_bounds[quad_i][1] <= crop_point[1] + crop_size:
                covered_at_least_one_quad = True
                break
        if covered_at_least_one_quad:
            good_crop_point = crop_point
            break
    if good_crop_point:
        cropped = stretched[good_crop_point[1] * output_scale:good_crop_point[1] * output_scale + crop_size*output_scale, good_crop_point[0] * output_scale:good_crop_point[0] * output_scale + crop_size*output_scale]
        quads -= np.array(good_crop_point)
        int_quads = quads.round().astype(int)
        minAreaRects = [cv2.minAreaRect(int_quad) for int_quad in int_quads]
        # cropped = cv2.resize(cropped, None, fx=0.25, fy=0.25)  # TODO comment!!!
        # cv2.polylines(cropped, int_quads, True, (0, 255, 255))   # TODO comment!!!
        training_mask = np.ones((crop_size, crop_size), dtype=float)  # TODO take NOT CARE texts into account
        classification = np.zeros((crop_size, crop_size), dtype=float)
        regression = np.zeros((4,) + classification.shape, dtype=float)
        tmp_regression = np.empty(classification.shape, dtype=float)
        thetas = np.zeros(classification.shape, dtype=float)
        for quad_i in range(len(minAreaRects)):
            minAreaRect = minAreaRects[quad_i]
            #assert (minAreaRect[2] >= -90) and (minAreaRect[2] <= 0), 'angle: {}'.format(minAreaRect[2])
            shrinkage = min(minAreaRect[1][0], minAreaRect[1][1]) * 0.6
            shrunk_minAreaRect = minAreaRect[0], (minAreaRect[1][0] - shrinkage, minAreaRect[1][1] - shrinkage), minAreaRect[2]
            poly = cv2.boxPoints(minAreaRect)
            if minAreaRect[2] >= -45:
                poly = np.array([poly[1], poly[2], poly[3], poly[0]])
            else:
                poly = np.array([poly[2], poly[3], poly[0], poly[1]])
            angle_cos = (poly[2, 0] - poly[3, 0]) / np.sqrt((poly[2, 0] - poly[3, 0])**2 + (poly[2, 1] - poly[3, 1])**2 + 1e-5)  # TODO tg or ctg
            angle = np.arccos(angle_cos)
            if poly[2, 1] > poly[3, 1]:
                angle *= -1
            angle += 45 * np.pi / 180  # [0, pi/2] for learning, actually [-pi/4, pi/4]
            int_poly = poly.round().astype(int)
            if smaller_bounds[quad_i][0] >= good_crop_point[0] and smaller_bounds[quad_i][1] >= good_crop_point[1] \
                    and bigger_bounds[quad_i][0] <= good_crop_point[0] + crop_size and bigger_bounds[quad_i][1] <= good_crop_point[1] + crop_size:
                tmp_regression.fill(0)
                cv2.fillConvexPoly(classification, int_poly, 1)
                cv2.fillConvexPoly(training_mask, int_poly, 0)
                if texts[quad_i]:  # dont account ###
                    cv2.fillConvexPoly(training_mask, np.int0(cv2.boxPoints(shrunk_minAreaRect)), 1)
                cv2.fillConvexPoly(tmp_regression, int_poly, 1)
                points = np.nonzero(tmp_regression)
                pointsT = np.transpose(points)
                for point in pointsT:
                    for plane in range(3):  # TODO looks that it is really slow
                        regression[(plane,) + tuple(point)] = point_dist_to_line(int_poly[plane], int_poly[plane + 1], np.array((point[1], point[0])))
                    regression[(3,) + tuple(point)] = point_dist_to_line(int_poly[3], int_poly[0], np.array((point[1], point[0])))
                thetas[points] = angle
            else:
                cv2.fillConvexPoly(training_mask, int_poly, 0)
        cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        permuted = np.transpose(cropped, (2, 0, 1))
        permuted = torch.from_numpy(permuted).float()

        permuted = normalizer(permuted)

        return permuted, torch.from_numpy(classification).float(), torch.from_numpy(regression).float(), torch.from_numpy(thetas).float(), torch.from_numpy(training_mask).float()
    else:
        # print('could not find good crop')
        return icdar[torch.randint(low=0, high=len(icdar), size=(1,), dtype=torch.int16).item()]


class ICDAR2015(torch.utils.data.Dataset):
    def __init__(self, root, transform):
        self.transform = transform
        self.root = root
        self.img_dir = 'ch4_training_images'
        self.labels_dir = 'ch4_training_localization_transcription_gt'
        self.image_prefix = []
        self.pattern = re.compile('^' + '(\\d+),' * 8 + '(.+)$')
        self.normalizer = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                           std=[0.229, 0.224, 0.225])
        for dirEntry in os.scandir(os.path.join(root, 'ch4_training_images')):
            if str(dirEntry.name) != 'img_636.jpg' and str(dirEntry.name) != 'img_624.jpg':
                self.image_prefix.append(dirEntry.name[:-4])

    def __len__(self):
        return len(self.image_prefix)

    def __getitem__(self, idx):
        img = cv2.imread(os.path.join(os.path.join(self.root, self.img_dir), self.image_prefix[idx] + '.jpg'), cv2.IMREAD_COLOR).astype(np.float32)
        quads = []
        texts = []
        lines = [line.rstrip('\n') for line in open(os.path.join(os.path.join(self.root, self.labels_dir), 'gt_' + self.image_prefix[idx] + '.txt'),
                                                    encoding='utf-8-sig')]
        for line in lines:
            matches = self.pattern.findall(line)[0]
            numbers = np.array(matches[:8], dtype=float)
            quads.append(numbers.reshape((4, 2)))
            texts.append('###' != matches[8])
        for trainable in texts:
            if trainable:  # at least one trainable quad
                return transform(img, np.stack(quads), texts, self.normalizer, self)
        # print('no trainable quads', self.image_prefix[idx])
        return self[torch.randint(low=0, high=len(self), size=(1,), dtype=torch.int16).item()]


class SynthText(torch.utils.data.Dataset):
    def __init__(self, root, transform):
        self.transform = transform
        self.root = root
        self.labels = scipy.io.loadmat(os.path.join(root, 'gt.mat'))
        #sample_path = labels['imnames'][0, 1][0]
        #sample_boxes = np.transpose(labels['wordBB'][0, 1], (2, 1, 0))
        self.pattern = re.compile('^' + '(\\d+),' * 8 + '(.+)$')
        self.normalizer = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                           std=[0.229, 0.224, 0.225])

    def __len__(self):
        return self.labels['imnames'].shape[1] // 20

    def __getitem__(self, idx):
        idx = (idx * 20) + random.randint(0, 19)  # compensate dataset size, while maintain diversity
        img = cv2.imread(os.path.join(self.root, self.labels['imnames'][0, idx][0]), cv2.IMREAD_COLOR).astype(np.float32)
        quads = []
        texts = []
        coordinates = self.labels['wordBB'][0, idx]
        if len(coordinates.shape) == 2:
            coordinates = np.expand_dims(coordinates, axis=2)
        quads.extend(np.transpose(coordinates, (2, 1, 0)))
        texts.extend([True] * len(quads))

        return transform(img, np.stack(quads), texts, self.normalizer, self)


if '__main__' == __name__:
    icdar = ICDAR2015('C:\\Users\\vzlobin\\Documents\\repo\\FOTS.PyTorch\\data\\icdar\\icdar2015\\4.4\\training', transform)
    # dl = torch.utils.data.DataLoader(icdar, batch_size=4, shuffle=False, sampler=None, batch_sampler=None, num_workers=4, pin_memory = False, drop_last = False, timeout = 0, worker_init_fn = None)
    for image_i in range(len(icdar)):
        normalized, classification, regression, thetas, training_mask = icdar[image_i]
        permuted = normalized * torch.tensor([0.229, 0.224, 0.225])[:, None, None] + torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        cropped = permuted.permute(1, 2, 0).numpy()
        cv2.imshow('img', cropped[:, :, ::-1])
        cv2.imshow('training_mask', training_mask.numpy() * 255)
        cv2.waitKey(0)
        # cv2.imshow('', classification * 255)
        # cv2.waitKey(0)
        # for i in range(4):
        #     m = np.amax(regression[i])
        #     cv2.imshow('', regression[i, :, :] / m)
        #     cv2.waitKey(0)
        # minim = np.amin(thetas)
        # thetas = thetas - minim
        # m = np.amax(thetas)
        # cv2.imshow('', np.array(np.around(thetas * 255 / m), dtype=np.uint8))
        # cv2.waitKey(0)
