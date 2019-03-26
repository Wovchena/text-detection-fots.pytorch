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
from shapely.geometry import Polygon, box


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
    quads /= 4
    crop_point = (torch.randint(low=0, high=stretched.shape[1] // 4 - 160, size=(1,), dtype=torch.int16).item(),
                  torch.randint(low=0, high=stretched.shape[0] // 4 - 160, size=(1,), dtype=torch.int16).item())
    crop_box = box(crop_point[0], crop_point[1], crop_point[0] + 160, crop_point[1] + 160)

    training_mask = np.ones((160, 160), dtype=float)
    classification = np.zeros((160, 160), dtype=float)
    regression = np.zeros((4,) + classification.shape, dtype=float)
    tmp_cls = np.empty(classification.shape, dtype=float)
    thetas = np.zeros(classification.shape, dtype=float)

    for quad_id, quad in enumerate(quads):
        polygon = Polygon(quad)
        intersected_polygon = polygon.intersection(crop_box)
        if type(intersected_polygon) is Polygon:
            minAreaRect = cv2.minAreaRect(quad.astype(np.float32))
            shrinkage = min(minAreaRect[1][0], minAreaRect[1][1]) * 0.6
            intersected_quad = np.array(intersected_polygon.exterior.coords[:-1])
            intersected_quad -= crop_point
            intersected_minAreaRect = cv2.minAreaRect(intersected_quad.astype(np.float32))
            intersected_minAreaRect_boxPoints = cv2.boxPoints(intersected_minAreaRect)
            cv2.fillConvexPoly(training_mask, intersected_minAreaRect_boxPoints.round().astype(int), 0)
            shrunk_width_and_height = (intersected_minAreaRect[1][0] - shrinkage, intersected_minAreaRect[1][1] - shrinkage)
            if shrunk_width_and_height[0] >= 0 and shrunk_width_and_height[1] >= 0 and texts[quad_id]:
                shrunk_minAreaRect = intersected_minAreaRect[0], shrunk_width_and_height, intersected_minAreaRect[2]

                poly = intersected_minAreaRect_boxPoints
                if intersected_minAreaRect[2] >= -45:
                    poly = np.array([poly[1], poly[2], poly[3], poly[0]])
                else:
                    poly = np.array([poly[2], poly[3], poly[0], poly[1]])
                angle_cos = (poly[2, 0] - poly[3, 0]) / np.sqrt(
                    (poly[2, 0] - poly[3, 0]) ** 2 + (poly[2, 1] - poly[3, 1]) ** 2 + 1e-5)  # TODO tg or ctg
                angle = np.arccos(angle_cos)
                if poly[2, 1] > poly[3, 1]:
                    angle *= -1
                angle += 45 * np.pi / 180  # [0, pi/2] for learning, actually [-pi/4, pi/4]

                tmp_cls.fill(0)
                round_shrink_minAreaRect_boxPoints = cv2.boxPoints(shrunk_minAreaRect)
                cv2.fillConvexPoly(tmp_cls, round_shrink_minAreaRect_boxPoints.round(out=round_shrink_minAreaRect_boxPoints).astype(int), 1)
                cv2.rectangle(tmp_cls, (0, 0), (tmp_cls.shape[1] - 1, tmp_cls.shape[0] - 1), 0, thickness=int(round(shrinkage * 2)))
                if 0 == np.count_nonzero(tmp_cls) and 0.1 < torch.rand(1).item():
                    return icdar[torch.randint(low=0, high=len(icdar), size=(1,), dtype=torch.int16).item()]

                classification += tmp_cls
                training_mask += tmp_cls
                thetas += tmp_cls * angle

                points = np.nonzero(tmp_cls)
                pointsT = np.transpose(points)
                for point in pointsT:
                    for plane in range(3):  # TODO looks that it is really slow
                        regression[(plane,) + tuple(point)] = point_dist_to_line(poly[plane], poly[plane + 1],
                                                                                 np.array((point[1], point[0])))
                    regression[(3,) + tuple(point)] = point_dist_to_line(poly[3], poly[0],
                                                                         np.array((point[1], point[0])))
    cropped = stretched[crop_point[1] * 4:crop_point[1] * 4 + 640, crop_point[0] * 4:crop_point[0] * 4 + 640]
    cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB).astype(np.float64) / 255
    permuted = np.transpose(cropped, (2, 0, 1))
    permuted = torch.from_numpy(permuted).float()
    permuted = normalizer(permuted)
    return permuted, torch.from_numpy(classification).float(), torch.from_numpy(regression).float(), torch.from_numpy(
        thetas).float(), torch.from_numpy(training_mask).float()


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
        return transform(img, np.stack(quads), texts, self.normalizer, self)


class SynthText(torch.utils.data.Dataset):
    def __init__(self, root, train, transform):
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
        cv2.imshow('orig', cropped[:, :, ::-1])
        cropped = cv2.resize(cropped, (160, 160))
        cv2.imshow('img', cropped[:, :, ::-1] * training_mask.numpy()[:, :, None])
        cv2.imshow('training_mask', training_mask.numpy() * 255)
        cv2.imshow('classification', classification.numpy() * 255)
        cv2.waitKey(0)
        regression = regression.numpy()
        for i in range(4):
            m = np.amax(regression[i])
            cv2.imshow('', regression[i, :, :] / m)
            cv2.waitKey(0)
        thetas = thetas.numpy()
        minim = np.amin(thetas)
        m = np.amax(thetas)
        print(m * 180 / np.pi)
        cv2.imshow('', np.array(np.around(thetas * 255 / m * 180 / np.pi), dtype=np.uint8))
        cv2.waitKey(0)
