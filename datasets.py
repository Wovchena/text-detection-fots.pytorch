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
import shapely


def point_dist_to_line(p1, p2, p3):
    """Compute the distance from p3 to p2-p1."""
    if not np.array_equal(p1, p2):
        return np.abs(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)
    else:
        return np.linalg.norm(p3 - p1)

def has_selfintersection(quad):
    bottom_right_l1 = quad[:2].max(axis=0)
    upper_left_l1 = quad[:2].min(axis=0)
    bottom_right_l2 = quad[2:].max(axis=0)
    upper_left_l2 = quad[2:].min(axis=0)
    if bottom_right_l1[0] < upper_left_l2[0] or bottom_right_l1[1] < upper_left_l2[1]\
            or bottom_right_l2[0] < upper_left_l1[0] or bottom_right_l2[1] < upper_left_l1[1]:
        return False

    h = np.hstack((quad, np.ones((4, 1)))) # h for homogeneous
    l1 = np.cross(h[0], h[1])           # get first line
    l2 = np.cross(h[2], h[3])           # get second line
    x, y, z = np.cross(l1, l2)          # point of intersection
    if z == 0:                          # lines are parallel
        return False
    x = x/z
    y = y/z
    return max(upper_left_l1[0], upper_left_l2[0]) <= x <= min(bottom_right_l1[0], bottom_right_l2[0])\
        and max(upper_left_l1[1], upper_left_l2[1]) <= y <= min(bottom_right_l1[1], bottom_right_l2[1])


IN_OUT_RATIO = 4
IN_SIDE = 640
OUT_SIDE = IN_SIDE // IN_OUT_RATIO


def transform(im, quads, texts, normalizer, data_set):
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

    quads /= IN_OUT_RATIO

    training_mask = np.ones((OUT_SIDE, OUT_SIDE), dtype=float)
    classification = np.zeros((OUT_SIDE, OUT_SIDE), dtype=float)
    regression = np.zeros((4,) + classification.shape, dtype=float)
    tmp_cls = np.empty(classification.shape, dtype=float)
    thetas = np.zeros(classification.shape, dtype=float)

    # crop
    crop_max_y = stretched.shape[0] // IN_OUT_RATIO - OUT_SIDE  # since Synth has some low images, there is a chance that y coord of crop can be zero only
    if 0 != crop_max_y:
        crop_point = (torch.randint(low=0, high=stretched.shape[1] // IN_OUT_RATIO - OUT_SIDE, size=(1,), dtype=torch.int16).item(),
                      torch.randint(low=0, high=stretched.shape[0] // IN_OUT_RATIO - OUT_SIDE, size=(1,), dtype=torch.int16).item())
    else:
        crop_point = (torch.randint(low=0, high=stretched.shape[1] // IN_OUT_RATIO - OUT_SIDE, size=(1,), dtype=torch.int16).item(),
                      0)
    crop_box = box(crop_point[0], crop_point[1], crop_point[0] + OUT_SIDE, crop_point[1] + OUT_SIDE)

    for quad_id, quad in enumerate(quads):
        polygon = Polygon(quad)
        try:
            intersected_polygon = polygon.intersection(crop_box)
        except shapely.errors.TopologicalError:  # some points of quads in Synth can be in wrong order
            quad[1], quad[2] = quad[2], quad[1]
            polygon = Polygon(quad)
            intersected_polygon = polygon.intersection(crop_box)
        if type(intersected_polygon) is Polygon:
            intersected_quad = np.array(intersected_polygon.exterior.coords[:-1])
            intersected_quad -= crop_point
            intersected_minAreaRect = cv2.minAreaRect(intersected_quad.astype(np.float32))
            intersected_minAreaRect_boxPoints = cv2.boxPoints(intersected_minAreaRect)
            cv2.fillConvexPoly(training_mask, intersected_minAreaRect_boxPoints.round().astype(int), 0)
            minAreaRect = cv2.minAreaRect(quad.astype(np.float32))
            shrinkage = min(minAreaRect[1][0], minAreaRect[1][1]) * 0.6
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

                classification += tmp_cls
                training_mask += tmp_cls
                thetas += tmp_cls * angle

                points = np.nonzero(tmp_cls)
                pointsT = np.transpose(points)
                for point in pointsT:
                    for plane in range(3):  # TODO widht - dist, height - other dist and more percise dist
                        regression[(plane,) + tuple(point)] = point_dist_to_line(poly[plane], poly[plane + 1], np.array((point[1], point[0]))) * IN_OUT_RATIO
                    regression[(3,) + tuple(point)] = point_dist_to_line(poly[3], poly[0], np.array((point[1], point[0]))) * IN_OUT_RATIO
    if 0 == np.count_nonzero(classification) and 0.1 < torch.rand(1).item():
        return data_set[torch.randint(low=0, high=len(data_set), size=(1,), dtype=torch.int16).item()]
    # avoiding training on black corners decreases hmean, see d9c727a8defbd1c8022478ae798c907ccd2fa0b2
    cropped = stretched[crop_point[1] * IN_OUT_RATIO:crop_point[1] * IN_OUT_RATIO + IN_SIDE, crop_point[0] * IN_OUT_RATIO:crop_point[0] * IN_OUT_RATIO + IN_SIDE]
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
        self.broken_image_ids = set()
        #sample_path = labels['imnames'][0, 1][0]
        #sample_boxes = np.transpose(labels['wordBB'][0, 1], (2, 1, 0))
        self.pattern = re.compile('^' + '(\\d+),' * 8 + '(.+)$')
        self.normalizer = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                           std=[0.229, 0.224, 0.225])

    def __len__(self):
        return self.labels['imnames'].shape[1] // 105  # there are more than 105 text images for each source image

    def __getitem__(self, idx):
        idx = (idx * 105) + random.randint(0, 104)  # compensate dataset size, while maintain diversity
        if idx in self.broken_image_ids:
            return self[torch.randint(low=0, high=len(self), size=(1,), dtype=torch.int16).item()]
        img = cv2.imread(os.path.join(self.root, self.labels['imnames'][0, idx][0]), cv2.IMREAD_COLOR).astype(np.float32)
        if 180 >= img.shape[0]:  # image is too low, it will not be possible to crop 640x640 after transformations
            self.broken_image_ids.add(idx)
            return self[torch.randint(low=0, high=len(self), size=(1,), dtype=torch.int16).item()]
        coordinates = self.labels['wordBB'][0, idx]
        if len(coordinates.shape) == 2:
            coordinates = np.expand_dims(coordinates, axis=2)
        transposed = np.transpose(coordinates, (2, 1, 0))
        if (transposed > 0).all() and (transposed[:, :, 1] < img.shape[1]).all() and (transposed[:, :, 1] < img.shape[0]).all():
            if ((transposed[:, 0] != transposed[:, 1]).all() and
                (transposed[:, 0] != transposed[:, 2]).all() and
                (transposed[:, 0] != transposed[:, 3]).all() and
                (transposed[:, 1] != transposed[:, 2]).all() and
                (transposed[:, 1] != transposed[:, 3]).all() and
                (transposed[:, 2] != transposed[:, 3]).all()):  # boxes can be in a form [p1, p1, p2, p2], while we need [p1, p2, p3, p4]
                    return transform(img, transposed, (True, ) * len(transposed), self.normalizer, self)
        self.broken_image_ids.add(idx)
        return self[torch.randint(low=0, high=len(self), size=(1,), dtype=torch.int16).item()]


if '__main__' == __name__:
    icdar = ICDAR2015('C:\\Users\\vzlobin\\Documents\\repo\\FOTS.PyTorch\\data\\icdar\\icdar2015\\4.4\\training', transform)
    # dl = torch.utils.data.DataLoader(icdar, batch_size=4, shuffle=False, sampler=None, batch_sampler=None, num_workers=4, pin_memory = False, drop_last = False, timeout = 0, worker_init_fn = None)
    for image_i in range(len(icdar)):
        normalized, classification, regression, thetas, training_mask = icdar[image_i]
        permuted = normalized * torch.tensor([0.229, 0.224, 0.225])[:, None, None] + torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        cropped = permuted.permute(1, 2, 0).numpy()
        cv2.imshow('orig', cv2.resize(cropped[:, :, ::-1], (640, 640)))
        cropped = cv2.resize(cropped, (160, 160))
        cv2.imshow('img', cv2.resize(cropped[:, :, ::-1] * training_mask.numpy()[:, :, None], (640, 640)))
        cv2.imshow('training_mask', cv2.resize(training_mask.numpy() * 255, (640, 640)))
        cv2.imshow('classification', cv2.resize(classification.numpy() * 255, (640, 640)))
        regression = regression.numpy()
        for i in range(4):
            m = np.amax(regression[i])
            if 0 != m:
                cv2.imshow(str(i), cv2.resize(regression[i, :, :] / m, (640, 640)))
            else:
                cv2.imshow(str(i), cv2.resize(regression[i, :, :], (640, 640)))
        thetas = thetas.numpy()
        minim = np.amin(thetas)
        m = np.amax(thetas)
        print(m * 180 / np.pi)
        cv2.imshow('angle', cv2.resize(np.array(np.around(thetas * 255 / m * 180 / np.pi), dtype=np.uint8), (640, 640)))
        cv2.waitKey(0)
