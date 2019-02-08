import torch
import torch.utils.data
import os
import cv2
import numpy as np
import re


def point_dist_to_line(p1, p2, p3):
    # compute the distance from p3 to p1-p2
    if not np.array_equal(p1, p2):
        return np.abs(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)
    else:
        return np.linalg.norm(p3 - p1)


def transform(im, quads, texts, file_name, icdar):
    # upscale
    upscaled = cv2.resize(im, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    quads = quads * 2
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
    goodPoint = None
    for _ in range(50):  # TODO we almost never get crops with texts with such approach
        point = (int(torch.randint(low=0, high=nW - 640, size=(1,)).item()),
                 int(torch.randint(low=0, high=nH - 640, size=(1,)).item()))  # (x, y)
        intersect = False
        for bbox in quads:
            mins = np.amin(bbox, axis=0, out=None, keepdims=False)
            maxes = np.amax(bbox, axis=0, out=None, keepdims=False)
            if mins[0] < point[0] < maxes[0] or mins[1] < point[1] < maxes[1] or mins[0] < point[0] + 640 < maxes[0] or mins[1] < point[1] + 640 < maxes[1]:
                intersect = True
                break
        if not intersect:
            goodPoint = point
            break
    if goodPoint:
        # stretched = im
        # goodPoint = (0, 0)
        cropped = stretched[goodPoint[1]:goodPoint[1] + 640, goodPoint[0]:goodPoint[0] + 640]
        quads -= np.array(goodPoint)
        quads = np.int0(quads / 4)
        rboxes = []
        for quad in quads:
            rboxes.append(cv2.minAreaRect(quad))
        # cropped = cv2.resize(cropped, None, fx=0.25, fy=0.25)
        # cv2.polylines(cropped, quads, True, (0, 255, 255))
        classification = np.zeros((160, 160), dtype=cropped.dtype)
        training_mask = np.ones(classification.shape, dtype=cropped.dtype)  # TODO take NOT CARE texts into account
        regression = np.zeros((4,) + classification.shape, dtype=float)
        tmp_regression = np.empty(classification.shape, dtype=cropped.dtype)
        thetas = np.zeros(classification.shape, dtype=float)
        for rbox in rboxes:
            tmp_regression.fill(0)
            poly = cv2.boxPoints(rbox)
            int_poly = np.int0(poly)
            cv2.fillConvexPoly(classification, int_poly, 1)
            cv2.fillConvexPoly(training_mask, int_poly, 0)
            shrunk_rbox = rbox[0], (rbox[1][0] * 0.4, rbox[1][1] * 0.4), rbox[2]
            cv2.fillConvexPoly(training_mask, np.int0(cv2.boxPoints(shrunk_rbox)), 1) # TODO use shrunk poly
            cv2.fillConvexPoly(tmp_regression, int_poly, 1)
            points = np.nonzero(tmp_regression)
            pointsT = np.transpose(points)
            for point in pointsT:
                for plane in range(3):  # TODO looks that it is really slow
                    regression[(plane, ), tuple(point)] = point_dist_to_line(poly[plane], poly[plane + 1], point)
                regression[(plane, ) + tuple(point)] = point_dist_to_line(poly[plane], poly[0], point)
            thetas[points] = rbox[2]
        permuted = np.transpose(cropped, (2, 1, 0))  # TODO check if I should swap w and h
        return torch.from_numpy(permuted).float(), torch.from_numpy(classification).float(),  torch.from_numpy(regression).float(), torch.from_numpy(thetas).float(), torch.from_numpy(training_mask).float(), file_name
        # return cropped, classification, regression, thetas, training_mask, file_name
    else:
        print('could not find')
        return icdar[int(torch.randint(low=0, high=len(icdar), size=(1,)).item())]


class ICDAR2015(torch.utils.data.Dataset):
    def __init__(self, root, train, transform):
        self.transform = transform
        self.root = root
        self.img_dir = 'ch4_training_images'
        self.labels_dir = 'ch4_training_localization_transcription_gt'
        self.image_prefix = []
        self.pattern = re.compile('^' + '(\\d+),' * 8 + '(.+)$')
        if train: # TODO else
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
            texts.append('###' == matches[8])

        return transform(img, np.stack(quads), texts, self.image_prefix[idx], self)


if '__main__' == __name__:
    import torch
    icdar = ICDAR2015('C:\\Users\\vzlobin\\Documents\\repo\\FOTS.PyTorch\\data\\icdar\\icdar2015\\4.4\\training', True, transform)
    # dl = torch.utils.data.DataLoader(icdar, batch_size=2, shuffle=False, sampler=None, batch_sampler=None, num_workers=1, pin_memory = False, drop_last = False, timeout = 0, worker_init_fn = None)
    # for cropped, classification, regression, thetas, training_mask, file_names in dl:
    #     print(file_names)
    crop, classif, regression, thetas, training_mask, file_name = icdar[0]
    cv2.imshow('', crop / 255)
    cv2.waitKey(0)
    # print(crop.shape)
    # print(classif.shape)
    cv2.imshow('', training_mask * 255)
    cv2.waitKey(0)
    # m = np.amax(regression)
    # for i in range(4):
    #     cv2.imshow('', np.array(np.around(regression[:, :, i] * 255 / m), dtype=np.uint8))
    #     cv2.waitKey(0)
    # minim = np.amin(thetas)
    # print(minim)
    # thetas = thetas - minim
    # m = np.amax(thetas)
    # print(m)
    # print(thetas)
    #
    # cv2.imshow('', np.array(np.around(thetas * 255 / m), dtype=np.uint8))
    # cv2.waitKey(0)
