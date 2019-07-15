import cv2
import numpy as np

from modules.nms import nms_locality, standard_nms


def parse_polys(cls, distances, angle, confidence_threshold=0.5, intersection_threshold=0.3, img=None):
    polys = []
    height, width = cls.shape

    #bin_cls = cls > confidence_threshold
    #t_dist = distances[0].copy()
    #t_dist[bin_cls == False] = 0
    #b_dist = distances[2].copy()
    #b_dist[bin_cls == False] = 0
    #cv2.imshow('cls', cls)
    #cv2.imshow('t_dist', t_dist.astype(np.uint8))
    #cv2.imshow('b_dist', b_dist.astype(np.uint8))
    #
    #thr_cls = cls.copy()
    #thr_cls[bin_cls == False] = 0
    #thr_cls = (thr_cls * 255).astype(np.uint8)
    #thr_cls = cv2.cvtColor(thr_cls, cv2.COLOR_GRAY2BGR)

    IN_OUT_RATIO = 4
    for y in range(height):
        for x in range(width):
            if cls[y, x] < confidence_threshold:
                continue
            #thr_cls_copy = thr_cls.copy()
            #thr_cls_copy[y, x] = [0, 0, 255]

            #a,b,c,d = distances[0, y, x], distances[1, y, x], distances[2, y, x], distances[3, y, x]
            poly_height = distances[0, y, x] + distances[2, y, x]
            poly_width = distances[1, y, x] + distances[3, y, x]
            #cv2.line(thr_cls_copy, (10, 10), (10 + int(poly_width), 10), (0, 255, 0))
            #cv2.line(thr_cls_copy, (10, 10), (10, int(10 + poly_height)), (0, 255, 0))

            poly_angle = angle[y, x] - np.pi / 4
            x_rot = x * np.cos(-poly_angle) + y * np.sin(-poly_angle)
            y_rot = -x * np.sin(-poly_angle) + y * np.cos(-poly_angle)
            poly_y_center = y_rot * IN_OUT_RATIO + (poly_height / 2 - distances[0, y, x])
            poly_x_center = x_rot * IN_OUT_RATIO - (poly_width / 2 - distances[1, y, x])
            poly = [
                int(((poly_x_center - poly_width / 2) * np.cos(poly_angle) + (poly_y_center - poly_height / 2) * np.sin(poly_angle))),
                int((-(poly_x_center - poly_width / 2) * np.sin(poly_angle) + (poly_y_center - poly_height / 2) * np.cos(poly_angle))),
                int(((poly_x_center + poly_width / 2) * np.cos(poly_angle) + (poly_y_center - poly_height / 2) * np.sin(poly_angle))),
                int((-(poly_x_center + poly_width / 2) * np.sin(poly_angle) + (poly_y_center - poly_height / 2) * np.cos(poly_angle))),
                int(((poly_x_center + poly_width / 2) * np.cos(poly_angle) + (poly_y_center + poly_height / 2) * np.sin(poly_angle))),
                int((-(poly_x_center + poly_width / 2) * np.sin(poly_angle) + (poly_y_center + poly_height / 2) * np.cos(poly_angle))),
                int(((poly_x_center - poly_width / 2) * np.cos(poly_angle) + (poly_y_center + poly_height / 2) * np.sin(poly_angle))),
                int((-(poly_x_center - poly_width / 2) * np.sin(poly_angle) + (poly_y_center + poly_height / 2) * np.cos(poly_angle))),
                cls[y, x]
            ]
            #pts = np.array(poly[:8]).reshape((4, 2)).astype(np.int32)
            #cv2.line(thr_cls_copy, (pts[0, 0], pts[0, 1]), (pts[1, 0], pts[1, 1]), color=(0, 255, 0))
            #cv2.line(thr_cls_copy, (pts[1, 0], pts[1, 1]), (pts[2, 0], pts[2, 1]), color=(0, 255, 0))
            #cv2.line(thr_cls_copy, (pts[2, 0], pts[2, 1]), (pts[3, 0], pts[3, 1]), color=(0, 255, 0))
            #cv2.line(thr_cls_copy, (pts[3, 0], pts[3, 1]), (pts[0, 0], pts[0, 1]), color=(0, 255, 0))
            #cv2.imshow('tmp', thr_cls_copy)
            #cv2.waitKey()

            polys.append(poly)

    polys = nms_locality(np.array(polys), intersection_threshold)
    if img is not None:
        for poly in polys:
            pts = np.array(poly[:8]).reshape((4, 2)).astype(np.int32)
            cv2.line(img, (pts[0, 0], pts[0, 1]), (pts[1, 0], pts[1, 1]), color=(0, 255, 0))
            cv2.line(img, (pts[1, 0], pts[1, 1]), (pts[2, 0], pts[2, 1]), color=(0, 255, 0))
            cv2.line(img, (pts[2, 0], pts[2, 1]), (pts[3, 0], pts[3, 1]), color=(0, 255, 0))
            cv2.line(img, (pts[3, 0], pts[3, 1]), (pts[0, 0], pts[0, 1]), color=(0, 255, 0))
        cv2.imshow('polys', img)
        cv2.waitKey()
    return polys
