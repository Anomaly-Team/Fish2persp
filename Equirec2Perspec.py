import os
import sys
import cv2
import numpy as np
from fish2pano import fish2pano
import pickle

def xyz2lonlat(xyz):
    atan2 = np.arctan2
    asin = np.arcsin

    norm = np.linalg.norm(xyz, axis=-1, keepdims=True)
    xyz_norm = xyz / norm
    x = xyz_norm[..., 0:1]
    y = xyz_norm[..., 1:2]
    z = xyz_norm[..., 2:]

    lon = atan2(x, z)
    lat = asin(y)
    lst = [lon, lat]

    out = np.concatenate(lst, axis=-1)
    return out

def lonlat2XY(lonlat, shape):
    X = (lonlat[..., 0:1] / (2 * np.pi) + 0.5) * (shape[1] - 1)
    Y = (lonlat[..., 1:] / (np.pi) + 0.5) * (shape[0] - 1)
    lst = [X, Y]
    out = np.concatenate(lst, axis=-1)

    return out 


class Equirectangular:
    def __init__(self, img_name):
        # self._img = cv2.imread(img_name, cv2.IMREAD_COLOR)
        # [self._height, self._width, _] = self._img.shape
        #cp = self._img.copy()  
        #w = self._width
        #self._img[:, :w/8, :] = cp[:, 7*w/8:, :]
        #self._img[:, w/8:, :] = cp[:, :7*w/8, :]
        'init'
    

def GetPerspective(img, FOV, THETA, PHI, height, width):
    #
    # THETA is left/right angle, PHI is up/down angle, both in degree
    #

    f = 0.5 * width * 1 / np.tan(0.5 * FOV / 180.0 * np.pi)
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    K = np.array([
            [f, 0, cx],
            [0, f, cy],
            [0, 0,  1],
        ], np.float32)
    K_inv = np.linalg.inv(K)
    
    x = np.arange(width)
    y = np.arange(height)
    x, y = np.meshgrid(x, y)
    z = np.ones_like(x)
    
    
    xyz = np.concatenate([x[..., None], y[..., None], z[..., None]], axis=-1)
    xyz = xyz @ K_inv.T

    y_axis = np.array([0.0, 1.0, 0.0], np.float32)
    x_axis = np.array([1.0, 0.0, 0.0], np.float32)
    R1, _ = cv2.Rodrigues(y_axis * np.radians(THETA))
    R2, _ = cv2.Rodrigues(np.dot(R1, x_axis) * np.radians(PHI))
    R = R2 @ R1
    xyz = xyz @ R.T
    lonlat = xyz2lonlat(xyz) 
    XY = lonlat2XY(lonlat, shape=img.shape).astype(np.float32)
    # persp = cv2.remap(self._img, XY[..., 0], XY[..., 1], cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)

    return XY[..., 0], XY[..., 1]


def fish2persp(img, FOV, THETA, PHI, height, width, pano_x, pano_y):
    #
    # THETA is left/right angle, PHI is up/down angle, both in degree
    #

    f = 0.5 * width * 1 / np.tan(0.5 * FOV / 180.0 * np.pi)
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    K = np.array([
            [f, 0, cx],
            [0, f, cy],
            [0, 0,  1],
        ], np.float32)
    K_inv = np.linalg.inv(K)
    
    x = np.arange(width)
    y = np.arange(height)
    x, y = np.meshgrid(x, y)
    z = np.ones_like(x)
    
    
    xyz = np.concatenate([x[..., None], y[..., None], z[..., None]], axis=-1)
    xyz = xyz @ K_inv.T

    y_axis = np.array([0.0, 1.0, 0.0], np.float32)
    x_axis = np.array([1.0, 0.0, 0.0], np.float32)
    R1, _ = cv2.Rodrigues(y_axis * np.radians(THETA))
    R2, _ = cv2.Rodrigues(np.dot(R1, x_axis) * np.radians(PHI))
    R = R2 @ R1
    xyz = xyz @ R.T
    lonlat = xyz2lonlat(xyz) 
    # _, pano_x, pano_y = fish2pano(img)
    pano_height, pano_width, _ = pano_x.shape
    pano_shape = (pano_height*2, pano_width, 1)
    persp_XY = lonlat2XY(lonlat, shape=pano_shape).astype(np.float32)
    persp_X = persp_XY[..., 0]
    persp_Y = persp_XY[..., 1]
    res_x = np.zeros((height, width, 1), np.float32)
    res_y = np.zeros((height, width, 1), np.float32)
    
    # pano2perspective
    for i in range(height):
        for j in range(width):
            res_x[i, j] = pano_x[int(persp_Y[i,j]) - img.shape[0] // 2, int(persp_X[i, j])]
            res_y[i, j] = pano_y[int(persp_Y[i,j]) - img.shape[0] // 2, int(persp_X[i, j])]
    
    return res_x, res_y

