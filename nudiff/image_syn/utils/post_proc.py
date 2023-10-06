import cv2
import numpy as np
import scipy.ndimage as ndi
from scipy.ndimage import filters, measurements
from scipy.ndimage.morphology import (
    binary_dilation,
    binary_fill_holes,
    distance_transform_cdt,
    distance_transform_edt,
)

from skimage.segmentation import watershed
from nudiff.image_syn.hovernet.utils import get_bounding_box, remove_small_objects

def watershed_inst(mask, raw=True):
    if raw:
        sem, edge = (mask[:,:,0] + 1) / 2, (mask[:,:,2] + 1) / 2
        sem[sem > 0.5] = 1
        sem[sem <= 0.5] = 0
        edge[edge > 0.5] = 1
        edge[edge <= 0.5] = 0
    else:
        sem_, edge_ = mask[:,:,0] / 255, mask[:,:,2] / 255
        sem = np.zeros_like(sem_)
        edge = np.zeros_like(edge_)
        sem[sem_ > 0.5] = 1
        edge[edge_ > 0.5] = 1
    distance = ndi.distance_transform_edt(sem)
    mask_ = np.zeros_like(sem)
    mask_[(edge > 0.5) | (sem > 0.5)] = 1
    mask_ = mask_ - edge
    markers, _ = ndi.label(mask_)
    labels = watershed(-distance, markers, mask=sem)
    return labels

def proc_np_hv_(blb_raw, h_dir_raw, v_dir_raw):

    # processing
    blb = np.array(blb_raw >= 0.5, dtype=np.int32)

    blb = measurements.label(blb)[0]
    blb = remove_small_objects(blb, min_size=10)
    blb[blb > 0] = 1  # background is 0 already

    h_dir = cv2.normalize(
        h_dir_raw, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
    )
    v_dir = cv2.normalize(
        v_dir_raw, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
    )

    sobelh = cv2.Sobel(h_dir, cv2.CV_64F, 1, 0, ksize=21)
    sobelv = cv2.Sobel(v_dir, cv2.CV_64F, 0, 1, ksize=21)

    sobelh = 1 - (
        cv2.normalize(
            sobelh, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
        )
    )
    sobelv = 1 - (
        cv2.normalize(
            sobelv, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
        )
    )

    overall = np.maximum(sobelh, sobelv)
    overall = overall - (1 - blb)
    overall[overall < 0] = 0

    dist = (1.0 - overall) * blb
    ## nuclei values form mountains so inverse to get basins
    dist = -cv2.GaussianBlur(dist, (3, 3), 0)

    overall = np.array(overall >= 0.4, dtype=np.int32)

    marker = blb - overall
    marker[marker < 0] = 0
    marker = binary_fill_holes(marker).astype("uint8")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    marker = cv2.morphologyEx(marker, cv2.MORPH_OPEN, kernel)
    marker = measurements.label(marker)[0]
    marker = remove_small_objects(marker, min_size=10)

    proced_pred = watershed(dist, markers=marker, mask=blb)

    return proced_pred

def proc_np_hv(pred):
    """Process Nuclei Prediction with XY Coordinate Map.

    Args:
        pred: prediction output, assuming
              channel 0 contain probability map of nuclei
              channel 1 containing the regressed X-map
              channel 2 containing the regressed Y-map

    """
    pred = np.array(pred, dtype=np.float32)
    blb_raw = pred[..., 0]
    
    # permutation
    max_inst = 0
    cases1 = [1, 2]
    cases2 = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    final_proced_pred = np.zeros_like(pred).astype(np.int32)
    for case1 in cases1:
        if case1 == 1:
            h_dir_raw_ = pred[..., 1]
            v_dir_raw_ = pred[..., 2]
        elif case1 == 2:
            h_dir_raw_ = pred[..., 2]
            v_dir_raw_ = pred[..., 1]
        for case2 in cases2:
            h_dir_raw = h_dir_raw_ * case2[0]
            v_dir_raw = v_dir_raw_ * case2[1]
            proced_pred = proc_np_hv_(blb_raw, h_dir_raw, v_dir_raw)
            num_inst = len(np.unique(proced_pred) ) - 1
            if num_inst > max_inst:
                max_inst = num_inst
                final_proced_pred = proced_pred
    
    return final_proced_pred

def get_instance_map(mask, mask_type='hv'):
    if mask_type == 'sdm':
        inst = watershed_inst(mask)
    elif mask_type == 'hv':
        pred = mask.copy() / 255
        pred[:,:,1:] = pred[:,:,1:] * 2 - 1
        inst = proc_np_hv(pred)
    else:
        raise NotImplementedError
    return inst
