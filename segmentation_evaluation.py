from sklearn.metrics import confusion_matrix
import numpy as np
import pathlib
import cv2
import os

def evaluate(preds,gt):
    VOID_CLASS = 0
    mask_valid = (preds != VOID_CLASS) & (gt != VOID_CLASS)
    preds = preds[mask_valid]
    gt    = gt[mask_valid]
    # Must have all the required classes (no class skipping or subsets of gt)
    assert len(np.unique(gt)) == np.max(gt)
    conf_mat = confusion_matrix(gt,preds)
    norm_conf_mat = np.transpose(np.transpose(conf_mat)/conf_mat.astype(np.float).sum(axis=1))
    print('Class Accuracies:{}'.format(np.diagonal(norm_conf_mat)))
    class_average_accuracy = np.mean(np.diagonal(norm_conf_mat))
    print('Class Average   :{}'.format(class_average_accuracy))
    pixel_accuracy = np.sum(np.diagonal(conf_mat))/np.sum(conf_mat)
    print('Pixel Accuracy  :{}'.format(pixel_accuracy))
    ious =  np.zeros(len(np.unique(gt)))
    for class_id in np.unique(gt):
        if class_id > VOID_CLASS:
            class_id -= 1
        ious[class_id] = (conf_mat[class_id,class_id] / (np.sum(conf_mat[class_id,:]) + np.sum(conf_mat[:,class_id]) - conf_mat[class_id,class_id]))
    mean_iou = np.mean(ious)
    print('Mean IoU        :{}'.format(mean_iou))

def evaluate_lists(pred_list,gt_list):
    assert(len(pred_list) == len(gt_list))
    all_preds = None
    all_gts   = None
    gt_shape  = None
    for idx,(in_pred,in_gt) in enumerate(zip(pred_list,gt_list)):
        gt = cv2.resize(in_gt,(in_pred.shape[1],in_pred.shape[0]),interpolation=cv2.INTER_NEAREST)
        pred = in_pred.flatten()
        gt   = gt.flatten()
        if all_preds is None:
            all_gts   = np.zeros((len(gt_list),len(gt))).astype(np.int32)
            all_preds = np.zeros((len(gt_list),len(gt))).astype(np.int32)
            gt_shape  = len(gt)
        else:
            assert(len(gt) == gt_shape)
        all_preds[idx] = pred
        all_gts[idx] = gt
    evaluate(all_preds.flatten(),all_gts.flatten())
