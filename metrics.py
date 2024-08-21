import numpy as np
import torch
from skimage import measure
import mxnet as mx
from mxnet import nd
from mxnet.metric import EvalMetric
import threading
       
class mIoU():
    
    def __init__(self):
        super(mIoU, self).__init__()
        self.reset()

    def update(self, preds, labels):
        # print('come_ininin')

        correct, labeled = batch_pix_accuracy(preds, labels)
        inter, union = batch_intersection_union(preds, labels)
        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        self.total_union += union
        self.num += 1
        self.total_IoU += 1.0 * inter / (np.spacing(1) + union)


    def get(self):
        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        nIoU = self.total_IoU / self.num
        return float(pixAcc), mIoU, float(nIoU)

    def reset(self):
        self.total_inter = 0
        self.total_union = 0
        self.total_correct = 0
        self.total_label = 0
        self.total_IoU = 0
        self.num = 0

class PD_FA():
    def __init__(self,):
        super(PD_FA, self).__init__()
        self.image_area_total = []
        self.image_area_match = []
        self.dismatch_pixel = 0
        self.all_pixel = 0
        self.PD = 0
        self.target= 0
    def update(self, preds, labels, size):
        predits  = np.array((preds).cpu()).astype('int64')
        labelss = np.array((labels).cpu()).astype('int64') 

        image = measure.label(predits, connectivity=2)
        coord_image = measure.regionprops(image)
        label = measure.label(labelss , connectivity=2)
        coord_label = measure.regionprops(label)

        self.target    += len(coord_label)
        self.image_area_total = []
        self.distance_match   = []
        self.dismatch         = []

        for K in range(len(coord_image)):
            area_image = np.array(coord_image[K].area)
            self.image_area_total.append(area_image)

        true_img = np.zeros(predits.shape)
        for i in range(len(coord_label)):
            centroid_label = np.array(list(coord_label[i].centroid))
            for m in range(len(coord_image)):
                centroid_image = np.array(list(coord_image[m].centroid))
                distance = np.linalg.norm(centroid_image - centroid_label)
                area_image = np.array(coord_image[m].area)
                if distance < 3:
                    self.distance_match.append(distance)
                    true_img[coord_image[m].coords[:,0], coord_image[m].coords[:,1]] = 1
                    del coord_image[m]
                    break

        self.dismatch_pixel += (predits - true_img).sum()
        self.all_pixel +=size[0]*size[1]
        self.PD +=len(self.distance_match)

    def get(self):
        Final_FA =  self.dismatch_pixel / self.all_pixel
        Final_PD =  self.PD /self.target
        return Final_PD, float(Final_FA.cpu().detach().numpy())

    def reset(self):
        self.FA  = np.zeros([self.bins+1])
        self.PD  = np.zeros([self.bins+1])

def batch_pix_accuracy(output, target):   
    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")

    assert output.shape == target.shape, "Predict and Label Shape Don't Match"
    predict = (output > 0).float()
    pixel_labeled = (target > 0).float().sum()
    pixel_correct = (((predict == target).float())*((target > 0)).float()).sum()
    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled

def batch_intersection_union(output, target):
    mini = 1
    maxi = 1
    nbins = 1
    predict = (output > 0).float()
    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")
    intersection = predict * ((predict == target).float())

    area_inter, _  = np.histogram(intersection.cpu(), bins=nbins, range=(mini, maxi))
    area_pred,  _  = np.histogram(predict.cpu(), bins=nbins, range=(mini, maxi))
    area_lab,   _  = np.histogram(target.cpu(), bins=nbins, range=(mini, maxi))
    area_union     = area_pred + area_lab - area_inter

    assert (area_inter <= area_union).all(), \
        "Error: Intersection area should be smaller than Union area"
    return area_inter, area_union


class ROCMetric(EvalMetric):
    """Computes pixAcc and mIoU metric scores
    """
    def __init__(self, bins):
        super(ROCMetric, self).__init__('ROC')
        self.lock = threading.Lock()
        self.bins = bins
        self.tp_arr = np.zeros(self.bins+1)
        self.pos_arr = np.zeros(self.bins+1)
        self.fp_arr = np.zeros(self.bins+1)
        self.neg_arr = np.zeros(self.bins+1)
        self.fn_tp_arr = np.zeros(self.bins + 1)
        self.nclass = 1
        # self.reset()

    def update(self, preds, labels):
        """Updates the internal evaluation result.

        Parameters
        ----------
        labels : 'NDArray' or list of `NDArray`
            The labels of the data.

        preds : 'NDArray' or list of `NDArray`
            Predicted values.
        """
        preds = mx.nd.array(preds.detach().numpy())
        labels = mx.nd.array(labels.detach().numpy())
        def evaluate_worker(self, label, pred):
            for iBin in range(self.bins+1):
                score_thresh = (iBin + 0.0) / self.bins
                # print(iBin, "-th, score_thresh: ", score_thresh)
                i_tp, i_pos, i_fp, i_neg, i_fn = cal_tp_pos_fp_neg(pred, label, self.nclass,
                                                             score_thresh)
                # print("i_tp: ", i_tp)
                # print("i_fp: ", i_fp)




                with self.lock:
                    self.tp_arr[iBin] += i_tp
                    self.pos_arr[iBin] += i_pos
                    self.fp_arr[iBin] += i_fp
                    self.neg_arr[iBin] += i_neg
                    self.fn_tp_arr[iBin] += i_tp + i_fn

        if isinstance(preds, mx.nd.NDArray):
            evaluate_worker(self, labels, preds)
        elif isinstance(preds, (list, tuple)):
            threads = [threading.Thread(target=evaluate_worker,
                                        args=(self, label, pred),
                                       )
                       for (label, pred) in zip(labels, preds)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

    def get(self):
        """Gets the current evaluation result.

        Returns
        -------
        metrics : tuple of float
            pixAcc and mIoU
        """
        # print("self.total_correct: ", self.total_correct)
        # print("self.total_label: ", self.total_label)
        # print("self.total_union: ", self.total_union)
        # pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        tp_rates = self.tp_arr / (self.pos_arr + 0.001)
        fp_rates = self.fp_arr / (self.neg_arr + 0.001)

        precision = self.tp_arr / (self.tp_arr + self.fp_arr + 0.001)
        recall = self.tp_arr / (self.fn_tp_arr + 0.001)


        return tp_rates, fp_rates, recall, precision


def cal_tp_pos_fp_neg(output, target, nclass, score_thresh):
    """mIoU"""
    # inputs are NDarray, output 4D, target 3D
    # the category 0 is ignored class, typically for background / boundary
    mini = 1
    maxi = 1 # nclass
    nbins = 1 # nclass

    predict = (output.asnumpy() >= score_thresh).astype('int64') # P
    # predict = (output.asnumpy() > 0).astype('int64')  # P
    if len(target.shape) == 3:
        target = nd.expand_dims(target, axis=1).asnumpy().astype('int64') # T
    elif len(target.shape) == 4:
        target = target.asnumpy().astype('int64')  # T
    else:
        raise ValueError("Unknown target dimension")
    intersection = predict * (predict == target)  # TP
    tp = intersection.sum()
    fp = (predict * (predict != target)).sum()  # FP
    tn = ((1 - predict) * (predict == target)).sum()  # TN
    fn = ((predict != target) * (1 - predict)).sum()  # FN
    pos = tp + fn
    neg = fp + tn

    return tp, pos, fp, neg, fn

def cal_tp_tn_fp_fn(output, target, score_thresh):
    """mIoU"""
    # inputs are NDarray, output 4D, target 3D
    # the category 0 is ignored class, typically for background / boundary
    predict = (output.detach().numpy() > score_thresh).astype('int64')  # P
    # predict = (output.asnumpy() > 0).astype('int64')  # P
    if len(target.shape) == 3:
        target = nd.expand_dims(target, axis=1).asnumpy().astype('int64')  # T
    elif len(target.shape) == 4:
        target = target.detach().numpy().astype('int64')  # T
    else:
        raise ValueError("Unknown target dimension")
    intersection = predict * (predict == target)  # TP
    tp = intersection.sum()
    fp = (predict * (predict != target)).sum()  # FP
    tn = ((1 - predict) * (predict == target)).sum()  # TN
    fn = ((predict != target) * (1 - predict)).sum()  # FN

    return tp, tn, fp, fn


class F_Measure():
    def __init__(self):
        super(F_Measure, self).__init__()
        self.reset()

    def update(self, preds, labels):
        tp, tn, fp, fn = cal_tp_tn_fp_fn(preds, labels, 0.5)

        recall = tp / (tp + fn + np.spacing(1))
        precision = tp / (tp + fp + np.spacing(1))
        f1 = 2 * recall * precision / (recall + precision + np.spacing(1))
        self.F1.append(f1)
        self.num += 1


    def get(self):
        F1_out = sum(self.F1) / self.num

        return float(F1_out)

    def reset(self):
        self.num = 0
        self.F1 = []
