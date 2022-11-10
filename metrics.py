from skimage.metrics import mean_squared_error, contingency_table
from skimage import measure
import numpy as np
import importlib
import torch

from utils import get_logger, expand_as_one_hot, convert_to_numpy
from losses import compute_per_channel_dice

logger = get_logger('EvalMetric')

def precision(tp, fp):
    return tp / (tp + fp) if tp > 0 else 0


def recall(tp, fn):
    return tp / (tp + fn) if tp > 0 else 0


def accuracy(tp, fp, fn):
    return tp / (tp + fp + fn) if tp > 0 else 0


def f1(tp, fp, fn):
    return (2 * tp) / (2 * tp + fp + fn) if tp > 0 else 0


def _relabel(input):
    _, unique_labels = np.unique(input, return_inverse=True)
    return unique_labels.reshape(input.shape)

class DiceCoefficient:
    """Computes Dice Coefficient and then simply taking the average.
    Input is expected to be probabilities instead of logits.
    This metric is mostly useful when channels contain the same semantic class (e.g. affinities computed with different offsets).
    DO NOT USE this metric when training with DiceLoss, otherwise the results will be biased towards the loss.
    """

    def __init__(self, epsilon=1e-6, **kwargs):
        self.epsilon = epsilon

    def __call__(self, input, target):
        # Average across channels in order to get the final score
        return torch.mean(compute_per_channel_dice(input, target, epsilon=self.epsilon))


class MeanIoU:
    """
    Computes IoU for each class separately and then averages over all classes.
    """

    def __init__(self, threshold, skip_channels=(), ignore_index=None, **kwargs):
        """
        :param skip_channels: list/tuple of channels to be ignored from the IoU computation
        :param ignore_index: id of the label to be ignored from IoU computation
        """
        self.ignore_index = ignore_index
        self.skip_channels = skip_channels
        self.threshold = threshold

    def __call__(self, input, target):
        """
        :param input: 4D probability maps torch float tensor (NxDxHxW)
        :param target: 4D ground truth torch tensor (NxDxHxW)
        :return: intersection over union averaged over all channels
        """
        assert input.size() == target.size()

        per_batch_iou = []
        for _input, _target in zip(input, target):
            binary_prediction = self._binarize_predictions(_input)

            # convert to uint8 just in case
            binary_prediction = binary_prediction.byte()
            _target = _target.byte()
                
            iou = self._jaccard_index(binary_prediction, _target)

            mean_iou = torch.mean(torch.tensor(iou))
            per_batch_iou.append(mean_iou)

        return torch.mean(torch.tensor(per_batch_iou))

    def _binarize_predictions(self, input):
        """
        Puts 1 for the class/channel with the highest probability and 0 in other channels. Returns byte tensor of the
        same size as the input tensor.
        """
        
        # for single channel input just threshold the probability map
        result = input > self.threshold 
        return result.long()

    def _jaccard_index(self, prediction, target):
        """
        Computes IoU for a given target and prediction tensors
        """
        return torch.sum(prediction & target).float() / torch.clamp(torch.sum(prediction | target).float(), min=1e-8)


class SegmentationMetrics:
    """
    Computes precision, recall, accuracy, f1 score for a given ground truth and predicted segmentation.
    Contingency table for a given ground truth and predicted segmentation is computed eagerly upon construction
    of the instance of `SegmentationMetrics`.

    Args:
        target (ndarray): ground truth segmentation
        input (ndarray): predicted segmentation
    """

    def __init__(self, threshold=0.1, **kwargs):
        self.iou_threshold = threshold
        self.iou = MeanIoU(threshold)
        self.dice = DiceCoefficient()
        self.metric = kwargs['name']
        self.metrics = ['MeanIoU', 'dice']
        

    def __call__(self, input, target):
        """
        Computes precision, recall, accuracy, f1 score at a given IoU threshold
        """
        # ignore background
        # iou_matrix = _iou_matrix(target,input)     
        
        # iou_matrix = iou_matrix[1:, 1:]
        # detection_matrix = (iou_matrix > self.iou_threshold).astype(np.uint8)
        # n_gt, n_seg = detection_matrix.shape

        # # if the iou_matrix is empty or all values are 0
        # trivial = min(n_gt, n_seg) == 0 or np.all(detection_matrix == 0)
        # if trivial:
        #     tp = fp = fn = 0
        # else:
        #     # count non-zero rows to get the number of TP
        #     tp = np.count_nonzero(detection_matrix.sum(axis=1))
        #     # count zero rows to get the number of FN
        #     fn = n_gt - tp
        #     # count zero columns to get the number of FP
        #     fp = n_seg - np.count_nonzero(detection_matrix.sum(axis=0))

        # import pdb; pdb.set_trace()
        return {
            # 'precision': precision(tp, fp),
            # 'recall': recall(tp, fn),
            # 'accuracy': accuracy(tp, fp, fn),
            # 'f1': f1(tp, fp, fn),
            'MeanIoU': self.iou(input,target), 
            'dice': self.dice(input,target)
        }


def get_evaluation_metric(config):
    """
    Returns the evaluation metric function based on provided configuration
    :param config: (dict) a top level configuration object containing the 'eval_metric' key
    :return: an instance of the evaluation metric
    """

    def _metric_class(class_name):
        m = importlib.import_module('metrics')
        clazz = getattr(m, class_name)
        return clazz

    assert 'eval_metric' in config, 'Could not find evaluation metric configuration'
    metric_config = config['eval_metric']
    metric_class = _metric_class('SegmentationMetrics')
    return metric_class(**metric_config)
