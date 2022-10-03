from skimage.metrics import mean_squared_error, contingency_table
from skimage import measure
import numpy as np
import importlib
import torch

from utils import get_logger, expand_as_one_hot, convert_to_numpy
from losses import compute_per_channel_dice

logger = get_logger('EvalMetric')

def precision(tp, fp, fn):
    return tp / (tp + fp) if tp > 0 else 0


def recall(tp, fp, fn):
    return tp / (tp + fn) if tp > 0 else 0


def accuracy(tp, fp, fn):
    return tp / (tp + fp + fn) if tp > 0 else 0


def f1(tp, fp, fn):
    return (2 * tp) / (2 * tp + fp + fn) if tp > 0 else 0


def _relabel(input):
    _, unique_labels = np.unique(input, return_inverse=True)
    return unique_labels.reshape(input.shape)

class DiceCoefficient:
    """Computes Dice Coefficient.
    Generalized to multiple channels by computing per-channel Dice Score
    (as described in https://arxiv.org/pdf/1707.03237.pdf) and then simply taking the average.
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

    def __init__(self, skip_channels=(), ignore_index=None, **kwargs):
        """
        :param skip_channels: list/tuple of channels to be ignored from the IoU computation
        :param ignore_index: id of the label to be ignored from IoU computation
        """
        self.ignore_index = ignore_index
        self.skip_channels = skip_channels

    def __call__(self, input, target):
        """
        :param input: 5D probability maps torch float tensor (NxCxDxHxW)
        :param target: 4D or 5D ground truth torch tensor. 4D (NxDxHxW) tensor will be expanded to 5D as one-hot
        :return: intersection over union averaged over all channels
        """
        assert input.dim() == 5

        n_classes = input.size()[1]

        if target.dim() == 4:
            target = expand_as_one_hot(target, C=n_classes, ignore_index=self.ignore_index)

        assert input.size() == target.size()

        per_batch_iou = []
        for _input, _target in zip(input, target):
            binary_prediction = self._binarize_predictions(_input, n_classes)

            if self.ignore_index is not None:
                # zero out ignore_index
                mask = _target == self.ignore_index
                binary_prediction[mask] = 0
                _target[mask] = 0

            # convert to uint8 just in case
            binary_prediction = binary_prediction.byte()
            _target = _target.byte()

            per_channel_iou = []
            for c in range(n_classes):
                if c in self.skip_channels:
                    continue

                per_channel_iou.append(self._jaccard_index(binary_prediction[c], _target[c]))

            assert per_channel_iou, "All channels were ignored from the computation"
            mean_iou = torch.mean(torch.tensor(per_channel_iou))
            per_batch_iou.append(mean_iou)

        return torch.mean(torch.tensor(per_batch_iou))

    def _binarize_predictions(self, input, n_classes):
        """
        Puts 1 for the class/channel with the highest probability and 0 in other channels. Returns byte tensor of the
        same size as the input tensor.
        """
        if n_classes == 1:
            # for single channel input just threshold the probability map
            result = input > 0.5
            return result.long()

        _, max_index = torch.max(input, dim=0, keepdim=True)
        return torch.zeros_like(input, dtype=torch.uint8).scatter_(0, max_index, 1)

    def _jaccard_index(self, prediction, target):
        """
        Computes IoU for a given target and prediction tensors
        """
        return torch.sum(prediction & target).float() / torch.clamp(torch.sum(prediction | target).float(), min=1e-8)


class GenericAveragePrecision:
    def __init__(self, min_instance_size=None, use_last_target=False, metric='ap', **kwargs):
        self.min_instance_size = min_instance_size
        self.use_last_target = use_last_target
        assert metric in ['ap', 'acc']
        if metric == 'ap':
            # use AveragePrecision
            self.metric = AveragePrecision()
        else:
            # use Accuracy at 0.5 IoU
            self.metric = Accuracy(iou_threshold=0.5)

    def __call__(self, input, target):
        if target.dim() == 5:
            if self.use_last_target:
                target = target[:, -1, ...]  # 4D
            else:
                # use 1st target channel
                target = target[:, 0, ...]  # 4D

        input1 = input2 = input
        multi_head = isinstance(input, tuple)
        if multi_head:
            input1, input2 = input

        input1, input2, target = convert_to_numpy(input1, input2, target)

        batch_aps = []
        i_batch = 0
        # iterate over the batch
        for inp1, inp2, tar in zip(input1, input2, target):
            if multi_head:
                inp = (inp1, inp2)
            else:
                inp = inp1

            segs = self.input_to_seg(inp, tar)  # expects 4D
            assert segs.ndim == 4
            # convert target to seg
            tar = self.target_to_seg(tar)

            # filter small instances if necessary
            tar = self._filter_instances(tar)

            # compute average precision per channel
            segs_aps = [self.metric(self._filter_instances(seg), tar) for seg in segs]

            logger.info(f'Batch: {i_batch}. Max Average Precision for channel: {np.argmax(segs_aps)}')
            # save max AP
            batch_aps.append(np.max(segs_aps))
            i_batch += 1

        return torch.tensor(batch_aps).mean()

    def _filter_instances(self, input):
        """
        Filters instances smaller than 'min_instance_size' by overriding them with 0-index
        :param input: input instance segmentation
        """
        if self.min_instance_size is not None:
            labels, counts = np.unique(input, return_counts=True)
            for label, count in zip(labels, counts):
                if count < self.min_instance_size:
                    input[input == label] = 0
        return input

    def input_to_seg(self, input, target=None):
        raise NotImplementedError

    def target_to_seg(self, target):
        return target


class BlobsAveragePrecision(GenericAveragePrecision):
    """
    Computes Average Precision given foreground prediction and ground truth instance segmentation.
    """

    def __init__(self, thresholds=None, metric='ap', min_instance_size=None, input_channel=0, **kwargs):
        super().__init__(min_instance_size=min_instance_size, use_last_target=True, metric=metric)
        if thresholds is None:
            thresholds = [0.4, 0.5, 0.6, 0.7, 0.8]
        assert isinstance(thresholds, list)
        self.thresholds = thresholds
        self.input_channel = input_channel

    def input_to_seg(self, input, target=None):
        input = input[self.input_channel]
        segs = []
        for th in self.thresholds:
            # threshold and run connected components
            mask = (input > th).astype(np.uint8)
            seg = measure.label(mask, background=0, connectivity=1)
            segs.append(seg)
        return np.stack(segs)

def _iou_matrix(gt, seg):
    # relabel gt and seg for smaller memory footprint of contingency table
    gt = _relabel(gt)
    seg = _relabel(seg)

    # get number of overlapping pixels between GT and SEG
    n_inter = contingency_table(gt, seg).A

    # number of pixels for GT instances
    n_gt = n_inter.sum(axis=1, keepdims=True)
    # number of pixels for SEG instances
    n_seg = n_inter.sum(axis=0, keepdims=True)

    # number of pixels in the union between GT and SEG instances
    n_union = n_gt + n_seg - n_inter

    iou_matrix = n_inter / n_union
    # make sure that the values are within [0,1] range
    assert 0 <= np.min(iou_matrix) <= np.max(iou_matrix) <= 1

    return iou_matrix


class SegmentationMetrics:
    """
    Computes precision, recall, accuracy, f1 score for a given ground truth and predicted segmentation.
    Contingency table for a given ground truth and predicted segmentation is computed eagerly upon construction
    of the instance of `SegmentationMetrics`.

    Args:
        gt (ndarray): ground truth segmentation
        seg (ndarray): predicted segmentation
    """

    def __init__(self, gt, seg):
        self.iou_matrix = _iou_matrix(gt, seg)

    def metrics(self, iou_threshold):
        """
        Computes precision, recall, accuracy, f1 score at a given IoU threshold
        """
        # ignore background
        iou_matrix = self.iou_matrix[1:, 1:]
        detection_matrix = (iou_matrix > iou_threshold).astype(np.uint8)
        n_gt, n_seg = detection_matrix.shape

        # if the iou_matrix is empty or all values are 0
        trivial = min(n_gt, n_seg) == 0 or np.all(detection_matrix == 0)
        if trivial:
            tp = fp = fn = 0
        else:
            # count non-zero rows to get the number of TP
            tp = np.count_nonzero(detection_matrix.sum(axis=1))
            # count zero rows to get the number of FN
            fn = n_gt - tp
            # count zero columns to get the number of FP
            fp = n_seg - np.count_nonzero(detection_matrix.sum(axis=0))

        return {
            'precision': precision(tp, fp, fn),
            'recall': recall(tp, fp, fn),
            'accuracy': accuracy(tp, fp, fn),
            'f1': f1(tp, fp, fn)
        }


class Accuracy:
    """
    Computes accuracy between ground truth and predicted segmentation a a given threshold value.
    Defined as: AC = TP / (TP + FP + FN).
    Kaggle DSB2018 calls it Precision, see:
    https://www.kaggle.com/stkbailey/step-by-step-explanation-of-scoring-metric.
    """

    def __init__(self, iou_threshold):
        self.iou_threshold = iou_threshold

    def __call__(self, input_seg, gt_seg):
        metrics = SegmentationMetrics(gt_seg, input_seg).metrics(self.iou_threshold)
        return metrics['accuracy']


class AveragePrecision:
    """
    Average precision taken for the IoU range (0.5, 0.95) with a step of 0.05 as defined in:
    https://www.kaggle.com/stkbailey/step-by-step-explanation-of-scoring-metric
    """

    def __init__(self):
        self.iou_range = np.linspace(0.50, 0.95, 10)

    def __call__(self, input_seg, gt_seg):
        # compute contingency_table
        sm = SegmentationMetrics(gt_seg, input_seg)
        # compute accuracy for each threshold
        acc = [sm.metrics(iou)['accuracy'] for iou in self.iou_range]
        # return the average
        return np.mean(acc)


class MSE:
    """
    Computes MSE between input and target
    """

    def __init__(self, **kwargs):
        pass

    def __call__(self, input, target):
        input, target = convert_to_numpy(input, target)
        return mean_squared_error(input, target)


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
    metric_class = _metric_class(metric_config['name'])
    return metric_class(**metric_config)

