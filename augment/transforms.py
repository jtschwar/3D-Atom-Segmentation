from scipy.ndimage import rotate, map_coordinates, gaussian_filter, convolve
from skimage.segmentation import find_boundaries
from skimage.filters import gaussian
from skimage import measure
import importlib, random
import numpy as np
import torch

# Using fixed random state for reproducibility; 
GLOBAL_RANDOM_STATE = np.random.RandomState(47)

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, m):
        for t in self.transforms:
            m = t(m)
        return m

class RandomFlip:
    """
    Randomly flips the image across the given axes. Image can be either 3D (DxHxW) or 4D (CxDxHxW).

    When creating make sure that the provided RandomStates are consistent between raw and labeled datasets,
    otherwise the models won't converge.
    """

    def __init__(self, random_state, axis_prob=0.5, **kwargs):
        assert random_state is not None, 'RandomState cannot be None'
        self.random_state = random_state
        self.axes = (0, 1, 2)
        self.axis_prob = axis_prob

    def __call__(self, m):
        assert m.ndim in [3, 4], 'Supports only 3D (DxHxW) or 4D (CxDxHxW) images'

        for axis in self.axes:
            if self.random_state.uniform() > self.axis_prob:
                if m.ndim == 3:
                    m = np.flip(m, axis)
                else:
                    channels = [np.flip(m[c], axis) for c in range(m.shape[0])]
                    m = np.stack(channels, axis=0)

        return m

class RandomRotate90:
    """
    Rotate an array by 90 degrees around a randomly chosen plane. Image can be either 3D (DxHxW) or 4D (CxDxHxW).

    When creating make sure that the provided RandomStates are consistent between raw and labeled datasets,
    otherwise the models won't converge.

    IMPORTANT: assumes DHW axis order (that's why rotation is performed across (1,2) axis)
    """

    def __init__(self, random_state, **kwargs):
        self.random_state = random_state
        # always rotate around z-axis
        self.axis = (1, 2)

    def __call__(self, m):
        assert m.ndim in [3, 4], 'Supports only 3D (DxHxW) or 4D (CxDxHxW) images'

        # pick number of rotations at random
        k = self.random_state.randint(0, 4)
        # rotate k times around a given plane
        if m.ndim == 3:
            m = np.rot90(m, k, self.axis)
        else:
            channels = [np.rot90(m[c], k, self.axis) for c in range(m.shape[0])]
            m = np.stack(channels, axis=0)

        return m

class RandomRotate:
    """
    Rotate an array by a random degrees from taken from (-angle_spectrum, angle_spectrum) interval.
    Rotation axis is picked at random from the list of provided axes.
    """

    def __init__(self, random_state, angle_spectrum=30, axes=None, mode='reflect', order=0, **kwargs):
        if axes is None:
            axes = [(1, 0), (2, 1), (2, 0)]
        else:
            assert isinstance(axes, list) and len(axes) > 0

        self.random_state = random_state
        self.angle_spectrum = angle_spectrum
        self.axes = axes
        self.mode = mode
        self.order = order

    def __call__(self, m):
        axis = self.axes[self.random_state.randint(len(self.axes))]
        angle = self.random_state.randint(-self.angle_spectrum, self.angle_spectrum)

        if m.ndim == 3:
            m = rotate(m, angle, axes=axis, reshape=False, order=self.order, mode=self.mode, cval=-1)
        else:
            channels = [rotate(m[c], angle, axes=axis, reshape=False, order=self.order, mode=self.mode, cval=-1) for c
                        in range(m.shape[0])]
            m = np.stack(channels, axis=0)

        return m

class LabelsToMask:
    """
    Returns binary mask from labeled image, i.e. every label greater than 0 is treated as foreground.
    """

    def __init__(self, append_label=False, cross_entropy=False, **kwargs):
        self.cross_entropy = cross_entropy
        self.append_label = append_label

    def __call__(self, m):
        assert m.ndim == 3

        # get the segmentation mask
        mask = (m > 0).astype('uint8')
        results = [mask]

        if self.append_label:
            results.append(m)

        return np.stack(results, axis=0)

class Standardize:
    """
    Apply Z-score normalization to a given input tensor, i.e. re-scaling the values to be 0-mean and 1-std.
    """

    def __init__(self, eps=1e-10, mean=None, std=None, channelwise=False, **kwargs):
        if mean is not None or std is not None:
            assert mean is not None and std is not None
        self.mean = mean
        self.std = std
        self.eps = eps
        self.channelwise = channelwise

    def __call__(self, m):
        if self.mean is not None:
            mean, std = self.mean, self.std
        else:
            if self.channelwise:
                # normalize per-channel
                axes = list(range(m.ndim))
                # average across channels
                axes = tuple(axes[1:])
                mean = np.mean(m, axis=axes, keepdims=True)
                std = np.std(m, axis=axes, keepdims=True)
            else:
                mean = np.mean(m)
                std = np.std(m)

        return (m - mean) / np.clip(std, a_min=self.eps, a_max=None)

class Normalize:
    """
    Apply simple min-max scaling to a given input tensor, i.e. shrinks the range of the data in a fixed range of [-1, 1].
    """

    def __init__(self, min_value, max_value, **kwargs):
        assert max_value > min_value
        self.min_value = min_value
        self.value_range = max_value - min_value

    def __call__(self, m):
        norm_0_1 = (m - self.min_value) / self.value_range
        return np.clip(2 * norm_0_1 - 1, -1, 1)

class AdditiveGaussianNoise:
    def __init__(self, random_state, scale=(0.0, 1.0), execution_probability=0.1, **kwargs):
        self.execution_probability = execution_probability
        self.random_state = random_state
        self.scale = scale

    def __call__(self, m):
        if self.random_state.uniform() < self.execution_probability:
            std = self.random_state.uniform(self.scale[0], self.scale[1])
            gaussian_noise = self.random_state.normal(0, std, size=m.shape)
            return m + gaussian_noise
        return m

class AdditivePoissonNoise:
    def __init__(self, random_state, lam=(0.0, 1.0), execution_probability=0.1, **kwargs):
        self.execution_probability = execution_probability
        self.random_state = random_state
        self.lam = lam

    def __call__(self, m):
        if self.random_state.uniform() < self.execution_probability:
            lam = self.random_state.uniform(self.lam[0], self.lam[1])
            poisson_noise = self.random_state.poisson(lam, size=m.shape)
            return m + poisson_noise
        return m

class ToTensor:
    """
    Converts a given input numpy.ndarray into torch.Tensor. Adds additional 'channel' axis when the input is 3D
    and expand_dims=True (use for raw data of the shape (D, H, W)).
    """

    def __init__(self, expand_dims, dtype=np.float32, **kwargs):
        self.expand_dims = expand_dims
        self.dtype = dtype

    def __call__(self, m):
        assert m.ndim in [3, 4], 'Supports only 3D (DxHxW) or 4D (CxDxHxW) images'
        # add channel dimension
        if self.expand_dims and m.ndim == 3:
            m = np.expand_dims(m, axis=0)

        return torch.from_numpy(m.astype(dtype=self.dtype))

class Identity:
    def __init__(self, **kwargs):
        pass

    def __call__(self, m):
        return m

class LabelToTensor:
    def __call__(self, m):
        m = np.array(m)
        return torch.from_numpy(m.astype(dtype='int64'))

class GaussianBlur3D:
    def __init__(self, sigma=[.1, 2.], execution_probability=0.5, **kwargs):
        self.sigma = sigma
        self.execution_probability = execution_probability

    def __call__(self, x):
        if random.random() < self.execution_probability:
            sigma = random.uniform(self.sigma[0], self.sigma[1])
            x = gaussian(x, sigma=sigma)
            return x
        return x

class Transformer:
    def __init__(self, phase_config, base_config):
        self.phase_config = phase_config
        self.config_base = base_config
        self.seed = GLOBAL_RANDOM_STATE.randint(10000000)

    def raw_transform(self):
        return self._create_transform('raw')

    def label_transform(self):
        return self._create_transform('label')

    @staticmethod
    def _transformer_class(class_name):
        m = importlib.import_module('augment.transforms')
        clazz = getattr(m, class_name)
        return clazz

    def _create_transform(self, name):
        assert name in self.phase_config, f'Could not find {name} transform'
        return Compose([
            self._create_augmentation(c) for c in self.phase_config[name]
        ])

    def _create_augmentation(self, c):
        config = dict(self.config_base)
        config.update(c)
        config['random_state'] = np.random.RandomState(self.seed)
        aug_class = self._transformer_class(config['name'])
        return aug_class(**config)


