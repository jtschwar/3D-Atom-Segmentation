from torch.utils.data import Dataset
from itertools import chain
import h5py, glob, os
import numpy as np

from datasets.utils import get_slice_builder, calculate_stats
import augment.transforms as transforms
from utils import get_logger

logger = get_logger('HDF5Dataset')

class HDF5Dataset(Dataset):
    """
    Loads data from all of the H5 files into the memory. Fast but might consume a lot of memory.
    Implementation of torch.utils.data.Dataset backed by the HDF5 files, which iterates over the raw and label datasets
    patch by patch with a given stride.
    """

    def __init__(self, file_path,
                 phrase,
                 slice_builder_config,
                 transformer_config,
                 mirror_padding=(16, 32, 32),
                 raw_internal_path='raw',
                 label_internal_path='label',
                 global_normalization=True):
        """
        :param file_path: path to H5 file containing raw data as well as labels and per pixel weights (optional)
        :param phrase: 'train' for training, 'val' for validation, 'test' for testing; data augmentation is performed
            only during the 'train' phrase
        :para'/home/adrian/workspace/ilastik-datasets/VolkerDeconv/train'm slice_builder_config: configuration of the SliceBuilder
        :param transformer_config: data augmentation configuration
        :param mirror_padding (int or tuple): number of voxels padded to each axis
        :param raw_internal_path (str or list): H5 internal path to the raw dataset
        :param label_internal_path (str or list): H5 internal path to the label dataset
        """

        if phrase in ['train', 'val']:
            mirror_padding = None

        if mirror_padding is not None:
            if isinstance(mirror_padding, int):
                mirror_padding = (mirror_padding,) * 3
            else:
                assert len(mirror_padding) == 3, f"Invalid mirror_padding: {mirror_padding}"

        self.mirror_padding = mirror_padding
        self.phrase = phrase
        self.file_path = file_path

        input_file = self.create_h5_file(file_path)

        self.raw = self.fetch_and_check(input_file, raw_internal_path)

        if global_normalization:
            stats = calculate_stats(self.raw)
        else:
            stats = {'pmin': None, 'pmax': None, 'mean': None, 'std': None}

        self.transformer = transforms.Transformer(transformer_config, stats)
        self.raw_transform = self.transformer.raw_transform()

        if phrase != 'test':
            # create label/weight transform only in train/val phrase
            self.label_transform = self.transformer.label_transform()
            self.label = self.fetch_and_check(input_file, label_internal_path)

            self._check_volume_sizes(self.raw, self.label)
        else:
            # 'test' phrase used only for predictions so ignore the label dataset
            self.label = None

            # add mirror padding if needed
            if self.mirror_padding is not None:
                z, y, x = self.mirror_padding
                pad_width = ((z, z), (y, y), (x, x))
                if self.raw.ndim == 4:
                    channels = [np.pad(r, pad_width=pad_width, mode='reflect') for r in self.raw]
                    self.raw = np.stack(channels)
                else:
                    self.raw = np.pad(self.raw, pad_width=pad_width, mode='reflect')

        # build slice indices for raw and label data sets
        slice_builder = get_slice_builder(self.raw, self.label,slice_builder_config)
        self.raw_slices = slice_builder.raw_slices
        self.label_slices = slice_builder.label_slices

        self.patch_count = len(self.raw_slices)
        logger.info(f'Number of patches: {self.patch_count}')

    def __len__(self):
        return self.patch_count

    @staticmethod
    def create_h5_file(file_path):
        return h5py.File(file_path, 'r')

    @staticmethod
    def fetch_and_check(input_file, internal_path):
        ds = input_file[internal_path][:]
        if ds.ndim == 2:
            # expand dims if 2d
            ds = np.expand_dims(ds, axis=0)
        return ds

    def __getitem__(self, idx):
        if idx >= len(self):
            raise StopIteration

        # get the slice for a given index 'idx'
        raw_idx = self.raw_slices[idx]
        # get the raw data patch for a given slice
        raw_patch_transformed = self.raw_transform(self.raw[raw_idx])

        if self.phrase == 'test':
            # discard the channel dimension in the slices: predictor requires only the spatial dimensions of the volume
            if len(raw_idx) == 4:
                raw_idx = raw_idx[1:]
            return raw_patch_transformed, raw_idx
        else:
            # get the slice for a given index 'idx'
            label_idx = self.label_slices[idx]
            label_patch_transformed = self.label_transform(self.label[label_idx])

            # return the transformed raw and label patches
            return raw_patch_transformed, label_patch_transformed

    @staticmethod
    def _check_volume_sizes(raw, label):
        def _volume_shape(volume):
            if volume.ndim == 3:
                return volume.shape
            return volume.shape[1:]

        assert raw.ndim in [3, 4], 'Raw dataset must be 3D (DxHxW) or 4D (CxDxHxW)'
        assert label.ndim in [3, 4], 'Label dataset must be 3D (DxHxW) or 4D (CxDxHxW)'

        assert _volume_shape(raw) == _volume_shape(label), 'Raw and labels have to be of the same size'

    @classmethod
    def create_datasets(cls, dataset_config, phrase=None):
        phrase_config = dataset_config[phrase]

        # load data augmentation configuration
        transformer_config = phrase_config['transformer']
        # load slice builder config
        slice_builder_config = phrase_config['slice_builder']
        # load files to process
        file_paths = phrase_config['file_paths']
        # file_paths may contain both files and directories; if the file_path is a directory all H5 files inside
        # are going to be included in the final file_paths
        file_paths = cls.traverse_h5_paths(file_paths)

        datasets = []
        for file_path in file_paths:
            try:
                logger.info(f'Loading {phrase} set from: {file_path}...')
                dataset = cls(file_path=file_path,
                              phrase=phrase,
                              slice_builder_config=slice_builder_config,
                              transformer_config=transformer_config,
                              mirror_padding=dataset_config.get('mirror_padding', None),
                              raw_internal_path=dataset_config.get('raw_internal_path', 'raw'),
                              label_internal_path=dataset_config.get('label_internal_path', 'label'),
                              global_normalization=dataset_config.get('global_normalization', None))
                datasets.append(dataset)
            except Exception:
                logger.error(f'Skipping {phrase} set: {file_path}', exc_info=True)
        return datasets

    @staticmethod
    def traverse_h5_paths(file_paths):
        assert isinstance(file_paths, list)
        results = []
        for file_path in file_paths:
            if os.path.isdir(file_path):
                # if file path is a directory take all H5 files in that directory
                iters = [glob.glob(os.path.join(file_path, ext)) for ext in ['*.h5', '*.hdf', '*.hdf5', '*.hd5']]
                for fp in chain(*iters):
                    results.append(fp)
            else:
                results.append(file_path)
        return results


