# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
from torch.utils.data import Sampler


class RandomTileSampler(Sampler):
    """
    A sampler that randomly samples from a list of geospatial image files with optional weighting.

    This sampler is designed for geospatial datasets where different tiles may have
    varying importance or content density. It allows for weighted sampling to focus
    training on more relevant areas.

    Args:
        image_fns (list): List of image file paths to sample from
        length (int): Number of samples to draw in one epoch
        weights (list, optional): Sample weights for each image file. If None, uniform weights are used.
    """

    def __init__(self, image_fns, length, weights=None):

        self.image_fns = image_fns
        self.length = length
        self.tile_sample_weights = []

        if weights is not None:
            assert len(weights) == len(image_fns)
            self.tile_sample_weights = weights
        else:
            self.tile_sample_weights = np.ones(len(image_fns))

        self.tile_sample_weights = np.array(self.tile_sample_weights)
        self.tile_sample_weights = (
            self.tile_sample_weights / self.tile_sample_weights.sum()
        )
        self.num_tiles = len(self.tile_sample_weights)

    def __iter__(self):
        indices = np.random.choice(
            self.num_tiles, size=len(self), replace=True, p=self.tile_sample_weights
        )
        for i in indices:
            yield i

    def __len__(self):
        return self.length


class GridGeoSampler(Sampler):
    def __init__(self, image_fn_indices, patch_size=4096, tile_size=4096):
        """Initialize the GridGeoSampler with a fixed tile size.

        Args:
            image_fns: List of image file paths or related data.
            image_fn_indices: Indices of images to process.
            patch_size: Size of each patch (default: 256).
            tile_size: Standardized tile size for all images (default: 4096).
        """
        self.image_fn_indices = image_fn_indices
        self.stride = patch_size
        self.tile_size = tile_size

        self.indices = []
        for i in self.image_fn_indices:

            # Use tile_size for width and height
            width, height = self.tile_size, self.tile_size

            # Iterate over the grid to create patches
            for y in list(range(0, height - patch_size, self.stride)) + [
                height - patch_size
            ]:
                for x in list(range(0, width - patch_size, self.stride)) + [
                    width - patch_size
                ]:
                    self.indices.append((i, y, x))

        self.num_chips = len(self.indices)

    def __iter__(self):
        for index in self.indices:
            yield index

    def __len__(self):
        return self.num_chips
