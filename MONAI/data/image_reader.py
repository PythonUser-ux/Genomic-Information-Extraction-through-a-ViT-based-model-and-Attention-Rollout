# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from torch.utils.data._utils.collate import np_str_obj_array_pattern

from monai.config import DtypeLike, KeysCollection, PathLike
from monai.data.utils import correct_nifti_header_if_necessary, is_supported_format, orientation_ras_lps
from monai.transforms.utility.array import EnsureChannelFirst
from monai.utils import ensure_tuple, ensure_tuple_rep, optional_import, require_pkg

if TYPE_CHECKING:
    import itk
    import nibabel as nib
    import nrrd
    from nibabel.nifti1 import Nifti1Image
    from PIL import Image as PILImage

    has_nrrd = has_itk = has_nib = has_pil = True
else:
    itk, has_itk = optional_import("itk", allow_namespace_pkg=True)
    nib, has_nib = optional_import("nibabel")
    Nifti1Image, _ = optional_import("nibabel.nifti1", name="Nifti1Image")
    PILImage, has_pil = optional_import("PIL.Image")
    nrrd, has_nrrd = optional_import("nrrd", allow_namespace_pkg=True)

OpenSlide, _ = optional_import("openslide", name="OpenSlide")
CuImage, _ = optional_import("cucim", name="CuImage")
TiffFile, _ = optional_import("tifffile", name="TiffFile")

__all__ = ["ImageReader", "ITKReader", "NibabelReader", "NumpyReader", "PILReader", "WSIReader", "NrrdReader"]


class ImageReader(ABC):
    """
    An abstract class defines APIs to load image files.

    Typical usage of an implementation of this class is:

    .. code-block:: python

        image_reader = MyImageReader()
        img_obj = image_reader.read(path_to_image)
        img_data, meta_data = image_reader.get_data(img_obj)

    - The `read` call converts image filenames into image objects,
    - The `get_data` call fetches the image data, as well as metadata.
    - A reader should implement `verify_suffix` with the logic of checking the input filename
      by the filename extensions.

    """

    @abstractmethod
    def verify_suffix(self, filename: Union[Sequence[PathLike], PathLike]) -> bool:
        """
        Verify whether the specified `filename` is supported by the current reader.
        This method should return True if the reader is able to read the format suggested by the
        `filename`.

        Args:
            filename: file name or a list of file names to read.
                if a list of files, verify all the suffixes.

        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")

    @abstractmethod
    def read(self, data: Union[Sequence[PathLike], PathLike], **kwargs) -> Union[Sequence[Any], Any]:
        """
        Read image data from specified file or files.
        Note that it returns a data object or a sequence of data objects.

        Args:
            data: file name or a list of file names to read.
            kwargs: additional args for actual `read` API of 3rd party libs.

        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")

    @abstractmethod
    def get_data(self, img) -> Tuple[np.ndarray, Dict]:
        """
        Extract data array and metadata from loaded image and return them.
        This function must return two objects, the first is a numpy array of image data,
        the second is a dictionary of metadata.

        Args:
            img: an image object loaded from an image file or a list of image objects.

        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")


def _copy_compatible_dict(from_dict: Dict, to_dict: Dict):
    if not isinstance(to_dict, dict):
        raise ValueError(f"to_dict must be a Dict, got {type(to_dict)}.")
    if not to_dict:
        for key in from_dict:
            datum = from_dict[key]
            if isinstance(datum, np.ndarray) and np_str_obj_array_pattern.search(datum.dtype.str) is not None:
                continue
            to_dict[key] = datum
    else:
        affine_key, shape_key = "affine", "spatial_shape"
        if affine_key in from_dict and not np.allclose(from_dict[affine_key], to_dict[affine_key]):
            raise RuntimeError(
                "affine matrix of all images should be the same for channel-wise concatenation. "
                f"Got {from_dict[affine_key]} and {to_dict[affine_key]}."
            )
        if shape_key in from_dict and not np.allclose(from_dict[shape_key], to_dict[shape_key]):
            raise RuntimeError(
                "spatial_shape of all images should be the same for channel-wise concatenation. "
                f"Got {from_dict[shape_key]} and {to_dict[shape_key]}."
            )


def _stack_images(image_list: List, meta_dict: Dict):
    if len(image_list) <= 1:
        return image_list[0]
    if meta_dict.get("original_channel_dim", None) not in ("no_channel", None):
        channel_dim = int(meta_dict["original_channel_dim"])
        return np.concatenate(image_list, axis=channel_dim)
    # stack at a new first dim as the channel dim, if `'original_channel_dim'` is unspecified
    meta_dict["original_channel_dim"] = 0
    return np.stack(image_list, axis=0)


@require_pkg(pkg_name="itk")
class ITKReader(ImageReader):
    """
    Load medical images based on ITK library.
    All the supported image formats can be found at:
    https://github.com/InsightSoftwareConsortium/ITK/tree/master/Modules/IO
    The loaded data array will be in C order, for example, a 3D image NumPy
    array index order will be `CDWH`.

    Args:
        channel_dim: the channel dimension of the input image, default is None.
            This is used to set original_channel_dim in the metadata, EnsureChannelFirstD reads this field.
            If None, `original_channel_dim` will be either `no_channel` or `-1`.

                - Nifti file is usually "channel last", so there is no need to specify this argument.
                - PNG file usually has `GetNumberOfComponentsPerPixel()==3`, so there is no need to specify this argument.

        series_name: the name of the DICOM series if there are multiple ones.
            used when loading DICOM series.
        reverse_indexing: whether to use a reversed spatial indexing convention for the returned data array.
            If ``False``, the spatial indexing follows the numpy convention;
            otherwise, the spatial indexing convention is reversed to be compatible with ITK. Default is ``False``.
            This option does not affect the metadata.
        series_meta: whether to load the metadata of the DICOM series (using the metadata from the first slice).
            This flag is checked only when loading DICOM series. Default is ``False``.
        affine_lps_to_ras: whether to convert the affine matrix from "LPS" to "RAS". Defaults to ``True``.
            Set to ``True`` to be consistent with ``NibabelReader``, otherwise the affine matrix remains in the ITK convention.
        kwargs: additional args for `itk.imread` API. more details about available args:
            https://github.com/InsightSoftwareConsortium/ITK/blob/master/Wrapping/Generators/Python/itk/support/extras.py

    """

    def __init__(
        self,
        channel_dim: Optional[int] = None,
        series_name: str = "",
        reverse_indexing: bool = False,
        series_meta: bool = False,
        affine_lps_to_ras: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.kwargs = kwargs
        self.channel_dim = channel_dim
        self.series_name = series_name
        self.reverse_indexing = reverse_indexing
        self.series_meta = series_meta
        self.affine_lps_to_ras = affine_lps_to_ras

    def verify_suffix(self, filename: Union[Sequence[PathLike], PathLike]) -> bool:
        """
        Verify whether the specified file or files format is supported by ITK reader.

        Args:
            filename: file name or a list of file names to read.
                if a list of files, verify all the suffixes.

        """
        return has_itk

    def read(self, data: Union[Sequence[PathLike], PathLike], **kwargs):
        """
        Read image data from specified file or files, it can read a list of images
        and stack them together as multi-channel data in `get_data()`.
        If passing directory path instead of file path, will treat it as DICOM images series and read.
        Note that the returned object is ITK image object or list of ITK image objects.

        Args:
            data: file name or a list of file names to read,
            kwargs: additional args for `itk.imread` API, will override `self.kwargs` for existing keys.
                More details about available args:
                https://github.com/InsightSoftwareConsortium/ITK/blob/master/Wrapping/Generators/Python/itkExtras.py

        """
        img_ = []

        filenames: Sequence[PathLike] = ensure_tuple(data)
        kwargs_ = self.kwargs.copy()
        kwargs_.update(kwargs)
        for name in filenames:
            name = f"{name}"
            if Path(name).is_dir():
                # read DICOM series
                # https://itk.org/ITKExamples/src/IO/GDCM/ReadDICOMSeriesAndWrite3DImage
                names_generator = itk.GDCMSeriesFileNames.New()
                names_generator.SetUseSeriesDetails(True)
                names_generator.AddSeriesRestriction("0008|0021")  # Series Date
                names_generator.SetDirectory(name)
                series_uid = names_generator.GetSeriesUIDs()

                if len(series_uid) < 1:
                    raise FileNotFoundError(f"no DICOMs in: {name}.")
                if len(series_uid) > 1:
                    warnings.warn(f"the directory: {name} contains more than one DICOM series.")
                series_identifier = series_uid[0] if not self.series_name else self.series_name
                name = names_generator.GetFileNames(series_identifier)

                _obj = itk.imread(name, **kwargs_)
                if self.series_meta:
                    _reader = itk.ImageSeriesReader.New(FileNames=name)
                    _reader.Update()
                    _meta = _reader.GetMetaDataDictionaryArray()
                    if len(_meta) > 0:
                        # TODO: using the first slice's meta. this could be improved to filter unnecessary tags.
                        _obj.SetMetaDataDictionary(_meta[0])
                img_.append(_obj)
            else:
                img_.append(itk.imread(name, **kwargs_))
        return img_ if len(filenames) > 1 else img_[0]

    def get_data(self, img):
        """
        Extract data array and metadata from loaded image and return them.
        This function returns two objects, first is numpy array of image data, second is dict of metadata.
        It constructs `affine`, `original_affine`, and `spatial_shape` and stores them in meta dict.
        When loading a list of files, they are stacked together at a new dimension as the first dimension,
        and the metadata of the first image is used to represent the output metadata.

        Args:
            img: an ITK image object loaded from an image file or a list of ITK image objects.

        """
        img_array: List[np.ndarray] = []
        compatible_meta: Dict = {}

        for i in ensure_tuple(img):
            data = self._get_array_data(i)
            img_array.append(data)
            header = self._get_meta_dict(i)
            header["original_affine"] = self._get_affine(i, self.affine_lps_to_ras)
            header["affine"] = header["original_affine"].copy()
            header["spatial_shape"] = self._get_spatial_shape(i)
            if self.channel_dim is None:  # default to "no_channel" or -1
                header["original_channel_dim"] = "no_channel" if len(data.shape) == len(header["spatial_shape"]) else -1
            else:
                header["original_channel_dim"] = self.channel_dim
            _copy_compatible_dict(header, compatible_meta)

        return _stack_images(img_array, compatible_meta), compatible_meta

    def _get_meta_dict(self, img) -> Dict:
        """
        Get all the metadata of the image and convert to dict type.

        Args:
            img: an ITK image object loaded from an image file.

        """
        img_meta_dict = img.GetMetaDataDictionary()
        meta_dict = {key: img_meta_dict[key] for key in img_meta_dict.GetKeys() if not key.startswith("ITK_")}

        meta_dict["spacing"] = np.asarray(img.GetSpacing())
        return meta_dict

    def _get_affine(self, img, lps_to_ras: bool = True):
        """
        Get or construct the affine matrix of the image, it can be used to correct
        spacing, orientation or execute spatial transforms.

        Args:
            img: an ITK image object loaded from an image file.
            lps_to_ras: whether to convert the affine matrix from "LPS" to "RAS". Defaults to True.

        """
        direction = itk.array_from_matrix(img.GetDirection())
        spacing = np.asarray(img.GetSpacing())
        origin = np.asarray(img.GetOrigin())

        direction = np.asarray(direction)
        sr = min(max(direction.shape[0], 1), 3)
        affine: np.ndarray = np.eye(sr + 1)
        affine[:sr, :sr] = direction[:sr, :sr] @ np.diag(spacing[:sr])
        affine[:sr, -1] = origin[:sr]
        if lps_to_ras:
            affine = orientation_ras_lps(affine)
        return affine

    def _get_spatial_shape(self, img):
        """
        Get the spatial shape of `img`.

        Args:
            img: an ITK image object loaded from an image file.

        """
        sr = itk.array_from_matrix(img.GetDirection()).shape[0]
        sr = max(min(sr, 3), 1)
        _size = list(itk.size(img))
        if self.channel_dim is not None:
            _size.pop(self.channel_dim)
        return np.asarray(_size[:sr])

    def _get_array_data(self, img):
        """
        Get the raw array data of the image, converted to Numpy array.

        Following PyTorch conventions, the returned array data has contiguous channels,
        e.g. for an RGB image, all red channel image pixels are contiguous in memory.
        The last axis of the returned array is the channel axis.

        See also:

            - https://github.com/InsightSoftwareConsortium/ITK/blob/v5.2.1/Modules/Bridge/NumPy/wrapping/PyBuffer.i.in

        Args:
            img: an ITK image object loaded from an image file.

        """
        np_img = itk.array_view_from_image(img, keep_axes=False)
        if img.GetNumberOfComponentsPerPixel() == 1:  # handling spatial images
            return np_img if self.reverse_indexing else np_img.T
        # handling multi-channel images
        return np_img if self.reverse_indexing else np.moveaxis(np_img.T, 0, -1)


@require_pkg(pkg_name="nibabel")
class NibabelReader(ImageReader):
    """
    Load NIfTI format images based on Nibabel library.

    Args:
        as_closest_canonical: if True, load the image as closest to canonical axis format.
        squeeze_non_spatial_dims: if True, non-spatial singletons will be squeezed, e.g. (256,256,1,3) -> (256,256,3)
        channel_dim: the channel dimension of the input image, default is None.
            this is used to set original_channel_dim in the metadata, EnsureChannelFirstD reads this field.
            if None, `original_channel_dim` will be either `no_channel` or `-1`.
            most Nifti files are usually "channel last", no need to specify this argument for them.
        dtype: dtype of the output data array when loading with Nibabel library.
        kwargs: additional args for `nibabel.load` API. more details about available args:
            https://github.com/nipy/nibabel/blob/master/nibabel/loadsave.py

    """

    def __init__(
        self,
        channel_dim: Optional[int] = None,
        as_closest_canonical: bool = False,
        squeeze_non_spatial_dims: bool = False,
        dtype: DtypeLike = np.float32,
        **kwargs,
    ):
        super().__init__()
        self.channel_dim = channel_dim
        self.as_closest_canonical = as_closest_canonical
        self.squeeze_non_spatial_dims = squeeze_non_spatial_dims
        self.dtype = dtype
        self.kwargs = kwargs

    def verify_suffix(self, filename: Union[Sequence[PathLike], PathLike]) -> bool:
        """
        Verify whether the specified file or files format is supported by Nibabel reader.

        Args:
            filename: file name or a list of file names to read.
                if a list of files, verify all the suffixes.

        """
        suffixes: Sequence[str] = ["nii", "nii.gz"]
        return has_nib and is_supported_format(filename, suffixes)

    def read(self, data: Union[Sequence[PathLike], PathLike], **kwargs):
        """
        Read image data from specified file or files, it can read a list of images
        and stack them together as multi-channel data in `get_data()`.
        Note that the returned object is Nibabel image object or list of Nibabel image objects.

        Args:
            data: file name or a list of file names to read.
            kwargs: additional args for `nibabel.load` API, will override `self.kwargs` for existing keys.
                More details about available args:
                https://github.com/nipy/nibabel/blob/master/nibabel/loadsave.py

        """
        img_: List[Nifti1Image] = []

        filenames: Sequence[PathLike] = ensure_tuple(data)
        kwargs_ = self.kwargs.copy()
        kwargs_.update(kwargs)
        for name in filenames:
            img = nib.load(name, **kwargs_)
            img = correct_nifti_header_if_necessary(img)
            img_.append(img)
        return img_ if len(filenames) > 1 else img_[0]

    def get_data(self, img):
        """
        Extract data array and metadata from loaded image and return them.
        This function returns two objects, first is numpy array of image data, second is dict of metadata.
        It constructs `affine`, `original_affine`, and `spatial_shape` and stores them in meta dict.
        When loading a list of files, they are stacked together at a new dimension as the first dimension,
        and the metadata of the first image is used to present the output metadata.

        Args:
            img: a Nibabel image object loaded from an image file or a list of Nibabel image objects.

        """
        img_array: List[np.ndarray] = []
        compatible_meta: Dict = {}

        for i in ensure_tuple(img):
            header = self._get_meta_dict(i)
            header["affine"] = self._get_affine(i)
            header["original_affine"] = self._get_affine(i)
            header["as_closest_canonical"] = self.as_closest_canonical
            if self.as_closest_canonical:
                i = nib.as_closest_canonical(i)
                header["affine"] = self._get_affine(i)
            header["spatial_shape"] = self._get_spatial_shape(i)
            data = self._get_array_data(i)
            if self.squeeze_non_spatial_dims:
                for d in range(len(data.shape), len(header["spatial_shape"]), -1):
                    if data.shape[d - 1] == 1:
                        data = data.squeeze(axis=d - 1)
            img_array.append(data)
            if self.channel_dim is None:  # default to "no_channel" or -1
                header["original_channel_dim"] = "no_channel" if len(data.shape) == len(header["spatial_shape"]) else -1
            else:
                header["original_channel_dim"] = self.channel_dim
            _copy_compatible_dict(header, compatible_meta)

        return _stack_images(img_array, compatible_meta), compatible_meta

    def _get_meta_dict(self, img) -> Dict:
        """
        Get the all the metadata of the image and convert to dict type.

        Args:
            img: a Nibabel image object loaded from an image file.

        """
        # swap to little endian as PyTorch doesn't support big endian
        try:
            header = img.header.as_byteswapped("<")
        except ValueError:
            header = img.header
        return dict(header)

    def _get_affine(self, img):
        """
        Get the affine matrix of the image, it can be used to correct
        spacing, orientation or execute spatial transforms.

        Args:
            img: a Nibabel image object loaded from an image file.

        """
        return np.array(img.affine, copy=True)

    def _get_spatial_shape(self, img):
        """
        Get the spatial shape of image data, it doesn't contain the channel dim.

        Args:
            img: a Nibabel image object loaded from an image file.

        """
        # swap to little endian as PyTorch doesn't support big endian
        try:
            header = img.header.as_byteswapped("<")
        except ValueError:
            header = img.header
        dim = header.get("dim", None)
        if dim is None:
            dim = header.get("dims")  # mgh format?
            dim = np.insert(dim, 0, 3)
        ndim = dim[0]
        size = list(dim[1:])
        if self.channel_dim is not None:
            size.pop(self.channel_dim)
        spatial_rank = max(min(ndim, 3), 1)
        return np.asarray(size[:spatial_rank])

    def _get_array_data(self, img):
        """
        Get the raw array data of the image, converted to Numpy array.

        Args:
            img: a Nibabel image object loaded from an image file.

        """
        _array = np.array(img.get_fdata(dtype=self.dtype))
        img.uncache()
        return _array


class NumpyReader(ImageReader):
    """
    Load NPY or NPZ format data based on Numpy library, they can be arrays or pickled objects.
    A typical usage is to load the `mask` data for classification task.
    It can load part of the npz file with specified `npz_keys`.

    Args:
        npz_keys: if loading npz file, only load the specified keys, if None, load all the items.
            stack the loaded items together to construct a new first dimension.
        channel_dim: if not None, explicitly specify the channel dim, otherwise, treat the array as no channel.
        kwargs: additional args for `numpy.load` API except `allow_pickle`. more details about available args:
            https://numpy.org/doc/stable/reference/generated/numpy.load.html

    """

    def __init__(self, npz_keys: Optional[KeysCollection] = None, channel_dim: Optional[int] = None, **kwargs):
        super().__init__()
        if npz_keys is not None:
            npz_keys = ensure_tuple(npz_keys)
        self.npz_keys = npz_keys
        self.channel_dim = channel_dim
        self.kwargs = kwargs

    def verify_suffix(self, filename: Union[Sequence[PathLike], PathLike]) -> bool:
        """
        Verify whether the specified file or files format is supported by Numpy reader.

        Args:
            filename: file name or a list of file names to read.
                if a list of files, verify all the suffixes.
        """
        suffixes: Sequence[str] = ["npz", "npy"]
        return is_supported_format(filename, suffixes)

    def read(self, data: Union[Sequence[PathLike], PathLike], **kwargs):
        """
        Read image data from specified file or files, it can read a list of data files
        and stack them together as multi-channel data in `get_data()`.
        Note that the returned object is Numpy array or list of Numpy arrays.

        Args:
            data: file name or a list of file names to read.
            kwargs: additional args for `numpy.load` API except `allow_pickle`, will override `self.kwargs` for existing keys.
                More details about available args:
                https://numpy.org/doc/stable/reference/generated/numpy.load.html

        """
        img_: List[Nifti1Image] = []

        filenames: Sequence[PathLike] = ensure_tuple(data)
        kwargs_ = self.kwargs.copy()
        kwargs_.update(kwargs)
        for name in filenames:
            img = np.load(name, allow_pickle=True, **kwargs_)
            if Path(name).name.endswith(".npz"):
                # load expected items from NPZ file
                npz_keys = [f"arr_{i}" for i in range(len(img))] if self.npz_keys is None else self.npz_keys
                for k in npz_keys:
                    img_.append(img[k])
            else:
                img_.append(img)

        return img_ if len(img_) > 1 else img_[0]

    def get_data(self, img):
        """
        Extract data array and metadata from loaded image and return them.
        This function returns two objects, first is numpy array of image data, second is dict of metadata.
        It constructs `affine`, `original_affine`, and `spatial_shape` and stores them in meta dict.
        When loading a list of files, they are stacked together at a new dimension as the first dimension,
        and the metadata of the first image is used to represent the output metadata.

        Args:
            img: a Numpy array loaded from a file or a list of Numpy arrays.

        """
        img_array: List[np.ndarray] = []
        compatible_meta: Dict = {}
        if isinstance(img, np.ndarray):
            img = (img,)

        for i in ensure_tuple(img):
            header = {}
            if isinstance(i, np.ndarray):
                # if `channel_dim` is None, can not detect the channel dim, use all the dims as spatial_shape
                spatial_shape = np.asarray(i.shape)
                if isinstance(self.channel_dim, int):
                    spatial_shape = np.delete(spatial_shape, self.channel_dim)
                header["spatial_shape"] = spatial_shape
            img_array.append(i)
            header["original_channel_dim"] = self.channel_dim if isinstance(self.channel_dim, int) else "no_channel"
            _copy_compatible_dict(header, compatible_meta)

        return _stack_images(img_array, compatible_meta), compatible_meta


@require_pkg(pkg_name="PIL")
class PILReader(ImageReader):
    """
    Load common 2D image format (supports PNG, JPG, BMP) file or files from provided path.

    Args:
        converter: additional function to convert the image data after `read()`.
            for example, use `converter=lambda image: image.convert("LA")` to convert image format.
        kwargs: additional args for `Image.open` API in `read()`, mode details about available args:
            https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.open
    """

    def __init__(self, converter: Optional[Callable] = None, **kwargs):
        super().__init__()
        self.converter = converter
        self.kwargs = kwargs

    def verify_suffix(self, filename: Union[Sequence[PathLike], PathLike]) -> bool:
        """
        Verify whether the specified file or files format is supported by PIL reader.

        Args:
            filename: file name or a list of file names to read.
                if a list of files, verify all the suffixes.
        """
        suffixes: Sequence[str] = ["png", "jpg", "jpeg", "bmp"]
        return has_pil and is_supported_format(filename, suffixes)

    def read(self, data: Union[Sequence[PathLike], PathLike, np.ndarray], **kwargs):
        """
        Read image data from specified file or files, it can read a list of images
        and stack them together as multi-channel data in `get_data()`.
        Note that the returned object is PIL image or list of PIL image.

        Args:
            data: file name or a list of file names to read.
            kwargs: additional args for `Image.open` API in `read()`, will override `self.kwargs` for existing keys.
                Mode details about available args:
                https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.open

        """
        img_: List[PILImage.Image] = []

        filenames: Sequence[PathLike] = ensure_tuple(data)
        kwargs_ = self.kwargs.copy()
        kwargs_.update(kwargs)
        for name in filenames:
            img = PILImage.open(name, **kwargs_)
            if callable(self.converter):
                img = self.converter(img)
            img_.append(img)

        return img_ if len(filenames) > 1 else img_[0]

    def get_data(self, img):
        """
        Extract data array and metadata from loaded image and return them.
        This function returns two objects, first is numpy array of image data, second is dict of metadata.
        It computes `spatial_shape` and stores it in meta dict.
        When loading a list of files, they are stacked together at a new dimension as the first dimension,
        and the metadata of the first image is used to represent the output metadata.
        Note that it will swap axis 0 and 1 after loading the array because the `HW` definition in PIL
        is different from other common medical packages.

        Args:
            img: a PIL Image object loaded from a file or a list of PIL Image objects.

        """
        img_array: List[np.ndarray] = []
        compatible_meta: Dict = {}

        for i in ensure_tuple(img):
            header = self._get_meta_dict(i)
            header["spatial_shape"] = self._get_spatial_shape(i)
            data = np.moveaxis(np.asarray(i), 0, 1)
            img_array.append(data)
            header["original_channel_dim"] = "no_channel" if len(data.shape) == len(header["spatial_shape"]) else -1
            _copy_compatible_dict(header, compatible_meta)

        return _stack_images(img_array, compatible_meta), compatible_meta

    def _get_meta_dict(self, img) -> Dict:
        """
        Get the all the metadata of the image and convert to dict type.
        Args:
            img: a PIL Image object loaded from an image file.

        """
        return {"format": img.format, "mode": img.mode, "width": img.width, "height": img.height}

    def _get_spatial_shape(self, img):
        """
        Get the spatial shape of image data, it doesn't contain the channel dim.
        Args:
            img: a PIL Image object loaded from an image file.
        """
        return np.asarray((img.width, img.height))


class WSIReader(ImageReader):
    """
    Read whole slide images and extract patches.

    Args:
        backend: backend library to load the images, available options: "cuCIM", "OpenSlide" and "TiffFile".
        level: the whole slide image level at which the image is extracted. (default=0)
            This is overridden if the level argument is provided in `get_data`.
        kwargs: additional args for backend reading API in `read()`, more details in `cuCIM`, `TiffFile`, `OpenSlide`:
            https://github.com/rapidsai/cucim/blob/v21.12.00/cpp/include/cucim/cuimage.h#L100.
            https://github.com/cgohlke/tifffile.
            https://openslide.org/api/python/#openslide.OpenSlide.

    Note:
        While "cuCIM" and "OpenSlide" backends both can load patches from large whole slide images
        without loading the entire image into memory, "TiffFile" backend needs to load the entire image into memory
        before extracting any patch; thus, memory consideration is needed when using "TiffFile" backend for
        patch extraction.

    """

    def __init__(self, backend: str = "OpenSlide", level: int = 0, **kwargs):
        super().__init__()
        self.backend = backend.lower()
        func = require_pkg(self.backend)(self._set_reader)
        self.wsi_reader = func(self.backend)
        self.level = level
        self.kwargs = kwargs

    @staticmethod
    def _set_reader(backend: str):
        if backend == "openslide":
            return OpenSlide
        if backend == "cucim":
            return CuImage
        if backend == "tifffile":
            return TiffFile
        raise ValueError("`backend` should be 'cuCIM', 'OpenSlide' or 'TiffFile'.")

    def verify_suffix(self, filename: Union[Sequence[PathLike], PathLike]) -> bool:
        """
        Verify whether the specified file or files format is supported by WSI reader.

        Args:
            filename: file name or a list of file names to read.
                if a list of files, verify all the suffixes.
        """
        return is_supported_format(filename, ["tif", "tiff"])

    def read(self, data: Union[Sequence[PathLike], PathLike, np.ndarray], **kwargs):
        """
        Read image data from given file or list of files.

        Args:
            data: file name or a list of file names to read.
            kwargs: additional args for backend reading API in `read()`, will override `self.kwargs` for existing keys.
                more details in `cuCIM`, `TiffFile`, `OpenSlide`:
                https://github.com/rapidsai/cucim/blob/v21.12.00/cpp/include/cucim/cuimage.h#L100.
                https://github.com/cgohlke/tifffile.
                https://openslide.org/api/python/#openslide.OpenSlide.

        Returns:
            image object or list of image objects

        """
        img_: List = []

        filenames: Sequence[PathLike] = ensure_tuple(data)
        kwargs_ = self.kwargs.copy()
        kwargs_.update(kwargs)
        for name in filenames:
            img = self.wsi_reader(name, **kwargs_)
            if self.backend == "openslide":
                img.shape = (img.dimensions[1], img.dimensions[0], 3)
            img_.append(img)

        return img_ if len(filenames) > 1 else img_[0]

    def get_data(
        self,
        img,
        location: Tuple[int, int] = (0, 0),
        size: Optional[Tuple[int, int]] = None,
        level: Optional[int] = None,
        dtype: DtypeLike = np.uint8,
        grid_shape: Tuple[int, int] = (1, 1),
        patch_size: Optional[Union[int, Tuple[int, int]]] = None,
    ):
        """
        Extract regions as numpy array from WSI image and return them.

        Args:
            img: a WSIReader image object loaded from a file, or list of CuImage objects
            location: (top, left) tuple giving the top left pixel in the level 0 reference frame,
            or list of tuples (default=(0, 0))
            size: (height, width) tuple giving the region size, or list of tuples (default to full image size)
            This is the size of image at the given level (`level`)
            level: the level number, or list of level numbers (default=0)
            dtype: the data type of output image
            grid_shape: (row, columns) tuple define a grid to extract patches on that
            patch_size: (height, width) the size of extracted patches at the given level
        """
        # Verify inputs
        if level is None:
            level = self.level
        max_level = self._get_max_level(img)
        if level > max_level:
            raise ValueError(f"The maximum level of this image is {max_level} while level={level} is requested)!")

        # Extract a region or the entire image
        region = self._extract_region(img, location=location, size=size, level=level, dtype=dtype)

        # Add necessary metadata
        metadata: Dict = {}
        metadata["spatial_shape"] = np.asarray(region.shape[:-1])
        metadata["original_channel_dim"] = -1

        # Make it channel first
        region = EnsureChannelFirst()(region, metadata)

        # Split into patches
        if patch_size is None:
            patches = region
        else:
            tuple_patch_size = ensure_tuple_rep(patch_size, 2)
            patches = self._extract_patches(
                region, patch_size=tuple_patch_size, grid_shape=grid_shape, dtype=dtype  # type: ignore
            )

        return patches, metadata

    def _get_max_level(self, img_obj):
        """
        Return the maximum number of levels in the whole slide image
        Args:
            img: the whole slide image object

        """
        if self.backend == "openslide":
            return img_obj.level_count - 1
        if self.backend == "cucim":
            return img_obj.resolutions["level_count"] - 1
        if self.backend == "tifffile":
            return len(img_obj.pages) - 1

    def _get_image_size(self, img, size, level, location):
        """
        Calculate the maximum region size for the given level and starting location (if size is None).
        Note that region size in OpenSlide and cuCIM are WxH (but the final image output would be HxW)
        """
        if size is not None:
            return size[::-1]

        max_size = []
        downsampling_factor = []
        if self.backend == "openslide":
            downsampling_factor = img.level_downsamples[level]
            max_size = img.level_dimensions[level]
        elif self.backend == "cucim":
            downsampling_factor = img.resolutions["level_downsamples"][level]
            max_size = img.resolutions["level_dimensions"][level]

        # subtract the top left corner of the patch (at given level) from maximum size
        location_at_level = (round(location[1] / downsampling_factor), round(location[0] / downsampling_factor))
        size = [max_size[i] - location_at_level[i] for i in range(len(max_size))]

        return size

    def _extract_region(
        self,
        img_obj,
        size: Optional[Tuple[int, int]],
        location: Tuple[int, int] = (0, 0),
        level: int = 0,
        dtype: DtypeLike = np.uint8,
    ):
        if self.backend == "tifffile":
            # Read the entire image
            if size is not None:
                raise ValueError(
                    f"TiffFile backend reads the entire image only, so size '{size}'' should not be provided!",
                    "For more flexibility or extracting regions, please use cuCIM or OpenSlide backend.",
                )
            if location != (0, 0):
                raise ValueError(
                    f"TiffFile backend reads the entire image only, so location '{location}' should not be provided!",
                    "For more flexibility and extracting regions, please use cuCIM or OpenSlide backend.",
                )
            region = img_obj.asarray(level=level)
        else:
            # Get region size to be extracted
            region_size = self._get_image_size(img_obj, size, level, location)
            # reverse the order of location's dimensions to become WxH (for cuCIM and OpenSlide)
            region_location = location[::-1]
            # Extract a region (or the entire image)
            region = img_obj.read_region(location=region_location, size=region_size, level=level)

        region = self.convert_to_rgb_array(region, dtype)
        return region

    def convert_to_rgb_array(self, raw_region, dtype: DtypeLike = np.uint8):
        """Convert to RGB mode and numpy array"""
        if self.backend == "openslide":
            # convert to RGB
            raw_region = raw_region.convert("RGB")

        # convert to numpy (if not already in numpy)
        raw_region = np.asarray(raw_region, dtype=dtype)

        # check if the image has three dimensions (2D + color)
        if raw_region.ndim != 3:
            raise ValueError(
                f"The input image dimension should be 3 but {raw_region.ndim} is given. "
                "`WSIReader` is designed to work only with 2D colored images."
            )

        # check if the color channel is 3 (RGB) or 4 (RGBA)
        if raw_region.shape[-1] not in [3, 4]:
            raise ValueError(
                f"There should be three or four color channels but {raw_region.shape[-1]} is given. "
                "`WSIReader` is designed to work only with 2D colored images."
            )

        # remove alpha channel if exist (RGBA)
        if raw_region.shape[-1] > 3:
            raw_region = raw_region[..., :3]

        return raw_region

    def _extract_patches(
        self,
        region: np.ndarray,
        grid_shape: Tuple[int, int] = (1, 1),
        patch_size: Optional[Tuple[int, int]] = None,
        dtype: DtypeLike = np.uint8,
    ):
        if patch_size is None and grid_shape == (1, 1):
            return region

        n_patches = grid_shape[0] * grid_shape[1]
        region_size = region.shape[1:]

        if patch_size is None:
            patch_size = (region_size[0] // grid_shape[0], region_size[1] // grid_shape[1])

        # split the region into patches on the grid and center crop them to patch size
        flat_patch_grid = np.zeros((n_patches, 3, patch_size[0], patch_size[1]), dtype=dtype)
        start_points = [
            np.round(region_size[i] * (0.5 + np.arange(grid_shape[i])) / grid_shape[i] - patch_size[i] / 2).astype(int)
            for i in range(2)
        ]
        idx = 0
        for y_start in start_points[1]:
            for x_start in start_points[0]:
                x_end = x_start + patch_size[0]
                y_end = y_start + patch_size[1]
                flat_patch_grid[idx] = region[:, x_start:x_end, y_start:y_end]
                idx += 1

        return flat_patch_grid


@dataclass
class NrrdImage:
    """Class to wrap nrrd image array and metadata header"""

    array: np.ndarray
    header: dict


@require_pkg(pkg_name="nrrd")
class NrrdReader(ImageReader):
    """
    Load NRRD format images based on pynrrd library.

    Args:
        channel_dim: the channel dimension of the input image, default is None.
            This is used to set original_channel_dim in the metadata, EnsureChannelFirstD reads this field.
            If None, `original_channel_dim` will be either `no_channel` or `0`.
            NRRD files are usually "channel first".
        dtype: dtype of the data array when loading image.
        index_order: Specify whether the returned data array should be in C-order (‘C’) or Fortran-order (‘F’).
            Numpy is usually in C-order, but default on the NRRD header is F
        kwargs: additional args for `nrrd.read` API. more details about available args:
            https://github.com/mhe/pynrrd/blob/master/nrrd/reader.py

    """

    def __init__(
        self,
        channel_dim: Optional[int] = None,
        dtype: Union[np.dtype, type, str, None] = np.float32,
        index_order: str = "F",
        **kwargs,
    ):
        self.channel_dim = channel_dim
        self.dtype = dtype
        self.index_order = index_order
        self.kwargs = kwargs

    def verify_suffix(self, filename: Union[Sequence[PathLike], PathLike]) -> bool:
        """
        Verify whether the specified `filename` is supported by pynrrd reader.

        Args:
            filename: file name or a list of file names to read.
                if a list of files, verify all the suffixes.

        """
        suffixes: Sequence[str] = ["nrrd", "seg.nrrd"]
        return has_nrrd and is_supported_format(filename, suffixes)

    def read(self, data: Union[Sequence[PathLike], PathLike], **kwargs) -> Union[Sequence[Any], Any]:
        """
        Read image data from specified file or files.
        Note that it returns a data object or a sequence of data objects.

        Args:
            data: file name or a list of file names to read.
            kwargs: additional args for actual `read` API of 3rd party libs.

        """
        img_: List = []
        filenames: Sequence[PathLike] = ensure_tuple(data)
        kwargs_ = self.kwargs.copy()
        kwargs_.update(kwargs)
        for name in filenames:
            nrrd_image = NrrdImage(*nrrd.read(name, index_order=self.index_order, *kwargs_))
            img_.append(nrrd_image)
        return img_ if len(filenames) > 1 else img_[0]

    def get_data(self, img: Union[NrrdImage, List[NrrdImage]]) -> Tuple[np.ndarray, Dict]:
        """
        Extract data array and metadata from loaded image and return them.
        This function must return two objects, the first is a numpy array of image data,
        the second is a dictionary of metadata.

        Args:
            img: a `NrrdImage` loaded from an image file or a list of image objects.

        """
        img_array: List[np.ndarray] = []
        compatible_meta: Dict = {}

        for i in ensure_tuple(img):
            data = i.array.astype(self.dtype)
            img_array.append(data)
            header = dict(i.header)
            if self.index_order == "C":
                header = self._convert_f_to_c_order(header)
            header["original_affine"] = self._get_affine(i)
            header = self._switch_lps_ras(header)
            header["affine"] = header["original_affine"].copy()
            header["spatial_shape"] = header["sizes"]
            [header.pop(k) for k in ("sizes", "space origin", "space directions")]  # rm duplicated data in header

            if self.channel_dim is None:  # default to "no_channel" or -1
                header["original_channel_dim"] = "no_channel" if len(data.shape) == len(header["spatial_shape"]) else 0
            else:
                header["original_channel_dim"] = self.channel_dim
            _copy_compatible_dict(header, compatible_meta)

        return _stack_images(img_array, compatible_meta), compatible_meta

    def _get_affine(self, img: NrrdImage) -> np.ndarray:
        """
        Get the affine matrix of the image, it can be used to correct
        spacing, orientation or execute spatial transforms.

        Args:
            img: A `NrrdImage` loaded from image file

        """
        direction = img.header["space directions"]
        origin = img.header["space origin"]

        x, y = direction.shape
        affine_diam = min(x, y) + 1
        affine: np.ndarray = np.eye(affine_diam)
        affine[:x, :y] = direction
        affine[: (affine_diam - 1), -1] = origin  # len origin is always affine_diam - 1
        return affine

    def _switch_lps_ras(self, header: dict) -> dict:
        """
        For compatibility with nibabel, switch from LPS to RAS. Adapt affine matrix and
        `space` argument in header accordingly. If no information of space is given in the header,
        LPS is assumed and thus converted to RAS. If information about space is given,
        but is not LPS, the unchanged header is returned.

        Args:
            header: The image metadata as dict

        """
        if "space" not in header or header["space"] == "left-posterior-superior":
            header["space"] = "right-anterior-superior"
            header["original_affine"] = orientation_ras_lps(header["original_affine"])
        return header

    def _convert_f_to_c_order(self, header: dict) -> dict:
        """
        All header fields of a NRRD are specified in `F` (Fortran) order, even if the image was read as C-ordered array.
        1D arrays of header['space origin'] and header['sizes'] become inverted, e.g, [1,2,3] -> [3,2,1]
        The 2D Array for header['space directions'] is transposed: [[1,0,0],[0,2,0],[0,0,3]] -> [[3,0,0],[0,2,0],[0,0,1]]
        For more details refer to: https://pynrrd.readthedocs.io/en/latest/user-guide.html#index-ordering

        Args:
            header: The image metadata as dict

        """

        header["space directions"] = np.rot90(np.flip(header["space directions"], 0))
        header["space origin"] = header["space origin"][::-1]
        header["sizes"] = header["sizes"][::-1]
        return header
