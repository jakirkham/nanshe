"""
The module ``xtiff`` provides support for conversion from TIFF to HDF5.

===============================================================================
Overview
===============================================================================
The module ``xtiff`` implements a relatively simplistic form of conversion from
TIFF to HDF5. Preserves the description fields from the metadata if found as a
list under the attribute `descriptions`. Additionally, keeps track of the TIFF
filenames stitched together and the offsets of each TIFF file as the attributes
`filenames` and `offsets`, respectively.

.. todo::

    Currently, this only keeps one channel and works on one Z-plane. It would \
    be nice to relax these constraints and these features in the future.

===============================================================================
API
===============================================================================
"""


__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "$Jun 26, 2014 11:40:54 EDT$"


import collections
import os

import numpy
import h5py

import vigra
import vigra.impex

from nanshe.util import iters, xglob, prof, xnumpy
from nanshe.io import hdf5

import tifffile

from past.builtins import unicode


# Get the logger
trace_logger = prof.getTraceLogger(__name__)



@prof.log_call(trace_logger)
def get_multipage_tiff_shape_dtype(new_tiff_filename):
    """
        Gets the info about the shape (including page number as time)
        and dtype.

        Args:
            new_tiff_filename(str):             the TIFF file to get info about

        Returns:
            (collections.OrderedDict):          an ordered dictionary with
                                                "shape" first and "dtype"
                                                (type) second.
    """

    shape_dtype_result = collections.OrderedDict(
        [("shape", None), ("dtype", None)])

    new_tiff_file_info = vigra.impex.ImageInfo(new_tiff_filename)
    new_tiff_file_number_pages = vigra.impex.numberImages(new_tiff_filename)

    new_tiff_file_shape = new_tiff_file_info.getShape()
    shape_dtype_result["shape"] = new_tiff_file_shape[
        :-1] + (new_tiff_file_number_pages,) + new_tiff_file_shape[-1:]

    shape_dtype_result["dtype"] = new_tiff_file_info.getDtype()

    return(shape_dtype_result)


@prof.log_call(trace_logger)
def get_multipage_tiff_shape_dtype_transformed(new_tiff_filename,
                                               axis_order="zyxtc",
                                               pages_to_channel=1):
    """
        Gets the info about the shape and dtype after some transformations
        have been performed.

        Args:
            new_tiff_filename(str):             the TIFF file to get info about
            axis_order(str):                    the desired axis order
                                                when reshaped

            pages_to_channel(int):              number of channels to divide
                                                from the pages

        Returns:
            (collections.OrderedDict):          an ordered dictionary with
                                                "shape" first and "dtype"
                                                (type) second.
    """

    assert (pages_to_channel > 0)
    assert (len(axis_order) == 5)
    assert all([_ in axis_order for _ in "zyxtc"])

    new_tiff_file_shape, new_tiff_file_dtype = list(get_multipage_tiff_shape_dtype(
        new_tiff_filename
    ).values())

    # Correct if the tiff is missing dims by adding singletons
    if (len(new_tiff_file_shape) == 5):
        pass
    elif (len(new_tiff_file_shape) == 4):
        new_tiff_file_shape = (1,) + new_tiff_file_shape
    else:
        raise Exception(
            "Invalid dimensionality for TIFF. Found shape to be \"" +
            repr(new_tiff_file_shape) + "\"."
        )

    # Correct if some pages are for different channels
    if (pages_to_channel != 1):
        new_tiff_file_shape = new_tiff_file_shape[:-2] + \
                              (new_tiff_file_shape[-2] // pages_to_channel,
                               pages_to_channel * new_tiff_file_shape[-1],)

    # Correct the axis order
    if (axis_order != "zyxtc"):
        vigra_ordering = dict(iters.reverse_each_element(enumerate("zyxtc")))

        new_tiff_file_shape_transposed = []
        for each_axis_label in axis_order:
            new_tiff_file_shape_transposed.append(
                new_tiff_file_shape[vigra_ordering[each_axis_label]]
            )

        new_tiff_file_shape = tuple(new_tiff_file_shape_transposed)

    shape_dtype_result = collections.OrderedDict(
        [("shape", None), ("dtype", None)])

    shape_dtype_result["shape"] = new_tiff_file_shape
    shape_dtype_result["dtype"] = new_tiff_file_dtype

    return(shape_dtype_result)


@prof.log_call(trace_logger)
def get_standard_tiff_array(new_tiff_filename,
                            axis_order="tzyxc",
                            pages_to_channel=1,
                            memmap=False):
    """
        Reads a tiff file and returns a standard 5D array.

        Args:
            new_tiff_filename(str):             the TIFF file to read in

            axis_order(int):                    how to order the axes (by
                                                default returns "tzyxc").

            pages_to_channel(int):              if channels are not normally
                                                stored in the channel variable,
                                                but are stored as pages (or as
                                                a mixture), then this will
                                                split neighboring pages into
                                                separate channels. (by default
                                                is 1 so changes nothing)

            memmap(bool):                       allows one to load the array
                                                using a memory mapped file as
                                                opposed to reading it directly.
                                                (by default is False)

        Returns:
            (numpy.ndarray or numpy.memmap):    an array with the axis order
                                                specified.
    """

    assert (pages_to_channel > 0)

    with tifffile.TiffFile(new_tiff_filename) as new_tiff_file:
        new_tiff_array = new_tiff_file.asarray(memmap=memmap)

    # Add a singleton channel if none is present.
    if new_tiff_array.ndim == 3:
        new_tiff_array = new_tiff_array[None]

    # Fit the old VIGRA style array. (may try to remove in the future)
    new_tiff_array = new_tiff_array.transpose(
        tuple(iters.irange(new_tiff_array.ndim - 1, 1, -1)) + (1, 0)
    )

    # Check to make sure the dimensions are ok
    if (new_tiff_array.ndim == 5):
        pass
    elif (new_tiff_array.ndim == 4):
        # Has no z. So, add this.
        new_tiff_array = xnumpy.add_singleton_axis_beginning(new_tiff_array)
    else:
        raise Exception(
            "Invalid dimensionality for TIFF. Found shape to be \"" +
            repr(new_tiff_array.shape) + "\"."
        )

    # Some people use pages to hold time and channel data. So, we need to
    # restructure it. However, if they are properly structuring their TIFF
    # file, then they shouldn't incur a penalty.
    if pages_to_channel > 1:
        new_tiff_array = new_tiff_array.reshape(
            new_tiff_array.shape[:-2] +
            (new_tiff_array.shape[-2] // pages_to_channel,
             pages_to_channel * new_tiff_array.shape[-1],)
        )

    new_tiff_array = xnumpy.tagging_reorder_array(
        new_tiff_array,
        from_axis_order="zyxtc",
        to_axis_order=axis_order,
        to_copy=True
    )

    return(new_tiff_array)


@prof.log_call(trace_logger)
def get_standard_tiff_data(new_tiff_filename,
                           axis_order="tzyxc",
                           pages_to_channel=1,
                           memmap=False):
    """
        Reads a tiff file and returns a standard 5D array and the metadata.

        Args:
            new_tiff_filename(str):             the TIFF file to read in

            axis_order(int):                    how to order the axes (by
                                                default returns "tzyxc").

            pages_to_channel(int):              if channels are not normally
                                                stored in the channel variable,
                                                but are stored as pages (or as
                                                a mixture), then this will
                                                split neighboring pages into
                                                separate channels. (by default
                                                is 1 so changes nothing)

            memmap(bool):                       allows one to load the array
                                                using a memory mapped file as
                                                opposed to reading it directly.
                                                (by default is False)

        Returns:
            (ndarray/memmap, ndarray):          an array with the axis order
                                                specified and description
                                                metadata.
    """

    assert (pages_to_channel > 0)

    new_tiff_description = []
    with tifffile.TiffFile(new_tiff_filename) as new_tiff_file:
        new_tiff_array = new_tiff_file.asarray(memmap=memmap)

        for i in iters.irange(
                0,
                len(new_tiff_file),
                pages_to_channel
        ):
            new_tiff_description.append([])
            for j in iters.irange(pages_to_channel):
                each_page = new_tiff_file[i + j]
                each_metadata = each_page.tags
                each_desc = u""

                try:
                    each_desc = unicode(
                        each_metadata["image_description"].value
                    )
                except KeyError:
                    pass

                new_tiff_description[-1].append(
                    each_desc
                )

    new_tiff_description = numpy.array(new_tiff_description)


    # Add a singleton channel if none is present.
    if new_tiff_array.ndim == 3:
        new_tiff_array = new_tiff_array[None]

    # Fit the old VIGRA style array. (may try to remove in the future)
    new_tiff_array = new_tiff_array.transpose(
        tuple(iters.irange(new_tiff_array.ndim - 1, 1, -1)) + (1, 0)
    )

    # Check to make sure the dimensions are ok
    if (new_tiff_array.ndim == 5):
        pass
    elif (new_tiff_array.ndim == 4):
        # Has no z. So, add this.
        new_tiff_array = xnumpy.add_singleton_axis_beginning(new_tiff_array)
    elif (new_tiff_array.ndim == 2):
        # Has no z, channel or time. So, add these.
        new_tiff_array = xnumpy.add_singleton_axis_beginning(new_tiff_array)
        new_tiff_array = xnumpy.add_singleton_axis_end(new_tiff_array)
        new_tiff_array = xnumpy.add_singleton_axis_end(new_tiff_array)
    else:
        raise Exception(
            "Invalid dimensionality for TIFF. Found shape to be \"" +
            repr(new_tiff_array.shape) + "\"."
        )

    # Some people use pages to hold time and channel data. So, we need to
    # restructure it. However, if they are properly structuring their TIFF
    # file, then they shouldn't incur a penalty.
    if pages_to_channel > 1:
        new_tiff_array = new_tiff_array.reshape(
            new_tiff_array.shape[:-2] +
            (new_tiff_array.shape[-2] // pages_to_channel,
             pages_to_channel * new_tiff_array.shape[-1],)
        )

    new_tiff_array = xnumpy.tagging_reorder_array(
        new_tiff_array,
        from_axis_order="zyxtc",
        to_axis_order=axis_order,
        to_copy=True
    )

    # Currently is `tc` order so convert it to the expected order.
    if axis_order.index("t") > axis_order.index("c"):
        new_tiff_description = new_tiff_description.T.copy()

    return(new_tiff_array, new_tiff_description)


@prof.log_call(trace_logger)
def convert_tiffs(new_tiff_filenames,
                  new_hdf5_pathname,
                  axis=0,
                  channel=0,
                  z_index=0,
                  pages_to_channel=1,
                  memmap=False):
    """
        Convert a stack of tiffs to an HDF5 file.

        Args:
            new_tiff_filenames(list or str):    takes a str for a single file
                                                or a list of strs for
                                                filenames to combine (allows
                                                regex).

            new_hdf5_pathname(str):             the HDF5 file and location to
                                                store the dataset.

            axis(int):                          which axis to concatenate
                                                along.

            channel(int):                       which channel to select for the
                                                HDF5 (can only keep one).

            z_index(int):                       which z value to take (the
                                                algorithm is not setup for 3D
                                                data yet)

            pages_to_channel(int):              if channels are not normally
                                                stored in the channel variable,
                                                but are stored as pages, then
                                                this will split neighboring
                                                pages into separate channels.

            memmap(bool):                       allows one to load the array
                                                using a memory mapped file as
                                                opposed to reading it directly.
                                                (by default is False)
    """

    assert (pages_to_channel > 0)

    # Get the axes that do not change
    static_axes = numpy.array(list(iters.xrange_with_skip(
        3, to_skip=axis
    )))

    # if it is only a single str, make it a singleton list
    if isinstance(new_tiff_filenames, (bytes, unicode)):
        new_tiff_filenames = [new_tiff_filenames]

    # Expand any regex in path names
    new_tiff_filenames = xglob.expand_pathname_list(*new_tiff_filenames)

    # Extract the offset and descriptions for storage.
    new_hdf5_dataset_filenames = list()
    new_hdf5_dataset_offsets = list()

    # Determine the shape and dtype to use for the dataset (so that everything
    # will fit).
    new_hdf5_dataset_shape = numpy.zeros((3,), dtype=int)
    new_hdf5_dataset_dtype = bool
    for each_new_tiff_filename in new_tiff_filenames:
        # Add each filename.
        new_hdf5_dataset_filenames.append(unicode(each_new_tiff_filename))

        # Get all of the offsets.
        new_hdf5_dataset_offsets.append(new_hdf5_dataset_shape[axis])

        # Get the shape and type of each frame.
        each_new_tiff_file_shape, each_new_tiff_file_dtype = list(get_multipage_tiff_shape_dtype_transformed(
            each_new_tiff_filename,
            axis_order="cztyx",
            pages_to_channel=pages_to_channel
        ).values())
        each_new_tiff_file_shape = each_new_tiff_file_shape[2:]

        # Find the increase on the merge axis. Find the largest shape for the
        # rest.
        each_new_tiff_file_shape = numpy.array(each_new_tiff_file_shape)
        new_hdf5_dataset_shape[axis] += each_new_tiff_file_shape[axis]
        new_hdf5_dataset_shape[static_axes] = numpy.array(
            [
                new_hdf5_dataset_shape[static_axes],
                each_new_tiff_file_shape[static_axes]
            ]
        ).max(axis=0)

        # Finds the best type that everything can be cast to without loss of
        # precision.
        if not numpy.can_cast(each_new_tiff_file_dtype, new_hdf5_dataset_dtype):
            if numpy.can_cast(new_hdf5_dataset_dtype, each_new_tiff_file_dtype):
                new_hdf5_dataset_dtype = each_new_tiff_file_dtype
            else:
                raise Exception(
                    "Cannot find safe conversion between" +
                    " new_hdf5_dataset_dtype = " +
                    repr(new_hdf5_dataset_dtype) +
                    " and each_new_tiff_file_dtype = " +
                    repr(each_new_tiff_file_dtype) + "."
                )

    # Convert to arrays.
    new_hdf5_dataset_filenames = numpy.array(new_hdf5_dataset_filenames)
    new_hdf5_dataset_offsets = numpy.array(new_hdf5_dataset_offsets)

    # Convert to standard forms
    new_hdf5_dataset_shape = tuple(new_hdf5_dataset_shape)
    new_hdf5_dataset_dtype = numpy.dtype(new_hdf5_dataset_dtype)

    # Get all the needed locations for the HDF5 file and dataset
    new_hdf5_filename, new_hdf5_dataset_name = hdf5.serializers.split_hdf5_path(new_hdf5_pathname)
    new_hdf5_groupname = os.path.dirname(new_hdf5_dataset_name)

    # Dump all datasets to the file
    with h5py.File(new_hdf5_filename, "a") as new_hdf5_file:
        new_hdf5_group = new_hdf5_file.require_group(new_hdf5_groupname)
        new_hdf5_dataset = new_hdf5_group.create_dataset(
            new_hdf5_dataset_name,
            new_hdf5_dataset_shape,
            new_hdf5_dataset_dtype,
            chunks=True
        )
        new_hdf5_dataset.attrs.create(
            "filenames",
            new_hdf5_dataset_filenames,
            shape=new_hdf5_dataset_filenames.shape,
            dtype=h5py.special_dtype(vlen=unicode)
        )
        new_hdf5_dataset.attrs["offsets"] = new_hdf5_dataset_offsets
        # Workaround required due to this issue
        # ( https://github.com/h5py/h5py/issues/289 ).
        new_hdf5_descriptions_dataset = new_hdf5_group.create_dataset(
            "_".join([new_hdf5_dataset_name, "descriptions"]),
            shape=new_hdf5_dataset_shape[0:1],
            dtype=h5py.special_dtype(vlen=unicode)
        )
        new_hdf5_dataset.attrs["descriptions"] = (
            new_hdf5_descriptions_dataset.file.filename +
            new_hdf5_descriptions_dataset.name
        )

        new_hdf5_dataset_axis_pos = 0
        for each_new_tiff_filename in new_tiff_filenames:
            # Log the filename in case something goes wrong.
            trace_logger.info(
                "Now appending TIFF: \"" + str(each_new_tiff_filename) + "\""
            )

            # Read the data in the format specified.
            each_new_tiff_array, each_new_tiff_description = get_standard_tiff_data(
                each_new_tiff_filename,
                axis_order="cztyx",
                pages_to_channel=pages_to_channel,
                memmap=memmap
            )

            # Take channel and z selection
            # TODO: Could we drop the channel constraint?
            # TODO: Want to drop z constraint.
            each_new_tiff_array = each_new_tiff_array[channel, z_index]
            each_new_tiff_description = each_new_tiff_description[channel]

            # Store into the current slice and go to the next one.
            new_hdf5_dataset_axis_pos_next = new_hdf5_dataset_axis_pos + \
                                             len(each_new_tiff_array)
            new_hdf5_dataset[new_hdf5_dataset_axis_pos:new_hdf5_dataset_axis_pos_next] = each_new_tiff_array
            new_hdf5_descriptions_dataset[new_hdf5_dataset_axis_pos:new_hdf5_dataset_axis_pos_next] = each_new_tiff_description
            new_hdf5_dataset_axis_pos = new_hdf5_dataset_axis_pos_next
