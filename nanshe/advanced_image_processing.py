# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "$Apr 30, 2014 5:23:37PM$"



# Generally useful and fast to import so done immediately.
import numpy

# For image processing.
import scipy
import scipy.interpolate
import scipy.ndimage
import scipy.spatial
import scipy.spatial.distance

# More image processing...
import skimage
import skimage.measure
import skimage.feature
import skimage.morphology
import skimage.segmentation

import additional_generators

# To allow for more advanced iteration patterns
import itertools

# Allows for deep and shallow copies.
import copy

# Need for opening
import vigra
import vigra.filters
import vigra.analysis

# Need in order to have logging information no matter what.
import debugging_tools

import expanded_numpy

# Short function to process image data.
import simple_image_processing

# To remove noise from the basis images
import denoising

# Wavelet transformation operations
import wavelet_transform

import HDF5_recorder


# Get the logger
logger = debugging_tools.logging.getLogger(__name__)


@debugging_tools.log_call(logger)
@HDF5_recorder.static_array_debug_recorder
def remove_zeroed_lines(new_data,
                        erosion_shape,
                        dilation_shape,
                        **parameters):
    """
        Due to registration errors, there will sometimes be lines that are zero.
        To correct this, we find an interpolated value and

        Args:
            new_data(numpy.ndarray):            data to remove lines from (first axis is time).
            erosion_shape(numpy.ndarray):       shape of the erosion element (will be filled with 1).
            dilation_shape(numpy.ndarray):      shape of the dilation element (will be filled with 1).
            **parameters(dict):                 essentially unused (catches unneeded arguments).

        Returns:
            numpy.ndarray:                      a new array with the zeroed lines interpolated away.
    """

    result = numpy.zeros(new_data.shape)

    # Get an outline of the region around the parts of the image that contain zeros
    erosion_structure = numpy.ones(tuple(erosion_shape))
    dilation_structure = numpy.ones(tuple(dilation_shape))

    points = numpy.array(numpy.meshgrid(*[numpy.arange(_) for _ in new_data.shape[1:]], indexing="ij"))

    zero_masks_dilated = numpy.zeros(new_data.shape, dtype=bool)
    zero_masks_eroded = numpy.zeros(new_data.shape, dtype=bool)
    zero_masks_outline = numpy.zeros(new_data.shape, dtype=bool)

    for i in xrange(new_data.shape[0]):
        new_data_i = new_data[i]
        zero_mask_i = (new_data_i == 0)

        zero_mask_i_dilated = skimage.morphology.binary_dilation(zero_mask_i, dilation_structure).astype(bool)
        zero_mask_i_eroded = skimage.morphology.binary_erosion(zero_mask_i, erosion_structure).astype(bool)
        zero_mask_i_outline = zero_mask_i_dilated - zero_mask_i_eroded

        zero_masks_dilated[i] = zero_mask_i_dilated
        zero_masks_eroded[i] = zero_mask_i_eroded
        zero_masks_outline[i] = zero_mask_i_outline

        # Get the points that correspond to those
        zero_mask_i_outline_points = numpy.array(zero_mask_i_outline.nonzero()).transpose()

        new_data_i_zero_mask_interpolation = numpy.zeros(new_data_i.shape)
        if zero_mask_i_outline.any():
            new_data_i_zero_mask_interpolation = scipy.interpolate.griddata(zero_mask_i_outline_points,
                                                                            new_data_i[zero_mask_i_outline],
                                                                            tuple(points),
                                                                            method = "linear")

            # Only need to check for nan in our case.
            new_data_i_zero_mask_interpolation = numpy.where(numpy.isnan(new_data_i_zero_mask_interpolation),
                                                             0,
                                                             new_data_i_zero_mask_interpolation)

        result[i] = numpy.where(zero_mask_i, new_data_i_zero_mask_interpolation, new_data_i)

    remove_zeroed_lines.recorders.array_debug_recorder["zero_masks_dilated"] = zero_masks_dilated
    remove_zeroed_lines.recorders.array_debug_recorder["zero_masks_eroded"] = zero_masks_eroded
    remove_zeroed_lines.recorders.array_debug_recorder["zero_masks_outline"] = zero_masks_outline

    return(result)


@debugging_tools.log_call(logger)
@HDF5_recorder.static_array_debug_recorder
def extract_f0(new_data,
               step_size,
               half_window_size,
               which_quantile,
               temporal_smoothing_gaussian_filter_stdev,
               spatial_smoothing_gaussian_filter_stdev,
               bias = None,
               **parameters):
    """
        Attempts to find a new baseline for the given data.

        Args:
            new_data(numpy.ndarray):                    array of data for finding new baseline (first axis is time).
            temporal_smoothing_gaussian_filter_stdev:   stdev for gaussian filter to convolve over time.
            spatial_smoothing_gaussian_filter_stdev:    stdev for gaussian filter to convolve over all other dimensions.
            **parameters(dict):                         essentially unused (catches unneeded arguments).

        Returns:
            numpy.ndarray:                              a new array with a new baseline.
    """

    # Add the bias param
    if bias is None:
        new_data_biased = new_data - new_data.min() + 1
    else:
        new_data_biased = new_data + bias

    # TODO: Check to see if norm is acceptable as 1.0 or if it must be 0.0.
    temporal_smoothing_gaussian_filter = vigra.filters.gaussianKernel(temporal_smoothing_gaussian_filter_stdev,
                                                                      1.0,
                                                                      5)
    # TODO: Check what border treatment to use
    temporal_smoothing_gaussian_filter.setBorderTreatment(vigra.filters.BorderTreatmentMode.BORDER_TREATMENT_REFLECT)

    new_data_temporally_smoothed = vigra.filters.convolveOneDimension(new_data_biased.astype(numpy.float32),
                                                                      0,
                                                                      temporal_smoothing_gaussian_filter)

    if (step_size > new_data_temporally_smoothed.shape[0]):
        raise Exception("The step size provided, " + repr(step_size) + ", is larger than the number of frames in the data, " + repr(new_data_temporally_smoothed.shape[0]) + ".")

    which_quantile_len = None
    try:
        which_quantile_len = len(which_quantile)
    except TypeError:
        # Does not have len
        which_quantile_len = 1

    # Only allowed to have one quantile.
    if (which_quantile_len > 1):
        raise Exception("Provided more than one quantile \"" + repr(which_quantile) + "\".")

    window_centers = numpy.arange(0, new_data_temporally_smoothed.shape[0], step_size)

    if window_centers[-1] != new_data_temporally_smoothed.shape[0]:
        window_centers = numpy.append(window_centers, new_data_temporally_smoothed.shape[0])

    def window_shape_iterator(window_centers = window_centers):
        for each_window_center in window_centers:
            each_window_lower = max(window_centers[0], each_window_center - half_window_size)
            each_window_upper = min(window_centers[-1], each_window_center + half_window_size)

            yield( (each_window_lower, each_window_center, each_window_upper) )

    which_quantile = expanded_numpy.get_quantiles(which_quantile)

    window_quantiles = numpy.zeros( (window_centers.shape[0],) + which_quantile.shape + new_data_temporally_smoothed.shape[1:] )
    for i, (each_window_lower, each_window_center, each_window_upper) in enumerate(window_shape_iterator()):
        new_data_temporally_smoothed_i = new_data_temporally_smoothed[each_window_lower:each_window_upper]

        each_quantile = expanded_numpy.quantile(new_data_temporally_smoothed_i.reshape(new_data_temporally_smoothed_i.shape[0], -1), which_quantile, axis=0)
        each_quantile = each_quantile.reshape(each_quantile.shape[0], *new_data_temporally_smoothed_i.shape[1:])

        # Are there bad values in our result (shouldn't be if axis=None)
        if each_quantile.mask.any():
            msg = "Found erroneous regions in quantile calculation. Dropping in HDF5 logger."

            logger.error(msg)
            extract_f0.recorders.array_debug_recorder["each_quantile"] = each_quantile
            raise Exception(msg)
        else:
            each_quantile = each_quantile.data

        window_quantiles[i] = each_quantile

    # Should only be one quantile. Drop the singleton dimension.
    window_quantiles = window_quantiles[:, 0]

    quantile_interpolator = scipy.interpolate.interp1d(window_centers, window_quantiles, axis=0)
    new_data_quantiled = quantile_interpolator(numpy.arange(new_data_temporally_smoothed.shape[0]))

    # TODO: Check to see if norm is acceptable as 1.0 or if it must be 0.0.
    spatial_smoothing_gaussian_filter = vigra.filters.gaussianKernel(spatial_smoothing_gaussian_filter_stdev,
                                                                     1.0,
                                                                     5)

    # TODO: Check what border treatment to use
    spatial_smoothing_gaussian_filter.setBorderTreatment(vigra.filters.BorderTreatmentMode.BORDER_TREATMENT_REFLECT)

    new_data_spatially_smoothed = new_data_quantiled
    for d in xrange(1, new_data_quantiled.ndim):
        new_data_spatially_smoothed = vigra.filters.convolveOneDimension(new_data_spatially_smoothed.astype(numpy.float32),
                                                                         d,
                                                                         spatial_smoothing_gaussian_filter)

    extract_f0.recorders.array_debug_recorder["new_data_spatially_smoothed"] = new_data_spatially_smoothed

    new_data_baselined = (new_data_biased - new_data_spatially_smoothed) / new_data_spatially_smoothed

    return(new_data_baselined)


@debugging_tools.log_call(logger)
@HDF5_recorder.static_array_debug_recorder
def normalize_data(new_data, **parameters):
    """
        Removes the mean from each image and normalizes each image as if they were vectors.

        Args:
            new_data(numpy.ndarray):                array of data for normalization (first axis is time).
            **parameters(dict):                     contains arguments for simple_image_processing.renormalized_images.

        Returns:
            numpy.ndarray:                          data that has been normalized.

        Examples:
            >>> a = numpy.zeros((2,2,2,))
            >>> a[1,1,1] = 1
            >>> a[0,0,0] = 1
            >>> normalize_data(a, **{"simple_image_processing.renormalized_images" : { "ord" : 2 }})
            array([[[ 0.8660254 , -0.28867513],
                    [-0.28867513, -0.28867513]],
            <BLANKLINE>
                   [[-0.28867513, -0.28867513],
                    [-0.28867513,  0.8660254 ]]])
    """

    # Remove the mean of each row vector
    new_data_mean_zeroed = simple_image_processing.zeroed_mean_images(new_data)

    # Renormalize each row vector using some specified normalization
    new_data_renormalized = simple_image_processing.renormalized_images(new_data_mean_zeroed,
                                                                        **parameters["simple_image_processing.renormalized_images"])

    return(new_data_renormalized)


@debugging_tools.log_call(logger)
@HDF5_recorder.static_array_debug_recorder
def preprocess_data(new_data, **parameters):
    """
        Performs all preprocessing steps that are specified (remove_zeroed_lines, bias, extract_f0, and
        wavelet_transform).
        
        Args:
            new_data(numpy.ndarray):            array of data for generating a dictionary (first axis is time).
            bias(float):                        how much to shift the result by (note normalize_data will zero mean).
            **parameters(dict):                 additional parameters for each step of preprocessing.
        
        Returns:
            dict:                               the dictionary found.
    """

    # TODO: Add preprocessing step wavelet transform, F_0, remove lines, etc.

    # Remove lines
    new_data_maybe_lines_removed = None
    if "remove_zeroed_lines" in parameters:
        remove_zeroed_lines.recorders.array_debug_recorder = preprocess_data.recorders.array_debug_recorder
        new_data_maybe_lines_removed = remove_zeroed_lines(new_data,
                                                           **parameters["remove_zeroed_lines"])
        preprocess_data.recorders.array_debug_recorder["images_lines_removed"] = new_data_maybe_lines_removed
        preprocess_data.recorders.array_debug_recorder["images_lines_removed_max"] = new_data_maybe_lines_removed.max(axis = 0)
    else:
        new_data_maybe_lines_removed = new_data

    new_data_maybe_f0_result = None
    if "extract_f0" in parameters:
        extract_f0.recorders.array_debug_recorder = preprocess_data.recorders.array_debug_recorder
        new_data_maybe_f0_result = extract_f0(new_data_maybe_lines_removed,
                                              **parameters["extract_f0"])
        preprocess_data.recorders.array_debug_recorder["images_f0"] = new_data_maybe_f0_result
        preprocess_data.recorders.array_debug_recorder["images_f0_max"] = new_data_maybe_f0_result.max(axis = 0)
    else:
        new_data_maybe_f0_result = new_data_maybe_lines_removed

    new_data_maybe_wavelet_result = None
    if "wavelet_transform" in parameters:
        wavelet_transform.wavelet_transform.recorders.array_debug_recorder = preprocess_data.recorders.array_debug_recorder
        new_data_maybe_wavelet_result = wavelet_transform.wavelet_transform(new_data_maybe_f0_result,
                                                                            **parameters["wavelet_transform"])[-1]
        preprocess_data.recorders.array_debug_recorder["images_wavelet_transformed"] = new_data_maybe_wavelet_result
        preprocess_data.recorders.array_debug_recorder["images_wavelet_transformed_max"] = new_data_maybe_wavelet_result.max(axis = 0)
    else:
        new_data_maybe_wavelet_result = new_data_maybe_f0_result

    normalize_data.recorders.array_debug_recorder = preprocess_data.recorders.array_debug_recorder
    new_data_normalized = normalize_data(new_data_maybe_wavelet_result,
                                         **parameters["normalize_data"])
    preprocess_data.recorders.array_debug_recorder["images_normalized"] = new_data_normalized
    preprocess_data.recorders.array_debug_recorder["images_normalized_max"] = new_data_normalized.max(axis = 0)

    return(new_data_normalized)


@debugging_tools.log_call(logger)
@HDF5_recorder.static_array_debug_recorder
def generate_dictionary(new_data, **parameters):
    """
        Generates a dictionary using the data and parameters given for trainDL.
        
        Args:
            new_data(numpy.ndarray):            array of data for generating a dictionary (first axis is time).
            **parameters(dict):                 passed directly to spams.trainDL.
        
        Returns:
            dict:                               the dictionary found.
    """

    import spams_sandbox

    # It takes a loooong time to load spams.
    # So, we shouldn't do this until we are sure that we are ready to generate the dictionary
    # (i.e. the user supplied a bad config file, /images does not exist, etc.).
    # Note it caches the import so subsequent calls should not make it any slower.
    import spams

    # Maybe should copy data so as not to change the original.
    # new_data_processed = new_data.copy()
    new_data_processed = new_data

    # Reshape data into a matrix (each image is now a column vector)
    new_data_processed = numpy.reshape(new_data_processed, (new_data_processed.shape[0], -1))
    new_data_processed = numpy.asmatrix(new_data_processed)
    new_data_processed = new_data_processed.transpose()

    # Spams requires all matrices to be fortran.
    new_data_processed = numpy.asfortranarray(new_data_processed)

    # Simply trains the dictionary. Does not return sparse code.
    # Need to look into generating the sparse code given the dictionary, spams.nmf? (may be too slow))
    new_dictionary = spams_sandbox.spams_sandbox.call_multiprocessing_array_spams_trainDL(new_data_processed,
                                                                                          **parameters["spams.trainDL"])

    # Fix dictionary so that the first index will be the particular image.
    # The rest will be the shape of an image (same as input shape).
    new_dictionary = new_dictionary.transpose()
    new_dictionary = numpy.asarray(new_dictionary).reshape((parameters["spams.trainDL"]["K"],) + new_data.shape[1:])

    return(new_dictionary)


@debugging_tools.log_call(logger)
def region_properties_scikit_image(new_label_image, *args, **kwargs):
    """
        Grabs region properties from a label image.
        
        Args:
            new_label_image(numpy.ndarray):      label image used for generating properties.
            args(list):                          additional position arguments to pass skimage.measure.regionprops.
            **parameters(dict):                  additional keyword arguments to pass skimage.measure.regionprops.

        Note:
            Uses all the same options in skimage.measure.regionprops. If a property is not specified, then it won't be
            returned.
        
        Returns:
            numpy.ndarray:                       a structured array of all the properties found for each label.
        
        
        Examples:
            
            >>> region_properties_scikit_image(numpy.zeros((2,2), dtype=int))
            array([], 
                  dtype=[('label', '<i8'), ('area', '<f8'), ('centroid', '<f8', (2,))])
            
            >>> region_properties_scikit_image(numpy.ones((2,2), dtype=int))
            array([(1, 4.0, [0.5, 0.5])], 
                  dtype=[('label', '<i8'), ('area', '<f8'), ('centroid', '<f8', (2,))])
            
            >>> region_properties_scikit_image(numpy.ones((3,3), dtype=int))
            array([(1, 9.0, [1.0, 1.0])], 
                  dtype=[('label', '<i8'), ('area', '<f8'), ('centroid', '<f8', (2,))])

            >>> region_properties_scikit_image(numpy.eye(3, dtype=int))
            array([(1, 3.0, [1.0, 1.0])], 
                  dtype=[('label', '<i8'), ('area', '<f8'), ('centroid', '<f8', (2,))])
    """

    region_properties_type_dict = {
        "area": numpy.float64,
        "bbox": numpy.int64,
        "centroid": numpy.float64,
        "convex_area": numpy.int64,
        "convex_image": numpy.bool_,
        "coords": numpy.int64,
        "eccentricity": numpy.float64,
        "equivalent_diameter": numpy.float64,
        "euler_number": numpy.int64,
        "filled_area": numpy.int64,
        "filled_image": numpy.bool_,
        "image": numpy.bool_,
        "inertia_tensor": numpy.float64,
        "inertia_tensor_eigvals": numpy.float64,
        "intensity_image": numpy.float64,
        "label": numpy.int64,
        "local_centroid": numpy.float64,
        "major_axis_length": numpy.float64,
        "max_intensity": numpy.float64,
        "mean_intensity": numpy.float64,
        "min_intensity": numpy.float64,
        "minor_axis_length": numpy.float64,
        "moments": numpy.float64,
        "moments_central": numpy.float64,
        "moments_hu": numpy.float64,
        "moments_normalized": numpy.float64,
        "orientation": numpy.float64,
        "perimeter": numpy.float64,
        "solidity": numpy.float64,
        "weighted_centroid": numpy.float64,
        "weighted_local_centroid": numpy.float64,
        "weighted_moments": numpy.float64,
        "weighted_moments_central": numpy.float64,
        "weighted_moments_hu": numpy.float64,
        "weighted_moments_normalized": numpy.float64
    }

    region_properties_shape_dict = {
        "area": (),
        "bbox": (4,),
        "centroid": (new_label_image.ndim,),
        "convex_area": (),
        "convex_image": (-1, -1),
        "coords": (-1, 2),
        "eccentricity": (),
        "equivalent_diameter": (),
        "euler_number": (),
        "filled_area": (),
        "filled_image": (-1, -1),
        "image": (-1, -1),
        "inertia_tensor": (2, 2),
        "inertia_tensor_eigvals": (2,),
        "intensity_image": (-1, -1),
        "label": (),
        "local_centroid": (2,),
        "major_axis_length": (),
        "max_intensity": (),
        "mean_intensity": (),
        "min_intensity": (),
        "minor_axis_length": (),
        "moments": (4, 4),
        "moments_central": (4, 4),
        "moments_hu": (7,),
        "moments_normalized": (4, 4),
        "orientation": (),
        "perimeter": (),
        "solidity": (),
        "weighted_centroid": (2,),
        "weighted_local_centroid": (2,),
        "weighted_moments": (4, 4),
        "weighted_moments_central": (4, 4),
        "weighted_moments_hu": (7,),
        "weighted_moments_normalized": (4, 4)
    }

    region_properties_ndim_dict = {
        "area": 0,
        "bbox": 1,
        "centroid": 1,
        "convex_area": 0,
        "convex_image": 2,
        "coords": 2,
        "eccentricity": 0,
        "equivalent_diameter": 0,
        "euler_number": 0,
        "filled_area": 0,
        "filled_image": 2,
        "image": 2,
        "inertia_tensor": 2,
        "inertia_tensor_eigvals": 1,
        "intensity_image": 2,
        "label": 0,
        "local_centroid": 1,
        "major_axis_length": 0,
        "max_intensity": 0,
        "mean_intensity": 0,
        "min_intensity": 0,
        "minor_axis_length": 0,
        "moments": 2,
        "moments_central": 2,
        "moments_hu": 1,
        "moments_normalized": 2,
        "orientation": 0,
        "perimeter": 0,
        "solidity": 0,
        "weighted_centroid": 1,
        "weighted_local_centroid": 1,
        "weighted_moments": 2,
        "weighted_moments_central": 2,
        "weighted_moments_hu": 1,
        "weighted_moments_normalized": 2
    }

    array_properties = [_k for _k, _v in region_properties_ndim_dict.items() if _v > 0]
    fixed_shape_properties = [_k for _k, _v in region_properties_shape_dict.items() if -1 not in _v]
    varied_shape_properties = [_k for _k, _v in region_properties_shape_dict.items() if -1 in _v]

    new_label_image_props = None
    new_label_image_props_with_arrays = None
    new_label_image_props_with_arrays_values = None
    new_label_image_props_with_arrays_dtype = None

    properties = None
    if (len(args)) and (args[0]):
        properties = args[0]
        args = args[1:]
    elif (len(kwargs)) and ("properties" in kwargs):
        properties = kwargs["properties"]
        del kwargs["properties"]
    else:
        properties = ["area", "centroid"]

    if ( (properties == "all") or (properties == None) ):
        properties = region_properties_type_dict.keys()

    intensity_image = None
    if (len(args)) and (args[0]):
        intensity_image = args[0]
        args = args[1:]
    elif (len(kwargs)) and ("intensity_image" in kwargs):
        intensity_image = kwargs["intensity_image"]
        del kwargs["intensity_image"]
    else:
        pass

    # Remove duplicates and make sure label is at the front.
    properties = set(properties)
    allowed_properties = set(region_properties_type_dict.keys())
    disallowed_properties = properties.difference(allowed_properties)

    if len(disallowed_properties):
        disallowed_properties = sorted(disallowed_properties)
        raise Exception("Recieved \"" + repr(
            len(disallowed_properties)) + "\" properties that are not allowed, which are \"" + repr(
            disallowed_properties) + "\".")

    properties.discard("label")
    properties = ["label"] + sorted(properties)

    if new_label_image.size:
        # This gives a list of dictionaries. However, this is not very usable.
        # So, we will convert this to a structured NumPy array.
        # In future versions, the properties argument will be removed.
        # It does not need to be passed to retain functionality of this function.
        # new_label_image_props = skimage.measure.regionprops(new_label_image, intensity_image, *args, **kwargs)
        new_label_image_props = skimage.measure.regionprops(label_image = new_label_image,
                                                            properties = properties,
                                                            intensity_image = intensity_image,
                                                            *args, **kwargs)

        new_label_image_props_with_arrays = []
        for i in xrange(len(new_label_image_props)):
            new_label_image_props_with_arrays.append({})

            for each_key in properties:
                if each_key in array_properties:
                    new_label_image_props_with_arrays[i][each_key] = numpy.array(new_label_image_props[i][each_key])
                else:
                    new_label_image_props_with_arrays[i][each_key] = new_label_image_props[i][each_key]

        # Holds the values from props.
        new_label_image_props_with_arrays_values = []

        # Holds the types from props.
        new_label_image_props_with_arrays_dtype = []

        if len(new_label_image_props_with_arrays):
            # Get types for all properties as a dictionary
            for each_name in properties:
                each_sample_value = new_label_image_props_with_arrays[0][each_name]
                each_type = type(each_sample_value)
                each_shape = tuple()

                if isinstance(each_sample_value, numpy.ndarray) and (each_name in fixed_shape_properties):
                    each_type = each_sample_value.dtype
                    each_shape = each_sample_value.shape
                else:
                    each_type = numpy.dtype(each_type)

                new_label_image_props_with_arrays_dtype.append((each_name, each_type, each_shape))

            # Store the values to place in NumPy structured array in order.
            new_label_image_props_with_arrays_values = []
            for j in xrange(len(new_label_image_props_with_arrays)):
                # Add all values in order of keys from the dictionary.
                new_label_image_props_with_arrays_values.append([])
                for each_new_label_image_props_with_arrays_dtype in new_label_image_props_with_arrays_dtype:

                    each_name, each_type, each_shape = each_new_label_image_props_with_arrays_dtype

                    if each_shape:
                        new_label_image_props_with_arrays_values[j].append(
                            new_label_image_props_with_arrays[j][each_name].tolist())
                    else:
                        new_label_image_props_with_arrays_values[j].append(
                            new_label_image_props_with_arrays[j][each_name])

                # NumPy will expect a tuple for each set of values.
                new_label_image_props_with_arrays_values[j] = tuple(new_label_image_props_with_arrays_values[j])

    if (not new_label_image.size) or (not len(new_label_image_props_with_arrays)):
        new_label_image_props_with_arrays_dtype = []
        for each_key in properties:
            each_type = region_properties_type_dict[each_key]
            each_shape = region_properties_shape_dict[each_key]

            if each_key in varied_shape_properties:
                each_type = numpy.object_
                each_shape = tuple()

            new_label_image_props_with_arrays_dtype.append((each_key, each_type, each_shape))

    new_label_image_props_with_arrays_dtype = numpy.dtype(new_label_image_props_with_arrays_dtype)

    # Replace the properties with the structured array.
    new_label_image_props_with_arrays = numpy.array(new_label_image_props_with_arrays_values,
                                                    dtype = new_label_image_props_with_arrays_dtype)

    return(new_label_image_props_with_arrays)

@debugging_tools.log_call(logger)
def region_properties(new_label_image, *args, **kwargs):
    """
        Grabs region properties from a label image.

        Args:
            new_label_image(numpy.ndarray):      label image used for generating properties.
            args(list):                          additional position arguments to pass skimage.measure.regionprops.
            **parameters(dict):                  additional keyword arguments to pass skimage.measure.regionprops.

        Note:
            Uses all the same options in skimage.measure.regionprops. If a property is not specified, then it won't be
            returned.

        Returns:
            numpy.ndarray:                       a structured array of all the properties found for each label.


        Examples:

            >>> region_properties(numpy.zeros((2,2), dtype=int)) # doctest: +NORMALIZE_WHITESPACE
            array([], 
                  dtype=[('label', '<i8'), ('area', '<f8'), ('centroid', '<f8', (2,))])

            >>> region_properties(numpy.ones((2,2), dtype=int)) # doctest: +NORMALIZE_WHITESPACE
            array([(1, 4.0, [0.5, 0.5])], 
                  dtype=[('label', '<i8'), ('area', '<f8'), ('centroid', '<f8', (2,))])

            >>> region_properties(numpy.ones((3,3), dtype=int)) # doctest: +NORMALIZE_WHITESPACE
            array([(1, 9.0, [1.0, 1.0])], 
                  dtype=[('label', '<i8'), ('area', '<f8'), ('centroid', '<f8', (2,))])

            >>> region_properties(numpy.eye(3, dtype=int))
            array([(1, 3.0, [1.0, 1.0])], 
                  dtype=[('label', '<i8'), ('area', '<f8'), ('centroid', '<f8', (2,))])
    """

    return(region_properties_scikit_image(new_label_image, *args, **kwargs))

@debugging_tools.log_call(logger)
def get_neuron_dtype(shape, dtype):
    """
        Gets the type based on properties of an image.

        Args:
            new_image(numpy.ndarray):     a single sample image (only spatial dimension).

        Returns:
            list:                         a list that can be converted to a numpy.dtype using numpy.ndtype's constructor.

        Examples:
            >>> get_neuron_dtype((3,), numpy.float64) #doctest: +NORMALIZE_WHITESPACE
            [('mask', <type 'numpy.bool_'>, (3,)),
             ('contour', <type 'numpy.bool_'>, (3,)),
             ('image', <type 'numpy.float64'>, (3,)),
             ('area', <type 'numpy.float64'>),
             ('max_F', <type 'numpy.float64'>),
             ('gaussian_mean', <type 'numpy.float64'>, (1,)),
             ('gaussian_cov', <type 'numpy.float64'>, (1, 1)),
             ('centroid', <type 'numpy.float64'>, (1,))]

            >>> get_neuron_dtype((2, 3), numpy.float64) #doctest: +NORMALIZE_WHITESPACE
            [('mask', <type 'numpy.bool_'>, (2, 3)),
             ('contour', <type 'numpy.bool_'>, (2, 3)),
             ('image', <type 'numpy.float64'>, (2, 3)),
             ('area', <type 'numpy.float64'>),
             ('max_F', <type 'numpy.float64'>),
             ('gaussian_mean', <type 'numpy.float64'>, (2,)),
             ('gaussian_cov', <type 'numpy.float64'>, (2, 2)),
             ('centroid', <type 'numpy.float64'>, (2,))]

    """

    ndim = len(shape)

    neurons_dtype = [("mask", numpy.bool8, shape),
                     ("contour", numpy.bool8, shape),
                     ("image", numpy.dtype(dtype).type, shape),
                     ("area", numpy.float64),
                     ("max_F", numpy.float64),
                     ("gaussian_mean", numpy.float64, (ndim,)),
                     ("gaussian_cov", numpy.float64, (ndim, ndim,)),
                     ("centroid", numpy.dtype(dtype).type, (ndim,))]

    return(neurons_dtype)


@debugging_tools.log_call(logger)
def get_empty_neuron(shape, dtype):
    """
        Gets a numpy structured array using the type from get_neuron_dtype that has no contents.

        Args:
            new_image(numpy.ndarray):     an empty numpy structured array with all types needed for a neuron.

        Returns:
            numpy.ndarray:                a numpy structured array with no contents and type from get_neuron_dtype.

        Examples:
            >>> get_empty_neuron((3,), numpy.float64) #doctest: +NORMALIZE_WHITESPACE
            array([], dtype=[('mask', '?', (3,)),
                             ('contour', '?', (3,)),
                             ('image', '<f8', (3,)),
                             ('area', '<f8'),
                             ('max_F', '<f8'),
                             ('gaussian_mean', '<f8', (1,)),
                             ('gaussian_cov', '<f8', (1, 1)),
                             ('centroid', '<f8', (1,))])

            >>> get_empty_neuron((2, 3), numpy.float64) #doctest: +NORMALIZE_WHITESPACE
            array([], dtype=[('mask', '?', (2, 3)),
                             ('contour', '?', (2, 3)),
                             ('image', '<f8', (2, 3)),
                             ('area', '<f8'),
                             ('max_F', '<f8'),
                             ('gaussian_mean', '<f8', (2,)),
                             ('gaussian_cov', '<f8', (2, 2)),
                             ('centroid', '<f8', (2,))])

    """

    neurons_dtype = get_neuron_dtype(shape=shape, dtype=dtype)
    neurons = numpy.zeros((0,), dtype = neurons_dtype)

    return(neurons)


@debugging_tools.log_call(logger)
def get_one_neuron(shape, dtype):
    """
        Gets a numpy structured array using the type from get_neuron_dtype that has one neuron with all zeros.

        Args:
            new_image(numpy.ndarray):     an empty numpy structured array with all types needed for a neuron.

        Returns:
            numpy.ndarray:                a numpy structured array with one neuron using type from get_neuron_dtype.

        Examples:
            >>> get_one_neuron((3,), numpy.float) #doctest: +NORMALIZE_WHITESPACE
            array([ ([False, False, False],
                     [False, False, False],
                     [0.0, 0.0, 0.0],
                     0.0,
                     0.0,
                     [0.0],
                     [[0.0]],
                     [0.0])],
                   dtype=[('mask', '?', (3,)),
                          ('contour', '?', (3,)),
                          ('image', '<f8', (3,)),
                          ('area', '<f8'),
                          ('max_F', '<f8'),
                          ('gaussian_mean', '<f8', (1,)),
                          ('gaussian_cov', '<f8', (1, 1)),
                          ('centroid', '<f8', (1,))])


            >>> get_one_neuron((2, 3), numpy.float) #doctest: +NORMALIZE_WHITESPACE
            array([ ([[False, False, False],
                      [False, False, False]],
                     [[False, False, False],
                      [False, False, False]],
                     [[0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0]],
                     0.0,
                     0.0,
                     [0.0, 0.0],
                    [[0.0, 0.0],
                     [0.0, 0.0]],
                    [0.0, 0.0])],
                   dtype=[('mask', '?', (2, 3)),
                          ('contour', '?', (2, 3)),
                          ('image', '<f8', (2, 3)),
                          ('area', '<f8'),
                          ('max_F', '<f8'),
                          ('gaussian_mean', '<f8', (2,)),
                          ('gaussian_cov', '<f8', (2, 2)),
                          ('centroid', '<f8', (2,))])
    """

    neurons_dtype = get_neuron_dtype(shape=shape, dtype=dtype)
    neurons = numpy.zeros((1,), dtype = neurons_dtype)

    return(neurons)


@debugging_tools.log_call(logger)
def generate_local_maxima_vigra(new_intensity_image):
    """
        Creates a mask the same size as the intensity image with local maxima as True and background False.
        Uses vigra's vigra.analysis.extendedLocalMaxima.
        
        Args:
            new_intensity_image(numpy.ndarray):     The image to find local maxima for (ideally, all axes are
                                                    spatial; however, it will treat all axes the same).

        Returns:
            numpy.ndarray:                          A mask of the local maxima.
    """

    local_maxima_mask = vigra.analysis.extendedLocalMaxima(new_intensity_image.astype(numpy.float32)).astype(bool)

    return(local_maxima_mask)


@debugging_tools.log_call(logger)
def generate_local_maxima_scikit_image(new_intensity_image, local_max_neighborhood_size = 1):
    """
        Creates a mask the same size as the intensity image with local maxima as True and background False.
        Uses scikit image's skimage.feature.peak_local_max.
        
        Args:
            new_intensity_image(numpy.ndarray):     The image to find local maxima for (ideally, all axes are
                                                    spatial; however, it will treat all axes the same).

            local_max_neighborhood_size(int):       Size of the neighborhood to check for a local maxima.

        Returns:
            numpy.ndarray:                          A mask of the local maxima.
    """

    local_maxima_neighborhood = numpy.ones((2 * local_max_neighborhood_size + 1,) * new_intensity_image.ndim)
    local_maxima_mask = skimage.feature.peak_local_max(new_intensity_image, footprint = local_maxima_neighborhood,
                                                       labels = (new_intensity_image > 0).astype(int), indices = False)

    return(local_maxima_mask)


@debugging_tools.log_call(logger)
def generate_local_maxima(new_intensity_image):
    """
        Creates a mask the same size as the intensity image with local maxima as True and background False.
        
        Args:
            new_intensity_image(numpy.ndarray):     The image to find local maxima for (ideally, all axes are
                                                    spatial; however, it will treat all axes the same).

        Returns:
            numpy.ndarray:                          A mask of the local maxima.
    """

    return(generate_local_maxima_vigra(new_intensity_image))


@debugging_tools.log_call(logger)
def extended_region_local_maxima_properties(new_intensity_image, new_label_image, **kwargs):
    """
        Generates local maxima along with other properties for each labeled region
        (therefore at least one entry per label).
        Gets a label image if not provided by using the threshhold (if not provided is zero).

        Note:
            This is similar to region_properties. It differs in that it keeps track of all local maxima. So, it is no
            longer one-to-one in labels, but is one-to-one in local maxima.

        Args:
            new_intensity_image(numpy.ndarray):     The intensity image to find local maxima for (ideally, all axes are
                                                    spatial; however, it will treat all axes the same).

            new_label_image(numpy.ndarray):         The label image to find local maxima for (ideally, all axes are
                                                    spatial; however, it will treat all axes the same).

        Returns:
            numpy.ndarray:                          A mask of the local maxima.
    """

    # Remove the background
    new_image_mask = (new_label_image != 0)
    new_intensity_image_masked = ( ~new_image_mask * new_intensity_image.min() ) + (
        new_image_mask * new_intensity_image )

    # Get a mask of the local maxima (that only includes local maxima not in the background)
    local_maxima_mask = generate_local_maxima(new_intensity_image_masked)

    # Count the local maxima and give them different labels
    local_maxima_labeled = scipy.ndimage.label(local_maxima_mask.astype(int))[0]

    # Generate the properties of the labeled regions
    labeled_props = region_properties(new_label_image, **kwargs)

    # Generate the properties of the local maxima
    local_maxima_props = region_properties(local_maxima_labeled, properties = ["label", "centroid"])

    # We want to have a few more type present in our NumPy structured array. To do this, we collect the existing types
    # into a list and then add our new types onto the end. Finally, we make the new structured array type from the list
    # we have.
    props_dtype = []

    # Place on the label props dtype first
    labeled_props_dtype = expanded_numpy.numpy_array_dtype_list(labeled_props)
    props_dtype.extend(labeled_props_dtype)

    # Then add new fields.
    props_dtype.append(("local_max", int, new_intensity_image.ndim))
    props_dtype.append(("intensity", new_intensity_image.dtype))

    # Makes a new properties array that contains enough entries to hold the old one and has all the types we desire.
    new_local_maxima_props = numpy.zeros(local_maxima_props.shape, dtype = numpy.dtype(props_dtype))

    # Take the centroids and old labels as the label and local max
    new_local_maxima_props["label"] = local_maxima_props["label"]
    new_local_maxima_props["local_max"] = local_maxima_props["centroid"].round().astype(int)

    # Replace the old structured array with the enlarged version.
    local_maxima_props = new_local_maxima_props

    # Stores the intensity at the local max for easy retrieval
    # Replace the labels by using the values from the label image
    if local_maxima_props.size:
        local_maxima_props["intensity"] = new_intensity_image[tuple(local_maxima_props["local_max"].T)]
        local_maxima_props["label"] = new_label_image[tuple(local_maxima_props["local_max"].T)]

    # Now, we want to merge the other properties in with our local maxima
    # But, we will skip it if there are no local maxima.
    if local_maxima_props.size:
        for each_i, each_label in enumerate(labeled_props["label"]):
            each_label_props_mask = (local_maxima_props["label"] == each_label)

            for each_new_prop_name, _, __ in labeled_props_dtype:
                local_maxima_props[each_new_prop_name][each_label_props_mask] = labeled_props[each_new_prop_name][each_i]

    return(local_maxima_props)


@debugging_tools.log_class(logger)
@HDF5_recorder.class_static_array_debug_recorder
class ExtendedRegionProps(object):
    """
        Using the results of extended_region_local_maxima_properties with given intensity and label images, the
        ExtendedRegionProps attempts to make it easy to get information about local maxima in the image. Further, it
        provides methods to allow removal of local maxima. When removing local maxima, it will updated the given label
        image to reflect labels that have been removed completely and will automatically renumber the label image (as
        some functions expect a label image with sequential numbering, no missing label number). This provides a simple
        way to maintain a label image, whose local maxima one is more interested in directly manipulating.

        Note:
            Adding local maxima is not allowed for and wouldn't make sense given its dependency on the intensity image.
    """

    def __init__(self, new_intensity_image,
                 new_label_image,
                 properties = ["centroid"]):
        """
            Construct an ExtendedRegionProps instance.

            This uses extended_region_local_maxima_properties to determine local maxima and properties for the labeled
            region. After doing this, it is possible to remove local maxima and update the label image to reflect this
            change.

            Args:
                new_intensity_image(numpy.ndarray):     The intensity image to find local maxima for (ideally, all axes are
                                                        spatial; however, it will treat all axes the same).

                new_label_image(numpy.ndarray):         The label image to find local maxima for (ideally, all axes are
                                                        spatial; however, it will treat all axes the same).

                properties(list):                       used by extended_region_local_maxima_properties to generate
                                                        initial properties.
        """

        # Copied to ensure purity. Would not want to change outside values.
        self.intensity_image = new_intensity_image.copy()
        self.label_image = new_label_image.copy()

        self.image_mask = (self.label_image > 0)
        self.props = None
        self.count = None

        logger.debug("Finding the local maxima and properties...")

        self.props = extended_region_local_maxima_properties(self.intensity_image,
                                                             self.label_image,
                                                             properties = properties)

        logger.debug("Found the local maxima and properties.")

        # Remove maxima in the background
        background_maxima_mask = (self.props["label"] == 0)
        if (numpy.any(background_maxima_mask)):
            # There shouldn't be any maximums in the background. This should never happen.
            logger.warning(
                "Found \"" + (background_maxima_mask).sum() + "\" maximum(s) found in the background (label 0).")
            # Remove the 0 labels
            self.props = self.props[background_maxima_mask]

        del background_maxima_mask

        # Stores the number of times a particular label maxima appears.
        self.count = numpy.zeros((self.label_image.max(),), dtype = [("label", int), ("count", int)])
        # Get all the labels used in the label image
        self.count["label"] = numpy.arange(1, self.label_image.max() + 1)
        # Get the count of those labels (excluding zero as it is background)
        self.count["count"] = numpy.bincount(self.props["label"], minlength = len(self.count) + 1)[1:]

        if self.count.size and numpy.any(self.count["count"] == 0):
            # All labels should have a local maximum. If they don't, this could be a problem.

            failed_labels = self.count["label"][self.count["count"] == 0]
            failed_labels_list = failed_labels.tolist()
            failed_label_msg = "Label(s) not found in local maxima. For labels = " + repr(failed_labels_list) + "."

            logger.warning(failed_label_msg)

            ExtendedRegionProps.recorders.array_debug_recorder["intensity_image"] = self.intensity_image
            ExtendedRegionProps.recorders.array_debug_recorder["label_image"] = self.label_image

            if self.props.size:
                ExtendedRegionProps.recorders.array_debug_recorder["props"] = self.props

            ExtendedRegionProps.recorders.array_debug_recorder["count"] = self.count
            ExtendedRegionProps.recorders.array_debug_recorder["masks"] = \
                expanded_numpy.all_permutations_equal(failed_labels, self.label_image)
            ExtendedRegionProps.recorders.array_debug_recorder["masks_labels"] = failed_labels

            # Renumber labels. This way there are no labels without local maxima.
            self.renumber_labels()

        logger.debug("Refinined properties for local maxima.")


    def get_local_max_index_array(self):
        """
            Gets a numpy style index array from the local maxima present.

            Returns:
                tuple:      a numpy styled index array, which can be used on the intensity image, or label image.
        """

        return(tuple(self.props["local_max"].T))


    def get_local_max_mask(self):
        """
            Gets a bool numpy array that is True at the locations of the local maxima.

            Returns:
                numpy.ndarray:      a bool numpy array that is True where a local max is and False everywhere else.
        """

        # Returns a label image containing each local max and its label.
        new_local_max_mask = numpy.zeros(self.label_image.shape, dtype = self.label_image.dtype)

        # Set the given local maxima to be the same as their labels
        new_local_max_mask[self.get_local_max_index_array()] = True

        return(new_local_max_mask)


    def get_local_max_label_image(self):
        """
            Gets an int numpy array that contains the labels of the local maxima.

            Returns:
                numpy.ndarray:      an int numpy array that contains the labels of the local maxima and is zero in the
                                    background.
        """

        # Returns a label image containing each local max and its labels.
        new_local_max_label_image = numpy.zeros(self.label_image.shape, dtype = self.label_image.dtype)

        # Set the given local maxima to be the same as their labels
        new_local_max_label_image[self.get_local_max_index_array()] = self.props["label"]

        return(new_local_max_label_image)


    def remove_prop_mask(self, remove_prop_indices_mask):
        """
            Removes the local maxima by the mask over their indices (only False elements will be kept).

            Note:
                If a label is removed completely, the label image, intensity image, and image mask will be updated.
                Also, the count (of local maxima per label) will be updated, regardless.

            Args:
                remove_prop_indices_mask(numpy.ndarray):     A mask of the indices to drop (True for the ones to remove,
                                                             and False for the ones to keep).
        """

        # Get the labels to remove
        remove_labels = self.props["label"][remove_prop_indices_mask]
        # Get how many of each label to remove
        label_count_to_remove = numpy.bincount(remove_labels, minlength = len(self.count) + 1)[1:]

        # Take a subset of the label props that does not include the removal mask
        # (copying may not be necessary as the mask may be as effective)
        self.props = self.props[~remove_prop_indices_mask].copy()
        # Reduce the count by the number of each label
        self.count["count"] -= label_count_to_remove

        # Mask over self.count to find labels that do not have local maxima
        inactive_label_count_mask = (self.count["count"] == 0)

        # Are there labels that do not exist now? If so, we will dump them.
        if inactive_label_count_mask.any():
            # Find the labels to remove from the label image and mask and remove them
            labels_to_remove = self.count["label"][inactive_label_count_mask]
            labels_to_remove_mask = expanded_numpy.contains(self.label_image, labels_to_remove)
            self.label_image[labels_to_remove_mask] = 0
            self.image_mask[labels_to_remove_mask] = 0
            self.intensity_image[labels_to_remove_mask] = 0

            # Renumber all labels sequentially starting with the label image
            self.renumber_labels()


    def remove_prop_indices(self, *i):
        """
            Removes the local maxima of the indices listed (like remove_prop_mask except with indices).

            Note:
                If a label is removed completely, the label image, intensity image, and image mask will be updated.
                Also, the count (of local maxima per label) will be updated, regardless.

            Args:
                *i(int):     A mask of the indices to drop (True for the ones to remove,
                                                             and False for the ones to keep).
        """

        # A mask of the indices to remove
        remove_prop_indices_mask = numpy.zeros((len(self.props),), dtype = bool)
        remove_prop_indices_mask[numpy.array(i)] = True

        self.remove_prop_mask(remove_prop_indices_mask)

    def renumber_labels(self):
        """
            An internal method to be called after a label is completely removed from the local maxima.

            Ensures all labels are in sequential order in the label image, count, and that all local maxima have their
            labels updated.


            Note:
                Calling this externally will not make a difference as long as removal has occurred through the usual
                mechanisms.
        """

        # Renumber all labels sequentially starting with the label image
        self.label_image[:], forward_label_mapping, reverse_label_mapping = skimage.segmentation.relabel_sequential(
            self.label_image
        )
        # new_label_image, forward_label_mapping, reverse_label_mapping = advanced_numpy.renumber_label_image(
        #   self.label_image
        # )

        # Remove zero from the mappings as it is background and remains the same
        forward_label_mapping = forward_label_mapping[forward_label_mapping != 0]
        reverse_label_mapping = reverse_label_mapping[reverse_label_mapping != 0]

        # Find which of the old labels appear in self.props["label"] (skip 0)
        props_reverse_mapped = expanded_numpy.all_permutations_equal(reverse_label_mapping, self.props["label"])
        # Get the new labels by noting they must range from 0 to the length of reverse_label_mapping.
        # Skip zero as it is not necessary to check for it.
        new_labels = numpy.arange(1, len(reverse_label_mapping) + 1)
        # Expand new_labels into a view of the same size as props_reverse_mapped
        new_labels_expanded = expanded_numpy.expand_view(new_labels, reps_after = self.props["label"].shape)
        # Replace the labels with the matches and combine them
        self.props["label"] = (props_reverse_mapped * new_labels_expanded).sum(axis = 0)

        # Get a mask over the labels to find what is contained
        new_count_mask = expanded_numpy.contains(self.count["label"], reverse_label_mapping)
        # Move the values of the count into the proper lower labels and zero everything else
        self.count["count"][:len(reverse_label_mapping)] = self.count["count"][new_count_mask]
        self.count["count"][len(reverse_label_mapping):] = 0

        # (copying may not be necessary as the slice may be as effective)
        self.count = self.count[:len(reverse_label_mapping)].copy()


@debugging_tools.log_call(logger)
def remove_low_intensity_local_maxima(local_maxima, percentage_pixels_below_max, **parameters):
    """
        Removes the local maxima for regions, which have too many pixels that are below the intensity of their local
        maxima.

        Args:
            local_maxima(ExtendedRegionProps):      The local maxima to refine

            percentage_pixels_below_max(float):     The percentage of pixels out of a region that are below the
                                                    local maxima of the region

            **parameters(dict):                     essentially unused (catches unneeded arguments).

        Returns:
            ExtendedRegionProps:                    a new instance with the local maxima removed
    """

    # Deleting local maxima that does not exceed the 90th percentile of the pixel intensities
    low_intensities__local_maxima_label_mask__to_remove = numpy.zeros(local_maxima.props.shape, dtype = bool)
    for i in xrange(len(local_maxima.props)):
        # Get the region with the label matching the maximum
        each_region_image_wavelet_mask = (local_maxima.label_image == local_maxima.props["label"][i])
        each_region_image_wavelet = local_maxima.intensity_image[each_region_image_wavelet_mask]

        # Get the number of pixels in that region
        each_region_image_wavelet_num_pixels = float(each_region_image_wavelet.size)

        # Get the value of the max for that region
        each_region_image_wavelet_local_max_value = local_maxima.props["intensity"][i]

        # Get a mask of the pixels below that max for that region
        each_region_image_wavelet_num_pixels_below_max = float(
            (each_region_image_wavelet < each_region_image_wavelet_local_max_value).sum())

        # Get a ratio of the number of pixels below that max for that region
        each_region_image_wavelet_ratio_pixels = each_region_image_wavelet_num_pixels_below_max / each_region_image_wavelet_num_pixels

        # If the ratio clears our threshhold, keep this label. Otherwise, eliminate it.
        low_intensities__local_maxima_label_mask__to_remove[i] = (
            each_region_image_wavelet_ratio_pixels < percentage_pixels_below_max)

    new_local_maxima = copy.copy(local_maxima)

    new_local_maxima.remove_prop_mask(low_intensities__local_maxima_label_mask__to_remove)

    logger.debug("Removed low intensity maxima that are too close.")

    return(new_local_maxima)


@debugging_tools.log_call(logger)
def remove_too_close_local_maxima(local_maxima, min_local_max_distance, **parameters):
    """
        Removes local maxima that are too close to each other.

        Args:
            local_maxima(ExtendedRegionProps):      The local maxima to refine

            min_local_max_distance(float):          The minimum allowed distance between local maxima

            **parameters(dict):                     essentially unused (catches unneeded arguments).

        Returns:
            ExtendedRegionProps:                    a new instance with the local maxima removed
    """

    # Deleting close local maxima below 16 pixels
    too_close__local_maxima_label_mask__to_remove = numpy.zeros(local_maxima.props.shape, dtype = bool)

    # Find the distance between every local max (efficiently)
    local_maxima_pairs = numpy.array(list(itertools.combinations(xrange(len(local_maxima.props)), 2)))
    local_maxima_distances = scipy.spatial.distance.pdist(local_maxima.props["local_max"], metric = "euclidean")

    too_close_local_maxima_labels_mask = local_maxima_distances < min_local_max_distance
    too_close_local_maxima_pairs = local_maxima_pairs[too_close_local_maxima_labels_mask]

    for each_too_close_local_maxima_pairs in too_close_local_maxima_pairs:
        first_props_index, second_props_index = each_too_close_local_maxima_pairs

        if (local_maxima.props["label"][first_props_index] == local_maxima.props["label"][second_props_index]):
            if local_maxima.props["intensity"][first_props_index] < local_maxima.props["intensity"][second_props_index]:
                too_close__local_maxima_label_mask__to_remove[first_props_index] = True
            else:
                too_close__local_maxima_label_mask__to_remove[second_props_index] = True

    new_local_maxima = copy.copy(local_maxima)

    new_local_maxima.remove_prop_mask(too_close__local_maxima_label_mask__to_remove)

    logger.debug("Removed local maxima that are too close.")

    return(new_local_maxima)


@debugging_tools.log_call(logger)
@HDF5_recorder.static_array_debug_recorder
def wavelet_denoising(new_image,
                      accepted_region_shape_constraints,
                      use_watershed,
                      accepted_neuron_shape_constraints,
                      **parameters):
    """
        Performs wavelet denoising on the given dictionary.
        
        Args:
            new_image(numpy.ndarray):                   array of data for generating a dictionary (first axis is time).

            accepted_region_shape_constraints(dict):    a dictionary containing properties (that could be passed to
                                                        region_properties) under this should be a dictionary that
                                                        contains the keys min and/or max with a value for each.

            use_watershed(bool):                        whether to use watershed for segmentation (currently, this is
                                                        the only option).

            accepted_neuron_shape_constraints(dict):    a dictionary containing properties (that could be passed to
                                                        region_properties) under this should be a dictionary that
                                                        contains the keys min and/or max with a value for each.

            **parameters(dict):                         passed directly to spams.trainDL.
        
        Returns:
            numpy.ndarray:                              a structured array of candidate neurons.
    """

    neurons = get_empty_neuron(shape=new_image.shape, dtype=new_image.dtype)

    new_wavelet_image_denoised_segmentation = None

    logger.debug("Started wavelet denoising.")
    logger.debug("Removing noise...")

    # Contains a bool array with significant values True and noise False.
    new_image_noise_estimate = denoising.estimate_noise(new_image,
                                                        **parameters["denoising.estimate_noise"])

    # Dictionary with wavelet transform applied. Wavelet transform is the first index.
    new_wavelet_transformed_image = wavelet_transform.wavelet_transform(new_image,
                                                                        **parameters["wavelet_transform.wavelet_transform"])

    for i in xrange(len(new_wavelet_transformed_image)):
        wavelet_denoising.recorders.array_debug_recorder["new_wavelet_transformed_image"] = new_wavelet_transformed_image[i]

    # Contains a bool array with significant values True and noise False for all wavelet transforms.
    new_wavelet_transformed_image_significant_mask = denoising.significant_mask(new_wavelet_transformed_image,
                                                                                noise_estimate = new_image_noise_estimate,
                                                                                **parameters["denoising.significant_mask"])

    for i in xrange(len(new_wavelet_transformed_image_significant_mask)):
        wavelet_denoising.recorders.array_debug_recorder["new_wavelet_transformed_image_significant_mask"] = \
            new_wavelet_transformed_image_significant_mask[i]

    new_wavelet_image_mask = new_wavelet_transformed_image_significant_mask[-1].copy()

    # Creates a new dictionary without the noise
    new_wavelet_image_denoised = new_wavelet_transformed_image[-1].copy()
    new_wavelet_image_denoised *= new_wavelet_image_mask

    wavelet_denoising.recorders.array_debug_recorder["new_wavelet_image_denoised"] = new_wavelet_image_denoised

    logger.debug("Noise removed.")

    if new_wavelet_image_denoised.any():
        logger.debug("Frame has content other than noise.")

        logger.debug("Finding the label image...")

        # For holding the label image
        new_wavelet_image_denoised_labeled = scipy.ndimage.label(new_wavelet_image_denoised)[0]

        logger.debug("Found the label image.")
        logger.debug("Determining the properties of the label image...")

        # For holding the label image properties
        new_wavelet_image_denoised_labeled_props = region_properties(new_wavelet_image_denoised_labeled,
                                                                     properties = accepted_region_shape_constraints.keys())

        logger.debug("Determined the properties of the label image.")

        logger.debug("Finding regions that fail to meet some shape constraints...")

        not_within_bound = numpy.zeros(new_wavelet_image_denoised_labeled_props.shape, dtype = bool)

        # Go through each property and make sure they are within the bounds
        for each_prop, each_prop_constraints in accepted_region_shape_constraints.items():
            # If there is a lower bound, checks to see which are above it.
            is_lower_bounded_maybe = None
            if "min" in each_prop_constraints:
                lower_bound = each_prop_constraints["min"]
                is_lower_bounded_maybe = (lower_bound <= new_wavelet_image_denoised_labeled_props[each_prop])
            else:
                is_lower_bounded_maybe = numpy.ones(new_wavelet_image_denoised_labeled_props.shape, dtype = bool)

            # If there is an upper bound, checks to see which are below it.
            is_upper_bounded_maybe = None
            if "max" in each_prop_constraints:
                upper_bound = each_prop_constraints["max"]
                is_upper_bounded_maybe = (new_wavelet_image_denoised_labeled_props[each_prop] <= upper_bound)
            else:
                is_upper_bounded_maybe = numpy.ones(new_wavelet_image_denoised_labeled_props.shape, dtype = bool)

            # See whether both or neither bound is satisified.
            is_within_bound = is_lower_bounded_maybe & is_upper_bounded_maybe
            is_not_within_bound = ~is_within_bound

            # Collect the unbounded ones
            not_within_bound |= is_not_within_bound

        logger.debug("Found regions that fail to meet some shape constraints.")

        logger.debug("Reducing wavelet transform on regions outside of constraints...")

        # Get labels of the unbounded ones
        labels_not_within_bound = new_wavelet_image_denoised_labeled_props["label"][not_within_bound]

        # Iterate over the unbounded ones to fix any errors.
        for each_labels_not_within_bound in labels_not_within_bound:
            # Get a mask for the current label
            current_label_mask = ( new_wavelet_image_denoised_labeled == each_labels_not_within_bound )

            # Get a lower wavelet mask
            lower_wavelet_mask = new_wavelet_transformed_image_significant_mask[-2]

            # Replacement mask
            replacement_mask = current_label_mask & lower_wavelet_mask

            # Zero everything that is not in the replacement region and then use the lower transformed wavelet.
            new_wavelet_image_denoised_replacement = new_wavelet_transformed_image[-2] * replacement_mask

            # Overwrite the area in our old labeled mask to match this lower wavelet transform
            new_wavelet_image_mask[current_label_mask] = lower_wavelet_mask[current_label_mask]

            # However, overwrite the previously labeled area completely (will push more things into the background)
            new_wavelet_image_denoised[current_label_mask] = new_wavelet_image_denoised_replacement[current_label_mask]

        logger.debug("Reduced wavelet transform on regions outside of constraints...")

        logger.debug("Finding new label image...")

        new_wavelet_image_denoised_label_image = scipy.ndimage.label(new_wavelet_image_denoised > 0)[0]

        logger.debug("Found new label image.")

        # ExtendedRegionProps.recorders.array_debug_recorder = HDF5_recorder.HDF5EnumeratedArrayRecorder(
        #     wavelet_denoising.recorders.array_debug_recorder.hdf5_handle
        # )
        ExtendedRegionProps.recorders.array_debug_recorder = wavelet_denoising.recorders.array_debug_recorder
        local_maxima = ExtendedRegionProps(new_wavelet_image_denoised, new_wavelet_image_denoised_label_image)

        wavelet_denoising.recorders.array_debug_recorder["local_maxima_label_image"] = local_maxima.label_image
        wavelet_denoising.recorders.array_debug_recorder["local_maxima_label_image_contours"] = \
            expanded_numpy.generate_labeled_contours(local_maxima.label_image > 0)

        local_maxima = remove_low_intensity_local_maxima(local_maxima, **parameters["remove_low_intensity_local_maxima"])

        wavelet_denoising.recorders.array_debug_recorder["local_maxima_label_image"] = local_maxima.label_image
        wavelet_denoising.recorders.array_debug_recorder["local_maxima_label_image_contours"] = \
            expanded_numpy.generate_labeled_contours(local_maxima.label_image > 0)

        local_maxima = remove_too_close_local_maxima(local_maxima, **parameters["remove_too_close_local_maxima"])

        wavelet_denoising.recorders.array_debug_recorder["local_maxima_label_image"] = local_maxima.label_image
        wavelet_denoising.recorders.array_debug_recorder["local_maxima_label_image_contours"] = \
            expanded_numpy.generate_labeled_contours(local_maxima.label_image > 0)

        if local_maxima.props.size:
            if use_watershed:
                # ############### TODO: Revisit to make sure all of Ferran's algorithm is implemented and this is working properly.

                # Perform the watershed segmentation.

                # First perform disc opening on the image. (Actually, we don't do this according to Ferran.)
                # new_wavelet_image_denoised_opened = vigra.filters.discOpening(new_wavelet_image_denoised.astype(numpy.float32), radius = 1)

                # We could look for seeds using local maxima.
                # However, we already know what these should be as these are the local maxima we have found.
                new_wavelet_image_denoised_maxima = local_maxima.get_local_max_label_image()

                # Segment with watershed on minimum image
                # Use seeds from local maxima as local minima
                # Also, include mask
                new_wavelet_image_denoised_segmentation = skimage.morphology.watershed(local_maxima.intensity_image,
                                                                                       new_wavelet_image_denoised_maxima,
                                                                                       mask = (local_maxima.intensity_image > 0))

                wavelet_denoising.recorders.array_debug_recorder["watershed_segmentation"] = new_wavelet_image_denoised_segmentation
                wavelet_denoising.recorders.array_debug_recorder["watershed_segmentation_contours"] = \
                    expanded_numpy.generate_labeled_contours(new_wavelet_image_denoised_segmentation)

                watershed_local_maxima = ExtendedRegionProps(local_maxima.intensity_image,
                                                             new_wavelet_image_denoised_segmentation,
                                                             properties = ["centroid"] + accepted_neuron_shape_constraints.keys())

                wavelet_denoising.recorders.array_debug_recorder["watershed_local_maxima_label_image"] = \
                    watershed_local_maxima.label_image
                wavelet_denoising.recorders.array_debug_recorder["watershed_local_maxima_label_image_contours"] = \
                    expanded_numpy.generate_labeled_contours(watershed_local_maxima.label_image > 0)

                wavelet_denoising.recorders.array_debug_recorder["watershed_local_maxima_props"] = watershed_local_maxima.props
                wavelet_denoising.recorders.array_debug_recorder["watershed_local_maxima_count"] = watershed_local_maxima.count

                # Remove duplicates
                new_watershed_local_maxima_count_duplicates_mask = (watershed_local_maxima.count["count"] > 1)
                new_watershed_local_maxima_count_duplicate_labels = watershed_local_maxima.count["label"][
                    new_watershed_local_maxima_count_duplicates_mask]
                new_watershed_local_maxima_props_duplicates_mask = expanded_numpy.contains(
                    watershed_local_maxima.props["label"], new_watershed_local_maxima_count_duplicate_labels)
                watershed_local_maxima.remove_prop_mask(new_watershed_local_maxima_props_duplicates_mask)

                wavelet_denoising.recorders.array_debug_recorder["watershed_local_maxima_label_image"] = watershed_local_maxima.label_image
                wavelet_denoising.recorders.array_debug_recorder["watershed_local_maxima_label_image_contours"] = \
                                   expanded_numpy.generate_labeled_contours(watershed_local_maxima.label_image > 0)

                if watershed_local_maxima.props.size:
                    wavelet_denoising.recorders.array_debug_recorder["watershed_local_maxima_props"] = watershed_local_maxima.props
                if watershed_local_maxima.count.size:
                    wavelet_denoising.recorders.array_debug_recorder["watershed_local_maxima_count"] = watershed_local_maxima.count

                not_within_bound = numpy.zeros(watershed_local_maxima.props.shape, dtype = bool)

                # Go through each property and make sure they are within the bounds
                for each_prop, each_prop_constraints in accepted_neuron_shape_constraints.items():
                    # If there is a lower bound, checks to see which are above it.
                    is_lower_bounded_maybe = None
                    if "min" in each_prop_constraints:
                        lower_bound = each_prop_constraints["min"]
                        is_lower_bounded_maybe = (lower_bound <= watershed_local_maxima.props[each_prop])
                    else:
                        is_lower_bounded_maybe = numpy.ones(watershed_local_maxima.props.shape, dtype = bool)

                    # If there is an upper bound, checks to see which are below it.
                    is_upper_bounded_maybe = None
                    if "max" in each_prop_constraints:
                        upper_bound = each_prop_constraints["max"]
                        is_upper_bounded_maybe = (watershed_local_maxima.props[each_prop] <= upper_bound)
                    else:
                        is_upper_bounded_maybe = numpy.ones(watershed_local_maxima.props.shape, dtype = bool)

                    # See whether both or neither bound is satisified.
                    is_within_bound = is_lower_bounded_maybe & is_upper_bounded_maybe
                    is_not_within_bound = ~is_within_bound

                    # Collect the unbounded ones
                    not_within_bound |= is_not_within_bound

                # Get labels outside of bounds and remove them
                watershed_local_maxima.remove_prop_mask(not_within_bound)

                wavelet_denoising.recorders.array_debug_recorder["watershed_local_maxima_label_image"] = watershed_local_maxima.label_image
                wavelet_denoising.recorders.array_debug_recorder["watershed_local_maxima_label_image_contours"] = \
                                   expanded_numpy.generate_labeled_contours(watershed_local_maxima.label_image > 0)

                if watershed_local_maxima.props.size:
                    wavelet_denoising.recorders.array_debug_recorder["watershed_local_maxima_props"] = watershed_local_maxima.props
                if watershed_local_maxima.count.size:
                    wavelet_denoising.recorders.array_debug_recorder["watershed_local_maxima_count"] = watershed_local_maxima.count

                if watershed_local_maxima.props.size:
                    # Creates a NumPy structure array to store
                    neurons = numpy.zeros(len(watershed_local_maxima.props), dtype = neurons.dtype)

                    # Get masks for all cells
                    neurons["mask"] = expanded_numpy.all_permutations_equal(watershed_local_maxima.props["label"],
                                                                            new_wavelet_image_denoised_segmentation)

                    neurons["image"] = new_image * neurons["mask"]

                    neurons["area"] = watershed_local_maxima.props["area"]

                    neurons["max_F"] = neurons["image"].reshape((neurons["image"].shape[0], -1)).max(axis = 1)

                    for i in xrange(len(neurons)):
                        neuron_mask_i_points = numpy.array(neurons["mask"][i].nonzero())

                        neurons["contour"][i] = expanded_numpy.generate_contour(neurons["mask"][i])
                        neurons["gaussian_mean"][i] = neuron_mask_i_points.mean(axis = 1)
                        neurons["gaussian_cov"][i] = numpy.cov(neuron_mask_i_points)

                    neurons["centroid"] = watershed_local_maxima.props["centroid"]

                    if len(neurons) > 1:
                        logger.debug("Extracted neurons. Found " + str(len(neurons)) + " neurons.")
                    else:
                        logger.debug("Extracted a neuron. Found " + str(len(neurons)) + " neuron.")

                    wavelet_denoising.recorders.array_debug_recorder["new_neuron_set"] = neurons
            else:
                # ################### Some other kind of segmentation???
                # Talked to Ferran and he said don't worry about implementing this for now.
                # Does not seem to give noticeably better results.
                raise Exception("No other form of segmentation is implemented.")
        else:
            logger.debug("No local maxima left that are acceptable neurons.")
    else:
        logger.debug("Frame is only noise.")

    logger.debug("Finished making neurons for the current frame.")
    return(neurons)


@debugging_tools.log_call(logger)
@HDF5_recorder.static_array_debug_recorder
def extract_neurons(new_image, neuron_masks):
    """
        Extracts neurons from an image using a stack of masks.

        Args:
            new_image(numpy.ndarray):           spatial coordinates only (no time).

            neuron_masks(numpy.ndarray):        first index of denotes which mask and all others are spatial indices.

        Returns:
            numpy.ndarray:                      a stack of neurons in the same order as the masks.
    """

    neurons = numpy.zeros(len(neuron_masks), dtype = get_neuron_dtype(shape=new_image.shape, dtype=new_image.dtype))

    neurons["mask"] = neuron_masks
    neurons["image"] = new_image * neurons["mask"]
    neurons["area"] = neurons["mask"].reshape(neurons["mask"].shape[0], -1).sum(axis = 1)
    neurons["max_F"] = neurons["image"].reshape((neurons["image"].shape[0], -1)).max(axis = 1)

    for i in xrange(len(neurons)):
        neuron_mask_i_points = numpy.array(neurons["mask"][i].nonzero())

        neurons["contour"][i] = expanded_numpy.generate_contour(neurons["mask"][i])
        neurons["gaussian_mean"][i] = neuron_mask_i_points.mean(axis = 1)
        neurons["gaussian_cov"][i] = numpy.cov(neuron_mask_i_points)

    neurons["centroid"] = neurons["gaussian_mean"]

    return(neurons)


@debugging_tools.log_call(logger)
@HDF5_recorder.static_array_debug_recorder
def fuse_neurons(neuron_1,
                 neuron_2,
                 fraction_mean_neuron_max_threshold,
                 **parameters):
    """
        Merges the two neurons into one neuron, which is returned.

        Note:
            The first neuron (neuron_1) is preferred in tie-breaking situations.

        Args:
            neuron_1(numpy.ndarray):            numpy structured array (dtype get_neuron_dtype) containing the first
                                                neuron.

            neuron_2(numpy.ndarray):            numpy structured array (dtype get_neuron_dtype) containing the second
                                                neuron.

            **parameters(dict):                 dictionary of parameters

        Returns:
            numpy.ndarray:                      a new neuron that is the result of fusing the two.
    """

    fuse_neurons.recorders.array_debug_recorder["neuron_1"] = neuron_1
    fuse_neurons.recorders.array_debug_recorder["neuron_2"] = neuron_2

    assert (neuron_1.shape == neuron_2.shape == tuple())
    assert (neuron_1.dtype == neuron_2.dtype)

    mean_neuron = numpy.array([neuron_1["image"], neuron_2["image"]]).mean(axis = 0)
    mean_neuron_mask = mean_neuron > (fraction_mean_neuron_max_threshold * mean_neuron.max())

    # Gaussian mixture model ??? Skipped this.

    # Creates a NumPy structure array to store
    new_neuron = numpy.zeros(neuron_1.shape, dtype = neuron_1.dtype)

    new_neuron["mask"] = mean_neuron_mask

    new_neuron["contour"] = expanded_numpy.generate_contour(new_neuron["mask"])

    new_neuron["image"] = mean_neuron * new_neuron["mask"]

    new_neuron["area"] = (new_neuron["mask"] > 0).sum()

    new_neuron["max_F"] = new_neuron["image"].max()

    new_neuron_mask_points = numpy.array(new_neuron["mask"].nonzero())
    new_neuron["gaussian_mean"] = new_neuron_mask_points.mean(axis = 1)
    new_neuron["gaussian_cov"] = numpy.cov(new_neuron_mask_points)

    new_neuron["centroid"] = new_neuron["gaussian_mean"]

    fuse_neurons.recorders.array_debug_recorder["new_neuron"] = new_neuron

    return(new_neuron)


@debugging_tools.log_call(logger)
@HDF5_recorder.static_array_debug_recorder
def merge_neuron_sets(new_neuron_set_1,
                      new_neuron_set_2,
                      alignment_min_threshold,
                      overlap_min_threshold,
                      **parameters):
    """
        Merges the two sets of neurons into one. Appends neurons that cannot be merged with the existing set.

        In the functions that follow, it helps to think of each neuron as a vector (instead of as an image). So, dot
        products for vectors are important.

        Note:
            The first neuron set (new_neuron_set_1) is preferred and treated as the set to merge into. However, this
            function will not change the original argument.
        
        Args:
            new_neuron_set_1(numpy.ndarray):            numpy structured array (dtype get_neuron_dtype) containing the
                                                        first neuron set (preferred for tie breaking).

            new_neuron_set_2(numpy.ndarray):            numpy structured array (dtype get_neuron_dtype) containing the
                                                        second neuron set.

            alignment_min_threshold(float):             The minimum required cosine of the angle between two neurons for
                                                        them to be treated as candidates for merging (uses the function
                                                        expanded_numpy.dot_product_L2_normalized).

            overlap_min_threshold(numpy.ndarray):       The minimum required dot product (divided by the L1 norm of one
                                                        of the neurons) for them to be treated as candidates for merging
                                                        (uses the function expanded_numpy.dot_product_partially_normalized).

            **parameters(dict):                         dictionary of parameters
        
        Returns:
            numpy.ndarray:                              a numpy structured array that contains the result of merging the
                                                        two sets (or appending for neurons that could not be merged).
    """

    if new_neuron_set_1.size:
        merge_neuron_sets.recorders.array_debug_recorder["new_neuron_set_1"] = new_neuron_set_1

    if new_neuron_set_2.size:
        merge_neuron_sets.recorders.array_debug_recorder["new_neuron_set_2"] = new_neuron_set_2

    assert (new_neuron_set_1.dtype == new_neuron_set_2.dtype)


    # TODO: Reverse if statement so it is not nots
    if len(new_neuron_set_1) and len(new_neuron_set_2):
        logger.debug("Have 2 sets of neurons to merge.")

        new_neuron_set = new_neuron_set_1.copy()

        new_neuron_set_1_flattened = new_neuron_set_1["image"].reshape(
            new_neuron_set_1["image"].shape[0], -1)
        new_neuron_set_2_flattened = new_neuron_set_2["image"].reshape(
            new_neuron_set_2["image"].shape[0], -1)

        new_neuron_set_1_flattened_mask = new_neuron_set_1["mask"].reshape(new_neuron_set_1["mask"].shape[0], -1)
        new_neuron_set_2_flattened_mask = new_neuron_set_2["mask"].reshape(new_neuron_set_2["mask"].shape[0], -1)

        # Measure the normalized dot product between any two neurons (i.e. related to the angle of separation)
        new_neuron_set_angle = expanded_numpy.dot_product_L2_normalized(new_neuron_set_1_flattened,
                                                                        new_neuron_set_2_flattened)

        merge_neuron_sets.recorders.array_debug_recorder["new_neuron_set_angle"] = new_neuron_set_angle

        # Measure the distance between the two masks
        # (note distance relative to the total mask content of each mask individually)
        new_neuron_set_masks_overlaid_1, new_neuron_set_masks_overlaid_2 = expanded_numpy.dot_product_partially_normalized(
            new_neuron_set_1_flattened_mask, new_neuron_set_2_flattened_mask, ord = 1)

        merge_neuron_sets.recorders.array_debug_recorder["new_neuron_set_masks_overlaid_1"] = new_neuron_set_masks_overlaid_1
        merge_neuron_sets.recorders.array_debug_recorder["new_neuron_set_masks_overlaid_2"] = new_neuron_set_masks_overlaid_2

        # Now that the three measures for the correlation method have been found, we want to know,
        # which are the best correlated neurons between the two sets using these measures.
        # This done to find the neuron in new_neuron_set_1 that best matches each neuron in new_neuron_set_2.
        new_neuron_set_angle_all_optimal_i = new_neuron_set_angle.argmax(axis = 0)
        new_neuron_set_masks_overlaid_1_all_optimal_i = new_neuron_set_masks_overlaid_1.argmax(axis = 0)
        new_neuron_set_masks_overlaid_2_all_optimal_i = new_neuron_set_masks_overlaid_2.argmax(axis = 0)

        merge_neuron_sets.recorders.array_debug_recorder["new_neuron_set_angle_all_optimal_i"] = new_neuron_set_angle_all_optimal_i
        merge_neuron_sets.recorders.array_debug_recorder["new_neuron_set_masks_overlaid_1_all_optimal_i"] = \
                           new_neuron_set_masks_overlaid_1_all_optimal_i
        merge_neuron_sets.recorders.array_debug_recorder["new_neuron_set_masks_overlaid_2_all_optimal_i"] = \
                           new_neuron_set_masks_overlaid_2_all_optimal_i

        # Get all the j indices
        new_neuron_set_all_j = numpy.arange(len(new_neuron_set_2))

        merge_neuron_sets.recorders.array_debug_recorder["new_neuron_set_all_j"] = new_neuron_set_all_j

        # Get the maximum corresponding to the best matched paris from before
        new_neuron_set_angle_maxes = new_neuron_set_angle[(new_neuron_set_angle_all_optimal_i, new_neuron_set_all_j,)]
        new_neuron_set_masks_overlaid_1_maxes = new_neuron_set_masks_overlaid_1[
            (new_neuron_set_masks_overlaid_1_all_optimal_i, new_neuron_set_all_j,)]
        new_neuron_set_masks_overlaid_2_maxes = new_neuron_set_masks_overlaid_2[
            (new_neuron_set_masks_overlaid_2_all_optimal_i, new_neuron_set_all_j,)]

        merge_neuron_sets.recorders.array_debug_recorder["new_neuron_set_angle_maxes"] = new_neuron_set_angle_maxes
        merge_neuron_sets.recorders.array_debug_recorder["new_neuron_set_masks_overlaid_1_maxes"] = new_neuron_set_masks_overlaid_1_maxes
        merge_neuron_sets.recorders.array_debug_recorder["new_neuron_set_masks_overlaid_2_maxes"] = new_neuron_set_masks_overlaid_2_maxes

        # Store a list of the optimal neurons in the existing set to fuse with (by default set all values to -1)
        new_neuron_set_all_optimal_i = numpy.zeros((len(new_neuron_set_2),), dtype = int)
        new_neuron_set_all_optimal_i -= 1

        merge_neuron_sets.recorders.array_debug_recorder["new_neuron_set_all_optimal_i_0"] = new_neuron_set_all_optimal_i

        # Create the masks to use for getting the proper indices
        new_neuron_set_angle_maxes_significant = numpy.zeros((len(new_neuron_set_2),), dtype = bool)
        new_neuron_set_masks_overlaid_1_maxes_significant = numpy.zeros((len(new_neuron_set_2),), dtype = bool)
        new_neuron_set_masks_overlaid_2_maxes_significant = numpy.zeros((len(new_neuron_set_2),), dtype = bool)

        merge_neuron_sets.recorders.array_debug_recorder["new_neuron_set_angle_maxes_significant_0"] = new_neuron_set_angle_maxes_significant
        merge_neuron_sets.recorders.array_debug_recorder["new_neuron_set_masks_overlaid_1_maxes_significant_0"] = \
                           new_neuron_set_masks_overlaid_1_maxes_significant
        merge_neuron_sets.recorders.array_debug_recorder["new_neuron_set_masks_overlaid_2_maxes_significant_0"] = \
                           new_neuron_set_masks_overlaid_2_maxes_significant

        # Get masks that indicate which measurements have the best matching neuron
        new_neuron_set_angle_maxes_significant[
            new_neuron_set_angle_maxes > alignment_min_threshold] = True
        new_neuron_set_masks_overlaid_1_maxes_significant[~new_neuron_set_angle_maxes_significant & (
            new_neuron_set_masks_overlaid_2_maxes > overlap_min_threshold)] = True
        new_neuron_set_masks_overlaid_2_maxes_significant[
            ~new_neuron_set_angle_maxes_significant & ~new_neuron_set_masks_overlaid_1_maxes_significant & (
                new_neuron_set_masks_overlaid_1_maxes > overlap_min_threshold)] = True

        merge_neuron_sets.recorders.array_debug_recorder["new_neuron_set_angle_maxes_significant_1"] = new_neuron_set_angle_maxes_significant
        merge_neuron_sets.recorders.array_debug_recorder["new_neuron_set_masks_overlaid_1_maxes_significant_1"] = \
                           new_neuron_set_masks_overlaid_1_maxes_significant
        merge_neuron_sets.recorders.array_debug_recorder["new_neuron_set_masks_overlaid_2_maxes_significant_1"] = \
                           new_neuron_set_masks_overlaid_2_maxes_significant

        # Using the masks construct the best match neuron index for each case
        # After doing these three, new_neuron_set_all_optimal_i will contain either
        # the index of the neuron to fuse with in new_neuron_set for each 
        new_neuron_set_all_optimal_i[new_neuron_set_angle_maxes_significant] = new_neuron_set_angle_all_optimal_i[new_neuron_set_angle_maxes_significant]

        merge_neuron_sets.recorders.array_debug_recorder["new_neuron_set_all_optimal_i_1"] = new_neuron_set_all_optimal_i

        new_neuron_set_all_optimal_i[new_neuron_set_masks_overlaid_1_maxes_significant] = new_neuron_set_masks_overlaid_1_all_optimal_i[new_neuron_set_masks_overlaid_1_maxes_significant]

        merge_neuron_sets.recorders.array_debug_recorder["new_neuron_set_all_optimal_i_2"] = new_neuron_set_all_optimal_i

        new_neuron_set_all_optimal_i[new_neuron_set_masks_overlaid_2_maxes_significant] = new_neuron_set_masks_overlaid_2_all_optimal_i[new_neuron_set_masks_overlaid_2_maxes_significant]

        merge_neuron_sets.recorders.array_debug_recorder["new_neuron_set_all_optimal_i_3"] = new_neuron_set_all_optimal_i


        # Separate all the best matches that were found from those that were not.
        # Also, remove the -1 as they have served their purpose.
        new_neuron_set_all_optimal_i_found = (new_neuron_set_all_optimal_i != -1)
        new_neuron_set_all_j_fuse = new_neuron_set_all_j[new_neuron_set_all_optimal_i_found]
        new_neuron_set_all_j_append = new_neuron_set_all_j[~new_neuron_set_all_optimal_i_found]
        new_neuron_set_all_optimal_i = new_neuron_set_all_optimal_i[new_neuron_set_all_optimal_i_found]

        merge_neuron_sets.recorders.array_debug_recorder["new_neuron_set_all_optimal_i_found"] = new_neuron_set_all_optimal_i_found

        if new_neuron_set_all_j_fuse.size:
            merge_neuron_sets.recorders.array_debug_recorder["new_neuron_set_all_j_fuse"] = new_neuron_set_all_j_fuse

        if new_neuron_set_all_j_append.size:
            merge_neuron_sets.recorders.array_debug_recorder["new_neuron_set_all_j_append"] = new_neuron_set_all_j_append

        if new_neuron_set_all_optimal_i.size:
            merge_neuron_sets.recorders.array_debug_recorder["new_neuron_set_all_optimal_i_4"] = new_neuron_set_all_optimal_i

        # Fuse all the neurons that can be from new_neuron_set_2 to the new_neuron_set (composed of new_neuron_set_1)
        for i, j in itertools.izip(new_neuron_set_all_optimal_i, new_neuron_set_all_j_fuse):
            # fuse_neurons.recorders.array_debug_recorder = HDF5_recorder.HDF5EnumeratedArrayRecorder(
            #     merge_neuron_sets.recorders.array_debug_recorder.hdf5_handle
            # )
            fuse_neurons.recorders.array_debug_recorder = merge_neuron_sets.recorders.array_debug_recorder

            new_neuron_set[i] = fuse_neurons(new_neuron_set_1[i], new_neuron_set_2[j], **parameters["fuse_neurons"])

        logger.debug("Fused \"" + repr(len(new_neuron_set_all_j_fuse)) + "\" neurons to the existing set.")

        # Tack on the ones that must be appended
        new_neuron_set = numpy.hstack([new_neuron_set, new_neuron_set_2[new_neuron_set_all_j_append]])

        logger.debug("Added \"" + repr(len(new_neuron_set_all_j_append)) + "\" new neurons to the existing set.")

    elif len(new_neuron_set_1):
        logger.debug("Have 1 set of neurons to merge. Only the first set has neurons.")
        new_neuron_set = new_neuron_set_1
    elif len(new_neuron_set_2):
        logger.debug("Have 1 set of neurons to merge. Only the second set has neurons.")
        new_neuron_set = new_neuron_set_2
    else:
        logger.debug("Have 0 sets of neurons to merge.")
        new_neuron_set = new_neuron_set_1

    if new_neuron_set.size:
        merge_neuron_sets.recorders.array_debug_recorder["new_merged_neurons_set"] = new_neuron_set

    return(new_neuron_set)


@debugging_tools.log_call(logger)
@HDF5_recorder.static_array_debug_recorder
def postprocess_data(new_dictionary, **parameters):
    """
        Generates neurons from the dictionary.
        
        Args:
            new_dictionary(numpy.ndarray):        dictionary of basis images to analyze for neurons.
            **parameters(dict):                   dictionary of parameters
        
        Returns:
            numpy.ndarray:                        structured array with relevant information for each neuron found.
    """


    new_neurons_set = None


    # Puts each dictionary basis debug log into a separate group depending on which basis image it was a part of.
    def array_debug_recorder_enumerator(new_list):
        postprocess_data.recorders.array_debug_recorder["neuron_sets"] = None
        neuron_sets_array_debug_recorder = postprocess_data.recorders.array_debug_recorder["neuron_sets"]

        for i, i_str, each in additional_generators.filled_stringify_enumerate(new_list):
            neuron_sets_array_debug_recorder[i_str] = None
            yield ( (i, each, neuron_sets_array_debug_recorder[i_str]) )

    # Get all neurons for all images
    new_neurons_set = get_empty_neuron(shape=new_dictionary[0].shape, dtype=new_dictionary[0].dtype)
    unmerged_neuron_set = get_empty_neuron(shape=new_dictionary[0].shape, dtype=new_dictionary[0].dtype)
    for i, each_new_dictionary_image, each_array_debug_recorder in array_debug_recorder_enumerator(new_dictionary):
        wavelet_denoising.recorders.array_debug_recorder = postprocess_data.recorders.array_debug_recorder
        each_new_neuron_set = wavelet_denoising(each_new_dictionary_image,
                                                **parameters["wavelet_denoising"])

        logger.debug("Denoised a set of neurons from frame " + str(i + 1) + " of " + str(len(new_dictionary)) + ".")

        unmerged_neuron_set = numpy.hstack([unmerged_neuron_set, each_new_neuron_set])

        merge_neuron_sets.recorders.array_debug_recorder = postprocess_data.recorders.array_debug_recorder
        new_neurons_set = merge_neuron_sets(new_neurons_set, each_new_neuron_set,
                                            **parameters["merge_neuron_sets"])

        logger.debug("Merged a set of neurons from frame " + str(i + 1) + " of " + str(len(new_dictionary)) + ".")

    if unmerged_neuron_set.size:
        postprocess_data.recorders.array_debug_recorder["unmerged_neuron_set"] = unmerged_neuron_set

        unmerged_neuron_set_contours = unmerged_neuron_set["contour"].astype(numpy.uint64).copy()

        unmerged_neuron_set_contours *= expanded_numpy.expand_enumerate(unmerged_neuron_set_contours, start = 1)
        unmerged_neuron_set_contours = unmerged_neuron_set_contours.max(axis = 0)

        postprocess_data.recorders.array_debug_recorder["unmerged_neuron_set_contours"] = unmerged_neuron_set_contours

    if new_neurons_set.size:
        postprocess_data.recorders.array_debug_recorder["new_neurons_set"] = new_neurons_set

        new_neurons_set_contours = new_neurons_set["contour"].astype(numpy.uint64).copy()

        new_neurons_set_contours *= expanded_numpy.expand_enumerate(new_neurons_set_contours, start = 1)
        new_neurons_set_contours = new_neurons_set_contours.max(axis = 0)

        postprocess_data.recorders.array_debug_recorder["new_neurons_set_contours"] = new_neurons_set_contours


    return(new_neurons_set)
