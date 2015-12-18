"""
The ``wavelet`` module provides support for performing a wavelet transform.

===============================================================================
Overview
===============================================================================
Included are tools to aid in the computation of the wavelet transform. In
particular, construction of the kernel at different scales and it application
to the data. The kernel applied is based on the technique presented by
Reichinnek, et al. ( doi:`10.1016/j.neuroimage.2011.12.018`_ ).


.. _`10.1016/j.neuroimage.2011.12.018`: \
    http://dx.doi.org/10.1016/j.neuroimage.2011.12.018

===============================================================================
API
===============================================================================
"""


__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "$May 01, 2014 14:24:55 EDT$"


import warnings

import numpy

import vigra

from nanshe.io import hdf5
from nanshe.util.iters import irange
from nanshe.util.xnumpy import binomial_coefficients


# Need in order to have logging information no matter what.
from nanshe.util import prof


# Get the logger
trace_logger = prof.getTraceLogger(__name__)


@prof.log_call(trace_logger)
def binomial_1D_array_kernel(i, n=4):
    """
        Generates a 1D numpy array used to make the kernel for the wavelet
        transform.

        Args:
            i(int):               which scaling to use.
            n(int):               which row of Pascal's triangle to return.

        Returns:
            r(numpy.ndarray):     a 1D numpy array to use as the wavelet
                                  transform kernel.


        Examples:
            >>> binomial_1D_array_kernel(0, -2)
            array([], dtype=float64)

            >>> binomial_1D_array_kernel(0, 0)
            array([ 1.])

            >>> binomial_1D_array_kernel(0)
            array([ 0.0625,  0.25  ,  0.375 ,  0.25  ,  0.0625])

            >>> binomial_1D_array_kernel(0, 4)
            array([ 0.0625,  0.25  ,  0.375 ,  0.25  ,  0.0625])

            >>> binomial_1D_array_kernel(1, 4)
            array([ 0.0625,  0.25  ,  0.375 ,  0.25  ,  0.0625])

            >>> binomial_1D_array_kernel(2, 4)
            array([ 0.0625,  0.    ,  0.25  ,  0.    ,  0.375 ,  0.    ,  0.25  ,
                    0.    ,  0.0625])

            >>> binomial_1D_array_kernel(3, 4)
            array([ 0.0625,  0.    ,  0.    ,  0.    ,  0.25  ,  0.    ,  0.    ,
                    0.    ,  0.375 ,  0.    ,  0.    ,  0.    ,  0.25  ,  0.    ,
                    0.    ,  0.    ,  0.0625])

            >>> binomial_1D_array_kernel(4, 4)
            array([ 0.0625,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,
                    0.    ,  0.25  ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,
                    0.    ,  0.    ,  0.375 ,  0.    ,  0.    ,  0.    ,  0.    ,
                    0.    ,  0.    ,  0.    ,  0.25  ,  0.    ,  0.    ,  0.    ,
                    0.    ,  0.    ,  0.    ,  0.    ,  0.0625])

            >>> binomial_1D_array_kernel(2, 1)
            array([ 0.5,  0. ,  0.5])
    """

    # Below 1 is irrelevant.
    if i < 1:
        i = 1

    # Get the binomial coefficients.
    cs = list(binomial_coefficients(n))

    # Reuse the correction to `n` found by `binomial_coefficients`.
    n = len(cs) - 1

    # Get the right number of zeros to fill in
    zs = list(numpy.zeros(2 ** (i - 1) - 1, dtype=int))

    # Create the contents of the 1D kernel before normalization
    r = []
    if len(cs) > 1:
        for _ in cs[:-1]:
            r.append(_)
            r.extend(zs)

        r.append(cs[-1])
    else:
        r.extend(cs)

    r = numpy.array(r)
    r = r.astype(float)

    # Normalization on the L_1 norm.
    r /= 2 ** n

    return(r)


@prof.log_call(trace_logger)
def binomial_1D_vigra_kernel(i, n=4, border_treatment=vigra.filters.BorderTreatmentMode.BORDER_TREATMENT_REFLECT):
    """
        Generates a vigra.filters.Kernel1D using binomial_1D_array_kernel(i).

        Args:
            i(int):                                                 which
                                                                    scaling to
                                                                    use.

            n(int):                                                 which row
                                                                    of Pascal's
                                                                    triangle to
                                                                    return.

            border_treatment(vigra.filters.BorderTreatmentMode):    determines
                                                                    how to deal
                                                                    with the
                                                                    borders.

        Returns:
            k(vigra.filters.Kernel1D):                              a 1D vigra
                                                                    kernel to
                                                                    aid in
                                                                    computing
                                                                    the wavelet
                                                                    transform.


        Examples:
            >>> binomial_1D_vigra_kernel(1) # doctest: +ELLIPSIS
            <vigra.filters.Kernel1D object at 0x...>
    """

    # Generate the vector for the kernel
    h_kern = binomial_1D_array_kernel(i, n)

    # Determine the kernel center
    h_kern_half_width = (h_kern.size - 1) // 2

    # Default kernel
    k = vigra.filters.Kernel1D()
    # Center the kernel
    k.initExplicitly(-h_kern_half_width, h_kern_half_width, h_kern)
    # Set the border treatment mode.
    k.setBorderTreatment(border_treatment)

    return(k)


@prof.log_call(trace_logger)
@hdf5.record.static_array_debug_recorder
def transform(im0,
              scale=5,
              include_intermediates=False,
              include_lower_scales=False,
              out=None):
    """
        Performs integral steps of the wavelet transform on im0 up to the given
        scale. If scale is an iterable, then

        Args:
            im0(numpy.ndarray):                  the original image.
            scale(int or tuple of ints):         the scale of wavelet transform
                                                 to apply.

            include_intermediates(bool):         whether to return
                                                 intermediates or not
                                                 (default False).

            include_lower_scales(bool):          whether to include lower
                                                 scales or not (default False)
                                                 (ignored if
                                                 include_intermediates is True)

            out(numpy.ndarray):                  holds final result (cannot use
                                                 unless include_intermediates
                                                 is False or an AssertionError
                                                 will be raised.)

        Returns:
            W, out(tuple of numpy.ndarrays):     returns the final result of
                                                 the wavelet transform and
                                                 possibly other scales. Also,
                                                 may return the intermediates.


        Examples:
            >>> transform(numpy.eye(3, dtype = numpy.float32),
            ...     scale = 1,
            ...     include_intermediates = True,
            ...     include_lower_scales = True) # doctest: +NORMALIZE_WHITESPACE
            (array([[[ 0.59375, -0.375  , -0.34375],
                     [-0.375  ,  0.625  , -0.375  ],
                     [-0.34375, -0.375  ,  0.59375]]], dtype=float32),
             array([[[ 1.     ,  0.     ,  0.     ],
                     [ 0.     ,  1.     ,  0.     ],
                     [ 0.     ,  0.     ,  1.     ]],
                    [[ 0.40625,  0.375  ,  0.34375],
                     [ 0.375  ,  0.375  ,  0.375  ],
                     [ 0.34375,  0.375  ,  0.40625]]], dtype=float32))

            >>> transform(numpy.eye(3, dtype = numpy.float32),
            ...     scale = 1,
            ...     include_intermediates = False,
            ...     include_lower_scales = True)
            array([[[ 0.59375, -0.375  , -0.34375],
                    [-0.375  ,  0.625  , -0.375  ],
                    [-0.34375, -0.375  ,  0.59375]]], dtype=float32)

            >>> transform(numpy.eye(3, dtype = numpy.float32),
            ...     scale = 1,
            ...     include_intermediates = False,
            ...     include_lower_scales = False)
            array([[ 0.59375, -0.375  , -0.34375],
                   [-0.375  ,  0.625  , -0.375  ],
                   [-0.34375, -0.375  ,  0.59375]], dtype=float32)

            >>> transform(numpy.eye(3, dtype = numpy.float32),
            ...     scale = (0, 1),
            ...     include_intermediates = False,
            ...     include_lower_scales = False)
            array([[ 0.625, -0.25 , -0.125],
                   [-0.5  ,  0.5  , -0.5  ],
                   [-0.125, -0.25 ,  0.625]], dtype=float32)

            >>> out = numpy.zeros((3, 3), dtype = numpy.float32)
            >>> transform(numpy.eye(3, dtype = numpy.float32),
            ...     scale = 1,
            ...     include_intermediates = False,
            ...     include_lower_scales = False,
            ...     out = out)
            array([[ 0.59375, -0.375  , -0.34375],
                   [-0.375  ,  0.625  , -0.375  ],
                   [-0.34375, -0.375  ,  0.59375]], dtype=float32)
            >>> out
            array([[ 0.59375, -0.375  , -0.34375],
                   [-0.375  ,  0.625  , -0.375  ],
                   [-0.34375, -0.375  ,  0.59375]], dtype=float32)

            >>> out = numpy.eye(3, dtype = numpy.float32)
            >>> transform(out,
            ...     scale = 1,
            ...     include_intermediates = False,
            ...     include_lower_scales = False,
            ...     out = out)
            array([[ 0.59375, -0.375  , -0.34375],
                   [-0.375  ,  0.625  , -0.375  ],
                   [-0.34375, -0.375  ,  0.59375]], dtype=float32)
            >>> out
            array([[ 0.59375, -0.375  , -0.34375],
                   [-0.375  ,  0.625  , -0.375  ],
                   [-0.34375, -0.375  ,  0.59375]], dtype=float32)

            >>> out = numpy.empty((1, 3, 3), dtype = numpy.float32)
            >>> transform(numpy.eye(3, dtype = numpy.float32),
            ...     scale = 1,
            ...     include_intermediates = False,
            ...     include_lower_scales = True,
            ...     out = out) # doctest: +NORMALIZE_WHITESPACE
            array([[[ 0.59375, -0.375  , -0.34375],
                    [-0.375  ,  0.625  , -0.375  ],
                    [-0.34375, -0.375  ,  0.59375]]], dtype=float32)
            >>> out
            array([[[ 0.59375, -0.375  , -0.34375],
                    [-0.375  ,  0.625  , -0.375  ],
                    [-0.34375, -0.375  ,  0.59375]]], dtype=float32)

            >>> out = numpy.empty((1, 3, 3), dtype = numpy.float64)
            >>> transform(numpy.eye(3, dtype = numpy.float32),
            ...     scale = 1,
            ...     include_intermediates = False,
            ...     include_lower_scales = True,
            ...     out = out) # doctest: +NORMALIZE_WHITESPACE
            array([[[ 0.59375, -0.375  , -0.34375],
                    [-0.375  ,  0.625  , -0.375  ],
                    [-0.34375, -0.375  ,  0.59375]]])
            >>> out
            array([[[ 0.59375, -0.375  , -0.34375],
                    [-0.375  ,  0.625  , -0.375  ],
                    [-0.34375, -0.375  ,  0.59375]]])

            >>> out = numpy.eye(3, dtype = numpy.uint8)
            >>> transform(out,
            ...     scale = 1,
            ...     include_intermediates = False,
            ...     include_lower_scales = False,
            ...     out = out)
            array([[ 0.59375, -0.375  , -0.34375],
                   [-0.375  ,  0.625  , -0.375  ],
                   [-0.34375, -0.375  ,  0.59375]], dtype=float32)
            >>> out
            array([[1, 0, 0],
                   [0, 1, 0],
                   [0, 0, 1]], dtype=uint8)
    """

    if not issubclass(im0.dtype.type, numpy.float32):
        warnings.warn(
            "Provided im0 with type \"" + repr(im0.dtype.type) + "\". " +
            "Will be cast to type \"" + repr(numpy.float32) + "\"",
            RuntimeWarning
        )

        im0 = im0.astype(numpy.float32)

    # Make sure that we have scale as a list.
    # If it is not a list, then make a singleton list.
    try:
        scale = numpy.array(list(scale))

        assert (scale.ndim == 1), \
            "Scale should only have 1 dimension. " + \
            "Instead, got scale.ndim = \"" + str(scale.ndim) + "\"."

        assert (len(scale) == im0.ndim), \
            "Scale should have a value of each dimension of im0. " + \
            "Instead, got len(scale) = \"" + str(len(scale)) + "\" and " + \
            "im0.ndim = \"" + str(im0.ndim) + "\"."

    except TypeError:
        scale = numpy.repeat([scale], im0.ndim)


    imPrev = None
    imCur = None
    if include_intermediates:
        assert (out is None)

        W = numpy.zeros((scale.max(),) + im0.shape, dtype=numpy.float32)
        imOut = numpy.zeros(
            (scale.max() + 1,) + im0.shape, dtype=numpy.float32
        )
        imOut[0] = im0

        imCur = imOut[0]
        imPrev = imCur
    else:
        if include_lower_scales:
            if out is None:
                W = numpy.zeros(
                    (scale.max(),) + im0.shape, dtype=numpy.float32
                )
                out = W
            else:
                assert (out.shape == ((scale.max(),) + im0.shape))

                if not issubclass(out.dtype.type, numpy.float32):
                    warnings.warn(
                        "Provided out with type \"" + repr(out.dtype.type) +
                        "\". " +
                        "Will be cast to type \"" + repr(numpy.float32) + "\"",
                        RuntimeWarning
                    )

                W = out

            imPrev = numpy.empty_like(im0)
        else:
            if out is not None:
                assert (out.shape == im0.shape)

                if not issubclass(out.dtype.type, numpy.float32):
                    warnings.warn(
                        "Provided out with type \"" + repr(out.dtype.type) +
                        "\". " +
                        "Will be cast to type \"" + repr(numpy.float32) + "\"",
                        RuntimeWarning
                    )

                    out = im0.astype(numpy.float32)

                imPrev = out
            else:
                imPrev = numpy.empty_like(im0)
                out = imPrev

        imCur = im0.astype(numpy.float32)


    ident_ker = vigra.filters.Kernel1D()
    for i in irange(1, scale.max() + 1):
        if include_intermediates:
            imPrev = imCur
            imOut[i] = imOut[i - 1]
            imCur = imOut[i]
        else:
            imPrev[:] = imCur

        h_ker_i = binomial_1D_vigra_kernel(i)
        h_ker = tuple()
        for d in irange(len(scale)):
            if i <= scale[d]:
                h_ker += (h_ker_i,)
            else:
                h_ker += (ident_ker,)

        vigra.filters.convolve(imCur, h_ker, out=imCur)

        if include_intermediates or include_lower_scales:
            W[i - 1] = imPrev - imCur

    if include_intermediates:
        return((W, imOut))
    elif include_lower_scales:
        return(W)
    else:
        # Same as returning imPrev - imCur.
        # Except, it avoids generating another array to hold the result.
        out -= imCur
        return(out)
