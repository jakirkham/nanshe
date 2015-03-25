# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "$May 1, 2014 2:24:55PM$"


import warnings

import numpy
import scipy
import scipy.misc

import vigra

import HDF5_recorder


# Need in order to have logging information no matter what.
import debugging_tools


# Get the logger
logger = debugging_tools.logging.getLogger(__name__)


@debugging_tools.log_call(logger)
def binomial_coefficients(n):
    """
        Generates a row in Pascal's triangle (binomial coefficients).
        
        Args:
            n(int):      which row of Pascal's triangle to return.
        
        Returns:
            cs(numpy.ndarray): a numpy array containing the row of Pascal's triangle.
        
        
        Examples:
            >>> binomial_coefficients(0)
            array([1])
                   
            >>> binomial_coefficients(1)
            array([1, 1])
                   
            >>> binomial_coefficients(2)
            array([1, 2, 1])
                   
            >>> binomial_coefficients(4)
            array([1, 4, 6, 4, 1])
                   
            >>> binomial_coefficients(4.0)
            array([1, 4, 6, 4, 1])
    """

    # Must be integer
    n = int(n)

    # Below 0 is irrelevant. 
    if n < 0:
        n = 0

    # Get enough repeats of n to get each coefficent
    ns = numpy.repeat(n, n + 1)
    # Get all relevant k's
    ks = numpy.arange(n + 1)

    # Get all the coefficents in order
    cs = scipy.misc.comb(ns, ks)
    cs = numpy.around(cs)
    cs = cs.astype(int)

    return(cs)


@debugging_tools.log_call(logger)
def binomial_1D_array_kernel(i, n = 4):
    """
        Generates a 1D numpy array used to make the kernel for the wavelet transform.
        
        Args:
            i(int):      which scaling to use.
            n(int):      which row of Pascal's triangle to return.
        
        Returns:
            r(numpy.ndarray): a numpy array containing the row of Pascal's triangle.
        
        
        Examples:
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

    # Below 0 is irrelevant. 
    if n < 0:
        n = 0

    # Get the binomial coefficients.
    cs = list(binomial_coefficients(n))

    # Get the right number of zeros to fill in
    zs = list(numpy.zeros(2 ** (i - 1) - 1, dtype = int))

    # Create the contents of the 1D kernel before normalization
    r = []
    if len(cs) > 1:
        for _ in cs[:-1]:
            r.append(_)
            r.extend(zs)

        r.append(cs[-1])
    else:
        r.append(cs)

    r = numpy.array(r)
    r = r.astype(float)

    # Normalization on the L_1 norm.
    r /= 2 ** n

    return(r)


@debugging_tools.log_call(logger)
def binomial_1D_vigra_kernel(i, n = 4, border_treatment = vigra.filters.BorderTreatmentMode.BORDER_TREATMENT_REFLECT):
    """
        Generates a vigra.filters.Kernel1D using binomial_1D_array_kernel(i).
        
        Args:
            i(int):                                                 which scaling to use.
            n(int):                                                 which row of Pascal's triangle to return.
            border_treatment(vigra.filters.BorderTreatmentMode):    determines how to deal with the borders.
        
        Returns:
            k(numpy.ndarray): a numpy array containing the row of Pascal's triangle.
        
        
        Examples:
            >>> binomial_1D_vigra_kernel(1) # doctest: +ELLIPSIS
            <vigra.filters.Kernel1D object at 0x...>
    """

    # Generate the vector for the kernel
    h_kern = binomial_1D_array_kernel(i, n)

    # Determine the kernel center
    h_kern_half_width = (h_kern.size - 1) / 2

    # Default kernel
    k = vigra.filters.Kernel1D()
    # Center the kernel
    k.initExplicitly(-h_kern_half_width, h_kern_half_width, h_kern)
    # Set the border treatment mode.
    k.setBorderTreatment(border_treatment)

    return(k)


@debugging_tools.log_call(logger)
@HDF5_recorder.static_array_debug_recorder
def wavelet_transform(im0, scale = 5, include_intermediates = False, include_lower_scales = False, out = None):
    """
        performs integral steps of the wavelet transform on im0 up to the given scale. If scale is an iterable, then 
        
        Args:
            im0(numpy.ndarray):                                  the original image.
            scale(int):                                          the scale of wavelet transform to apply.
            include_intermediates(bool):                         whether to return intermediates or not (default False).
            include_lower_scales(bool):                          whether to include lower scales or not (default False)
                                                                 (ignored if include_intermediates is True).
            out(numpy.ndarray):                                  holds final result (cannot use unless
                                                                 include_intermediates is False or an AssertionError
                                                                 will be raised.)
        
        Returns:
            k(numpy.ndarray): a numpy array containing the row of Pascal's triangle.
        
        
        Examples:
            >>> wavelet_transform(numpy.eye(3, dtype = numpy.float32),
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

            >>> wavelet_transform(numpy.eye(3, dtype = numpy.float32),
            ...     scale = 1,
            ...     include_intermediates = False,
            ...     include_lower_scales = True)
            array([[[ 0.59375, -0.375  , -0.34375],
                    [-0.375  ,  0.625  , -0.375  ],
                    [-0.34375, -0.375  ,  0.59375]]], dtype=float32)

            >>> wavelet_transform(numpy.eye(3, dtype = numpy.float32),
            ...     scale = 1,
            ...     include_intermediates = False,
            ...     include_lower_scales = False)
            array([[ 0.59375, -0.375  , -0.34375],
                   [-0.375  ,  0.625  , -0.375  ],
                   [-0.34375, -0.375  ,  0.59375]], dtype=float32)

            >>> out = numpy.zeros((3, 3), dtype = numpy.float32)
            >>> wavelet_transform(numpy.eye(3, dtype = numpy.float32),
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
            >>> wavelet_transform(out,
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

            >>> out = numpy.eye(3, dtype = numpy.uint8)
            >>> wavelet_transform(out,
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
        warnings.warn("Provided im0 with type \"" + repr(im0.dtype.type) + "\". " +
                      "Will be cast to type \"" + repr(numpy.float32) + "\"", RuntimeWarning)

        im0 = im0.astype(numpy.float32)

    # Make sure that we have scale as a list.
    # If it is not a list, then make a singleton list.
    try:
        scale = numpy.array(list(scale))

        if scale.ndim > 1:
            raise Exception(
                "Scale should only have 1 dimension. Instead, got scale.ndim = \"" + str(scale.ndim) + "\".")

        if len(scale) != im0.ndim:
            raise Exception(
                    "Scale should have a value of each dimension of im0. Instead, got len(scale) = \"" + str(
                        len(scale)) + "\" and im0.ndim = \"" + str(im0.ndim) + "\".")

    except TypeError:
        scale = numpy.repeat([scale], im0.ndim)


    imPrev = None
    imCur = None
    if include_intermediates:
        assert (out is None)

        W = numpy.zeros((scale.max(),) + im0.shape, dtype = numpy.float32)
        imOut = numpy.zeros((scale.max() + 1,) + im0.shape, dtype = numpy.float32)
        imOut[0] = im0

        imCur = imOut[0]
        imPrev = imCur
    else:
        if include_lower_scales:
            if out is None:
                W = numpy.zeros((scale.max(),) + im0.shape, dtype = numpy.float32)
                out = W
            else:
                assert ( out.shape == ((scale.max(),) + im0.shape) )

                if not issubclass(out.dtype.type, numpy.float32):
                    warnings.warn("Provided out with type \"" + repr(out.dtype.type) + "\". " +
                                  "Will be cast to type \"" + repr(numpy.float32) + "\"", RuntimeWarning)

                W = out

            imPrev = numpy.empty_like(im0)
        else:
            if out is not None:
                assert (out.shape == im0.shape)

                if not issubclass(out.dtype.type, numpy.float32):
                    warnings.warn("Provided out with type \"" + repr(out.dtype.type) + "\". " +
                                  "Will be cast to type \"" + repr(numpy.float32) + "\"", RuntimeWarning)

                    out = im0.astype(numpy.float32)

                imPrev = out
            else:
                imPrev = numpy.empty_like(im0)
                out = imPrev

        imCur = im0.astype(numpy.float32)


    for i in xrange(1, scale.max() + 1):
        if include_intermediates:
            imPrev = imCur
            imOut[i] = imOut[i - 1]
            imCur = imOut[i]
        else:
            imPrev[:] = imCur

        h_ker = binomial_1D_vigra_kernel(i)

        for d in xrange(len(scale)):
            if i <= scale[d]:
                vigra.filters.convolveOneDimension(imCur, d, h_ker, out=imCur)

        if include_intermediates or include_lower_scales:
            W[i - 1] = imPrev - imCur

    if include_intermediates:
        return((W, imOut))
    elif include_lower_scales:
        return(W)
    else:
        # Same as returning imPrev - imCur. Except, it avoids generating another array to hold the result.
        out -= imCur
        return(out)