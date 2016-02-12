__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "$Aug 08, 2014 08:08:39 EDT$"


import nose
import nose.plugins
import nose.plugins.attrib

import json
import operator
import os
import shutil
import tempfile

import h5py
import numpy

import nanshe.util.iters
import nanshe.util.xnumpy
import nanshe.util.wrappers
import nanshe.io.hdf5.record

import nanshe.syn.data

import nanshe.learner


def setup_2d(a_callable):
    a_callable.config_a_block = {
        "debug" : True,
        "generate_neurons" : {
            "postprocess_data" : {
                "wavelet_denoising" : {
                    "remove_low_intensity_local_maxima" : {
                        "percentage_pixels_below_max" : 0.8
                    },
                    "wavelet.transform" : {
                        "scale" : 3
                    },
                    "accepted_region_shape_constraints" : {
                        "major_axis_length" : {
                            "max" : 25.0,
                            "min" : 0.0
                        }
                    },
                    "accepted_neuron_shape_constraints" : {
                        "eccentricity" : {
                            "max" : 0.9,
                            "min" : 0.0
                        },
                        "area" : {
                            "max" : 600,
                            "min" : 25
                        }
                    },
                    "estimate_noise" : {
                        "significance_threshold" : 3.0
                    },
                    "significant_mask" : {
                        "noise_threshold" : 2.0
                    },
                    "remove_too_close_local_maxima" : {
                        "min_local_max_distance" : 16.0
                    },
                    "use_watershed" : True
                },
                "merge_neuron_sets" : {
                    "alignment_min_threshold" : 0.6,
                    "fuse_neurons" : {
                        "fraction_mean_neuron_max_threshold" : 0.01
                    },
                    "overlap_min_threshold" : 0.6
                }
            },
            "run_stage" : "all",
            "preprocess_data" : {
                "normalize_data" : {
                    "renormalized_images" : {
                        "ord" : 2
                    }
                },
                "extract_f0" : {
                    "temporal_smoothing_gaussian_filter_stdev" : 0.0,
                    "temporal_smoothing_gaussian_filter_window_size" : 0.0,

                    "half_window_size" : 100,
                    "which_quantile" : 0.5,

                    "spatial_smoothing_gaussian_filter_stdev" : 0.0,
                    "spatial_smoothing_gaussian_filter_window_size" : 0.0
                },
                "remove_zeroed_lines" : {
                    "erosion_shape" : [
                        21,
                        1
                    ],
                    "dilation_shape" : [
                        1,
                        3
                    ]
                },
                "wavelet.transform" : {
                    "scale" : 3
                }
            },
            "generate_dictionary" : {
                "sklearn.decomposition.dict_learning_online" : {
                    "n_jobs" : 1,
                    "n_components" : 10,
                    "n_iter" : 100,
                    "batch_size" : 256,
                    "alpha" : 0.2
                }
            }
        }
    }

    a_callable.config_blocks = {
        "generate_neurons_blocks" : {
            "num_processes" : 4,
            "block_shape" : [10000, -1, -1],
            "num_blocks" : [-1, 2, 2],
            "half_border_shape" : [0, 5, 5],
            "half_window_shape" : [50, 20, 20],

            "debug" : True,

            "generate_neurons" : {
                "run_stage" : "all",

                "preprocess_data" : {
                    "remove_zeroed_lines" : {
                        "erosion_shape" : [21, 1],
                        "dilation_shape" : [1, 3]
                    },

                    "extract_f0" : {
                        "temporal_smoothing_gaussian_filter_stdev" : 0.0,
                        "temporal_smoothing_gaussian_filter_window_size" : 0.0,

                        "half_window_size" : 50,
                        "which_quantile" : 0.5,

                        "spatial_smoothing_gaussian_filter_stdev" : 0.0,
                        "spatial_smoothing_gaussian_filter_window_size" : 0.0
                    },

                    "wavelet.transform" : {
                        "scale" : 3
                    },

                    "normalize_data" : {
                        "renormalized_images": {
                            "ord" : 2
                        }
                    }
                },


                "generate_dictionary" : {
                    "sklearn.decomposition.dict_learning_online" : {
                        "n_jobs" : 1,
                        "n_components" : 10,
                        "n_iter" : 100,
                        "batch_size" : 256,
                        "alpha" : 0.2
                    }
                },


                "postprocess_data" : {

                    "wavelet_denoising" : {

                        "estimate_noise" : {
                            "significance_threshold" : 3.0
                        },

                        "significant_mask" : {
                            "noise_threshold" : 2.0
                        },

                        "wavelet.transform" : {
                            "scale" : 3
                        },

                        "accepted_region_shape_constraints" : {
                            "major_axis_length" : {
                                "min" : 0.0,
                                "max" : 25.0
                            }
                        },

                        "remove_low_intensity_local_maxima" : {
                            "percentage_pixels_below_max" : 0.8

                        },

                        "remove_too_close_local_maxima" : {
                            "min_local_max_distance"  : 100.0
                        },

                        "use_watershed" : True,

                        "accepted_neuron_shape_constraints" : {
                            "area" : {
                                "min" : 25,
                                "max" : 600
                            },

                            "eccentricity" : {
                                "min" : 0.0,
                                "max" : 0.9
                            }
                        }
                    },


                    "merge_neuron_sets" : {
                        "alignment_min_threshold" : 0.6,
                        "overlap_min_threshold" : 0.6,

                        "fuse_neurons" : {
                            "fraction_mean_neuron_max_threshold" : 0.01
                        }
                    }
                }
            }
        }
    }

    a_callable.config_blocks_drmaa = {
        "generate_neurons_blocks" : {
            "num_processes" : 4,
            "block_shape" : [10000, -1, -1],
            "num_blocks" : [-1, 2, 2],
            "half_border_shape" : [0, 5, 5],
            "half_window_shape" : [50, 20, 20],

            "use_drmaa" : True,
            "num_drmaa_cores" : 1,

            "debug" : True,

            "generate_neurons" : {
                "run_stage" : "all",

                "preprocess_data" : {
                    "remove_zeroed_lines" : {
                        "erosion_shape" : [21, 1],
                        "dilation_shape" : [1, 3]
                    },

                    "extract_f0" : {
                        "temporal_smoothing_gaussian_filter_stdev" : 0.0,
                        "temporal_smoothing_gaussian_filter_window_size" : 0.0,

                        "half_window_size" : 50,
                        "which_quantile" : 0.5,

                        "spatial_smoothing_gaussian_filter_stdev" : 0.0,
                        "spatial_smoothing_gaussian_filter_window_size" : 0.0
                    },

                    "wavelet.transform" : {
                        "scale" : 3
                    },

                    "normalize_data" : {
                        "renormalized_images": {
                            "ord" : 2
                        }
                    }
                },


                "generate_dictionary" : {
                    "sklearn.decomposition.dict_learning_online" : {
                        "n_jobs" : 1,
                        "n_components" : 10,
                        "n_iter" : 100,
                        "batch_size" : 256,
                        "alpha" : 0.2
                    }
                },


                "postprocess_data" : {

                    "wavelet_denoising" : {

                        "estimate_noise" : {
                            "significance_threshold" : 3.0
                        },

                        "significant_mask" : {
                            "noise_threshold" : 2.0
                        },

                        "wavelet.transform" : {
                            "scale" : 3
                        },

                        "accepted_region_shape_constraints" : {
                            "major_axis_length" : {
                                "min" : 0.0,
                                "max" : 25.0
                            }
                        },

                        "remove_low_intensity_local_maxima" : {
                            "percentage_pixels_below_max" : 0.8

                        },

                        "remove_too_close_local_maxima" : {
                            "min_local_max_distance"  : 100.0
                        },

                        "use_watershed" : True,

                        "accepted_neuron_shape_constraints" : {
                            "area" : {
                                "min" : 25,
                                "max" : 600
                            },

                            "eccentricity" : {
                                "min" : 0.0,
                                "max" : 0.9
                            }
                        }
                    },


                    "merge_neuron_sets" : {
                        "alignment_min_threshold" : 0.6,
                        "overlap_min_threshold" : 0.6,

                        "fuse_neurons" : {
                            "fraction_mean_neuron_max_threshold" : 0.01
                        }
                    }
                }
            }
        }
    }

    a_callable.temp_dir = tempfile.mkdtemp(dir=os.environ.get("TEMP", None))
    a_callable.temp_dir = os.path.abspath(a_callable.temp_dir)

    a_callable.hdf5_input_filename = os.path.join(a_callable.temp_dir, "input.h5")
    a_callable.hdf5_input_filepath = a_callable.hdf5_input_filename + "/" + "images"
    a_callable.hdf5_output_filename = os.path.join(a_callable.temp_dir, "output.h5")
    a_callable.hdf5_output_filepath = a_callable.hdf5_output_filename + "/"

    a_callable.config_a_block_filename = os.path.join(a_callable.temp_dir, "config_a_block.json")
    a_callable.config_blocks_filename = os.path.join(a_callable.temp_dir, "config_blocks.json")
    a_callable.config_blocks_drmaa_filename = os.path.join(a_callable.temp_dir, "config_blocks_drmaa.json")

    space = numpy.array([110, 110])
    radii = numpy.array([6, 6, 6, 6, 7, 6])
    magnitudes = numpy.array([15, 16, 15, 17, 16, 16])
    a_callable.points = numpy.array([
        [30, 24],
        [59, 65],
        [21, 65],
        [80, 78],
        [72, 16],
        [45, 32]
    ])

    bases_indices = [[1, 3, 4], [0, 2], [5]]
    linspace_length = 25

    masks = nanshe.syn.data.generate_hypersphere_masks(space, a_callable.points, radii)
    images = nanshe.syn.data.generate_gaussian_images(space, a_callable.points, radii/3.0, magnitudes) * masks

    bases_masks = numpy.zeros((len(bases_indices),) + masks.shape[1:], dtype=masks.dtype)
    bases_images = numpy.zeros((len(bases_indices),) + images.shape[1:], dtype=images.dtype)

    for i, each_basis_indices in enumerate(bases_indices):
        bases_masks[i] = masks[list(each_basis_indices)].max(axis=0)
        bases_images[i] = images[list(each_basis_indices)].max(axis=0)

    image_stack = None
    ramp = numpy.concatenate([numpy.linspace(0, 1, linspace_length), numpy.linspace(1, 0, linspace_length)])

    image_stack = numpy.zeros((bases_images.shape[0] * len(ramp),) + bases_images.shape[1:],
                                   dtype=bases_images.dtype)
    for i in nanshe.util.iters.irange(len(bases_images)):
        image_stack_slice = slice(i * len(ramp), (i+1) * len(ramp), 1)

        image_stack[image_stack_slice] = nanshe.util.xnumpy.all_permutations_operation(
            operator.mul,
            ramp,
            bases_images[i]
        )

    with h5py.File(a_callable.hdf5_input_filename, "w") as fid:
        fid["images"] = image_stack

    with h5py.File(a_callable.hdf5_output_filename, "w") as fid:
        pass

    with open(a_callable.config_a_block_filename, "w") as fid:
        json.dump(a_callable.config_a_block, fid)
        fid.write("\n")

    with open(a_callable.config_blocks_filename, "w") as fid:
        json.dump(a_callable.config_blocks, fid)
        fid.write("\n")

    with open(a_callable.config_blocks_drmaa_filename, "w") as fid:
        json.dump(a_callable.config_blocks_drmaa, fid)
        fid.write("\n")


def teardown_2d(a_callable):
    try:
        os.remove(a_callable.config_a_block_filename)
    except OSError:
        pass
    a_callable.config_a_block_filename = ""

    try:
        os.remove(a_callable.config_blocks_filename)
    except OSError:
        pass
    a_callable.config_blocks_filename = ""

    try:
        os.remove(a_callable.config_blocks_drmaa_filename)
    except OSError:
        pass
    a_callable.config_blocks_drmaa_filename = ""

    try:
        os.remove(a_callable.hdf5_input_filename)
    except OSError:
        pass
    a_callable.hdf5_input_filename = ""

    try:
        os.remove(a_callable.hdf5_output_filename)
    except OSError:
        pass
    a_callable.hdf5_output_filename = ""

    shutil.rmtree(a_callable.temp_dir)
    a_callable.temp_dir = ""


def setup_3d(a_callable):
    a_callable.config_a_block_3D = {
        "debug" : True,
        "generate_neurons" : {
            "postprocess_data" : {
                "wavelet_denoising" : {
                    "remove_low_intensity_local_maxima" : {
                        "percentage_pixels_below_max" : 0.8
                    },
                    "wavelet.transform" : {
                        "scale" : 3
                    },
                    "accepted_region_shape_constraints" : {
                        "major_axis_length" : {
                            "max" : 25.0,
                            "min" : 0.0
                        }
                    },
                    "accepted_neuron_shape_constraints" : {
                        "eccentricity" : {
                            "max" : 0.9,
                            "min" : 0.0
                        },
                        "area" : {
                            "max" : 15000,
                            "min" : 100
                        }
                    },
                    "estimate_noise" : {
                        "significance_threshold" : 3.0
                    },
                    "significant_mask" : {
                        "noise_threshold" : 2.0
                    },
                    "remove_too_close_local_maxima" : {
                        "min_local_max_distance" : 16.0
                    },
                    "use_watershed" : True
                },
                "merge_neuron_sets" : {
                    "alignment_min_threshold" : 0.6,
                    "fuse_neurons" : {
                        "fraction_mean_neuron_max_threshold" : 0.01
                    },
                    "overlap_min_threshold" : 0.6
                }
            },
            "run_stage" : "all",
            "preprocess_data" : {
                "normalize_data" : {
                    "renormalized_images" : {
                        "ord" : 2
                    }
                },
                "extract_f0" : {
                    "temporal_smoothing_gaussian_filter_stdev" : 0.0,
                    "temporal_smoothing_gaussian_filter_window_size" : 0.0,

                    "half_window_size" : 50,
                    "which_quantile" : 0.5,

                    "spatial_smoothing_gaussian_filter_stdev" : 0.0,
                    "spatial_smoothing_gaussian_filter_window_size" : 0.0
                },
                "wavelet.transform" : {
                    "scale" : 3
                }
            },
            "generate_dictionary" : {
                "sklearn.decomposition.dict_learning_online" : {
                    "n_jobs" : 1,
                    "n_components" : 10,
                    "n_iter" : 100,
                    "batch_size" : 256,
                    "alpha" : 0.2
                }
            }
        }
    }

    a_callable.config_blocks_3D = {
        "generate_neurons_blocks" : {
            "num_processes" : 4,
            "block_shape" : [10000, -1, -1, -1],
            "num_blocks" : [-1, 2, 2, 2],
            "half_border_shape" : [0, 5, 5, 5],
            "half_window_shape" : [50, 20, 20, 20],

            "debug" : True,

            "generate_neurons" : {
                "postprocess_data" : {
                    "wavelet_denoising" : {
                        "remove_low_intensity_local_maxima" : {
                            "percentage_pixels_below_max" : 0.8
                        },
                        "wavelet.transform" : {
                            "scale" : 3
                        },
                        "accepted_region_shape_constraints" : {
                            "major_axis_length" : {
                                "max" : 25.0,
                                "min" : 0.0
                            }
                        },
                        "accepted_neuron_shape_constraints" : {
                            "eccentricity" : {
                                "max" : 0.9,
                                "min" : 0.0
                            },
                            "area" : {
                                "max" : 15000,
                                "min" : 100
                            }
                        },
                        "estimate_noise" : {
                            "significance_threshold" : 3.0
                        },
                        "significant_mask" : {
                            "noise_threshold" : 2.0
                        },
                        "remove_too_close_local_maxima" : {
                            "min_local_max_distance" : 16.0
                        },
                        "use_watershed" : True
                    },
                    "merge_neuron_sets" : {
                        "alignment_min_threshold" : 0.6,
                        "fuse_neurons" : {
                            "fraction_mean_neuron_max_threshold" : 0.01
                        },
                        "overlap_min_threshold" : 0.6
                    }
                },
                "run_stage" : "all",
                "preprocess_data" : {
                    "normalize_data" : {
                        "renormalized_images" : {
                            "ord" : 2
                        }
                    },
                    "extract_f0" : {
                        "temporal_smoothing_gaussian_filter_stdev" : 0.0,
                        "temporal_smoothing_gaussian_filter_window_size" : 0.0,

                        "half_window_size" : 50,
                        "which_quantile" : 0.5,

                        "spatial_smoothing_gaussian_filter_stdev" : 0.0,
                        "spatial_smoothing_gaussian_filter_window_size" : 0.0
                    },
                    "wavelet.transform" : {
                        "scale" : 3
                    }
                },
                "generate_dictionary" : {
                    "sklearn.decomposition.dict_learning_online" : {
                        "n_jobs" : 1,
                        "n_components" : 10,
                        "n_iter" : 100,
                        "batch_size" : 256,
                        "alpha" : 0.2
                    }
                }
            }
        }
    }

    a_callable.config_blocks_3D_drmaa = {
        "generate_neurons_blocks" : {
            "num_processes" : 4,
            "block_shape" : [10000, -1, -1, -1],
            "num_blocks" : [-1, 2, 2, 2],
            "half_border_shape" : [0, 5, 5, 5],
            "half_window_shape" : [50, 20, 20, 20],

            "use_drmaa" : True,
            "num_drmaa_cores" : 1,

            "debug" : True,

            "generate_neurons" : {
                "postprocess_data" : {
                    "wavelet_denoising" : {
                        "remove_low_intensity_local_maxima" : {
                            "percentage_pixels_below_max" : 0.8
                        },
                        "wavelet.transform" : {
                            "scale" : 3
                        },
                        "accepted_region_shape_constraints" : {
                            "major_axis_length" : {
                                "max" : 25.0,
                                "min" : 0.0
                            }
                        },
                        "accepted_neuron_shape_constraints" : {
                            "eccentricity" : {
                                "max" : 0.9,
                                "min" : 0.0
                            },
                            "area" : {
                                "max" : 15000,
                                "min" : 100
                            }
                        },
                        "estimate_noise" : {
                            "significance_threshold" : 3.0
                        },
                        "significant_mask" : {
                            "noise_threshold" : 2.0
                        },
                        "remove_too_close_local_maxima" : {
                            "min_local_max_distance" : 16.0
                        },
                        "use_watershed" : True
                    },
                    "merge_neuron_sets" : {
                        "alignment_min_threshold" : 0.6,
                        "fuse_neurons" : {
                            "fraction_mean_neuron_max_threshold" : 0.01
                        },
                        "overlap_min_threshold" : 0.6
                    }
                },
                "run_stage" : "all",
                "preprocess_data" : {
                    "normalize_data" : {
                        "renormalized_images" : {
                            "ord" : 2
                        }
                    },
                    "extract_f0" : {
                        "temporal_smoothing_gaussian_filter_stdev" : 0.0,
                        "temporal_smoothing_gaussian_filter_window_size" : 0.0,

                        "half_window_size" : 50,
                        "which_quantile" : 0.5,

                        "spatial_smoothing_gaussian_filter_stdev" : 0.0,
                        "spatial_smoothing_gaussian_filter_window_size" : 0.0
                    },
                    "wavelet.transform" : {
                        "scale" : 3
                    }
                },
                "generate_dictionary" : {
                    "sklearn.decomposition.dict_learning_online" : {
                        "n_jobs" : 1,
                        "n_components" : 10,
                        "n_iter" : 100,
                        "batch_size" : 256,
                        "alpha" : 0.2
                    }
                }
            }
        }
    }

    a_callable.temp_dir = tempfile.mkdtemp(dir=os.environ.get("TEMP", None))
    a_callable.temp_dir = os.path.abspath(a_callable.temp_dir)

    a_callable.hdf5_input_3D_filename = os.path.join(a_callable.temp_dir, "input_3D.h5")
    a_callable.hdf5_input_3D_filepath = a_callable.hdf5_input_3D_filename + "/" + "images"
    a_callable.hdf5_output_3D_filename = os.path.join(a_callable.temp_dir, "output_3D.h5")
    a_callable.hdf5_output_3D_filepath = a_callable.hdf5_output_3D_filename + "/"

    a_callable.config_a_block_3D_filename = os.path.join(a_callable.temp_dir, "config_a_block_3D.json")
    a_callable.config_blocks_3D_filename = os.path.join(a_callable.temp_dir, "config_blocks_3D.json")
    a_callable.config_blocks_3D_drmaa_filename = os.path.join(a_callable.temp_dir, "config_blocks_3D_drmaa.json")

    bases_indices = [[1, 3, 4], [0, 2], [5]]
    linspace_length = 25

    space3 = numpy.array([60, 60, 60])
    radii3 = numpy.array([4, 3, 3, 3, 4, 3])
    magnitudes3 = numpy.array([8, 8, 8, 8, 8, 8])
    a_callable.points3 = numpy.array([
        [15, 16, 17],
        [42, 21, 23],
        [45, 32, 34],
        [41, 41, 42],
        [36, 15, 41],
        [22, 16, 34]
    ])

    masks3 = nanshe.syn.data.generate_hypersphere_masks(space3, a_callable.points3, radii3)
    images3 = nanshe.syn.data.generate_gaussian_images(space3, a_callable.points3, radii3/3.0, magnitudes3) * masks3

    bases_masks3 = numpy.zeros((len(bases_indices),) + masks3.shape[1:], dtype=masks3.dtype)
    bases_images3 = numpy.zeros((len(bases_indices),) + images3.shape[1:], dtype=images3.dtype)

    for i, each_basis_indices in enumerate(bases_indices):
        bases_masks3[i] = masks3[list(each_basis_indices)].max(axis=0)
        bases_images3[i] = images3[list(each_basis_indices)].max(axis=0)

    image_stack3 = None
    ramp = numpy.concatenate([numpy.linspace(0, 1, linspace_length), numpy.linspace(1, 0, linspace_length)])

    image_stack3 = numpy.zeros(
        (bases_images3.shape[0] * len(ramp),) + bases_images3.shape[1:],
        dtype=bases_images3.dtype
    )
    for i in nanshe.util.iters.irange(len(bases_images3)):
        image_stack_slice3 = slice(i * len(ramp), (i+1) * len(ramp), 1)

        image_stack3[image_stack_slice3] = nanshe.util.xnumpy.all_permutations_operation(
            operator.mul,
            ramp,
            bases_images3[i]
        )

    with h5py.File(a_callable.hdf5_input_3D_filename, "w") as fid:
        fid["images"] = image_stack3

    with h5py.File(a_callable.hdf5_output_3D_filename, "w") as fid:
        pass

    with open(a_callable.config_a_block_3D_filename, "w") as fid:
        json.dump(a_callable.config_a_block_3D, fid)
        fid.write("\n")

    with open(a_callable.config_blocks_3D_filename, "w") as fid:
        json.dump(a_callable.config_blocks_3D, fid)
        fid.write("\n")

    with open(a_callable.config_blocks_3D_drmaa_filename, "w") as fid:
        json.dump(a_callable.config_blocks_3D_drmaa, fid)
        fid.write("\n")


def teardown_3d(a_callable):
    try:
        os.remove(a_callable.config_a_block_3D_filename)
    except OSError:
        pass
    a_callable.config_a_block_3D_filename = ""

    try:
        os.remove(a_callable.config_blocks_3D_filename)
    except OSError:
        pass
    a_callable.config_blocks_3D_filename = ""

    try:
        os.remove(a_callable.config_blocks_3D_drmaa_filename)
    except OSError:
        pass
    a_callable.config_blocks_3D_drmaa_filename = ""

    try:
        os.remove(a_callable.hdf5_input_3D_filename)
    except OSError:
        pass
    a_callable.hdf5_input_3D_filename = ""

    try:
        os.remove(a_callable.hdf5_output_3D_filename)
    except OSError:
        pass
    a_callable.hdf5_output_3D_filename = ""

    shutil.rmtree(a_callable.temp_dir)
    a_callable.temp_dir = ""


@nanshe.util.wrappers.with_setup_state(setup_2d, teardown_2d)
def test_main_1():
    executable = os.path.splitext(nanshe.learner.__file__)[0] + os.extsep + "py"

    argv = (executable, test_main_1.config_a_block_filename, test_main_1.hdf5_input_filepath, test_main_1.hdf5_output_filepath,)

    assert (0 == nanshe.learner.main(*argv))

    assert os.path.exists(test_main_1.hdf5_output_filename)

    with h5py.File(test_main_1.hdf5_output_filename, "r") as fid:
        assert ("neurons" in fid)

        neurons = fid["neurons"].value

    assert (len(test_main_1.points) == len(neurons))

    neuron_maxes = (neurons["image"] == nanshe.util.xnumpy.expand_view(neurons["max_F"], neurons["image"].shape[1:]))

    neuron_max_points = []
    for i in nanshe.util.iters.irange(len(neuron_maxes)):
        neuron_max_points.append(
            numpy.array(neuron_maxes[i].nonzero()).mean(axis=1).round().astype(int)
        )
    neuron_max_points = numpy.array(neuron_max_points)

    assert numpy.allclose(
        neurons["centroid"], neuron_max_points, rtol=0.0, atol=0.5
    )

    matched = dict()
    unmatched_points = numpy.arange(len(test_main_1.points))
    for i in nanshe.util.iters.irange(len(neuron_max_points)):
        new_unmatched_points = []
        for j in unmatched_points:
            if not (neuron_max_points[i] == test_main_1.points[j]).all():
                new_unmatched_points.append(j)
            else:
                matched[i] = j

        unmatched_points = new_unmatched_points

    assert (len(unmatched_points) == 0)


@nanshe.util.wrappers.with_setup_state(setup_2d, teardown_2d)
def test_main_2():
    executable = os.path.splitext(nanshe.learner.__file__)[0] + os.extsep + "py"

    argv = (executable, test_main_2.config_blocks_filename, test_main_2.hdf5_input_filepath, test_main_2.hdf5_output_filepath,)

    assert (0 == nanshe.learner.main(*argv))

    assert os.path.exists(test_main_2.hdf5_output_filename)

    with h5py.File(test_main_2.hdf5_output_filename, "r") as fid:
        assert ("neurons" in fid)

        neurons = fid["neurons"].value

    assert (len(test_main_2.points) == len(neurons))

    neuron_maxes = (neurons["image"] == nanshe.util.xnumpy.expand_view(neurons["max_F"], neurons["image"].shape[1:]))

    neuron_max_points = []
    for i in nanshe.util.iters.irange(len(neuron_maxes)):
        neuron_max_points.append(
            numpy.array(neuron_maxes[i].nonzero()).mean(axis=1).round().astype(int)
        )
    neuron_max_points = numpy.array(neuron_max_points)

    assert numpy.allclose(
        neurons["centroid"], neuron_max_points, rtol=0.0, atol=0.5
    )

    matched = dict()
    unmatched_points = numpy.arange(len(test_main_2.points))
    for i in nanshe.util.iters.irange(len(neuron_max_points)):
        new_unmatched_points = []
        for j in unmatched_points:
            if not (neuron_max_points[i] == test_main_2.points[j]).all():
                new_unmatched_points.append(j)
            else:
                matched[i] = j

        unmatched_points = new_unmatched_points

    assert (len(unmatched_points) == 0)


@nose.plugins.attrib.attr("DRMAA")
@nanshe.util.wrappers.with_setup_state(setup_2d, teardown_2d)
def test_main_3():
    # Attempt to import drmaa.
    # If it fails to import, either the user has no intent in using it or forgot to install it.
    # If it imports, but fails to find symbols, then the user has not set DRMAA_LIBRARY_PATH or does not have libdrmaa.so.
    try:
        import drmaa
    except ImportError:
        # python-drmaa is not installed.
        raise nose.SkipTest("Skipping test test_main_3. Was not able to import drmaa. To run this test, please pip or easy_install drmaa.")
    except RuntimeError:
        # The drmaa library was not specified, but python-drmaa is installed.
        raise nose.SkipTest("Skipping test test_main_3. Was able to import drmaa. However, the drmaa library could not be found. Please either specify the location of libdrmaa.so using the DRMAA_LIBRARY_PATH environment variable or disable/remove use_drmaa from the config file.")

    executable = os.path.splitext(nanshe.learner.__file__)[0] + os.extsep + "py"

    argv = (executable, test_main_3.config_blocks_drmaa_filename, test_main_3.hdf5_input_filepath, test_main_3.hdf5_output_filepath,)

    assert (0 == nanshe.learner.main(*argv))

    assert os.path.exists(test_main_3.hdf5_output_filename)

    with h5py.File(test_main_3.hdf5_output_filename, "r") as fid:
        assert ("neurons" in fid)

        neurons = fid["neurons"].value

    assert (len(test_main_3.points) == len(neurons))

    neuron_maxes = (neurons["image"] == nanshe.util.xnumpy.expand_view(neurons["max_F"], neurons["image"].shape[1:]))

    neuron_max_points = []
    for i in nanshe.util.iters.irange(len(neuron_maxes)):
        neuron_max_points.append(
            numpy.array(neuron_maxes[i].nonzero()).mean(axis=1).round().astype(int)
        )
    neuron_max_points = numpy.array(neuron_max_points)

    assert numpy.allclose(
        neurons["centroid"], neuron_max_points, rtol=0.0, atol=0.5
    )

    matched = dict()
    unmatched_points = numpy.arange(len(test_main_3.points))
    for i in nanshe.util.iters.irange(len(neuron_max_points)):
        new_unmatched_points = []
        for j in unmatched_points:
            if not (neuron_max_points[i] == test_main_3.points[j]).all():
                new_unmatched_points.append(j)
            else:
                matched[i] = j

        unmatched_points = new_unmatched_points

    assert (len(unmatched_points) == 0)


@nose.plugins.attrib.attr("3D")
@nanshe.util.wrappers.with_setup_state(setup_3d, teardown_3d)
def test_main_4():
    executable = os.path.splitext(nanshe.learner.__file__)[0] + os.extsep + "py"

    argv = (executable, test_main_4.config_a_block_3D_filename, test_main_4.hdf5_input_3D_filepath, test_main_4.hdf5_output_3D_filepath,)

    assert (0 == nanshe.learner.main(*argv))

    assert os.path.exists(test_main_4.hdf5_output_3D_filename)

    with h5py.File(test_main_4.hdf5_output_3D_filename, "r") as fid:
        assert ("neurons" in fid)

        neurons = fid["neurons"].value

    assert (len(test_main_4.points3) == len(neurons))

    neuron_maxes = (neurons["image"] == nanshe.util.xnumpy.expand_view(neurons["max_F"], neurons["image"].shape[1:]))

    neuron_max_points = []
    for i in nanshe.util.iters.irange(len(neuron_maxes)):
        neuron_max_points.append(
            numpy.array(neuron_maxes[i].nonzero()).mean(axis=1).round().astype(int)
        )
    neuron_max_points = numpy.array(neuron_max_points)

    assert numpy.allclose(
        neurons["centroid"], neuron_max_points, rtol=0.0, atol=2.0
    )

    matched = dict()
    unmatched_points = numpy.arange(len(test_main_4.points3))
    for i in nanshe.util.iters.irange(len(neuron_max_points)):
        new_unmatched_points = []
        for j in unmatched_points:
            if not (neuron_max_points[i] == test_main_4.points3[j]).all():
                new_unmatched_points.append(j)
            else:
                matched[i] = j

        unmatched_points = new_unmatched_points

    assert (len(unmatched_points) == 0)


@nose.plugins.attrib.attr("3D")
@nanshe.util.wrappers.with_setup_state(setup_3d, teardown_3d)
def test_main_5():
    executable = os.path.splitext(nanshe.learner.__file__)[0] + os.extsep + "py"

    argv = (executable, test_main_5.config_blocks_3D_filename, test_main_5.hdf5_input_3D_filepath, test_main_5.hdf5_output_3D_filepath,)

    assert (0 == nanshe.learner.main(*argv))

    assert os.path.exists(test_main_5.hdf5_output_3D_filename)

    with h5py.File(test_main_5.hdf5_output_3D_filename, "r") as fid:
        assert ("neurons" in fid)

        neurons = fid["neurons"].value

    assert (len(test_main_5.points3) == len(neurons))

    neuron_maxes = (neurons["image"] == nanshe.util.xnumpy.expand_view(neurons["max_F"], neurons["image"].shape[1:]))

    neuron_max_points = []
    for i in nanshe.util.iters.irange(len(neuron_maxes)):
        neuron_max_points.append(
            numpy.array(neuron_maxes[i].nonzero()).mean(axis=1).round().astype(int)
        )
    neuron_max_points = numpy.array(neuron_max_points)

    assert numpy.allclose(
        neurons["centroid"], neuron_max_points, rtol=0.0, atol=2.0
    )

    matched = dict()
    unmatched_points = numpy.arange(len(test_main_5.points3))
    for i in nanshe.util.iters.irange(len(neuron_max_points)):
        new_unmatched_points = []
        for j in unmatched_points:
            if not (neuron_max_points[i] == test_main_5.points3[j]).all():
                new_unmatched_points.append(j)
            else:
                matched[i] = j

        unmatched_points = new_unmatched_points

    assert (len(unmatched_points) == 0)


@nose.plugins.attrib.attr("3D", "DRMAA")
@nanshe.util.wrappers.with_setup_state(setup_3d, teardown_3d)
def test_main_6():
    # Attempt to import drmaa.
    # If it fails to import, either the user has no intent in using it or forgot to install it.
    # If it imports, but fails to find symbols, then the user has not set DRMAA_LIBRARY_PATH or does not have libdrmaa.so.
    try:
        import drmaa
    except ImportError:
        # python-drmaa is not installed.
        raise nose.SkipTest("Skipping test test_main_3. Was not able to import drmaa. To run this test, please pip or easy_install drmaa.")
    except RuntimeError:
        # The drmaa library was not specified, but python-drmaa is installed.
        raise nose.SkipTest("Skipping test test_main_3. Was able to import drmaa. However, the drmaa library could not be found. Please either specify the location of libdrmaa.so using the DRMAA_LIBRARY_PATH environment variable or disable/remove use_drmaa from the config file.")

    executable = os.path.splitext(nanshe.learner.__file__)[0] + os.extsep + "py"

    argv = (executable, test_main_6.config_blocks_3D_drmaa_filename, test_main_6.hdf5_input_3D_filepath, test_main_6.hdf5_output_3D_filepath,)

    assert (0 == nanshe.learner.main(*argv))

    assert os.path.exists(test_main_6.hdf5_output_3D_filename)

    with h5py.File(test_main_6.hdf5_output_3D_filename, "r") as fid:
        assert ("neurons" in fid)

        neurons = fid["neurons"].value

    assert (len(test_main_6.points3) == len(neurons))

    neuron_maxes = (neurons["image"] == nanshe.util.xnumpy.expand_view(neurons["max_F"], neurons["image"].shape[1:]))

    neuron_max_points = []
    for i in nanshe.util.iters.irange(len(neuron_maxes)):
        neuron_max_points.append(
            numpy.array(neuron_maxes[i].nonzero()).mean(axis=1).round().astype(int)
        )
    neuron_max_points = numpy.array(neuron_max_points)

    assert numpy.allclose(
        neurons["centroid"], neuron_max_points, rtol=0.0, atol=2.0
    )

    matched = dict()
    unmatched_points = numpy.arange(len(test_main_6.points3))
    for i in nanshe.util.iters.irange(len(neuron_max_points)):
        new_unmatched_points = []
        for j in unmatched_points:
            if not (neuron_max_points[i] == test_main_6.points3[j]).all():
                new_unmatched_points.append(j)
            else:
                matched[i] = j

        unmatched_points = new_unmatched_points

    assert (len(unmatched_points) == 0)


@nanshe.util.wrappers.with_setup_state(setup_2d, teardown_2d)
def test_generate_neurons_io_handler_1():
    nanshe.learner.generate_neurons_io_handler(test_generate_neurons_io_handler_1.hdf5_input_filepath, test_generate_neurons_io_handler_1.hdf5_output_filepath, test_generate_neurons_io_handler_1.config_a_block_filename)

    assert os.path.exists(test_generate_neurons_io_handler_1.hdf5_output_filename)

    with h5py.File(test_generate_neurons_io_handler_1.hdf5_output_filename, "r") as fid:
        assert ("neurons" in fid)

        neurons = fid["neurons"].value

    assert (len(test_generate_neurons_io_handler_1.points) == len(neurons))

    neuron_maxes = (neurons["image"] == nanshe.util.xnumpy.expand_view(neurons["max_F"], neurons["image"].shape[1:]))

    neuron_max_points = []
    for i in nanshe.util.iters.irange(len(neuron_maxes)):
        neuron_max_points.append(
            numpy.array(neuron_maxes[i].nonzero()).mean(axis=1).round().astype(int)
        )
    neuron_max_points = numpy.array(neuron_max_points)

    assert numpy.allclose(
        neurons["centroid"], neuron_max_points, rtol=0.0, atol=0.5
    )

    matched = dict()
    unmatched_points = numpy.arange(len(test_generate_neurons_io_handler_1.points))
    for i in nanshe.util.iters.irange(len(neuron_max_points)):
        new_unmatched_points = []
        for j in unmatched_points:
            if not (neuron_max_points[i] == test_generate_neurons_io_handler_1.points[j]).all():
                new_unmatched_points.append(j)
            else:
                matched[i] = j

        unmatched_points = new_unmatched_points

    assert (len(unmatched_points) == 0)


@nanshe.util.wrappers.with_setup_state(setup_2d, teardown_2d)
def test_generate_neurons_io_handler_2():
    nanshe.learner.generate_neurons_io_handler(test_generate_neurons_io_handler_2.hdf5_input_filepath, test_generate_neurons_io_handler_2.hdf5_output_filepath, test_generate_neurons_io_handler_2.config_blocks_filename)

    assert os.path.exists(test_generate_neurons_io_handler_2.hdf5_output_filename)

    with h5py.File(test_generate_neurons_io_handler_2.hdf5_output_filename, "r") as fid:
        assert ("neurons" in fid)

        neurons = fid["neurons"].value

    assert (len(test_generate_neurons_io_handler_2.points) == len(neurons))

    neuron_maxes = (neurons["image"] == nanshe.util.xnumpy.expand_view(neurons["max_F"], neurons["image"].shape[1:]))

    neuron_max_points = []
    for i in nanshe.util.iters.irange(len(neuron_maxes)):
        neuron_max_points.append(
            numpy.array(neuron_maxes[i].nonzero()).mean(axis=1).round().astype(int)
        )
    neuron_max_points = numpy.array(neuron_max_points)

    assert numpy.allclose(
        neurons["centroid"], neuron_max_points, rtol=0.0, atol=0.5
    )

    matched = dict()
    unmatched_points = numpy.arange(len(test_generate_neurons_io_handler_2.points))
    for i in nanshe.util.iters.irange(len(neuron_max_points)):
        new_unmatched_points = []
        for j in unmatched_points:
            if not (neuron_max_points[i] == test_generate_neurons_io_handler_2.points[j]).all():
                new_unmatched_points.append(j)
            else:
                matched[i] = j

        unmatched_points = new_unmatched_points

    assert (len(unmatched_points) == 0)


@nose.plugins.attrib.attr("DRMAA")
@nanshe.util.wrappers.with_setup_state(setup_2d, teardown_2d)
def test_generate_neurons_io_handler_3():
    # Attempt to import drmaa.
    # If it fails to import, either the user has no intent in using it or forgot to install it.
    # If it imports, but fails to find symbols, then the user has not set DRMAA_LIBRARY_PATH or does not have libdrmaa.so.
    try:
        import drmaa
    except ImportError:
        # python-drmaa is not installed.
        raise nose.SkipTest("Skipping test test_generate_neurons_io_handler_3. Was not able to import drmaa. To run this test, please pip or easy_install drmaa.")
    except RuntimeError:
        # The drmaa library was not specified, but python-drmaa is installed.
        raise nose.SkipTest("Skipping test test_generate_neurons_io_handler_3. Was able to import drmaa. However, the drmaa library could not be found. Please either specify the location of libdrmaa.so using the DRMAA_LIBRARY_PATH environment variable or disable/remove use_drmaa from the config file.")

    nanshe.learner.generate_neurons_io_handler(test_generate_neurons_io_handler_3.hdf5_input_filepath, test_generate_neurons_io_handler_3.hdf5_output_filepath, test_generate_neurons_io_handler_3.config_blocks_drmaa_filename)

    assert os.path.exists(test_generate_neurons_io_handler_3.hdf5_output_filename)

    with h5py.File(test_generate_neurons_io_handler_3.hdf5_output_filename, "r") as fid:
        assert ("neurons" in fid)

        neurons = fid["neurons"].value

    assert (len(test_generate_neurons_io_handler_3.points) == len(neurons))

    neuron_maxes = (neurons["image"] == nanshe.util.xnumpy.expand_view(neurons["max_F"], neurons["image"].shape[1:]))

    neuron_max_points = []
    for i in nanshe.util.iters.irange(len(neuron_maxes)):
        neuron_max_points.append(
            numpy.array(neuron_maxes[i].nonzero()).mean(axis=1).round().astype(int)
        )
    neuron_max_points = numpy.array(neuron_max_points)

    assert numpy.allclose(
        neurons["centroid"], neuron_max_points, rtol=0.0, atol=0.5
    )

    matched = dict()
    unmatched_points = numpy.arange(len(test_generate_neurons_io_handler_3.points))
    for i in nanshe.util.iters.irange(len(neuron_max_points)):
        new_unmatched_points = []
        for j in unmatched_points:
            if not (neuron_max_points[i] == test_generate_neurons_io_handler_3.points[j]).all():
                new_unmatched_points.append(j)
            else:
                matched[i] = j

        unmatched_points = new_unmatched_points

    assert (len(unmatched_points) == 0)


@nose.plugins.attrib.attr("3D")
@nanshe.util.wrappers.with_setup_state(setup_3d, teardown_3d)
def test_generate_neurons_io_handler_4():
    nanshe.learner.generate_neurons_io_handler(test_generate_neurons_io_handler_4.hdf5_input_3D_filepath, test_generate_neurons_io_handler_4.hdf5_output_3D_filepath, test_generate_neurons_io_handler_4.config_a_block_3D_filename)

    assert os.path.exists(test_generate_neurons_io_handler_4.hdf5_output_3D_filename)

    with h5py.File(test_generate_neurons_io_handler_4.hdf5_output_3D_filename, "r") as fid:
        assert ("neurons" in fid)

        neurons = fid["neurons"].value

    assert (len(test_generate_neurons_io_handler_4.points3) == len(neurons))

    neuron_maxes = (neurons["image"] == nanshe.util.xnumpy.expand_view(neurons["max_F"], neurons["image"].shape[1:]))

    neuron_max_points = []
    for i in nanshe.util.iters.irange(len(neuron_maxes)):
        neuron_max_points.append(
            numpy.array(neuron_maxes[i].nonzero()).mean(axis=1).round().astype(int)
        )
    neuron_max_points = numpy.array(neuron_max_points)

    assert numpy.allclose(
        neurons["centroid"], neuron_max_points, rtol=0.0, atol=2.0
    )

    matched = dict()
    unmatched_points = numpy.arange(len(test_generate_neurons_io_handler_4.points3))
    for i in nanshe.util.iters.irange(len(neuron_max_points)):
        new_unmatched_points = []
        for j in unmatched_points:
            if not (neuron_max_points[i] == test_generate_neurons_io_handler_4.points3[j]).all():
                new_unmatched_points.append(j)
            else:
                matched[i] = j

        unmatched_points = new_unmatched_points

    assert (len(unmatched_points) == 0)


@nose.plugins.attrib.attr("3D")
@nanshe.util.wrappers.with_setup_state(setup_3d, teardown_3d)
def test_generate_neurons_io_handler_5():
    nanshe.learner.generate_neurons_io_handler(test_generate_neurons_io_handler_5.hdf5_input_3D_filepath, test_generate_neurons_io_handler_5.hdf5_output_3D_filepath, test_generate_neurons_io_handler_5.config_blocks_3D_filename)

    assert os.path.exists(test_generate_neurons_io_handler_5.hdf5_output_3D_filename)

    with h5py.File(test_generate_neurons_io_handler_5.hdf5_output_3D_filename, "r") as fid:
        assert ("neurons" in fid)

        neurons = fid["neurons"].value

    assert (len(test_generate_neurons_io_handler_5.points3) == len(neurons))

    neuron_maxes = (neurons["image"] == nanshe.util.xnumpy.expand_view(neurons["max_F"], neurons["image"].shape[1:]))

    neuron_max_points = []
    for i in nanshe.util.iters.irange(len(neuron_maxes)):
        neuron_max_points.append(
            numpy.array(neuron_maxes[i].nonzero()).mean(axis=1).round().astype(int)
        )
    neuron_max_points = numpy.array(neuron_max_points)

    assert numpy.allclose(
        neurons["centroid"], neuron_max_points, rtol=0.0, atol=2.0
    )

    matched = dict()
    unmatched_points = numpy.arange(len(test_generate_neurons_io_handler_5.points3))
    for i in nanshe.util.iters.irange(len(neuron_max_points)):
        new_unmatched_points = []
        for j in unmatched_points:
            if not (neuron_max_points[i] == test_generate_neurons_io_handler_5.points3[j]).all():
                new_unmatched_points.append(j)
            else:
                matched[i] = j

        unmatched_points = new_unmatched_points

    assert (len(unmatched_points) == 0)


@nose.plugins.attrib.attr("3D", "DRMAA")
@nanshe.util.wrappers.with_setup_state(setup_3d, teardown_3d)
def test_generate_neurons_io_handler_6():
    # Attempt to import drmaa.
    # If it fails to import, either the user has no intent in using it or forgot to install it.
    # If it imports, but fails to find symbols, then the user has not set DRMAA_LIBRARY_PATH or does not have libdrmaa.so.
    try:
        import drmaa
    except ImportError:
        # python-drmaa is not installed.
        raise nose.SkipTest("Skipping test test_main_3. Was not able to import drmaa. To run this test, please pip or easy_install drmaa.")
    except RuntimeError:
        # The drmaa library was not specified, but python-drmaa is installed.
        raise nose.SkipTest("Skipping test test_main_3. Was able to import drmaa. However, the drmaa library could not be found. Please either specify the location of libdrmaa.so using the DRMAA_LIBRARY_PATH environment variable or disable/remove use_drmaa from the config file.")

    nanshe.learner.generate_neurons_io_handler(test_generate_neurons_io_handler_6.hdf5_input_3D_filepath, test_generate_neurons_io_handler_6.hdf5_output_3D_filepath, test_generate_neurons_io_handler_6.config_blocks_3D_drmaa_filename)

    assert os.path.exists(test_generate_neurons_io_handler_6.hdf5_output_3D_filename)

    with h5py.File(test_generate_neurons_io_handler_6.hdf5_output_3D_filename, "r") as fid:
        assert ("neurons" in fid)

        neurons = fid["neurons"].value

    assert (len(test_generate_neurons_io_handler_6.points3) == len(neurons))

    neuron_maxes = (neurons["image"] == nanshe.util.xnumpy.expand_view(neurons["max_F"], neurons["image"].shape[1:]))

    neuron_max_points = []
    for i in nanshe.util.iters.irange(len(neuron_maxes)):
        neuron_max_points.append(
            numpy.array(neuron_maxes[i].nonzero()).mean(axis=1).round().astype(int)
        )
    neuron_max_points = numpy.array(neuron_max_points)

    assert numpy.allclose(
        neurons["centroid"], neuron_max_points, rtol=0.0, atol=2.0
    )

    matched = dict()
    unmatched_points = numpy.arange(len(test_generate_neurons_io_handler_6.points3))
    for i in nanshe.util.iters.irange(len(neuron_max_points)):
        new_unmatched_points = []
        for j in unmatched_points:
            if not (neuron_max_points[i] == test_generate_neurons_io_handler_6.points3[j]).all():
                new_unmatched_points.append(j)
            else:
                matched[i] = j

        unmatched_points = new_unmatched_points

    assert (len(unmatched_points) == 0)


@nanshe.util.wrappers.with_setup_state(setup_2d, teardown_2d)
def test_generate_neurons_a_block_1():
    nanshe.learner.generate_neurons_a_block(test_generate_neurons_a_block_1.hdf5_input_filepath, test_generate_neurons_a_block_1.hdf5_output_filepath, **test_generate_neurons_a_block_1.config_a_block)

    assert os.path.exists(test_generate_neurons_a_block_1.hdf5_output_filename)

    with h5py.File(test_generate_neurons_a_block_1.hdf5_output_filename, "r") as fid:
        assert ("neurons" in fid)

        neurons = fid["neurons"].value

    assert (len(test_generate_neurons_a_block_1.points) == len(neurons))

    neuron_maxes = (neurons["image"] == nanshe.util.xnumpy.expand_view(neurons["max_F"], neurons["image"].shape[1:]))

    neuron_max_points = []
    for i in nanshe.util.iters.irange(len(neuron_maxes)):
        neuron_max_points.append(
            numpy.array(neuron_maxes[i].nonzero()).mean(axis=1).round().astype(int)
        )
    neuron_max_points = numpy.array(neuron_max_points)

    assert numpy.allclose(
        neurons["centroid"], neuron_max_points, rtol=0.0, atol=0.5
    )

    matched = dict()
    unmatched_points = numpy.arange(len(test_generate_neurons_a_block_1.points))
    for i in nanshe.util.iters.irange(len(neuron_max_points)):
        new_unmatched_points = []
        for j in unmatched_points:
            if not (neuron_max_points[i] == test_generate_neurons_a_block_1.points[j]).all():
                new_unmatched_points.append(j)
            else:
                matched[i] = j

        unmatched_points = new_unmatched_points

    assert (len(unmatched_points) == 0)


@nose.plugins.attrib.attr("3D")
@nanshe.util.wrappers.with_setup_state(setup_3d, teardown_3d)
def test_generate_neurons_a_block_2():
    nanshe.learner.generate_neurons_a_block(test_generate_neurons_a_block_2.hdf5_input_3D_filepath, test_generate_neurons_a_block_2.hdf5_output_3D_filepath, **test_generate_neurons_a_block_2.config_a_block_3D)

    assert os.path.exists(test_generate_neurons_a_block_2.hdf5_output_3D_filename)

    with h5py.File(test_generate_neurons_a_block_2.hdf5_output_3D_filename, "r") as fid:
        assert ("neurons" in fid)

        neurons = fid["neurons"].value

    assert (len(test_generate_neurons_a_block_2.points3) == len(neurons))

    neuron_maxes = (neurons["image"] == nanshe.util.xnumpy.expand_view(neurons["max_F"], neurons["image"].shape[1:]))

    neuron_max_points = []
    for i in nanshe.util.iters.irange(len(neuron_maxes)):
        neuron_max_points.append(
            numpy.array(neuron_maxes[i].nonzero()).mean(axis=1).round().astype(int)
        )
    neuron_max_points = numpy.array(neuron_max_points)

    assert numpy.allclose(
        neurons["centroid"], neuron_max_points, rtol=0.0, atol=2.0
    )

    matched = dict()
    unmatched_points = numpy.arange(len(test_generate_neurons_a_block_2.points3))
    for i in nanshe.util.iters.irange(len(neuron_max_points)):
        new_unmatched_points = []
        for j in unmatched_points:
            if not (neuron_max_points[i] == test_generate_neurons_a_block_2.points3[j]).all():
                new_unmatched_points.append(j)
            else:
                matched[i] = j

        unmatched_points = new_unmatched_points

    assert (len(unmatched_points) == 0)


@nanshe.util.wrappers.with_setup_state(setup_2d, teardown_2d)
def test_generate_neurons_blocks_1():
    nanshe.learner.generate_neurons_blocks(test_generate_neurons_blocks_1.hdf5_input_filepath, test_generate_neurons_blocks_1.hdf5_output_filepath, **test_generate_neurons_blocks_1.config_blocks["generate_neurons_blocks"])

    assert os.path.exists(test_generate_neurons_blocks_1.hdf5_output_filename)

    with h5py.File(test_generate_neurons_blocks_1.hdf5_output_filename, "r") as fid:
        assert ("neurons" in fid)

        neurons = fid["neurons"].value

    assert (len(test_generate_neurons_blocks_1.points) == len(neurons))

    neuron_maxes = (neurons["image"] == nanshe.util.xnumpy.expand_view(neurons["max_F"], neurons["image"].shape[1:]))

    neuron_max_points = []
    for i in nanshe.util.iters.irange(len(neuron_maxes)):
        neuron_max_points.append(
            numpy.array(neuron_maxes[i].nonzero()).mean(axis=1).round().astype(int)
        )
    neuron_max_points = numpy.array(neuron_max_points)

    assert numpy.allclose(
        neurons["centroid"], neuron_max_points, rtol=0.0, atol=0.5
    )

    matched = dict()
    unmatched_points = numpy.arange(len(test_generate_neurons_blocks_1.points))
    for i in nanshe.util.iters.irange(len(neuron_max_points)):
        new_unmatched_points = []
        for j in unmatched_points:
            if not (neuron_max_points[i] == test_generate_neurons_blocks_1.points[j]).all():
                new_unmatched_points.append(j)
            else:
                matched[i] = j

        unmatched_points = new_unmatched_points

    assert (len(unmatched_points) == 0)


@nose.plugins.attrib.attr("DRMAA")
@nanshe.util.wrappers.with_setup_state(setup_2d, teardown_2d)
def test_generate_neurons_blocks_2():
    # Attempt to import drmaa.
    # If it fails to import, either the user has no intent in using it or forgot to install it.
    # If it imports, but fails to find symbols, then the user has not set DRMAA_LIBRARY_PATH or does not have libdrmaa.so.
    try:
        import drmaa
    except ImportError:
        # python-drmaa is not installed.
        raise nose.SkipTest("Skipping test test_generate_neurons_blocks_2. Was not able to import drmaa. To run this test, please pip or easy_install drmaa.")
    except RuntimeError:
        # The drmaa library was not specified, but python-drmaa is installed.
        raise nose.SkipTest("Skipping test test_generate_neurons_blocks_2. Was able to import drmaa. However, the drmaa library could not be found. Please either specify the location of libdrmaa.so using the DRMAA_LIBRARY_PATH environment variable or disable/remove use_drmaa from the config file.")

    nanshe.learner.generate_neurons_blocks(test_generate_neurons_blocks_2.hdf5_input_filepath, test_generate_neurons_blocks_2.hdf5_output_filepath, **test_generate_neurons_blocks_2.config_blocks_drmaa["generate_neurons_blocks"])

    assert os.path.exists(test_generate_neurons_blocks_2.hdf5_output_filename)

    with h5py.File(test_generate_neurons_blocks_2.hdf5_output_filename, "r") as fid:
        assert ("neurons" in fid)

        neurons = fid["neurons"].value

    assert (len(test_generate_neurons_blocks_2.points) == len(neurons))

    neuron_maxes = (neurons["image"] == nanshe.util.xnumpy.expand_view(neurons["max_F"], neurons["image"].shape[1:]))

    neuron_max_points = []
    for i in nanshe.util.iters.irange(len(neuron_maxes)):
        neuron_max_points.append(
            numpy.array(neuron_maxes[i].nonzero()).mean(axis=1).round().astype(int)
        )
    neuron_max_points = numpy.array(neuron_max_points)

    assert numpy.allclose(
        neurons["centroid"], neuron_max_points, rtol=0.0, atol=0.5
    )

    matched = dict()
    unmatched_points = numpy.arange(len(test_generate_neurons_blocks_2.points))
    for i in nanshe.util.iters.irange(len(neuron_max_points)):
        new_unmatched_points = []
        for j in unmatched_points:
            if not (neuron_max_points[i] == test_generate_neurons_blocks_2.points[j]).all():
                new_unmatched_points.append(j)
            else:
                matched[i] = j

        unmatched_points = new_unmatched_points

    assert (len(unmatched_points) == 0)


@nose.plugins.attrib.attr("3D")
@nanshe.util.wrappers.with_setup_state(setup_3d, teardown_3d)
def test_generate_neurons_blocks_3():
    nanshe.learner.generate_neurons_blocks(test_generate_neurons_blocks_3.hdf5_input_3D_filepath, test_generate_neurons_blocks_3.hdf5_output_3D_filepath, **test_generate_neurons_blocks_3.config_blocks_3D["generate_neurons_blocks"])

    assert os.path.exists(test_generate_neurons_blocks_3.hdf5_output_3D_filename)

    with h5py.File(test_generate_neurons_blocks_3.hdf5_output_3D_filename, "r") as fid:
        assert ("neurons" in fid)

        neurons = fid["neurons"].value

    assert (len(test_generate_neurons_blocks_3.points3) == len(neurons))

    neuron_maxes = (neurons["image"] == nanshe.util.xnumpy.expand_view(neurons["max_F"], neurons["image"].shape[1:]))

    neuron_max_points = []
    for i in nanshe.util.iters.irange(len(neuron_maxes)):
        neuron_max_points.append(
            numpy.array(neuron_maxes[i].nonzero()).mean(axis=1).round().astype(int)
        )
    neuron_max_points = numpy.array(neuron_max_points)

    assert numpy.allclose(
        neurons["centroid"], neuron_max_points, rtol=0.0, atol=2.0
    )

    matched = dict()
    unmatched_points = numpy.arange(len(test_generate_neurons_blocks_3.points3))
    for i in nanshe.util.iters.irange(len(neuron_max_points)):
        new_unmatched_points = []
        for j in unmatched_points:
            if not (neuron_max_points[i] == test_generate_neurons_blocks_3.points3[j]).all():
                new_unmatched_points.append(j)
            else:
                matched[i] = j

        unmatched_points = new_unmatched_points

    assert (len(unmatched_points) == 0)


@nose.plugins.attrib.attr("3D", "DRMAA")
@nanshe.util.wrappers.with_setup_state(setup_3d, teardown_3d)
def test_generate_neurons_blocks_4():
    # Attempt to import drmaa.
    # If it fails to import, either the user has no intent in using it or forgot to install it.
    # If it imports, but fails to find symbols, then the user has not set DRMAA_LIBRARY_PATH or does not have libdrmaa.so.
    try:
        import drmaa
    except ImportError:
        # python-drmaa is not installed.
        raise nose.SkipTest("Skipping test test_generate_neurons_blocks_2. Was not able to import drmaa. To run this test, please pip or easy_install drmaa.")
    except RuntimeError:
        # The drmaa library was not specified, but python-drmaa is installed.
        raise nose.SkipTest("Skipping test test_generate_neurons_blocks_2. Was able to import drmaa. However, the drmaa library could not be found. Please either specify the location of libdrmaa.so using the DRMAA_LIBRARY_PATH environment variable or disable/remove use_drmaa from the config file.")

    nanshe.learner.generate_neurons_blocks(test_generate_neurons_blocks_4.hdf5_input_3D_filepath, test_generate_neurons_blocks_4.hdf5_output_3D_filepath, **test_generate_neurons_blocks_4.config_blocks_3D_drmaa["generate_neurons_blocks"])

    assert os.path.exists(test_generate_neurons_blocks_4.hdf5_output_3D_filename)

    with h5py.File(test_generate_neurons_blocks_4.hdf5_output_3D_filename, "r") as fid:
        assert ("neurons" in fid)

        neurons = fid["neurons"].value

    assert (len(test_generate_neurons_blocks_4.points3) == len(neurons))

    neuron_maxes = (neurons["image"] == nanshe.util.xnumpy.expand_view(neurons["max_F"], neurons["image"].shape[1:]))

    neuron_max_points = []
    for i in nanshe.util.iters.irange(len(neuron_maxes)):
        neuron_max_points.append(
            numpy.array(neuron_maxes[i].nonzero()).mean(axis=1).round().astype(int)
        )
    neuron_max_points = numpy.array(neuron_max_points)

    assert numpy.allclose(
        neurons["centroid"], neuron_max_points, rtol=0.0, atol=2.0
    )

    matched = dict()
    unmatched_points = numpy.arange(len(test_generate_neurons_blocks_4.points3))
    for i in nanshe.util.iters.irange(len(neuron_max_points)):
        new_unmatched_points = []
        for j in unmatched_points:
            if not (neuron_max_points[i] == test_generate_neurons_blocks_4.points3[j]).all():
                new_unmatched_points.append(j)
            else:
                matched[i] = j

        unmatched_points = new_unmatched_points

    assert (len(unmatched_points) == 0)


@nanshe.util.wrappers.with_setup_state(setup_2d, teardown_2d)
def test_generate_neurons_1():
    image_stack = None
    with h5py.File(test_generate_neurons_1.hdf5_input_filename, "r") as input_file_handle:
        image_stack = input_file_handle["images"][...]

    with h5py.File(test_generate_neurons_1.hdf5_output_filename, "a") as output_file_handle:
        output_group = output_file_handle["/"]

        # Get a debug logger for the HDF5 file (if needed)
        array_debug_recorder = nanshe.io.hdf5.record.generate_HDF5_array_recorder(
            output_group,
            group_name="debug",
            enable=test_generate_neurons_1.config_a_block["debug"],
            overwrite_group=False,
            recorder_constructor=nanshe.io.hdf5.record.HDF5EnumeratedArrayRecorder
        )

        # Saves intermediate result to make resuming easier
        resume_logger = nanshe.io.hdf5.record.generate_HDF5_array_recorder(
            output_group,
            recorder_constructor=nanshe.io.hdf5.record.HDF5ArrayRecorder,
            overwrite=True
        )

        nanshe.learner.generate_neurons.resume_logger = resume_logger
        nanshe.learner.generate_neurons.recorders.array_debug_recorder = array_debug_recorder
        nanshe.learner.generate_neurons(
            image_stack,
            **test_generate_neurons_1.config_a_block["generate_neurons"]
        )

    assert os.path.exists(test_generate_neurons_1.hdf5_output_filename)

    with h5py.File(test_generate_neurons_1.hdf5_output_filename, "r") as fid:
        assert ("neurons" in fid)

        neurons = fid["neurons"].value

    assert (len(test_generate_neurons_1.points) == len(neurons))

    neuron_maxes = (neurons["image"] == nanshe.util.xnumpy.expand_view(neurons["max_F"], neurons["image"].shape[1:]))

    neuron_max_points = []
    for i in nanshe.util.iters.irange(len(neuron_maxes)):
        neuron_max_points.append(
            numpy.array(neuron_maxes[i].nonzero()).mean(axis=1).round().astype(int)
        )
    neuron_max_points = numpy.array(neuron_max_points)

    assert numpy.allclose(
        neurons["centroid"], neuron_max_points, rtol=0.0, atol=0.5
    )

    matched = dict()
    unmatched_points = numpy.arange(len(test_generate_neurons_1.points))
    for i in nanshe.util.iters.irange(len(neuron_max_points)):
        new_unmatched_points = []
        for j in unmatched_points:
            if not (neuron_max_points[i] == test_generate_neurons_1.points[j]).all():
                new_unmatched_points.append(j)
            else:
                matched[i] = j

        unmatched_points = new_unmatched_points

    assert (len(unmatched_points) == 0)


@nose.plugins.attrib.attr("3D")
@nanshe.util.wrappers.with_setup_state(setup_3d, teardown_3d)
def test_generate_neurons_2():
    image_stack3 = None
    with h5py.File(test_generate_neurons_2.hdf5_input_3D_filename, "r") as input_file_handle:
        image_stack3 = input_file_handle["images"][...]

    with h5py.File(test_generate_neurons_2.hdf5_output_3D_filename, "a") as output_file_handle:
        output_group = output_file_handle["/"]

        # Get a debug logger for the HDF5 file (if needed)
        array_debug_recorder = nanshe.io.hdf5.record.generate_HDF5_array_recorder(
            output_group,
            group_name="debug",
            enable=test_generate_neurons_2.config_a_block_3D["debug"],
            overwrite_group=False,
            recorder_constructor=nanshe.io.hdf5.record.HDF5EnumeratedArrayRecorder
        )

        # Saves intermediate result to make resuming easier
        resume_logger = nanshe.io.hdf5.record.generate_HDF5_array_recorder(
            output_group,
            recorder_constructor=nanshe.io.hdf5.record.HDF5ArrayRecorder,
            overwrite=True
        )

        nanshe.learner.generate_neurons.resume_logger = resume_logger
        nanshe.learner.generate_neurons.recorders.array_debug_recorder = array_debug_recorder
        nanshe.learner.generate_neurons(
            image_stack3,
            **test_generate_neurons_2.config_a_block_3D["generate_neurons"]
        )

    assert os.path.exists(test_generate_neurons_2.hdf5_output_3D_filename)

    with h5py.File(test_generate_neurons_2.hdf5_output_3D_filename, "r") as fid:
        assert ("neurons" in fid)

        neurons = fid["neurons"].value

    assert (len(test_generate_neurons_2.points3) == len(neurons))

    neuron_maxes = (neurons["image"] == nanshe.util.xnumpy.expand_view(neurons["max_F"], neurons["image"].shape[1:]))

    neuron_max_points = []
    for i in nanshe.util.iters.irange(len(neuron_maxes)):
        neuron_max_points.append(
            numpy.array(neuron_maxes[i].nonzero()).mean(axis=1).round().astype(int)
        )
    neuron_max_points = numpy.array(neuron_max_points)

    assert numpy.allclose(
        neurons["centroid"], neuron_max_points, rtol=0.0, atol=2.0
    )

    matched = dict()
    unmatched_points = numpy.arange(len(test_generate_neurons_2.points3))
    for i in nanshe.util.iters.irange(len(neuron_max_points)):
        new_unmatched_points = []
        for j in unmatched_points:
            if not (neuron_max_points[i] == test_generate_neurons_2.points3[j]).all():
                new_unmatched_points.append(j)
            else:
                matched[i] = j

        unmatched_points = new_unmatched_points

    assert (len(unmatched_points) == 0)
