"""
The module ``learner`` provides the ability to run the segmentation algorithm.

===============================================================================
Overview
===============================================================================
The ``main`` function actually starts the algorithm and can
be called externally. The module allows for running multiple jobs through
|subprocess|_ or |drmaa|_. Configuration files for the learner are
provided in the examples_ and are entitled learner.

.. |subprocess| replace:: ``subprocess``
.. _subprocess: http://docs.python.org/2/library/subprocess.html
.. |drmaa| replace:: ``drmaa``
.. _drmaa: http://github.com/pygridtools/drmaa-python
.. _examples: http://github.com/nanshe-org/nanshe/tree/master/examples

===============================================================================
API
===============================================================================
"""


__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "$Apr 09, 2014 16:00:40 EDT$"


import os
import json
import itertools

try:
    import billiard as multiprocessing
    mp = multiprocessing.get_context("spawn")
except ImportError:
    import multiprocessing as mp

import subprocess
import time

# Generally useful and fast to import so done immediately.
import numpy

import h5py

import nanshe

# Need in order to have logging information no matter what.
from nanshe.util import prof

from nanshe.util import iters, xnumpy,\
    wrappers, pathHelpers

from nanshe.io import hdf5

# Short function to process image data.
from nanshe.imp import segment

# For IO. Right now, just includes read_parameters for reading a config file.
from nanshe.io import xjson


# Get the loggers
trace_logger = prof.getTraceLogger(__name__)
logger = prof.logging.getLogger(__name__)



@prof.log_call(trace_logger)
def generate_neurons_io_handler(input_filename,
                                output_filename,
                                parameters_filename):
    """
        Uses generate_neurons to process a input_filename (HDF5 dataset) and
        outputs results to an output_filename (HDF5 dataset). Also,

        Args:
            input_filename          HDF5 filename to read from (should be a
                                    path to a h5py.Dataset)

            output_filename         HDF5 filename to write to (should be a path
                                    to a h5py.Group)

            parameters_filename     JSON filename with parameters.
    """


    # Extract and validate file extensions.

    # Parse parameter filename and validate that the name is acceptable
    parameters_filename_details = pathHelpers.PathComponents(
        parameters_filename
    )
    # Clean up the extension so it fits the standard.
    if (parameters_filename_details.extension.lower().lstrip(os.extsep) not in ["json"]):
        raise Exception(
            "Parameter file with filename: \"" + parameters_filename + "\"" +
            " provided with an unknown file extension: \"" +
            parameters_filename_details.extension + "\". If it is a " +
            "supported format, please run the given file through " +
            "nanshe_converter first before proceeding."
        )

    # Parse the parameters from the json file.
    parameters = xjson.read_parameters(parameters_filename)

    if (len(parameters) == 1) and ("generate_neurons_blocks" in parameters):
        generate_neurons_blocks(
            input_filename,
            output_filename,
            **parameters["generate_neurons_blocks"]
        )
    else:
        generate_neurons_a_block(input_filename, output_filename, **parameters)


@prof.log_call(trace_logger)
def generate_neurons_a_block(input_filename,
                             output_filename,
                             debug=False,
                             **parameters):
    """
        Uses generate_neurons to process a input_filename (HDF5 dataset) and
        outputs results to an output_filename (HDF5 dataset).

        Args:
            input_filename          HDF5 filename to read from (should be a
                                    path to a h5py.Dataset)

            output_filename         HDF5 filename to write to (should be a path
                                    to a h5py.Group)

            parameters              how the run should be configured.
    """


    # Extract and validate file extensions.

    # Parse input filename and validate that the name is acceptable
    input_filename_details = pathHelpers.PathComponents(input_filename)
    # Clean up the extension so it fits the standard.
    if (input_filename_details.extension.lower().lstrip(os.extsep) not in ["h5", "hdf5", "he5"]):
        raise Exception(
            "Input file with filename: \"" + input_filename + "\"" +
            " provided with an unknown file extension: \"" +
            input_filename_details.extension + "\". If it is a supported " +
            "format, please run the given file through " +
            "nanshe_converter first before proceeding."
        )

    # Parse output filename and validate that the name is acceptable
    output_filename_details = pathHelpers.PathComponents(output_filename)
    # Clean up the extension so it fits the standard.
    if (output_filename_details.extension.lower().lstrip(os.extsep) not in ["h5", "hdf5", "he5"]):
        raise Exception(
            "Output file with filename: \"" + output_filename + "\"" +
            " provided with an unknown file extension: \"" +
            output_filename_details.extension + "\". If it is a supported " +
            "format, please run the given file through nanshe_converter " +
            "first before proceeding."
        )

    # Where the original images are.
    input_dataset_name = input_filename_details.internalPath

    # Name of the group where all data will be stored.
    output_group_name = output_filename_details.internalPath


    # Read the input data.
    original_images = None
    with h5py.File(input_filename_details.externalPath, "r") as input_file_handle:
        original_images = hdf5.serializers.read_numpy_structured_array_from_HDF5(
            input_file_handle, input_dataset_name)
        original_images = original_images.astype(numpy.float32)

    # Write out the output.
    with h5py.File(output_filename_details.externalPath, "a") as output_file_handle:
        # Create a new output directory if doesn't exists.
        output_file_handle.require_group(output_group_name)

        # Group where all data will be stored.
        output_group = output_file_handle[output_group_name]

        # Create a soft link to the original images. But use the appropriate type of soft link depending on whether
        # the input and output file are the same.
        if "original_images" not in output_group:
            if input_filename_details.externalPath == output_filename_details.externalPath:
                output_group["original_images"] = h5py.SoftLink(
                    input_dataset_name
                )
            else:
                output_group["original_images"] = h5py.ExternalLink(
                    input_filename_details.externalPath, input_dataset_name
                )

        # Get a debug logger for the HDF5 file (if needed)
        array_debug_recorder = hdf5.record.generate_HDF5_array_recorder(
            output_group,
            group_name="debug",
            enable=debug,
            overwrite_group=False,
            recorder_constructor=hdf5.record.HDF5EnumeratedArrayRecorder
        )

        # Saves intermediate result to make resuming easier
        resume_logger = hdf5.record.generate_HDF5_array_recorder(
            output_group,
            recorder_constructor=hdf5.record.HDF5ArrayRecorder,
            overwrite=True
        )

        # Generate the neurons and attempt to resume if possible
        generate_neurons.resume_logger = resume_logger
        generate_neurons.recorders.array_debug_recorder = array_debug_recorder
        generate_neurons(
            original_images=original_images, **parameters["generate_neurons"]
        )

        # Save the configuration parameters in the attributes as a string.
        if "parameters" not in output_group.attrs:
            # Write the configuration parameters in the attributes as a string.
            output_group.attrs["parameters"] = repr(
                dict(list(parameters.items()) + [("debug", debug)])
            )


@prof.log_call(trace_logger)
def generate_neurons_blocks(input_filename,
                            output_filename,
                            num_processes=mp.cpu_count(),
                            block_shape=None,
                            num_blocks=None,
                            half_window_shape=None,
                            half_border_shape=None,
                            use_drmaa=False,
                            num_drmaa_cores=16,
                            debug=False,
                            **parameters):
    # TODO: Move this function into a new module with its own command line interface.
    # TODO: Heavy refactoring required on this function.

    # Extract and validate file extensions.

    # Parse input filename and validate that the name is acceptable
    input_filename_details = pathHelpers.PathComponents(input_filename)
    # Clean up the extension so it fits the standard.
    if (input_filename_details.extension.lower().lstrip(os.extsep) not in ["h5", "hdf5", "he5"]):
        raise Exception(
            "Input file with filename: \"" + input_filename + "\"" +
            " provided with an unknown file extension: \"" +
            input_filename_details.extension + "\". If it is a supported " +
            "format, please run the given file through nanshe_converter " +
            "first before proceeding."
        )

    # Parse output filename and validate that the name is acceptable
    output_filename_details = pathHelpers.PathComponents(output_filename)
    # Clean up the extension so it fits the standard.
    if (output_filename_details.extension.lower().lstrip(os.extsep) not in ["h5", "hdf5", "he5"]):
        raise Exception(
            "Output file with filename: \"" + output_filename + "\"" +
            " provided with an unknown file extension: \"" +
            output_filename_details.extension + "\". If it is a supported " +
            "format, please run the given file through nanshe_converter " +
            "first before proceeding."
        )

    # Where the original images are.
    input_dataset_name = input_filename_details.internalPath

    # Name of the group where all data will be stored.
    output_group_name = output_filename_details.internalPath

    # Directory where individual block runs will be stored.
    intermediate_output_dir = output_filename_details.externalPath.rsplit(
        output_filename_details.extension, 1)[0] + "_blocks"


    # Read the input data.
    original_images_shape_array = None
    with h5py.File(input_filename_details.externalPath, "r") as input_file_handle:
        original_images_shape_array = numpy.array(
            input_file_handle[input_dataset_name].shape
        )

    # Get the amount of the border to slice
    half_border_shape_array = None
    if half_border_shape is None:
        half_border_shape_array = numpy.zeros(
            len(original_images_shape_array), dtype=int
        )
    else:
        assert (len(half_window_shape) == len(original_images_shape_array))

        half_border_shape_array = numpy.array(half_border_shape)

        # Should be of type integer
        assert (issubclass(half_border_shape_array.dtype.type, numpy.integer))

        # Should not cut along temporal portion.
        # Maybe replace with a warning.
        assert (half_border_shape[0] == 0)

    # TODO: Refactor to expanded_numpy.
    # Cuts boundaries from original_images_shape
    original_images_pared_shape_array = original_images_shape_array - \
                                        2*half_border_shape_array

    # At least one of them must be specified. If not some mixture of both.
    assert ((block_shape is not None) or (num_blocks is not None))

    # Size of the block to use by pixels
    block_shape_array = None
    block_shape_array_undefined = None
    if block_shape is None:
        block_shape_array = -numpy.ones(
            original_images_pared_shape_array.shape, dtype=int
        )
        block_shape_array_undefined = numpy.ones(
            original_images_pared_shape_array.shape, dtype=bool
        )
    else:
        # Should have the same number of values in each
        assert (len(original_images_pared_shape_array) == len(block_shape))

        block_shape_array = numpy.array(block_shape, dtype=int)

        # Should be of type integer
        assert issubclass(block_shape_array.dtype.type, numpy.integer)

        block_shape_array_undefined = (block_shape_array == -1)

    # Number of
    num_blocks_array = None
    num_blocks_array_undefined = None
    if num_blocks is None:
        num_blocks_array = - \
            numpy.ones(original_images_pared_shape_array.shape, dtype=int)
        num_blocks_array_undefined = numpy.ones(
            original_images_pared_shape_array.shape, dtype=bool)
    else:
        # Should have the same number of values in each
        assert (len(original_images_pared_shape_array) == len(num_blocks))

        num_blocks_array = numpy.array(num_blocks, dtype=int)

        # Should be of type integer
        assert issubclass(num_blocks_array.dtype.type, numpy.integer)

        num_blocks_array_undefined = (num_blocks_array == -1)

    # Want to ensure that both aren't defined.
    assert ~(~block_shape_array_undefined & ~num_blocks_array_undefined).all()

    # If both are undefined, then the block should span that dimension
    missing_both = (block_shape_array_undefined & num_blocks_array_undefined)
    block_shape_array[
        missing_both] = original_images_pared_shape_array[missing_both]
    num_blocks_array[missing_both] = 1
    # Thus, we have resolved these values and can continue.
    block_shape_array_undefined[missing_both] = False
    num_blocks_array_undefined[missing_both] = False

    # Replace undefined values in block_shape_array
    missing_block_shape_array, block_shape_array_remainder = divmod(
        original_images_pared_shape_array[block_shape_array_undefined],
        num_blocks_array[block_shape_array_undefined]
    )
    # Block shape must be well defined.
    assert (block_shape_array_remainder == 0).all()
    missing_block_shape_array = missing_block_shape_array.astype(int)
    block_shape_array[block_shape_array_undefined] = missing_block_shape_array

    # Replace undefined values in num_blocks_array
    missing_num_blocks_array, num_blocks_array_remainder = divmod(
        original_images_pared_shape_array[num_blocks_array_undefined],
        block_shape_array[num_blocks_array_undefined]
    )
    # Allow some blocks to be smaller
    missing_num_blocks_array += (num_blocks_array_remainder != 0).astype(int)
    num_blocks_array[num_blocks_array_undefined] = missing_num_blocks_array
    # Get the overlap window
    half_window_shape_array = None
    if half_window_shape is None:
        half_window_shape_array = block_shape_array / 2.0
    else:
        assert (len(half_window_shape) == len(
            original_images_pared_shape_array))

        half_window_shape_array = numpy.array(half_window_shape)

        assert issubclass(half_window_shape_array.dtype.type, numpy.integer)

    # Want to make our window size is at least as large as the one used for
    # the f0 calculation.
    if "extract_f0" in parameters["generate_neurons"]["preprocess_data"]:
        #assert (parameters["generate_neurons"]["preprocess_data"]["extract_f0"]["half_window_size"] == half_window_shape_array[0])
        assert (parameters["generate_neurons"]["preprocess_data"][
                "extract_f0"]["half_window_size"] <= half_window_shape_array[0])

    # Estimate bounds for each slice. Uses typical python [begin, end) for the
    # indices.
    estimated_bounds = numpy.zeros(
        tuple(num_blocks_array),
        dtype=(int, original_images_pared_shape_array.shape + (2,))
    )

    for each_block_indices in iters.index_generator(*num_blocks_array):
        for each_dim, each_block_dim_index in enumerate(each_block_indices):
            estimated_lower_bound = each_block_dim_index * block_shape_array[each_dim]
            estimated_upper_bound = (each_block_dim_index + 1) * block_shape_array[each_dim]

            estimated_bounds[each_block_indices][each_dim] = numpy.array([
                estimated_lower_bound, estimated_upper_bound
            ])

    original_images_pared_slices = numpy.zeros(
        estimated_bounds.shape[:-2],
        dtype=[("actual", int, estimated_bounds.shape[-2:]),
               ("windowed", int, estimated_bounds.shape[-2:]),
               ("windowed_stack_selection", int, estimated_bounds.shape[-2:]),
               ("windowed_block_selection", int, estimated_bounds.shape[-2:])])

    # Get the slice that is within bounds
    original_images_pared_slices["actual"] = estimated_bounds
    original_images_pared_slices["actual"][..., 0] = numpy.where(
        0 < original_images_pared_slices["actual"][..., 0],
        original_images_pared_slices["actual"][..., 0],
        0
    )
    original_images_pared_slices["actual"][..., 1] = numpy.where(
        original_images_pared_slices["actual"][..., 1] < original_images_pared_shape_array,
        original_images_pared_slices["actual"][..., 1],
        original_images_pared_shape_array
    )

    # Gets the defined half_window_size.
    window_addition = numpy.zeros(estimated_bounds.shape, dtype=int)
    window_addition[..., 0] = -half_window_shape_array
    window_addition[..., 1] = half_window_shape_array

    # Get the slice with a window added.
    original_images_pared_slices[
        "windowed"] = estimated_bounds + window_addition
    original_images_pared_slices["windowed"][..., 0] = numpy.where(
        0 < original_images_pared_slices["windowed"][..., 0],
        original_images_pared_slices["windowed"][..., 0],
        0
    )
    original_images_pared_slices["windowed"][..., 1] = numpy.where(
        original_images_pared_slices["windowed"][..., 1] < original_images_pared_shape_array,
        original_images_pared_slices["windowed"][..., 1],
        original_images_pared_shape_array
    )

    # Get the slice information to get the windowed block from the original
    # image stack.
    original_images_pared_slices["windowed_stack_selection"] = original_images_pared_slices["windowed"]
    original_images_pared_slices["windowed_stack_selection"] += xnumpy.expand_view(
        half_border_shape_array, reps_after=2
    )

    # Get slice information for the portion within original_images_pared_slices["windowed"],
    # which corresponds to original_images_pared_slices["actual"]
    #original_images_pared_slices["windowed_block_selection"][..., 0] = 0
    original_images_pared_slices["windowed_block_selection"][..., 1] = (
        original_images_pared_slices["actual"][..., 1] - original_images_pared_slices["actual"][..., 0]
    )
    original_images_pared_slices["windowed_block_selection"][:] += xnumpy.expand_view(
        original_images_pared_slices["actual"][..., 0] - original_images_pared_slices["windowed"][..., 0],
        reps_after=2
    )

    # Get a directory for intermediate results.
    try:
        os.mkdir(intermediate_output_dir)
    except OSError:
        # If it already exists, that is fine.
        pass

    intermediate_config = intermediate_output_dir + "/" + "config.json"

    # Overwrite the config file always
    with open(intermediate_config, "w") as fid:
        json.dump(
            dict(list(parameters.items()) + list({"debug" : debug}.items())),
            fid,
            indent=4,
            separators=(",", " : ")
        )
        fid.write("\n")

    # Construct an HDF5 file for each block
    input_filename_block = []
    output_filename_block = []
    stdout_filename_block = []
    stderr_filename_block = []
    with h5py.File(output_filename_details.externalPath, "a") as output_file_handle:
        # Create a new output directory if doesn't exists.
        output_file_handle.require_group(output_group_name)

        output_group = output_file_handle[output_group_name]

        if "original_images" not in output_group:
            if input_filename_details.externalPath == output_filename_details.externalPath:
                output_group["original_images"] = h5py.SoftLink(
                    input_dataset_name
                )
            else:
                output_group["original_images"] = h5py.ExternalLink(
                    input_filename_details.externalPath,
                    "/" + input_dataset_name
                )

        output_group.require_group("blocks")

        output_group_blocks = output_group["blocks"]

        input_file_handle = None
        try:
            # Skipping using region refs.
            input_file_handle = h5py.File(
                input_filename_details.externalPath, "r"
            )
        except IOError:
            # File is already open
            input_file_handle = output_file_handle

        for i, i_str, sequential_block_i in iters.filled_stringify_enumerate(
                original_images_pared_slices.flat
        ):
            intermediate_basename_i = intermediate_output_dir + "/" + i_str

            # Hold redirected stdout and stderr for each subprocess.
            stdout_filename_block.append(
                intermediate_basename_i + os.extsep + "out")
            stderr_filename_block.append(
                intermediate_basename_i + os.extsep + "err")

            # Ensure that the blocks are corrected to deal with trimming of the image stack
            # Must be done after the calculation of
            # original_images_pared_slices["windowed_block_selection"].
            sequential_block_i_windowed = sequential_block_i["windowed_stack_selection"]
            slice_i = tuple(
                slice(_1, _2, 1) for _1, _2 in sequential_block_i_windowed
            )

            if i_str not in output_group_blocks:
                output_group_blocks[i_str] = []
                output_group_blocks[i_str].attrs["filename"] = input_file_handle.filename
                output_group_blocks[i_str].attrs["dataset"] = input_dataset_name
                output_group_blocks[i_str].attrs["slice"] = str(slice_i)

            block_i = output_group_blocks[i_str]

            with h5py.File(intermediate_basename_i + os.extsep + "h5", "a") as each_block_file_handle:
                # Create a soft link to the original images. But use the appropriate type of soft link depending on whether
                # the input and output file are the same.
                if "original_images" not in each_block_file_handle:
                    each_block_file_handle["original_images"] = h5py.ExternalLink(
                        os.path.relpath(
                            block_i.file.filename, intermediate_output_dir
                        ),
                        block_i.name
                    )

                input_filename_block.append(
                    each_block_file_handle.filename + "/" + "original_images"
                )
                output_filename_block.append(
                    each_block_file_handle.filename + "/"
                )

        if input_file_handle != output_file_handle:
            input_file_handle.close()

    cur_module_dirpath = os.path.dirname(os.path.dirname(nanshe.__file__))
    cur_module_filepath = os.path.splitext(os.path.abspath(__file__))[0]
    cur_module_name = os.path.relpath(cur_module_filepath, cur_module_dirpath)
    cur_module_name = cur_module_name.replace(os.path.sep, ".")
    cur_module_filepath += os.extsep + "py"

    import sys

    python = sys.executable

    executable_run = ""
    executable_run += "from sys import argv, path, exit; "

    executable_run += "path[:] = [\"%s\"] + [_ for _ in path if _ != \"%s\"]; " % \
                      (cur_module_dirpath, cur_module_dirpath,)
    executable_run += "from %s import main; exit(main(*argv))" % \
                      (cur_module_name,)

    block_process_args_gen = iters.izip(
        itertools.repeat(python),
        itertools.repeat("-c"),
        itertools.repeat(executable_run),
        itertools.repeat(intermediate_config),
        input_filename_block,
        output_filename_block,
        stdout_filename_block,
        stderr_filename_block
    )

    if use_drmaa:
        # Attempt to import drmaa.
        # If it fails to import, either the user has no intent in using it or
        # forgot to install it. If it imports, but fails to find symbols,
        # then the user has not set DRMAA_LIBRARY_PATH or
        # does not have libdrmaa.so.
        try:
            import drmaa
        except ImportError:
            # python-drmaa is not installed.
            logger.error(
                "Was not able to import drmaa. " +
                "If this is meant to be run using the OpenGrid submission " +
                "system, then drmaa needs to be installed via pip or " +
                "easy_install."
            )
            raise
        except RuntimeError:
            # The drmaa library was not specified, but python-drmaa is
            # installed.
            logger.error(
                "Was able to import drmaa. " +
                "However, the drmaa library could not be found. Please " +
                "either specify the location of libdrmaa.so using the " +
                "DRMAA_LIBRARY_PATH environment variable or disable/remove " +
                "use_drmaa from the config file."
            )
            raise

        s=drmaa.Session()
        s.initialize()

        ready_processes = []
        for each_arg_pack in block_process_args_gen:
            ready_processes.append((each_arg_pack, s.createJobTemplate()))
            ready_processes[-1][1].jobName = os.path.basename(
                os.path.splitext(cur_module_filepath)[0]
            ) + "-" + os.path.basename(
                os.path.dirname(each_arg_pack[3].split(".h5")[0])
            ) + "-" + os.path.basename(each_arg_pack[3].split(".h5")[0])
            ready_processes[-1][1].remoteCommand = each_arg_pack[0]
            ready_processes[-1][1].args = each_arg_pack[1:-2]
            ready_processes[-1][1].jobEnvironment = os.environ
            ready_processes[-1][1].inputPath = "localhost:" + os.devnull
            ready_processes[-1][1].outputPath = "localhost:" + each_arg_pack[-2]
            ready_processes[-1][1].errorPath = "localhost:" + each_arg_pack[-1]
            ready_processes[-1][1].workingDirectory = os.getcwd()
            ready_processes[-1][1].nativeSpecification = "-pe batch " + str(num_drmaa_cores)


        running_processes = []
        for each_arg_pack, each_process_template in ready_processes:
            each_process_id = s.runJob(each_process_template)
            running_processes.append(
                (each_arg_pack, each_process_id, each_process_template)
            )
            logger.info(
                "Started new process ( \"" + " ".join(each_arg_pack) + "\" )."
            )

        start_queue_time = time.time()
        logger.info("Waiting for queued jobs to complete.")

        #finished_processes = []
        for each_arg_pack, each_process_id, each_process_template in running_processes:
            each_process_status = s.wait(each_process_id)

            if not each_process_status.hasExited:
                raise RuntimeError(
                    "The process (\"" + " ".join(each_arg_pack) +
                    "\") has exited prematurely."
                )

            logger.info(
                "Finished process ( \"" + " ".join(each_arg_pack) + "\" )."
            )
            s.deleteJobTemplate(each_process_template)
            #finished_processes.append((each_arg_pack, each_process_id))

        s.exit()

        end_queue_time = time.time()
        diff_queue_time = end_queue_time - start_queue_time

        logger.info(
            "Run time for queued jobs to complete is \""
            + str(diff_queue_time) + " s\"."
        )
    else:
        # TODO: Refactor into a separate class (have it return futures somehow)
        #finished_processes = []
        running_processes = []
        pool_tasks_empty = False
        while (not pool_tasks_empty) or len(running_processes):
            while (not pool_tasks_empty) and (len(running_processes) < num_processes):
                try:
                    each_arg_pack = next(block_process_args_gen)
                    each_arg_pack, each_stdout_filename, each_stderr_filename = each_arg_pack[:-2], each_arg_pack[-2], each_arg_pack[-1]
                    each_process = subprocess.Popen(
                        each_arg_pack,
                        stdout=open(each_stdout_filename, "w"),
                        stderr=open(each_stderr_filename, "w")
                    )

                    running_processes.append((each_arg_pack, each_process,))

                    logger.info(
                        "Started new process ( \"" + " ".join(each_arg_pack) + "\" )."
                    )
                except StopIteration:
                    pool_tasks_empty = True

            while ((not pool_tasks_empty) and
                       (len(running_processes) >= num_processes)) or \
                    (pool_tasks_empty and len(running_processes)):
                time.sleep(1)

                i = 0
                while i < len(running_processes):
                    if running_processes[i][1].poll() is not None:
                        logger.info(
                            "Finished process ( \"" +
                            " ".join(running_processes[i][0]) + "\" )."
                        )

                        #finished_processes.append(running_processes[i])
                        del running_processes[i]
                    else:
                        time.sleep(1)
                        i += 1

        # finished_processes = None

    start_time = time.time()
    logger.info("Starting merge over all blocks.")

    with h5py.File(output_filename_details.externalPath, "a") as output_file_handle:
        output_group = output_file_handle[output_group_name]

        new_neurons_set = segment.get_empty_neuron(
            shape=tuple(original_images_shape_array[1:]), dtype=float
        )

        for i, i_str, (output_filename_block_i, sequential_block_i) in iters.filled_stringify_enumerate(
                iters.izip(output_filename_block, original_images_pared_slices.flat)):
            windowed_slice_i = tuple(
                slice(_1, _2, 1) for _1, _2 in [(None, None)] + sequential_block_i["windowed_stack_selection"].tolist()[1:]
            )
            window_trimmed_i = tuple(
                slice(_1, _2, 1) for _1, _2 in sequential_block_i["windowed_block_selection"].tolist()
            )
            output_filename_block_i = output_filename_block_i.rstrip("/")

            with h5py.File(output_filename_block_i, "r") as each_block_file_handle:
                if "neurons" in each_block_file_handle:
                    neurons_block_i_smaller = hdf5.serializers.read_numpy_structured_array_from_HDF5(
                        each_block_file_handle, "/neurons"
                    )

                    neurons_block_i_windowed_count = numpy.squeeze(
                        numpy.apply_over_axes(
                            numpy.sum,
                            neurons_block_i_smaller["mask"].astype(float),
                            tuple(iters.irange(1, neurons_block_i_smaller["mask"].ndim))
                        )
                    )

                    if neurons_block_i_windowed_count.shape == tuple():
                        neurons_block_i_windowed_count = numpy.array(
                            [neurons_block_i_windowed_count])

                    neurons_block_i_non_windowed_count = numpy.squeeze(
                        numpy.apply_over_axes(
                            numpy.sum,
                            neurons_block_i_smaller["mask"][window_trimmed_i].astype(float),
                            tuple(iters.irange(1, neurons_block_i_smaller["mask"].ndim))
                        )
                    )

                    if neurons_block_i_non_windowed_count.shape == tuple():
                        neurons_block_i_non_windowed_count = numpy.array(
                            [neurons_block_i_non_windowed_count]
                        )

                    if len(neurons_block_i_non_windowed_count):
                        # Find ones that are inside the margins by more than
                        # half
                        neurons_block_i_acceptance = (
                            (neurons_block_i_non_windowed_count / neurons_block_i_windowed_count) > 0.5
                        )

                        logger.info(
                            "Accepted the following neurons %s from block %s."
                            % (
                                str(neurons_block_i_acceptance.nonzero()[0].tolist()),
                                i_str
                            )
                        )

                        # Take a subset of our previous neurons that are within
                        # the margins by half
                        neurons_block_i_accepted = neurons_block_i_smaller[neurons_block_i_acceptance]

                        neurons_block_i = numpy.zeros(
                            neurons_block_i_accepted.shape, dtype=new_neurons_set.dtype
                        )
                        neurons_block_i["mask"][windowed_slice_i] = neurons_block_i_accepted["mask"]
                        neurons_block_i["contour"][windowed_slice_i] = neurons_block_i_accepted["contour"]
                        neurons_block_i["image"][windowed_slice_i] = neurons_block_i_accepted["image"]

                        # Copy other properties
                        neurons_block_i["area"] = neurons_block_i_accepted["area"]
                        neurons_block_i["max_F"] = neurons_block_i_accepted["max_F"]
                        neurons_block_i["gaussian_mean"] = neurons_block_i_accepted["gaussian_mean"]
                        neurons_block_i["gaussian_cov"] = neurons_block_i_accepted["gaussian_cov"]
                        # TODO: Correct centroid to larger block position.
                        neurons_block_i["centroid"] = neurons_block_i_accepted["centroid"]
                        neurons_block_i["centroid"] += sequential_block_i["windowed_stack_selection"][1:, 0]

                        array_debug_recorder = hdf5.record.generate_HDF5_array_recorder(
                            output_group,
                            group_name="debug",
                            enable=debug,
                            overwrite_group=False,
                            recorder_constructor=hdf5.record.HDF5EnumeratedArrayRecorder
                        )

                        segment.merge_neuron_sets.recorders.array_debug_recorder = array_debug_recorder
                        new_neurons_set = segment.merge_neuron_sets(
                            new_neurons_set,
                            neurons_block_i,
                            **parameters["generate_neurons"]["postprocess_data"]["merge_neuron_sets"]
                        )
                    else:
                        logger.info(
                            "Accepted the following neurons %s from block %s." %
                            (
                                str([]),
                                i_str
                            )
                        )
                else:
                    logger.info(
                        "No neurons accepted as none were found for block"
                        " %s." %
                        i_str
                    )

        hdf5.serializers.create_numpy_structured_array_in_HDF5(
            output_group, "neurons", new_neurons_set, overwrite=True)

        if "parameters" not in output_group["neurons"].attrs:
            output_group["neurons"].attrs["parameters"] = repr(dict(
                list(parameters.items()) +
                [("block_shape", block_shape),
                 ("num_blocks", num_blocks),
                 ("half_window_shape", half_window_shape),
                 ("half_border_shape", half_border_shape),
                 ("use_drmaa", use_drmaa),
                 ("num_drmaa_cores", num_drmaa_cores),
                 ("debug", debug)]
            ))

    logger.info("Finished merge over all blocks.")
    end_time = time.time()

    diff_time = end_time - start_time

    logger.info(
        "Run time for merge over all blocks is \"" + str(diff_time) + " s\"."
    )


@prof.log_call(trace_logger)
@hdf5.record.static_subgrouping_array_recorders(array_debug_recorder=hdf5.record.EmptyArrayRecorder())
@wrappers.static_variables(resume_logger=hdf5.record.EmptyArrayRecorder())
def generate_neurons(original_images, run_stage="all", **parameters):
    if "original_images_max_projection" not in generate_neurons.recorders.array_debug_recorder:
        generate_neurons.recorders.array_debug_recorder["original_images_max_projection"] = xnumpy.add_singleton_op(
            numpy.max,
            original_images,
            axis=0
        )

    if "original_images_mean_projection" not in generate_neurons.recorders.array_debug_recorder:
        generate_neurons.recorders.array_debug_recorder["original_images_mean_projection"] = xnumpy.add_singleton_op(
            numpy.mean,
            original_images,
            axis=0
        )

    # Preprocess images
    new_preprocessed_images = generate_neurons.resume_logger.get(
        "preprocessed_images", None
    )
    if (new_preprocessed_images is None) or \
            (run_stage == "preprocessing") or \
            (run_stage == "all"):
        new_preprocessed_images = original_images.copy()
        segment.preprocess_data.recorders.array_debug_recorder = generate_neurons.recorders.array_debug_recorder
        new_preprocessed_images = segment.preprocess_data(
            new_preprocessed_images,
            out=new_preprocessed_images,
            **parameters["preprocess_data"]
        )
        generate_neurons.resume_logger["preprocessed_images"] = new_preprocessed_images

        if "preprocessed_images_max_projection" not in generate_neurons.recorders.array_debug_recorder:
            generate_neurons.recorders.array_debug_recorder["preprocessed_images_max_projection"] = xnumpy.add_singleton_op(
            numpy.max,
            new_preprocessed_images,
            axis=0
        )

    if run_stage == "preprocessing":
        return

    # Find the dictionary
    new_dictionary = generate_neurons.resume_logger.get("dictionary", None)
    if (new_dictionary is None) or \
            (run_stage == "dictionary") or \
            (run_stage == "all"):
        segment.generate_dictionary.recorders.array_debug_recorder = generate_neurons.recorders.array_debug_recorder
        new_dictionary = segment.generate_dictionary(
            new_preprocessed_images,
            **parameters["generate_dictionary"]
        )
        generate_neurons.resume_logger["dictionary"] = new_dictionary

        if "dictionary_max_projection" not in generate_neurons.recorders.array_debug_recorder:
            generate_neurons.recorders.array_debug_recorder["dictionary_max_projection"] = xnumpy.add_singleton_op(
            numpy.max,
            new_dictionary,
            axis=0
        )

    if run_stage == "dictionary":
        return

    # Find the neurons
    new_neurons = None
    new_neurons = generate_neurons.resume_logger.get("neurons", None)
    if (new_neurons is None) or \
            (run_stage == "postprocessing") or \
            (run_stage == "all"):
        segment.postprocess_data.recorders.array_debug_recorder = generate_neurons.recorders.array_debug_recorder
        new_neurons = segment.postprocess_data(
            new_dictionary,
            **parameters["postprocess_data"]
        )

        if new_neurons.size:
            generate_neurons.resume_logger["neurons"] = new_neurons

    if new_neurons.size == 0:
        logger.warning("No neurons were found in the data.")
    else:
        logger.info(
            "Found \"" + str(len(new_neurons)) +
            "\" neurons were found in the data."
        )


@prof.log_call(trace_logger)
def main(*argv):
    """
        Simple main function (like in C). Takes all arguments (as from
        sys.argv) and returns an exit status.

        Args:
            argv(list):     arguments (includes command line call).

        Returns:
            int:            exit code (0 if success)
    """

    # Only necessary if running main (normally if calling command line).
    # No point in importing otherwise.
    import argparse
    import threading

    memory_profiler_thread = threading.Thread(
        target=prof.memory_profiler, args=(logger,)
    )
    memory_profiler_thread.daemon = True
    memory_profiler_thread.start()

    argv = list(argv)

    # Creates command line parser
    parser = argparse.ArgumentParser(
        description="Parses input from the command line for a batch job."
    )

    # Takes a config file and then a series of one or more HDF5 files.
    parser.add_argument(
        "config_filename",
        metavar="CONFIG_FILE",
        type=str,
        help="JSON file that provides configuration options " +
             "for how to use dictionary learning on the input files."
    )
    parser.add_argument(
        "input_file",
        metavar="INPUT_FILE",
        type=str,
        nargs=1,
        help="HDF5 file with an array of images. " +
             "A dataset or video will be expected at the internal path. " +
             "Time must be the first dimension."
    )
    parser.add_argument(
        "output_file",
        metavar="OUTPUT_FILE",
        type=str,
        nargs=1,
        help="HDF5 file(s) to write output. " +
             "If a specific group is desired, " +
             "that should be included in the filename."
    )

    # Results of parsing arguments
    # (ignore the first one as it is the command line call).
    parsed_args = parser.parse_args(argv[1:])

    # Remove args from singleton lists
    parsed_args.input_file = parsed_args.input_file[0]
    parsed_args.output_file = parsed_args.output_file[0]

    # Runs the dictionary learning algorithm on each file with
    # the given parameters and saves the results in the given files.
    generate_neurons_io_handler(
        parsed_args.input_file,
        parsed_args.output_file,
        parsed_args.config_filename
    )

    return(0)
