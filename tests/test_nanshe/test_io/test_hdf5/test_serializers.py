__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "$Jul 28, 2014 11:50:37 EDT$"


import os
import shutil
import tempfile

import nose

import numpy
import h5py

import nanshe.io.hdf5.serializers
import nanshe.io.disk


class TestSerializers(object):
    def setup(self):
        self.temp_dir = tempfile.mkdtemp()

        self.temp_hdf5_file = h5py.File(os.path.join(self.temp_dir, "test.h5"), "w")

    def test_split_hdf5_path_0a(self):
        try:
            ext_path, int_path = nanshe.io.hdf5.serializers.split_hdf5_path("test.h5")
        except AssertionError:
            assert True
        else:
            assert False

    def test_split_hdf5_path_0b(self):
        try:
            ext_path, int_path = nanshe.io.hdf5.serializers.split_hdf5_path("test.eg/")
        except ValueError:
            assert True
        else:
            assert False

    def test_split_hdf5_path_1a(self):
        ext_path, int_path = nanshe.io.hdf5.serializers.split_hdf5_path("test.h5/")
        assert ext_path == "test.h5"
        assert int_path == "/"

    def test_split_hdf5_path_1b(self):
        ext_path, int_path = nanshe.io.hdf5.serializers.split_hdf5_path("test.hdf5/")
        assert ext_path == "test.hdf5"
        assert int_path == "/"

    def test_split_hdf5_path_1c(self):
        ext_path, int_path = nanshe.io.hdf5.serializers.split_hdf5_path("test.he5/")
        assert ext_path == "test.he5"
        assert int_path == "/"

    def test_split_hdf5_path_2(self):
        ext_path, int_path = nanshe.io.hdf5.serializers.split_hdf5_path("test.h5/a/b/c")
        assert ext_path == "test.h5"
        assert int_path == "/a/b/c"

    def test_split_hdf5_path_3(self):
        ext_path, int_path = nanshe.io.hdf5.serializers.split_hdf5_path("my_dir/test.h5/a/b/c")
        assert ext_path == "my_dir/test.h5"
        assert int_path == "/a/b/c"

    def test_create_numpy_structured_array_in_HDF5_1(self):
        data1 = numpy.random.random((10, 10))
        data2 = numpy.random.random((10, 10))

        while (data1 == data2).all():
            data2 = numpy.random.random((10, 10))

        nanshe.io.hdf5.serializers.create_numpy_structured_array_in_HDF5(self.temp_hdf5_file, "data", data1)

        assert ("data" in self.temp_hdf5_file)
        assert (data1 == self.temp_hdf5_file["data"].value).all()

        try:
            nanshe.io.hdf5.serializers.create_numpy_structured_array_in_HDF5(self.temp_hdf5_file, "data", data2)
        except:
            assert (True)
        else:
            assert (False)

        assert ("data" in self.temp_hdf5_file)
        assert (data1 == self.temp_hdf5_file["data"].value).all()

        try:
            nanshe.io.hdf5.serializers.create_numpy_structured_array_in_HDF5(self.temp_hdf5_file, "data", data2, overwrite=True)
        except:
            assert (False)
        else:
            assert (True)

        assert ("data" in self.temp_hdf5_file)
        assert (data2 == self.temp_hdf5_file["data"].value).all()


    def test_create_numpy_structured_array_in_HDF5_2(self):
        data1 = numpy.zeros((10, 10), dtype=[("a", float, 2), ("b", int, 3)])
        data1["a"] = numpy.random.random((10, 10, 2))
        data1["b"] = numpy.random.random_integers(0, 10, (10, 10, 3))

        data2 = numpy.zeros((10, 10), dtype=[("a", float, 2), ("b", int, 3)])
        data2["a"] = numpy.random.random((10, 10, 2))
        data2["b"] = numpy.random.random_integers(0, 10, (10, 10, 3))
        while (data1 == data2).all():
            data2["a"] = numpy.random.random((10, 10, 2))
            data2["b"] = numpy.random.random_integers(0, 10, (10, 10, 3))

        nanshe.io.hdf5.serializers.create_numpy_structured_array_in_HDF5(self.temp_hdf5_file, "data", data1)

        assert ("data" in self.temp_hdf5_file)
        assert (data1.dtype == self.temp_hdf5_file["data"].dtype)
        assert (data1.shape == self.temp_hdf5_file["data"].shape)
        assert (data1 == self.temp_hdf5_file["data"].value).all()

        try:
            nanshe.io.hdf5.serializers.create_numpy_structured_array_in_HDF5(self.temp_hdf5_file, "data", data2)
        except:
            assert (True)
        else:
            assert (False)

        assert ("data" in self.temp_hdf5_file)
        assert (data1.dtype == self.temp_hdf5_file["data"].dtype)
        assert (data1.shape == self.temp_hdf5_file["data"].shape)
        assert (data1 == self.temp_hdf5_file["data"].value).all()

        try:
            nanshe.io.hdf5.serializers.create_numpy_structured_array_in_HDF5(self.temp_hdf5_file, "data", data2, overwrite=True)
        except:
            assert (False)
        else:
            assert (True)

        assert ("data" in self.temp_hdf5_file)
        assert (data2.dtype == self.temp_hdf5_file["data"].dtype)
        assert (data2.shape == self.temp_hdf5_file["data"].shape)
        assert (data2 == self.temp_hdf5_file["data"].value).all()


    def test_create_numpy_structured_array_in_HDF5_3(self):
        data1 = numpy.random.random((10,)).tolist()
        data2 = numpy.random.random((10,)).tolist()

        while (numpy.asarray(data1) == numpy.asarray(data2)).all():
            data2 = numpy.random.random((10,)).tolist()

        nanshe.io.hdf5.serializers.create_numpy_structured_array_in_HDF5(self.temp_hdf5_file, "data", data1)

        assert ("data" in self.temp_hdf5_file)
        assert (numpy.asarray(data1) == self.temp_hdf5_file["data"].value).all()

        try:
            nanshe.io.hdf5.serializers.create_numpy_structured_array_in_HDF5(self.temp_hdf5_file, "data", data2)
        except:
            assert (True)
        else:
            assert (False)

        assert ("data" in self.temp_hdf5_file)
        assert (numpy.asarray(data1) == self.temp_hdf5_file["data"].value).all()

        try:
            nanshe.io.hdf5.serializers.create_numpy_structured_array_in_HDF5(self.temp_hdf5_file, "data", data2, overwrite=True)
        except:
            assert (False)
        else:
            assert (True)

        assert ("data" in self.temp_hdf5_file)
        assert (numpy.asarray(data2) == self.temp_hdf5_file["data"].value).all()


    def test_read_numpy_structured_array_from_HDF5_1(self):
        data1 = numpy.random.random((10, 10))

        nanshe.io.hdf5.serializers.create_numpy_structured_array_in_HDF5(self.temp_hdf5_file, "data", data1)

        data2 = nanshe.io.hdf5.serializers.read_numpy_structured_array_from_HDF5(self.temp_hdf5_file, "data")

        assert (data1.dtype == data2.dtype)
        assert (data1.shape == data2.shape)
        assert (data1 == data2).all()

        self.temp_hdf5_file["data_ref"] = self.temp_hdf5_file["data"].ref

        data3 = nanshe.io.hdf5.serializers.read_numpy_structured_array_from_HDF5(self.temp_hdf5_file, "data_ref")

        assert (data1.dtype == data3.dtype)
        assert (data1.shape == data3.shape)
        assert (data1 == data3).all()

        self.temp_hdf5_file["data_rref"] = self.temp_hdf5_file["data"].regionref[2:8, 2:8]

        data4 = nanshe.io.hdf5.serializers.read_numpy_structured_array_from_HDF5(self.temp_hdf5_file, "data_rref")

        assert (data1[2:8, 2:8].dtype == data4.dtype)
        assert (data1[2:8, 2:8].shape == data4.shape)
        assert (data1[2:8, 2:8] == data4).all()

        self.temp_hdf5_file2 = h5py.File(os.path.join(self.temp_dir, "test2.h5"), "w")

        self.temp_hdf5_file2["data_lref"] = self.temp_hdf5_file["data"].ref
        self.temp_hdf5_file2["data_lref"].attrs["filename"] = self.temp_hdf5_file.filename

        data5 = nanshe.io.hdf5.serializers.read_numpy_structured_array_from_HDF5(self.temp_hdf5_file2, "data_lref")

        assert (data1.dtype == data5.dtype)
        assert (data1.shape == data5.shape)
        assert (data1 == data5).all()

        self.temp_hdf5_file2["data_lrref"] = self.temp_hdf5_file["data"].regionref[2:8, 2:8]
        self.temp_hdf5_file2["data_lrref"].attrs["filename"] = self.temp_hdf5_file.filename

        data6 = nanshe.io.hdf5.serializers.read_numpy_structured_array_from_HDF5(self.temp_hdf5_file2, "data_lrref")

        assert (data1[2:8, 2:8].dtype == data6.dtype)
        assert (data1[2:8, 2:8].shape == data6.shape)
        assert (data1[2:8, 2:8] == data6).all()


    def test_read_numpy_structured_array_from_HDF5_2(self):
        data1 = numpy.zeros((10, 10), dtype=[("a", float, 2), ("b", int, 3)])
        data1["a"] = numpy.random.random((10, 10, 2))
        data1["b"] = numpy.random.random_integers(0, 10, (10, 10, 3))

        nanshe.io.hdf5.serializers.create_numpy_structured_array_in_HDF5(self.temp_hdf5_file, "data", data1)

        data2 = nanshe.io.hdf5.serializers.read_numpy_structured_array_from_HDF5(self.temp_hdf5_file, "data")

        assert (data1.dtype == data2.dtype)
        assert (data1.shape == data2.shape)
        assert (data1 == data2).all()

        self.temp_hdf5_file["data_ref"] = self.temp_hdf5_file["data"].ref

        data3 = nanshe.io.hdf5.serializers.read_numpy_structured_array_from_HDF5(self.temp_hdf5_file, "data_ref")

        assert (data1.dtype == data3.dtype)
        assert (data1.shape == data3.shape)
        assert (data1 == data3).all()


    def teardown(self):
        self.temp_hdf5_file.close()

        self.temp_hdf5_file = None

        nanshe.io.disk.rmtree(self.temp_dir)

        self.temp_dir = ""


class TestHdf5Wrapper(object):
    def setup(self):
        self.temp_dir = tempfile.mkdtemp()

        def my_add(a, b):
            return(a + b)

        self.func = my_add

        self.data_a = numpy.random.random((2, 3))
        self.data_b = numpy.random.random((2, 3))
        self.data_ab = self.func(self.data_a, self.data_b)

        self.filename_a = os.path.join(self.temp_dir, "data_a.h5")
        self.datasetname_a = "array"
        self.filepath_a = os.path.join(self.filename_a, self.datasetname_a)

        with h5py.File(self.filename_a, "w") as file_a:
            file_a[self.datasetname_a] = self.data_a

        self.filename_b = os.path.join(self.temp_dir, "data_b.h5")
        self.datasetname_b = "array"
        self.filepath_b = os.path.join(self.filename_b, self.datasetname_b)

        with h5py.File(self.filename_b, "w") as file_b:
            file_b[self.datasetname_b] = self.data_b

        self.filename_ab = os.path.join(self.temp_dir, "data_ab.h5")
        self.datasetname_ab = "array"
        self.filepath_ab = os.path.join(self.filename_ab, self.datasetname_ab)


    def test_hdf5_wrapper_0(self):
        new_func = nanshe.io.hdf5.serializers.hdf5_wrapper()(self.func)

        new_data_ab = new_func(self.data_a, self.data_b)

        assert (new_data_ab == self.data_ab).all()


    def test_hdf5_wrapper_1a(self):
        new_func = nanshe.io.hdf5.serializers.hdf5_wrapper(
            hdf5_args=[1],
            hdf5_kwargs=["b"]
        )(self.func)

        new_data_ab = new_func(self.data_a, self.filepath_b)

        assert (new_data_ab == self.data_ab).all()


    def test_hdf5_wrapper_1b(self):
        new_func = nanshe.io.hdf5.serializers.hdf5_wrapper(
            hdf5_args=[1],
            hdf5_kwargs=["b"]
        )(self.func)

        new_data_ab = new_func(self.data_a, b=self.filepath_b)

        assert (new_data_ab == self.data_ab).all()


    @nose.tools.raises(TypeError)
    def test_hdf5_wrapper_2(self):
        new_func = nanshe.io.hdf5.serializers.hdf5_wrapper()(self.func)

        new_data_ab = new_func(self.data_a, b=self.filepath_b)


    def test_hdf5_wrapper_3(self):
        new_func = nanshe.io.hdf5.serializers.hdf5_wrapper(
            hdf5_result=self.filepath_ab
        )(self.func)

        new_filepath_ab = new_func(self.data_a, self.data_b)

        assert (new_filepath_ab == self.filepath_ab)

        with h5py.File(self.filename_ab, "r") as file_ab:
            new_data_ab = file_ab[self.datasetname_ab][...]
            assert (new_data_ab == self.data_ab).all()


    def teardown(self):
        self.filepath_ab = None
        self.datasetname_ab = None
        self.filename_ab = None

        self.filepath_b = None
        self.datasetname_b = None
        self.filename_b = None

        self.filepath_a = None
        self.datasetname_a = None
        self.filename_a = None

        self.data_ab = None
        self.data_b = None
        self.data_a = None

        self.func = None

        shutil.rmtree(self.temp_dir)

        self.temp_dir = ""


class TestHDF5MaskedDataset(object):
    def setup(self):
        self.temp_dir = tempfile.mkdtemp()

        self.temp_hdf5_file = h5py.File(os.path.join(self.temp_dir, "test.h5"), "w")


    def test_create(self):
        shape = (2, 3)
        dtype = numpy.dtype(float)
        ndim = len(shape)
        size = numpy.prod(shape)

        a = nanshe.io.hdf5.serializers.HDF5MaskedDataset(
            self.temp_hdf5_file, shape=shape, dtype=dtype
        )

        assert (a.ndim == ndim)
        assert (len(a) == shape[0])
        assert (a.shape == shape)
        assert (a.size == size)
        assert (a.dtype == dtype)

        assert (a.data[...] == 0.0).all()
        assert (a.mask[...] == False).all()
        assert (a.fill_value[...] == 0).all()


    def test_create_modify_1(self):
        shape = (2, 3)
        dtype = numpy.dtype(int)
        ndim = len(shape)
        size = numpy.prod(shape)

        a = nanshe.io.hdf5.serializers.HDF5MaskedDataset(
            self.temp_hdf5_file, shape=shape, dtype=dtype
        )

        assert (a.ndim == ndim)
        assert (len(a) == shape[0])
        assert (a.shape == shape)
        assert (a.size == size)
        assert (a.dtype == dtype)

        assert (a.data[...] == 0).all()
        assert (a.mask[...] == False).all()
        assert (a.fill_value[...] == 0).all()

        b = numpy.ma.arange(size).reshape(shape).astype(dtype)
        b.mask = numpy.arange(size).reshape(shape[::-1]).T.copy() % 2
        b.fill_value = b.dtype.type(1)

        a.data[...] = b.data
        a.mask[...] = b.mask
        a.fill_value[...] = b.fill_value

        assert (a.ndim == b.ndim)
        assert (len(a) == len(b))
        assert (a.shape == b.shape)
        assert (a.size == b.size)
        assert (a.dtype == b.dtype)

        assert (a.data[...] == b.data).all()
        assert (a.mask[...] == b.mask).all()
        assert (a.fill_value[...] == b.fill_value).all()


    def test_create_modify_2(self):
        shape = (2, 3)
        dtype = numpy.dtype(int)
        ndim = len(shape)
        size = numpy.prod(shape)

        a = nanshe.io.hdf5.serializers.HDF5MaskedDataset(
            self.temp_hdf5_file, shape=shape, dtype=dtype
        )

        assert (a.ndim == ndim)
        assert (len(a) == shape[0])
        assert (a.shape == shape)
        assert (a.size == size)
        assert (a.dtype == dtype)

        assert (a.data[...] == 0).all()
        assert (a.mask[...] == False).all()
        assert (a.fill_value[...] == 0).all()

        b = numpy.ma.arange(size).reshape(shape).astype(dtype)
        b.mask = numpy.arange(size).reshape(shape[::-1]).T.copy() % 2
        b.fill_value = b.dtype.type(1)

        a.data = b.data
        a.mask = b.mask
        a.fill_value = b.fill_value

        assert (a.ndim == b.ndim)
        assert (len(a) == len(b))
        assert (a.shape == b.shape)
        assert (a.size == b.size)
        assert (a.dtype == b.dtype)

        assert (a.data[...] == b.data).all()
        assert (a.mask[...] == b.mask).all()
        assert (a.fill_value[...] == b.fill_value).all()


    def test_create_modify_3(self):
        shape = (2, 3)
        dtype = numpy.dtype(int)
        ndim = len(shape)
        size = numpy.prod(shape)

        a = nanshe.io.hdf5.serializers.HDF5MaskedDataset(
            self.temp_hdf5_file, shape=shape, dtype=dtype
        )

        assert (a.ndim == ndim)
        assert (len(a) == shape[0])
        assert (a.shape == shape)
        assert (a.size == size)
        assert (a.dtype == dtype)

        assert (a.data[...] == 0).all()
        assert (a.mask[...] == False).all()
        assert (a.fill_value[...] == 0).all()

        b = numpy.ma.arange(size).reshape(shape).astype(dtype)
        b.mask = numpy.arange(size).reshape(shape[::-1]).T.copy() % 2
        b.fill_value = b.dtype.type(1)

        a[...] = b

        assert (a.ndim == b.ndim)
        assert (len(a) == len(b))
        assert (a.shape == b.shape)
        assert (a.size == b.size)
        assert (a.dtype == b.dtype)

        assert (a.data[...] == b.data).all()
        assert (a.mask[...] == b.mask).all()
        assert (a.fill_value[...] == b.fill_value).all()


    def test_read(self):
        shape = (2, 3)
        dtype = numpy.dtype(int)
        ndim = len(shape)
        size = numpy.prod(shape)

        b = numpy.ma.arange(size).reshape(shape).astype(dtype)
        b.mask = numpy.arange(size).reshape(shape[::-1]).T.copy() % 2
        b.fill_value = b.dtype.type(1)

        self.temp_hdf5_file.create_dataset(
            "data",
            data=b.data,
            chunks=True
        )
        self.temp_hdf5_file.create_dataset(
            "mask",
            data=b.mask,
            chunks=True,
            compression="gzip",
            compression_opts=2
        )
        self.temp_hdf5_file.create_dataset(
            "fill_value",
            data=b.fill_value
        )

        a = nanshe.io.hdf5.serializers.HDF5MaskedDataset(
            self.temp_hdf5_file
        )

        assert (a.ndim == ndim)
        assert (len(a) == shape[0])
        assert (a.shape == shape)
        assert (a.size == size)
        assert (a.dtype == dtype)

        a[...] = b

        assert (a.ndim == b.ndim)
        assert (len(a) == len(b))
        assert (a.shape == b.shape)
        assert (a.size == b.size)
        assert (a.dtype == b.dtype)

        assert (a.data[...] == b.data).all()
        assert (a.mask[...] == b.mask).all()
        assert (a.fill_value[...] == b.fill_value).all()


    def teardown(self):
        self.temp_hdf5_file.close()

        self.temp_hdf5_file = None

        shutil.rmtree(self.temp_dir)

        self.temp_dir = ""
