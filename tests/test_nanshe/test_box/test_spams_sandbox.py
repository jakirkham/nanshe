from __future__ import print_function

__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "$Aug 05, 2014 17:38:58 EDT$"


import imp

import nose
import nose.plugins
import nose.plugins.attrib

import ctypes
import multiprocessing as mp

try:
    from queue import Queue
except ImportError:
    from Queue import Queue

import numpy

import nanshe.box.spams_sandbox

import nanshe.syn.data


has_spams = False
try:
    imp.find_module("spams")
    has_spams = True
except ImportError:
    pass


try:
    xrange
except NameError:
    xrange = range


class TestSpamsSandbox(object):
    def setup(self):
        self.p = numpy.array([[27, 51],
                              [66, 85],
                              [77, 45]])

        self.p3 = numpy.array([[27, 51, 37],
                               [66, 85, 25],
                               [77, 45, 73]])

        self.space = numpy.array((100, 100))
        self.space3 = numpy.array((100, 100, 100))
        self.radii = numpy.array((5, 6, 7))

        self.g = nanshe.syn.data.generate_hypersphere_masks(
            self.space, self.p, self.radii
        )

        self.g = self.g.reshape((self.g.shape[0], -1))
        self.g = self.g.transpose()
        self.g = numpy.asmatrix(self.g)
        self.g = numpy.asfortranarray(self.g)

        self.g3 = nanshe.syn.data.generate_hypersphere_masks(
            self.space3, self.p3, self.radii
        )

        self.g3 = self.g3.reshape((self.g3.shape[0], -1))
        self.g3 = self.g3.transpose()
        self.g3 = numpy.asmatrix(self.g3)
        self.g3 = numpy.asfortranarray(self.g3)

    def test_run_multiprocessing_queue_spams_trainDL_1(self):
        if not has_spams:
            raise nose.SkipTest(
                "Cannot run this test without SPAMS being installed."
            )

        out_queue = Queue()

        nanshe.box.spams_sandbox.run_multiprocessing_queue_spams_trainDL(
            out_queue,
            self.g.astype(float),
            **{
                "gamma2" : 0,
                "gamma1" : 0,
                "numThreads" : 1,
                "K" : self.g.shape[1],
                "iter" : 10,
                "modeD" : 0,
                "posAlpha" : True,
                "clean" : True,
                "posD" : True,
                "batchsize" : 256,
                "lambda1" : 0.2,
                "lambda2" : 0,
                "mode" : 2
            }
        )
        d = out_queue.get()

        d = (d != 0)

        self.g = self.g.transpose()
        d = d.transpose()

        assert (self.g.shape == d.shape)

        assert (self.g.astype(bool).max(axis=0) == d.astype(bool).max(axis=0)).all()

        unmatched_g = range(len(self.g))
        matched = dict()

        for i in xrange(len(d)):
            new_unmatched_g = []
            for j in unmatched_g:
                if not (d[i] == self.g[j]).all():
                    new_unmatched_g.append(j)
                else:
                    matched[i] = j

            unmatched_g = new_unmatched_g

        print(unmatched_g)

        assert (len(unmatched_g) == 0)

    @nose.plugins.attrib.attr("3D")
    def test_run_multiprocessing_queue_spams_trainDL_2(self):
        if not has_spams:
            raise nose.SkipTest(
                "Cannot run this test without SPAMS being installed."
            )

        out_queue = Queue()

        nanshe.box.spams_sandbox.run_multiprocessing_queue_spams_trainDL(
            out_queue,
            self.g3.astype(float),
            **{
                "gamma2" : 0,
                "gamma1" : 0,
                "numThreads" : 1,
                "K" : self.g3.shape[1],
                "iter" : 10,
                "modeD" : 0,
                "posAlpha" : True,
                "clean" : True,
                "posD" : True,
                "batchsize" : 256,
                "lambda1" : 0.2,
                "lambda2" : 0,
                "mode" : 2
            }
        )
        d3 = out_queue.get()

        d3 = (d3 != 0)

        self.g3 = self.g3.transpose()
        d3 = d3.transpose()

        assert (self.g3.shape == d3.shape)

        assert (self.g3.astype(bool).max(axis=0) == d3.astype(bool).max(axis=0)).all()

        unmatched_g3 = range(len(self.g3))
        matched = dict()

        for i in xrange(len(d3)):
            new_unmatched_g3 = []
            for j in unmatched_g3:
                if not (d3[i] == self.g3[j]).all():
                    new_unmatched_g3.append(j)
                else:
                    matched[i] = j

            unmatched_g3 = new_unmatched_g3

        print(unmatched_g3)

        assert (len(unmatched_g3) == 0)

    def test_run_multiprocessing_queue_spams_trainDL_3(self):
        if not has_spams:
            raise nose.SkipTest(
                "Cannot run this test without SPAMS being installed."
            )

        out_queue = Queue()

        nanshe.box.spams_sandbox.run_multiprocessing_queue_spams_trainDL(
            out_queue,
            self.g.astype(float),
            D=self.g.astype(float),
            **{
                "gamma2" : 0,
                "gamma1" : 0,
                "numThreads" : 1,
                "iter" : 10,
                "modeD" : 0,
                "posAlpha" : True,
                "clean" : True,
                "posD" : True,
                "batchsize" : 256,
                "lambda1" : 0.2,
                "lambda2" : 0,
                "mode" : 2
            }
        )
        d = out_queue.get()

        d = (d != 0)

        self.g = self.g.transpose()
        d = d.transpose()

        assert (self.g.shape == d.shape)

        assert (self.g.astype(bool).max(axis=0) == d.astype(bool).max(axis=0)).all()

        unmatched_g = range(len(self.g))
        matched = dict()

        for i in xrange(len(d)):
            new_unmatched_g = []
            for j in unmatched_g:
                if not (d[i] == self.g[j]).all():
                    new_unmatched_g.append(j)
                else:
                    matched[i] = j

            unmatched_g = new_unmatched_g

        print(unmatched_g)

        assert (len(unmatched_g) == 0)

        assert (self.g.astype(bool) == d.astype(bool)).all()

    @nose.plugins.attrib.attr("3D")
    def test_run_multiprocessing_queue_spams_trainDL_4(self):
        if not has_spams:
            raise nose.SkipTest(
                "Cannot run this test without SPAMS being installed."
            )

        out_queue = Queue()

        nanshe.box.spams_sandbox.run_multiprocessing_queue_spams_trainDL(
            out_queue,
            self.g3.astype(float),
            D=self.g3.astype(float),
            **{
                "gamma2" : 0,
                "gamma1" : 0,
                "numThreads" : 1,
                "iter" : 10,
                "modeD" : 0,
                "posAlpha" : True,
                "clean" : True,
                "posD" : True,
                "batchsize" : 256,
                "lambda1" : 0.2,
                "lambda2" : 0,
                "mode" : 2
            }
        )
        d3 = out_queue.get()

        d3 = (d3 != 0)

        self.g3 = self.g3.transpose()
        d3 = d3.transpose()

        assert (self.g3.shape == d3.shape)

        assert (self.g3.astype(bool).max(axis=0) == d3.astype(bool).max(axis=0)).all()

        unmatched_g3 = range(len(self.g3))
        matched = dict()

        for i in xrange(len(d3)):
            new_unmatched_g3 = []
            for j in unmatched_g3:
                if not (d3[i] == self.g3[j]).all():
                    new_unmatched_g3.append(j)
                else:
                    matched[i] = j

            unmatched_g3 = new_unmatched_g3

        print(unmatched_g3)

        assert (len(unmatched_g3) == 0)

        assert (self.g3.astype(bool) == d3.astype(bool)).all()

    def test_call_multiprocessing_queue_spams_trainDL_1(self):
        if not has_spams:
            raise nose.SkipTest(
                "Cannot run this test without SPAMS being installed."
            )

        d = nanshe.box.spams_sandbox.call_multiprocessing_queue_spams_trainDL(
            self.g.astype(float),
            **{
                "gamma2" : 0,
                "gamma1" : 0,
                "numThreads" : 1,
                "K" : self.g.shape[1],
                "iter" : 10,
                "modeD" : 0,
                "posAlpha" : True,
                "clean" : True,
                "posD" : True,
                "batchsize" : 256,
                "lambda1" : 0.2,
                "lambda2" : 0,
                "mode" : 2
            }
        )
        d = (d != 0)

        self.g = self.g.transpose()
        d = d.transpose()

        assert (self.g.shape == d.shape)

        assert (self.g.astype(bool).max(axis=0) == d.astype(bool).max(axis=0)).all()

        unmatched_g = range(len(self.g))
        matched = dict()

        for i in xrange(len(d)):
            new_unmatched_g = []
            for j in unmatched_g:
                if not (d[i] == self.g[j]).all():
                    new_unmatched_g.append(j)
                else:
                    matched[i] = j

            unmatched_g = new_unmatched_g

        print(unmatched_g)

        assert (len(unmatched_g) == 0)

    @nose.plugins.attrib.attr("3D")
    def test_call_multiprocessing_queue_spams_trainDL_2(self):
        if not has_spams:
            raise nose.SkipTest(
                "Cannot run this test without SPAMS being installed."
            )

        d3 = nanshe.box.spams_sandbox.call_multiprocessing_queue_spams_trainDL(
            self.g3.astype(float),
            **{
                "gamma2" : 0,
                "gamma1" : 0,
                "numThreads" : 1,
                "K" : self.g3.shape[1],
                "iter" : 10,
                "modeD" : 0,
                "posAlpha" : True,
                "clean" : True,
                "posD" : True,
                "batchsize" : 256,
                "lambda1" : 0.2,
                "lambda2" : 0,
                "mode" : 2
            }
        )
        d3 = (d3 != 0)

        self.g3 = self.g3.transpose()
        d3 = d3.transpose()

        assert (self.g3.shape == d3.shape)

        assert (self.g3.astype(bool).max(axis=0) == d3.astype(bool).max(axis=0)).all()

        unmatched_g3 = range(len(self.g3))
        matched = dict()

        for i in xrange(len(d3)):
            new_unmatched_g3 = []
            for j in unmatched_g3:
                if not (d3[i] == self.g3[j]).all():
                    new_unmatched_g3.append(j)
                else:
                    matched[i] = j

            unmatched_g3 = new_unmatched_g3

        print(unmatched_g3)

        assert (len(unmatched_g3) == 0)

    def test_call_multiprocessing_queue_spams_trainDL_3(self):
        if not has_spams:
            raise nose.SkipTest(
                "Cannot run this test without SPAMS being installed."
            )

        d = nanshe.box.spams_sandbox.call_multiprocessing_queue_spams_trainDL(
            self.g.astype(float),
            D=self.g.astype(float),
            **{
                "gamma2" : 0,
                "gamma1" : 0,
                "numThreads" : 1,
                "iter" : 10,
                "modeD" : 0,
                "posAlpha" : True,
                "clean" : True,
                "posD" : True,
                "batchsize" : 256,
                "lambda1" : 0.2,
                "lambda2" : 0,
                "mode" : 2
            }
        )
        d = (d != 0)

        self.g = self.g.transpose()
        d = d.transpose()

        assert (self.g.shape == d.shape)

        assert (self.g.astype(bool).max(axis=0) == d.astype(bool).max(axis=0)).all()

        unmatched_g = range(len(self.g))
        matched = dict()

        for i in xrange(len(d)):
            new_unmatched_g = []
            for j in unmatched_g:
                if not (d[i] == self.g[j]).all():
                    new_unmatched_g.append(j)
                else:
                    matched[i] = j

            unmatched_g = new_unmatched_g

        print(unmatched_g)

        assert (len(unmatched_g) == 0)

        assert (self.g.astype(bool) == d.astype(bool)).all()

    @nose.plugins.attrib.attr("3D")
    def test_call_multiprocessing_queue_spams_trainDL_4(self):
        if not has_spams:
            raise nose.SkipTest(
                "Cannot run this test without SPAMS being installed."
            )

        d3 = nanshe.box.spams_sandbox.call_multiprocessing_queue_spams_trainDL(
            self.g3.astype(float),
            D=self.g3.astype(float),
            **{
                "gamma2" : 0,
                "gamma1" : 0,
                "numThreads" : 1,
                "iter" : 10,
                "modeD" : 0,
                "posAlpha" : True,
                "clean" : True,
                "posD" : True,
                "batchsize" : 256,
                "lambda1" : 0.2,
                "lambda2" : 0,
                "mode" : 2
            }
        )
        d3 = (d3 != 0)

        self.g3 = self.g3.transpose()
        d3 = d3.transpose()

        assert (self.g3.shape == d3.shape)

        assert (self.g3.astype(bool).max(axis=0) == d3.astype(bool).max(axis=0)).all()

        unmatched_g3 = range(len(self.g3))
        matched = dict()

        for i in xrange(len(d3)):
            new_unmatched_g3 = []
            for j in unmatched_g3:
                if not (d3[i] == self.g3[j]).all():
                    new_unmatched_g3.append(j)
                else:
                    matched[i] = j

            unmatched_g3 = new_unmatched_g3

        print(unmatched_g3)

        assert (len(unmatched_g3) == 0)

        assert (self.g3.astype(bool) == d3.astype(bool)).all()

    def test_run_multiprocessing_array_spams_trainDL_1(self):
        if not has_spams:
            raise nose.SkipTest(
                "Cannot run this test without SPAMS being installed."
            )

        float_type = numpy.float64

        g_array_type = numpy.ctypeslib.ndpointer(
            dtype=float_type,
            ndim=self.g.ndim,
            shape=self.g.shape,
            flags=self.g.flags
        )
        g_array_ctype = type(numpy.ctypeslib.as_ctypes(numpy.dtype(
            g_array_type._dtype_.type
        ).type(0)[()]))
        g_array = mp.Array(
            g_array_ctype,
            numpy.product(g_array_type._shape_),
            lock=False
        )

        g_numpy_array = numpy.frombuffer(
            g_array,
            dtype=g_array_type._dtype_
        ).reshape(g_array_type._shape_)
        g_numpy_array[:] = self.g
        g_numpy_array = None

        result_array_type = numpy.ctypeslib.ndpointer(
            dtype=float_type,
            ndim=2,
            shape=(self.g.shape[0], self.g.shape[1])
        )
        result_array_ctype = type(numpy.ctypeslib.as_ctypes(numpy.dtype(
            result_array_type._dtype_.type
        ).type(0)[()]))
        result_array = mp.Array(
            result_array_ctype,
            numpy.product(result_array_type._shape_),
            lock=False
        )

        nanshe.box.spams_sandbox.run_multiprocessing_array_spams_trainDL(
            result_array_type,
            result_array,
            g_array_type,
            g_array,
            **{
                "gamma2" : 0,
                "gamma1" : 0,
                "numThreads" : 1,
                "K" : self.g.shape[1],
                "iter" : 10,
                "modeD" : 0,
                "posAlpha" : True,
                "clean" : True,
                "posD" : True,
                "batchsize" : 256,
                "lambda1" : 0.2,
                "lambda2" : 0,
                "mode" : 2
            }
        )
        d = numpy.frombuffer(
            result_array,
            dtype=float_type
        ).reshape(result_array_type._shape_).copy()
        d = (d != 0)

        self.g = self.g.transpose()
        d = d.transpose()

        assert (self.g.shape == d.shape)

        assert (self.g.astype(bool).max(axis=0) == d.astype(bool).max(axis=0)).all()

        unmatched_g = range(len(self.g))
        matched = dict()

        for i in xrange(len(d)):
            new_unmatched_g = []
            for j in unmatched_g:
                if not (d[i] == self.g[j]).all():
                    new_unmatched_g.append(j)
                else:
                    matched[i] = j

            unmatched_g = new_unmatched_g

        print(unmatched_g)

        assert (len(unmatched_g) == 0)

    @nose.plugins.attrib.attr("3D")
    def test_run_multiprocessing_array_spams_trainDL_2(self):
        if not has_spams:
            raise nose.SkipTest(
                "Cannot run this test without SPAMS being installed."
            )

        float_type = numpy.float64

        g3_array_type = numpy.ctypeslib.ndpointer(
            dtype=float_type,
            ndim=self.g3.ndim,
            shape=self.g3.shape,
            flags=self.g3.flags
        )
        g3_array_ctype = type(numpy.ctypeslib.as_ctypes(numpy.dtype(
            g3_array_type._dtype_.type
        ).type(0)[()]))
        g3_array = mp.Array(
            g3_array_ctype,
            numpy.product(g3_array_type._shape_),
            lock=False
        )

        g3_numpy_array = numpy.frombuffer(
            g3_array,
            dtype=g3_array_type._dtype_
        ).reshape(g3_array_type._shape_)
        g3_numpy_array[:] = self.g3
        g3_numpy_array = None

        result_array_type = numpy.ctypeslib.ndpointer(
            dtype=float_type,
            ndim=2,
            shape=(self.g3.shape[0], self.g3.shape[1])
        )
        result_array_ctype = type(numpy.ctypeslib.as_ctypes(numpy.dtype(
            result_array_type._dtype_.type
        ).type(0)[()]))
        result_array = mp.Array(
            result_array_ctype,
            numpy.product(result_array_type._shape_),
            lock=False
        )

        nanshe.box.spams_sandbox.run_multiprocessing_array_spams_trainDL(
            result_array_type,
            result_array,
            g3_array_type,
            g3_array,
            **{
                "gamma2" : 0,
                "gamma1" : 0,
                "numThreads" : 1,
                "K" : self.g3.shape[1],
                "iter" : 10,
                "modeD" : 0,
                "posAlpha" : True,
                "clean" : True,
                "posD" : True,
                "batchsize" : 256,
                "lambda1" : 0.2,
                "lambda2" : 0,
                "mode" : 2
            }
        )
        d3 = numpy.frombuffer(
            result_array,
            dtype=float_type
        ).reshape(result_array_type._shape_).copy()
        d3 = (d3 != 0)

        self.g3 = self.g3.transpose()
        d3 = d3.transpose()

        assert (self.g3.shape == d3.shape)

        assert (self.g3.astype(bool).max(axis=0) == d3.astype(bool).max(axis=0)).all()

        unmatched_g3 = range(len(self.g3))
        matched = dict()

        for i in xrange(len(d3)):
            new_unmatched_g3 = []
            for j in unmatched_g3:
                if not (d3[i] == self.g3[j]).all():
                    new_unmatched_g3.append(j)
                else:
                    matched[i] = j

            unmatched_g3 = new_unmatched_g3

        print(unmatched_g3)

        assert (len(unmatched_g3) == 0)

    def test_call_multiprocessing_array_spams_trainDL_1(self):
        if not has_spams:
            raise nose.SkipTest(
                "Cannot run this test without SPAMS being installed."
            )

        d = nanshe.box.spams_sandbox.call_multiprocessing_array_spams_trainDL(
            self.g.astype(float),
            **{
                "gamma2" : 0,
                "gamma1" : 0,
                "numThreads" : 1,
                "K" : self.g.shape[1],
                "iter" : 10,
                "modeD" : 0,
                "posAlpha" : True,
                "clean" : True,
                "posD" : True,
                "batchsize" : 256,
                "lambda1" : 0.2,
                "lambda2" : 0,
                "mode" : 2
            }
        )
        d = (d != 0)

        self.g = self.g.transpose()
        d = d.transpose()

        assert (self.g.shape == d.shape)

        assert (self.g.astype(bool).max(axis=0) == d.astype(bool).max(axis=0)).all()

        unmatched_g = range(len(self.g))
        matched = dict()

        for i in xrange(len(d)):
            new_unmatched_g = []
            for j in unmatched_g:
                if not (d[i] == self.g[j]).all():
                    new_unmatched_g.append(j)
                else:
                    matched[i] = j

            unmatched_g = new_unmatched_g

        print(unmatched_g)

        assert (len(unmatched_g) == 0)

    @nose.plugins.attrib.attr("3D")
    def test_call_multiprocessing_array_spams_trainDL_2(self):
        if not has_spams:
            raise nose.SkipTest(
                "Cannot run this test without SPAMS being installed."
            )

        d3 = nanshe.box.spams_sandbox.call_multiprocessing_array_spams_trainDL(
            self.g3.astype(float),
            **{
                "gamma2" : 0,
                "gamma1" : 0,
                "numThreads" : 1,
                "K" : self.g3.shape[1],
                "iter" : 10,
                "modeD" : 0,
                "posAlpha" : True,
                "clean" : True,
                "posD" : True,
                "batchsize" : 256,
                "lambda1" : 0.2,
                "lambda2" : 0,
                "mode" : 2
            }
        )
        d3 = (d3 != 0)

        self.g3 = self.g3.transpose()
        d3 = d3.transpose()

        assert (self.g3.shape == d3.shape)

        assert (self.g3.astype(bool).max(axis=0) == d3.astype(bool).max(axis=0)).all()

        unmatched_g3 = range(len(self.g3))
        matched = dict()

        for i in xrange(len(d3)):
            new_unmatched_g3 = []
            for j in unmatched_g3:
                if not (d3[i] == self.g3[j]).all():
                    new_unmatched_g3.append(j)
                else:
                    matched[i] = j

            unmatched_g3 = new_unmatched_g3

        print(unmatched_g3)

        assert (len(unmatched_g3) == 0)

    def test_run_multiprocessing_array_spams_trainDL_3(self):
        if not has_spams:
            raise nose.SkipTest(
                "Cannot run this test without SPAMS being installed."
            )

        float_type = numpy.float64

        g_array_type = numpy.ctypeslib.ndpointer(
            dtype=float_type,
            ndim=self.g.ndim,
            shape=self.g.shape,
            flags=self.g.flags
        )
        g_array_ctype = type(numpy.ctypeslib.as_ctypes(numpy.dtype(
            g_array_type._dtype_.type
        ).type(0)[()]))
        g_array = mp.Array(
            g_array_ctype,
            numpy.product(g_array_type._shape_),
            lock=False
        )

        g_numpy_array = numpy.frombuffer(
            g_array,
            dtype=g_array_type._dtype_
        ).reshape(g_array_type._shape_)
        g_numpy_array[:] = self.g
        g_numpy_array = None

        result_array_type = numpy.ctypeslib.ndpointer(
            dtype=float_type,
            ndim=2,
            shape=(self.g.shape[0], self.g.shape[1])
        )
        result_array_ctype = type(numpy.ctypeslib.as_ctypes(numpy.dtype(
            result_array_type._dtype_.type
        ).type(0)[()]))
        result_array = mp.Array(
            result_array_ctype,
            numpy.product(result_array_type._shape_),
            lock=False
        )

        nanshe.box.spams_sandbox.run_multiprocessing_array_spams_trainDL(
            result_array_type,
            result_array,
            g_array_type,
            g_array,
            False,
            g_array_type,
            g_array,
            **{
                "gamma2" : 0,
                "gamma1" : 0,
                "numThreads" : 1,
                "iter" : 10,
                "modeD" : 0,
                "posAlpha" : True,
                "clean" : True,
                "posD" : True,
                "batchsize" : 256,
                "lambda1" : 0.2,
                "lambda2" : 0,
                "mode" : 2
            }
        )
        d = numpy.frombuffer(
            result_array,
            dtype=float_type
        ).reshape(result_array_type._shape_).copy()
        d = (d != 0)

        self.g = self.g.transpose()
        d = d.transpose()

        assert (self.g.shape == d.shape)

        assert (self.g.astype(bool).max(axis=0) == d.astype(bool).max(axis=0)).all()

        unmatched_g = range(len(self.g))
        matched = dict()

        for i in xrange(len(d)):
            new_unmatched_g = []
            for j in unmatched_g:
                if not (d[i] == self.g[j]).all():
                    new_unmatched_g.append(j)
                else:
                    matched[i] = j

            unmatched_g = new_unmatched_g

        print(unmatched_g)

        assert (len(unmatched_g) == 0)

        assert (self.g.astype(bool) == d.astype(bool)).all()

    @nose.plugins.attrib.attr("3D")
    def test_run_multiprocessing_array_spams_trainDL_4(self):
        if not has_spams:
            raise nose.SkipTest(
                "Cannot run this test without SPAMS being installed."
            )

        float_type = numpy.float64

        g3_array_type = numpy.ctypeslib.ndpointer(
            dtype=float_type,
            ndim=self.g3.ndim,
            shape=self.g3.shape,
            flags=self.g3.flags
        )
        g3_array_ctype = type(numpy.ctypeslib.as_ctypes(numpy.dtype(
            g3_array_type._dtype_.type
        ).type(0)[()]))
        g3_array = mp.Array(
            g3_array_ctype,
            numpy.product(g3_array_type._shape_),
            lock=False
        )

        g3_numpy_array = numpy.frombuffer(
            g3_array,
            dtype=g3_array_type._dtype_
        ).reshape(g3_array_type._shape_)
        g3_numpy_array[:] = self.g3
        g3_numpy_array = None

        result_array_type = numpy.ctypeslib.ndpointer(
            dtype=float_type,
            ndim=2,
            shape=(self.g3.shape[0], self.g3.shape[1])
        )
        result_array_ctype = type(numpy.ctypeslib.as_ctypes(numpy.dtype(
            result_array_type._dtype_.type
        ).type(0)[()]))
        result_array = mp.Array(
            result_array_ctype,
            numpy.product(result_array_type._shape_),
            lock=False
        )

        nanshe.box.spams_sandbox.run_multiprocessing_array_spams_trainDL(
            result_array_type,
            result_array,
            g3_array_type,
            g3_array,
            False,
            g3_array_type,
            g3_array,
            **{
                "gamma2" : 0,
                "gamma1" : 0,
                "numThreads" : 1,
                "iter" : 10,
                "modeD" : 0,
                "posAlpha" : True,
                "clean" : True,
                "posD" : True,
                "batchsize" : 256,
                "lambda1" : 0.2,
                "lambda2" : 0,
                "mode" : 2
            }
        )
        d3 = numpy.frombuffer(
            result_array,
            dtype=float_type
        ).reshape(result_array_type._shape_).copy()
        d3 = (d3 != 0)

        self.g3 = self.g3.transpose()
        d3 = d3.transpose()

        assert (self.g3.shape == d3.shape)

        assert (self.g3.astype(bool).max(axis=0) == d3.astype(bool).max(axis=0)).all()

        unmatched_g3 = range(len(self.g3))
        matched = dict()

        for i in xrange(len(d3)):
            new_unmatched_g3 = []
            for j in unmatched_g3:
                if not (d3[i] == self.g3[j]).all():
                    new_unmatched_g3.append(j)
                else:
                    matched[i] = j

            unmatched_g3 = new_unmatched_g3

        print(unmatched_g3)

        assert (len(unmatched_g3) == 0)

        assert (self.g3.astype(bool) == d3.astype(bool)).all()

    def test_call_spams_trainDL_1(self):
        if not has_spams:
            raise nose.SkipTest(
                "Cannot run this test without SPAMS being installed."
            )

        d = nanshe.box.spams_sandbox.call_spams_trainDL(
            self.g.astype(float),
            **{
                "gamma2" : 0,
                "gamma1" : 0,
                "numThreads" : 1,
                "K" : self.g.shape[1],
                "iter" : 10,
                "modeD" : 0,
                "posAlpha" : True,
                "clean" : True,
                "posD" : True,
                "batchsize" : 256,
                "lambda1" : 0.2,
                "lambda2" : 0,
                "mode" : 2
            }
        )
        d = (d != 0)

        self.g = self.g.transpose()
        d = d.transpose()

        assert (self.g.shape == d.shape)

        assert (self.g.astype(bool).max(axis=0) == d.astype(bool).max(axis=0)).all()

        unmatched_g = range(len(self.g))
        matched = dict()

        for i in xrange(len(d)):
            new_unmatched_g = []
            for j in unmatched_g:
                if not (d[i] == self.g[j]).all():
                    new_unmatched_g.append(j)
                else:
                    matched[i] = j

            unmatched_g = new_unmatched_g

        print(unmatched_g)

        assert (len(unmatched_g) == 0)

    @nose.plugins.attrib.attr("3D")
    def test_call_spams_trainDL_2(self):
        if not has_spams:
            raise nose.SkipTest(
                "Cannot run this test without SPAMS being installed."
            )

        d3 = nanshe.box.spams_sandbox.call_spams_trainDL(
            self.g3.astype(float),
            **{
                "gamma2" : 0,
                "gamma1" : 0,
                "numThreads" : 1,
                "K" : self.g3.shape[1],
                "iter" : 10,
                "modeD" : 0,
                "posAlpha" : True,
                "clean" : True,
                "posD" : True,
                "batchsize" : 256,
                "lambda1" : 0.2,
                "lambda2" : 0,
                "mode" : 2
            }
        )
        d3 = (d3 != 0)

        self.g3 = self.g3.transpose()
        d3 = d3.transpose()

        assert (self.g3.shape == d3.shape)

        assert (self.g3.astype(bool).max(axis=0) == d3.astype(bool).max(axis=0)).all()

        unmatched_g3 = range(len(self.g3))
        matched = dict()

        for i in xrange(len(d3)):
            new_unmatched_g3 = []
            for j in unmatched_g3:
                if not (d3[i] == self.g3[j]).all():
                    new_unmatched_g3.append(j)
                else:
                    matched[i] = j

            unmatched_g3 = new_unmatched_g3

        print(unmatched_g3)

        assert (len(unmatched_g3) == 0)

    def test_call_spams_trainDL_3(self):
        if not has_spams:
            raise nose.SkipTest(
                "Cannot run this test without SPAMS being installed."
            )

        d = nanshe.box.spams_sandbox.call_spams_trainDL(
            self.g.astype(float),
            D=self.g.astype(float),
            **{
                "gamma2" : 0,
                "gamma1" : 0,
                "numThreads" : 1,
                "iter" : 10,
                "modeD" : 0,
                "posAlpha" : True,
                "clean" : True,
                "posD" : True,
                "batchsize" : 256,
                "lambda1" : 0.2,
                "lambda2" : 0,
                "mode" : 2
            }
        )
        d = (d != 0)

        self.g = self.g.transpose()
        d = d.transpose()

        assert (self.g.shape == d.shape)

        assert (self.g.astype(bool).max(axis=0) == d.astype(bool).max(axis=0)).all()

        unmatched_g = range(len(self.g))
        matched = dict()

        for i in xrange(len(d)):
            new_unmatched_g = []
            for j in unmatched_g:
                if not (d[i] == self.g[j]).all():
                    new_unmatched_g.append(j)
                else:
                    matched[i] = j

            unmatched_g = new_unmatched_g

        print(unmatched_g)

        assert (len(unmatched_g) == 0)

        assert (self.g.astype(bool) == d.astype(bool)).all()

    @nose.plugins.attrib.attr("3D")
    def test_call_spams_trainDL_4(self):
        if not has_spams:
            raise nose.SkipTest(
                "Cannot run this test without SPAMS being installed."
            )

        d3 = nanshe.box.spams_sandbox.call_spams_trainDL(
            self.g3.astype(float),
            D=self.g3.astype(float),
            **{
                "gamma2" : 0,
                "gamma1" : 0,
                "numThreads" : 1,
                "iter" : 10,
                "modeD" : 0,
                "posAlpha" : True,
                "clean" : True,
                "posD" : True,
                "batchsize" : 256,
                "lambda1" : 0.2,
                "lambda2" : 0,
                "mode" : 2
            }
        )
        d3 = (d3 != 0)

        self.g3 = self.g3.transpose()
        d3 = d3.transpose()

        assert (self.g3.shape == d3.shape)

        assert (self.g3.astype(bool).max(axis=0) == d3.astype(bool).max(axis=0)).all()

        unmatched_g3 = range(len(self.g3))
        matched = dict()

        for i in xrange(len(d3)):
            new_unmatched_g3 = []
            for j in unmatched_g3:
                if not (d3[i] == self.g3[j]).all():
                    new_unmatched_g3.append(j)
                else:
                    matched[i] = j

            unmatched_g3 = new_unmatched_g3

        print(unmatched_g3)

        assert (len(unmatched_g3) == 0)

        assert (self.g3.astype(bool) == d3.astype(bool)).all()

    def teardown(self):
        self.p = None
        self.space = None
        self.radii = None
        self.g = None

        self.p3 = None
        self.g3 = None
