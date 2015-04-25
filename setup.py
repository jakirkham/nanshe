from __future__ import print_function


__author__ = "John Kirkham <kirkhamj@janelia.hhmi.org>"
__date__ = "$Mar 30, 2015 23:17:09 EDT$"


from glob import glob
import os
import shutil
import sys

from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand
from setuptools.dist import Distribution

import versioneer


class NoseTestCommand(TestCommand):
    description = "Run unit tests using nosetests"
    user_options = []

    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import nose
        nose.run_exit(argv=["nosetests"])

build_requires = []
install_requires = []
tests_require = ["nose"]
sphinx_build_pdf = False
if len(sys.argv) == 1:
    pass
elif ("--help" in sys.argv) or ("-h" in sys.argv):
    pass
elif sys.argv[1] == "bdist_conda":
    from distutils.command.bdist_conda import CondaDistribution as Distribution

    build_requires = [
        "nomkl",
        "openblas",
        "fftw",
        "setuptools",
        "psutil",
        "billiard",
        "numpy",
        "scipy",
        "h5py",
        "bottleneck",
        "matplotlib",
        "pyfftw",
        "scikit-image",
        "scikit-learn",
        "mahotas",
        "vigra",
        "rank_filter"
    ]

    install_requires = [
        "nomkl",
        "openblas",
        "fftw",
        "setuptools",
        "psutil",
        "billiard",
        "numpy",
        "scipy",
        "h5py",
        "bottleneck",
        "matplotlib",
        "pyfftw",
        "scikit-image",
        "scikit-learn",
        "mahotas",
        "vigra",
        "rank_filter"
    ]

    if sys.version_info < (3, 2):
        build_requires += [
            "functools32"
        ]
        install_requires += [
            "functools32"
        ]
    if sys.version_info < (3,):
        build_requires += [
            "spams",
            "pyqt",
            "volumina"
        ]
        install_requires += [
            "spams",
            "pyqt",
            "volumina"
        ]
elif sys.argv[1] == "build_sphinx":
    import sphinx.apidoc

    sphinx.apidoc.main([
        sphinx.apidoc.__file__,
        "-f", "-T", "-e", "-M",
        "-o", "docs",
        ".", "setup.py", "tests", "versioneer.py"
    ])

    build_prefix_arg_index = None
    for each_build_arg in ["-b", "--builder"]:
        try:
            build_arg_index = sys.argv.index(each_build_arg)
        except ValueError:
            continue

        if sys.argv[build_arg_index + 1] == "pdf":
            sphinx_build_pdf = True
            sys.argv[build_arg_index + 1] = "latex"
elif sys.argv[1] == "clean":
    saved_rst_files = ["docs/index.rst", "docs/readme.rst", "docs/todo.rst"]

    tmp_rst_files = glob("docs/*.rst")

    print("removing 'docs/*.rst'")
    for each_saved_rst_file in saved_rst_files:
        print("skipping '" + each_saved_rst_file + "'")
        tmp_rst_files.remove(each_saved_rst_file)

    for each_tmp_rst_file in tmp_rst_files:
        os.remove(each_tmp_rst_file)

    if os.path.exists("build/sphinx/doctrees"):
        print("removing 'build/sphinx/doctrees'")
        shutil.rmtree("build/sphinx/doctrees")
    else:
        print("'build/sphinx/doctrees' does not exist -- can't clean it")

    if os.path.exists(".eggs"):
        print("removing '.eggs'")
        shutil.rmtree(".eggs")
    else:
        print("'.eggs' does not exist -- can't clean it")

    if (len(sys.argv) > 2) and (sys.argv[2] in ["-a", "--all"]):
        if os.path.exists("build/sphinx"):
            print("removing 'build/sphinx'")
            shutil.rmtree("build/sphinx")
        else:
            print("'build/sphinx' does not exist -- can't clean it")
elif sys.argv[1] == "develop":
    if (len(sys.argv) > 2) and (sys.argv[2] in ["-u", "--uninstall"]):
        if os.path.exists("nanshe.egg-info"):
            print("removing 'nanshe.egg-info'")
            shutil.rmtree("nanshe.egg-info")
        else:
            print("'nanshe.egg-info' does not exist -- can't clean it")

setup(
    name="nanshe",
    version=versioneer.get_version(),
    description="An image processing toolkit.",
    url="https://github.com/nanshe-org/nanshe",
    license="GPLv3",
    author="John Kirkham",
    author_email="kirkhamj@janelia.hhmi.org",
    scripts=glob("bin/*"),
    py_modules=["versioneer"],
    packages=find_packages(exclude=["tests*"]),
    distclass=Distribution,
    cmdclass=dict(sum([list(_.items()) for _ in [
        versioneer.get_cmdclass(),
        {"test": NoseTestCommand}
    ]], [])),
    build_requires=build_requires,
    install_requires=install_requires,
    tests_require=tests_require,
    test_suite="nose.collector",
    zip_safe=True,
    conda_import_tests=False,
    conda_command_tests=False
)

if sphinx_build_pdf:
    make_cmd = os.environ.get("MAKE", "make")
    cwd = os.getcwd()
    os.chdir("build/sphinx/latex")
    os.execlpe(make_cmd, "all", os.environ)
    os.chdir(cwd)
