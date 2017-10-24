#!/bin/bash

##########
# CONFIG #
##########

# Additional conda configuration
source "/opt/conda${PYTHON_VERSION:0:1}/bin/activate"
conda config --add channels nanshe

# Fix root environment to have the correct Python version
touch "$CONDA_ENV_PATH/conda-meta/pinned"
echo "python ${PYTHON_VERSION}.*" >> "$CONDA_ENV_PATH/conda-meta/pinned"
conda install -yq python=$PYTHON_VERSION

# Build the conda package for nanshe.
VERSION=`python setup.py --version`
echo "$VERSION"
python setup.py bdist_conda

# Setup environment for nanshe and install it with all dependencies.
conda create -yq --use-local -n testenv python=$PYTHON_VERSION nanshe==$VERSION
source activate testenv

# Optionally install GPL dependencies.
if [[ $USE_GPL == true ]]; then conda install -yq pyfftw python-spams; fi

# Install sphinx and friends to build documentation.
conda install -yq sphinx cloud_sptheme

# Install coverage and coveralls to generate and submit test coverage results for coveralls.io.
# Also, install docstring-coverage to get information about documentation coverage.
conda install -yq nose coverage nose-timer
conda install -yq docstring-coverage || true

# Clean up downloads as there are quite a few and they waste space/memory.
conda clean -tipsy

########
# TEST #
########

# Run tests. Skip 3D tests as they take too long (~1hr).
python setup.py nosetests --with-timer

# Build documentation.
python setup.py build_sphinx

# Get info on docstring coverage.
(hash docstring-coverage && docstring-coverage nanshe | tee .docstring-coverage) || true
