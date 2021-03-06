#This script was written by Dr. Shannon Quinn for distributing to the students, part of 
#CSCI 8360 : Data Science Practicum course, to let us create clusters with an 
#initialization action.
#
#A few additional packages were added at the end to download using `pip`as a part of the setup.
#
#For the purpose of testing, this file was put in a Google Storage Bucket, the access
#of which is limited to the developers. For reproducing with the `einstein` package,
#you are advised to upload this file to a Google Storage Bucket and use that link as
#an argument to the module. 

#!/bin/bash
set -e
# Modified from bootstrap-conda.sh script, see:
# https://bitbucket.org/bombora-datascience/bootstrap-conda

# Run on BOTH 'Master' and 'Worker' nodes
#ROLE=$(/usr/share/google/get_metadata_value attributes/dataproc-role)
#if [[ "${ROLE}" == 'Master' ]]; then
# Some variables to cut down on space. Modify these if you want.

URL_PREFIX=https://repo.continuum.io/archive/
ANACONDA_VARIANT=3
ANACONDA_VERSION=2019.03
OS_ARCH=Linux-x86_64
ANACONDA_FULL_NAME="Anaconda${ANACONDA_VARIANT}-${ANACONDA_VERSION}-${OS_ARCH}.sh"

CONDA_INSTALL_PATH="/opt/conda"
PROJ_DIR=${PWD}
LOCAL_CONDA_PATH=${PROJ_DIR}/anaconda

if [[ -f "/etc/profile.d/conda.sh" ]]; then
    echo "file /etc/profile.d/conda.sh exists! Dataproc has installed conda previously. Skipping install!"
    command -v conda >/dev/null && echo "conda command detected in $PATH"
else

    ANACONDA_SCRIPT_PATH="${PROJ_DIR}/${ANACONDA_FULL_NAME}"
    echo "Defined Anaconda script path: ${ANACONDA_SCRIPT_PATH}"

    if [[ -f "${ANACONDA_SCRIPT_PATH}" ]]; then
      echo "Found existing Anaconda script at: ${ANACONDA_SCRIPT_PATH}"
    else
      echo "Downloading Anaconda script to: ${ANACONDA_SCRIPT_PATH} ..."
      wget ${URL_PREFIX}${ANACONDA_FULL_NAME} -P "${PROJ_DIR}"
      echo "Downloaded ${ANACONDA_FULL_NAME}!"
      ls -al ${ANACONDA_SCRIPT_PATH}
      chmod 755 ${ANACONDA_SCRIPT_PATH}
    fi

    # 2. Install conda
    ## 2.1 Via bootstrap
    if [[ ! -d ${LOCAL_CONDA_PATH} ]]; then
        #blow away old symlink / default Anaconda install
        rm -rf "${LOCAL_CONDA_PATH}"
        # Install Anaconda
        echo "Installing ${ANACONDA_FULL_NAME} to ${CONDA_INSTALL_PATH}..."
        bash ${ANACONDA_SCRIPT_PATH} -b -p ${CONDA_INSTALL_PATH} -f
        chmod 755 ${CONDA_INSTALL_PATH}
        #create symlink
        ln -sf ${CONDA_INSTALL_PATH} "${LOCAL_CONDA_PATH}"
        chmod 755 "${LOCAL_CONDA_PATH}"
    else
        echo "Existing directory at path: ${LOCAL_CONDA_PATH}, skipping install!"
    fi
fi

## 2.2 Update PATH and conda...
echo "Setting environment variables..."
CONDA_BIN_PATH="${CONDA_INSTALL_PATH}/bin"
export PATH="${CONDA_BIN_PATH}:${PATH}"
echo "Updated PATH: ${PATH}"
echo "And also HOME: ${HOME}"
hash -r
which conda
conda config --set always_yes true --set changeps1 false

# Useful printout for debugging any issues with conda
conda info -a

## 2.3 Update global profiles to add the anaconda location to PATH
# based on: http://stackoverflow.com/questions/14637979/how-to-permanently-set-path-on-linux
# and also: http://askubuntu.com/questions/391515/changing-etc-environment-did-not-affect-my-environemtn-variables
# and this: http://askubuntu.com/questions/128413/setting-the-path-so-it-applies-to-all-users-including-root-sudo
echo "Updating global profiles to export anaconda bin location to PATH..."
if grep -ir "CONDA_BIN_PATH=${CONDA_BIN_PATH}" /etc/profile.d/conda.sh
    then
    echo "CONDA_BIN_PATH found in /etc/profile.d/conda.sh , skipping..."
else
    echo "Adding path definition to profiles..."
    echo "export CONDA_BIN_PATH=$CONDA_BIN_PATH" | tee -a /etc/profile.d/conda.sh #/etc/*bashrc /etc/profile
    echo 'export PATH=$CONDA_BIN_PATH:$PATH' | tee -a /etc/profile.d/conda.sh  #/etc/*bashrc /etc/profile

fi

# 2.3 Update global profiles to add the anaconda location to PATH
echo "Updating global profiles to export anaconda bin location to PATH and set PYTHONHASHSEED ..."
if grep -ir "export PYTHONHASHSEED=0" /etc/profile.d/conda.sh
    then
    echo "export PYTHONHASHSEED=0 detected in /etc/profile.d/conda.sh , skipping..."
else
    # Fix issue with Python3 hash seed.
    # Issue here: https://issues.apache.org/jira/browse/SPARK-13330 (fixed in Spark 2.2.0 release)
    # Fix here: http://blog.stuart.axelbrooke.com/python-3-on-spark-return-of-the-pythonhashseed/
    echo "Adding PYTHONHASHSEED=0 to profiles and spark-defaults.conf..."
    echo "export PYTHONHASHSEED=0" | tee -a  /etc/profile.d/conda.sh  #/etc/*bashrc  /usr/lib/spark/conf/spark-env.sh
    echo "spark.executorEnv.PYTHONHASHSEED=0" >> /etc/spark/conf/spark-defaults.conf
fi

## 3. Ensure that Anaconda Python and PySpark play nice
### http://blog.cloudera.com/blog/2015/09/how-to-prepare-your-apache-hadoop-cluster-for-pyspark-jobs/
echo "Ensure that Anaconda Python and PySpark play nice by all pointing to same Python distro..."
if grep -ir "export PYSPARK_PYTHON=$CONDA_BIN_PATH/python" /etc/profile.d/conda.sh
    then
    echo "export PYSPARK_PYTHON=$CONDA_BIN_PATH/python detected in /etc/profile.d/conda.sh , skipping..."
else
    echo "export PYSPARK_PYTHON=$CONDA_BIN_PATH/python" | tee -a  /etc/profile.d/conda.sh /etc/environment /usr/lib/spark/conf/spark-env.sh
fi

echo "Setting IPython as the default PySpark shell..."
echo "export PYSPARK_DRIVER_PYTHON=ipython" >> ${HOME}/.bashrc

echo "Finished bootstrapping via Anaconda, sourcing /etc/profile ..."
source /etc/profile

# Update everything.
conda update -y --all

# instaling required packages
echo "installing pip packages"
pip install pvlib siphon tables findspark pytest netCDF4 pyAesCrypt
