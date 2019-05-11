# CSCI 8360 - Solar Irradiance Prediction Using PySpark

## Synopsis
The goal of this project is to develop a pipeline which implements a variety of machine learning algorithms for predicting solar irradiance, on [Spark](https://spark.apache.org/docs/2.2.1/api/python/pyspark.html). The software's design centers on execution in [Google Cloud Platform](https://cloud.google.com/) along with accessing pertinent data from a Google Storage Bucket.

## Important Links
*	[Getting Started]()
* [Theory](https://github.com/dsp-uga/einstein/wiki)
*	[Contribution Guidelines](https://github.com/dsp-uga/einstein#contributing)
*	[Contributors](https://github.com/dsp-uga/einstein#contributors)


## Getting Started
This section describes the preqrequisites, and contains instructions, to get the project up and running.

### Setup
 This project can be easily set up on the Google Cloud Platform, by creating a cluster using Google's [Cloud DataProc](https://cloud.google.com/dataproc/) services. On doing so, '[Compute Engine VM Instances](https://console.cloud.google.com/compute/instances)' will be created, from among which, you can open the VM instance of the master node. 

**NOTE**: The `conda-dataproc-bootstrap.sh` script in the "scripts" directory needs to be used as an "Initialization Action" while creating the cluster.

The Initialization Action will set up a `conda` environment in all the VM instances, in which, Spark, latest version of Python, and the following libraries will be installed: `findspark`, `pytest`, `pvlib`, `siphon`,`tables`, `netCDF4`, `pyAesCrypt`.

To copy the `einstein` project to the VM instance, there are two alternatives:
* [Google Cloud SDK](https://cloud.google.com/sdk/install): When installed in the local machine, the user can copy the contents of this repository to the DataProc VM instance using the command:

  `> gcloud compute scp --recurse /complete/link/to/repository/* <user>@<instance_name-vm>:~/`
* The project directory can be compressed using the command: `> tar -zvcf einstein.tar.gz einstein`; the compressed folder can be uploaded into the VM instance using the `Upload File` option in the VM, and can further be decompressed using the command: `> tar -xzf einstein.tar.gz`. 

Run the following command: `bash einstein/scripts/setup.sh` to setup the bash utilities for the project.


### Usage
 To run the experiments and analyze solar irradiance predictions, the user can run the following command: 
 
`> einstein --options`
 
 The user can get a description of the options by using the command: `> einstein --help`.

### Built With
* [Python](https://www.python.org/)
* [PySpark](https://spark.apache.org/docs/2.3.1/api/python/index.html) - Python API for [Apache Spark](https://spark.apache.org/)
* [Spark MLLib](https://spark.apache.org/mllib/) 

## Contributing Guidelines
There are no specific guidelines for contributing, apart from a few general guidelines we tried to follow, such as:
* Code should follow PEP8 standards as closely as possible
* We use [Google-Style docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) to document the Python modules in this project.

If you see something that could be improved, send a pull request! 
We are always happy to look at improvements and new experiments, to ensure that `einstein`, as a project, is the best version of itself. 

If you think something should be done differently (or is just-plain-broken), please create an issue.

## Contributors
See the [Contributors](https://github.com/aashishyadavally/einstein/blob/master/CONTRIBUTORS.md) file for details about individual contributions.
