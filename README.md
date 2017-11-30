# TARDIS

This repository contains TARDIS the software that coregister embryos, proposed in our Mouse-Atlas article.

## Description of the repository
Folders:
  - TGMMlibraries: The class `lineageTree`, a container for lineage trees and Statistical Vector Flow (SVF). Can read output data from TGMM.
  - Transformation: A class for linear transformations manipulation writen by Christoph Gohlke from Laboratory for Fluorescence Dynamics in University of California (http://www.lfd.uci.edu/~gohlke/).
  - Annotation-example: Example of annotations for TARDIS. The xml files are MaMuT xml files (https://imagej.net/MaMuT).
  - csv-parameter-files: Example of parameterization csv files for TARDIS.
Python files:
  - TARDIS.py: python script to coregister two segmented embryos.

## Basic usage
The TARDIS.py python script proposed here can be run from a terminal the following way:

`python TARDIS.py`

It is then prompted to provide a parameter csv file (example provided in the folder csv-parameter-files). The path to the parameter file should be then typed in the terminal.

## Dependencies
Some dependecies are requiered:
  - general python dependecies:
    - numpy, scipy, pandas
  - TGMMlibraries:
    - No dependecies besides the general ones
  - TARDIS.py:
    - TGMMlibraries has to be installed and included in PYTHONPATH so it can be called in python as `from TGMMlibraries import lineageTree` for example
    - Transformations has to be installed and included in PYTHONPATH so it can be called in python as `from Transformations.transformations import quaternion_from_matrix` for example

To include the TGMM library in the python path one can use the following command in a terminal:
  - For Ubuntu: `echo 'export PYTHONPATH=/path/to/TGMMlibraries:$PYTHONPATH' >> ./.bashrc`
  - For MacOs: `echo 'export PYTHONPATH=/path/to/TGMMlibraries:$PYTHONPATH' >> ./.profile`
 
Similarly, to include the Transformations library in the python path one can use the following command in a terminal:
  - For Ubuntu: `echo 'export PYTHONPATH=/path/to/Transformations:$PYTHONPATH' >> ./.bashrc`
  - For MacOs: `echo 'export PYTHONPATH=/path/to/Transformations:$PYTHONPATH' >> ./.profile`
