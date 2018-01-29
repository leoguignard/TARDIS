# TARDIS

This repository contains TARDIS the software that coregister embryos, proposed in our Mouse-Atlas article.

## Description of the repository
Folders:
  - Annotation-example: Example of annotations for TARDIS. The xml files are MaMuT xml files (https://imagej.net/MaMuT).
  - csv-parameter-files: Example of parameterization csv files for TARDIS.

Python files:
  - TARDIS.py: python script to coregister two segmented embryos.

## Basic usage
The TARDIS.py python script proposed here can be run from a terminal the following way:

`python TARDIS.py`

It is then prompted to provide a parameter csv file (example provided in the folder csv-parameter-files). The path to the parameter file should be then typed in the terminal.

**If the package have been installed using setup.py, the script can be call just by typing `TARDIS.py` from anywhere in the terminal**

## Dependencies
Some dependecies are requiered:
  - general python dependecies:
    - numpy, scipy, pandas
  - TARDIS.py:
    - TGMMlibraries has to be installed (https://github.com/leoguignard/TGMMlibraries)
    - Transformations has to be installed (https://github.com/leoguignard/Transformations)
    
## Quick install
To quickly install the script so it can be call from the terminal and install too the common dependecies one can run
```shell
python setup.py install [--user]
```
Still will be remaining to install Transformations and TGMMlibraries packages.
