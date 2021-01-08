# uobrainflex
Code related to the project "Brain States and Flexible Behavior".

This Python package contains tools for converting data to NWB format, copying data to a cloud storage, and interacting with a DataJoint database. The package is organized as folllows:
* `examples/`: programs illustrating the use of this package.
* `matlab/`: Matlab scripts for interacting with the data created by this package.
* `scripts/`: programs for converting/copying data, intended to be run directly.
* `uobrainflex/`: the Python package.
  * `nwb/`: modules for creating NWB files.
  * `pipeline/`: modules defining the DataJoint pipeline.
  * `utils/`: utilities for interacting with cloud storage, DataJoint, etc.
