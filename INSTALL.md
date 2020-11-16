# Installation

It is recommended to install this package in a Python virtual environment, as described below.

## On Ubuntu Linux (20.04)

1. Install virtualenvwrapper to manage Python virtual environments:
   * `sudo apt install virtualenvwrapper`
1. Create a virtual environment:
   * `mkvirtualenv uobrainflex`
   * This should active the virtual environment, so `(uobrainflex)` should appear at the beginning of your prompt.
1. Choose or create a folder to install this package and go to that folder:
   * `mkdir ~/src/`
   * `cd ~/src/`
1. Clone the repository:
   * `git clone https://github.com/sjara/uobrainflex.git`
1. Install the package in editable/development mode:
   * `cd uobrainflex`
   * pip install -e ./
1. Create a configuration file and edit your settings:
   * `cp uobrainflex_TEMPLATE.conf uobrainflex.conf`
   * Open it with you favorite editor and change the settings.
1. Test the installation:
   * `cd scripts`
   * `python3 test_installation.py`


## On Windows
These instructions assume you have installed the following applications:
* Python (via the Anaconda Individual Edition): https://www.anaconda.com/products/individual#windows
* git (64bit): https://git-scm.com/download/win
  * Recommended: during installation, choose Nano as the default editor.

Open the Anaconda Powershell Prompt to follow the steps below:
1. Create a virtual environment:
   * conda create -n uobrainflex --clone base
1. Activate the virtual environment:
   * conda activate uobrainflex
   * If successful, `(uobrainflex)` should appear at the beginning of your prompt.
1. Install dependencies:
   * conda install -c conda-forge pynwb
   * conda install -c conda-forge datajoint 
1. Choose or create a folder to install this package and go to that folder:
   * `mkdir ~/src/`
   * `cd ~/src/`
1. Clone the repository:
   * `git clone https://github.com/sjara/uobrainflex.git`
1. Install the package in editable/development mode:
   * `cd uobrainflex`
   * `pip install -e ./`
1. Create a configuration file and edit your settings:
   * `cp uobrainflex_TEMPLATE.conf uobrainflex.conf`
   * Open it with you favorite editor and change the settings.
1. Test the installation:
   * `cd scripts`
   * `python test_installation.py`
