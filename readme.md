AMIGO: Asteroseismic Modelling of Gravity-mode Oscillations
===========================================================

Author: Timothy Van Reeth
        timothy.vanreeth@kuleuven.be

An object-oriented python module to help you to do forward asteroseismic modelling of observed (patterns of) gravito-inertial modes.


Instructions
------------

To install AMIGO, you need to:
- download the source code from https://github.com/TVanReeth/amigo.
- install the required python environment from the included configuration file amigo_py.yml.
- ensure you have access to the source code of a recent version of GYRE (v5.x or later; https://gyre.readthedocs.io/en/stable/). It does not have to be installed, but AMIGO relies on some of the data files that are included with the GYRE source code.


To run AMIGO, you need to:
- activate the amigo_py python environment.
- adapt the contents of the inlist UserInput_rotation.dat to your use case. Most importantly, 
    - the path to the GYRE directory has to be modified if you do not have access to the computer system of the Institute of Astronomy at KU Leuven (Belgium).
    - ensure that the path to the file with the observed g-mode patterns is complete and correct.
- enter the terminal command: 

    $ python computate_rotation.py


In the subdirectory /demo you can find:
- a backup of the inlist UserInput_rotation.dat for the demo.
- an example of a file with two period-spacing patterns, used in the demo.
- an additional python script (demo_plot_pattern.py) which illustrates how the code can be used to calculate and plot a single, specified g-mode period-spacing pattern.

An introduction to the physics implemented in AMIGO can be found in Van Reeth et al.(2016; https://ui.adsabs.harvard.edu/abs/2016A%26A...593A.120V/abstract). Please note that when this manuscript was written, the asymptotic spacings of g-mode patterns were still expressed as $\Delta \Pi_{l} = \Pi_0 / \sqrt{l(l+1)}$ rather than just the buoyancy travel time $\Pi_0$ (as is done in AMIGO).
