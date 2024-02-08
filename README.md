AMiGO: Asymptotic Modelling of Gravity-mode Oscillations
========================================================

Author: Timothy Van Reeth
        timothy.vanreeth@kuleuven.be

An object-oriented python module to help you to do forward asteroseismic 
modelling of observed (patterns of) gravito-inertial modes, using asymptotic
models.


Installation instructions
-------------------------

To install AMIGO, you need to:
1. download the source code from https://github.com/TVanReeth/amigo.
2. install the required python environment. AMIGO requires python >=3.9 and 
   <=3.12, and uses Poetry (https://python-poetry.org/docs/) to manage package 
   dependencies.
3. ensure you have access to the source code of a recent version of GYRE 
   (v6.x or later; https://gyre.readthedocs.io/en/stable/). It does not have to
   be installed, but AMIGO relies on some of the data files that are included 
   with the GYRE source code.


Using AMIGO
-----------

To run AMIGO, you need to:
1. activate the python environment.
2. adapt the contents of the inlist UserInput_rotation.dat to your use case. 
   Most importantly, the path to the GYRE directory has to be modified.
3. enter the terminal command: 
       python computate_rotation.py

In the subdirectory /demo you can find:
- a backup of the inlist UserInput_rotation.dat for the demo.
- an example of a file with two period-spacing patterns, used in the demo.
- an additional python script (demo_plot_pattern.py) which illustrates how the 
  code can be used to calculate and plot a single, specified g-mode 
  period-spacing pattern.


Acknowledgements
----------------

The AMIGO python package itself contains algorithms developed in different 
scientific studies. Please cite the appropriate manuscripts when using the 
different submodules. These include:

1. The modelling of observed period-spacing patterns by asymptotically fitting 
   the spacing between consecutive g-mode periods as a function of the periods,
   assuming the star is uniformly rotating, spherically symmetric, and 
   non-magnetic.
   
   Van Reeth et al., 2016, A&A 593, A120
   https://ui.adsabs.harvard.edu/abs/2016A%26A...593A.120V/abstract

2. Asymptotically modelling observed period-spacing patterns, allowing for 
   radially differential rotation.
   
   Mathis, 2009, A&A 506, 2
   https://ui.adsabs.harvard.edu/abs/2009A%26A...506..811M/abstract
   
   Van Reeth et al., 2018, A&A 618, A24
   https://ui.adsabs.harvard.edu/abs/2018A%26A...618A..24V/abstract

3. Asymptotically modelling observed period-spacing patterns, assuming uniform 
   rotation but taking into account small deformations of the star caused by 
   the centrifugal acceleration.
   
   Mathis & Prat, 2019, A&A 631 A26
   https://ui.adsabs.harvard.edu/abs/2019A%26A...631A..26M/abstract 
   
   Henneco et al., 2021, A&A 648, A97
   https://ui.adsabs.harvard.edu/abs/2021A%26A...648A..97H/abstract

4. Asymptotically modelling observed period-spacing patterns by fitting g-mode
   frequencies as a function of the (estimated) radial order. This allows us to
   analyse more sparse patterns.
   
   Van Reeth et al., 2022, A&A 662, A58 
   https://ui.adsabs.harvard.edu/abs/2022A%26A...662A..58V/abstract

Moreover, please also fulfill the acknowledgement requirements of the GYRE
software package (https://gyre.readthedocs.io/en/stable/), since AMIGO relies
on it.
