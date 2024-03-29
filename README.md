![AMiGO logo](https://github.com/TVanReeth/amigo/blob/master/doc/logo_white-bkg.png?raw=true)

AMiGO: Asymptotic Modelling of Gravity-mode Oscillations
========================================================

Author: Timothy Van Reeth
        timothy.vanreeth@kuleuven.be

AMiGO (Asymptotic Modelling of G-mode Oscillations) is a python package to (i) calculate
 theoretical asymptotic g-mode period-spacing patterns for rotating stars and (ii)
measure near-core rotation rates of observed stars by analysing their g-mode pulsations.
In AMiGO, (i) any mode trapping caused by the chemical structure of the star is ignored,
and (ii) it is assumed that the g-mode pulsations are in the asymptotic regime, that is, 
have pulsation frequencies << N , where N is the Bruntaisala frequency. Moreover,
the Traditional Approximation of Rotation (TAR) is used: the horizontal component of the 
rotation vector is ignored in the equation of motion. Finally, unless otherwise specified,
the star is assumed to be uniformly rotating and spherically symmetric.

AMiGO combines multiple algorithms, which have been described in separate scientific
publications (listed below). We refer the user to these publications and the
references therein for a more detailed of the scientific framework(s).

AMiGO allows the user to:
* determine the mode identification of observed g-mode period spacing patterns.
* measure (uniform) near-core rotation rates of observed stars by fitting their observed
g-mode period spacing patterns.
* measure (uniform) near-core rotation rates of observed stars by fitting individual ob-
served g-mode periods.
* calculate the effects of radially differential rotation on g-mode period-spacing patterns.
* account for the effects of the weak centrifugal acceleration on g-mode pulsation periods
in a uniformly rotating star.


Installation instructions
-------------------------

To install AMIGO, you need to:
1. install the required python environment. AMIGO requires python >=3.9 and 
   <=3.12, and uses Poetry (https://python-poetry.org/docs/) to manage package 
   dependencies. You can combine this with the virtual Python environment manager of your
   choice, e.g., Conda (https://conda.io/projects/conda/en/latest/index.html).

2. ensure you have access to the source code of a recent version of GYRE 
   (v6.x or later; https://gyre.readthedocs.io/en/stable/). It does not have to
   be installed, but AMIGO relies on some of the data files that are included 
   with the GYRE source code.
   
3. When the prerequisites are met, the git repository can be cloned into a directory <dir> of your choice by typing these commands into a terminal:
```
           $ cd <dir>
           $ git clone https://github.com/TVanReeth/amigo.git amigo
```
        
4. Add the directory <dir> as a Python path to your `~/.bashrc` file:
```
           $ export PYTHONPATH="${PYTHONPATH}:<dir>"
```
        
5. Activate the python virtual environment in which you want to install the required
   python packages. To avoid possible conflicting dependencies, we advice to build and
   activate a custom environment. E.g., with conda this can be done by typing:
```
           $ conda create -n amigo_py python=3.9
           $ conda activate amigo_py
```
   
6. Use Poetry to install the required Python packages with all their dependencies.
```
           $ cd amigo
           $ poetry install
```
   
7. Modify the parameters in the configuration file `<dir>/amigo/defaults/config.dat` as needed.

8. Optional: include the following alias in your `~/.bashrc` file:
```
        alias amigo=’python <dir>/amigo/amigo/compute_rotation.py’
```

Throughout the rest of this documentation, it is assumed that this alias command has
been defined.


Using AMIGO
-----------

AMiGO can be used as a standalone software package to model observed period-spacing patterns,
or (selected) subroutines can be called from a python script. 

Various use cases of AMiGO are demonstrated in the tutorials subdirectory. In tutorials 1 to 3,
AMiGO is run as a standalone software package, while in tutorials 4 to 7 selected subroutines
are called. It is assumed that the python environment in which AMiGO was installed, is activated.

To learn more about using AMiGO, we refer the user to these tutorials and the Amigo-documentation.pdf 
file in the /doc subdirectory.

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
software package (https://gyre.readthedocs.io/en/stable/). We are grateful
to Rich Townsend and all its other developers. Their hard work has allowed 
us to develop the AMiGO package.

