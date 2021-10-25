# Overview

This script performs the fitting of the peaks of the spectra of individual probes that have been saved in msgpack format. Each peak is fitted using a skewed pseudo-voigt (a sum of a Gaussian and Lorentzian rather than a convolution) profile as per many studies in the literature, which is described by a five parameters: amplitude, peak center, width, skewness, and the Gaussian and Lorentzian proportion (alpha). However, for the sake of interpretability/comparability,
the output parameters include only the fraction and skewness, with the intensity, position and fwhm calculated from the deconvoled model peaks individually. The spectra of the individual probes is generated using the script `raw_json_processing.py`, so run that first before using this script. Requires the lmfit, numdifftools, mpi4py, mpipool, str2bool and msgpack python packages. These can be convieniently installed by using the command `pip install -r requirements.txt`, where `requirements.txt` is a text file distributed along with this script. Note that mpi4py installation is non-trivial on windows, so please look into installing and configuring Microsoft MPI before attempting to use this script.

# How to run

This script must use parallel functionality (i.e. can make use of multiple CPU cores on multiple machines)! To execute a parallel computation on 4 cores (e.g. on a laptop), use the command `mpiexec -n 4 python peak_fitter.py inlist peak_ranges`, where `inlist` is an input text file containing the following variables:

INPUT_FOLDER_PATH - Folder where the compiled probe spectra are stored (e.g. /home/Destop/spectra/QC_processed/)
RESULTS_FOLDER_PATH - Folder where the results will be stored\
RESUME - Specifies whether the fitting should start from the beginning (False) or continue from the last saved result (True). The default is False.

and `peak_ranges` is a second input text file containing seven columns that specify the 

1) Probe number
2) start
3) end of the wavenumber ranges of the spectral regions that are used 
4) the number of peaks in each region
5) the expected position
6) intensity
7) index of a problematic peak (e.g. a blended shoulder region of another peak)
 
If the value of 5. is set to zero, it is assumed that there are no problmatic peaks. Incorrect specification of the any of the aforementioned parameters may cause the fitting to fail (i.e. give poor results), so it is important to have knowledge/plot of each spectral region before performing any fitting (a good practice in general for such problems).

# Other important notes

If RESULTS_FOLDER_PATH is not specified, the fitting results (i.e. model spectrum, parameters and errors) will be stored in csv files (due to its tiny memory footprint) in a parallel folder named 'Fit_Results' to INPUT_FOLDER_PATH (e.g. if INPUT_FOLDER_PATH = /home/Destop/spectra/QC_processed/, OUTPUT_FOLDER_PATH defaults to /home/Destop/spectra/Fit_Results/).

@author: Sanjay Sekaran
