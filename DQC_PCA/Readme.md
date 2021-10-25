# Overview

This script performs a principal component analysis (PCA) of the input spectra, using either the raw intensities as input or the parameters of the major peaks in the spectrum. The peak parameters are obtained by the fitting of the peaks of the spectra of individual probes. Each peak is fitted using a skewed pseudo-voigt (a sum of a Gaussian and Lorentzian rather than a convolution) profile as per many studies in the literature, which is described by a five parameters: amplitude, peak center, width, skewness, and the Gaussian and Lorentzian proportion (alpha). However, for the sake of interpretability/comparability, the output parameters include only the fraction and skewness, with the intensity, position and fwhm calculated from the deconvoled model peaks individually. Requires the lmfit and numdifftools python packages.

# Requirements

The code requires that the data used as input (the new spectra to be fitted/decomposed) be placed in the folder 'Data', with the training spectral data (saved as msgpack files) placed in the folder 'Train_Spectra/Compiled/'. The parameters used for training are to be saved in the main folder under the name 'Dispensing_QC_train.csv'. Note that this script is able to handle multiple batches of dispensed chip spectra. This requires the individual batches to be placed in appropriately named subfolders within the 'Data' folder, which are then used to name the batches. This script also uses a modified version of the pca
python package (pca_mod.py), which leverages the scikit-learn package for PCA calculations.

# How to run

The code is executed using the command python Dispensing_QC_param_PCA.py peak_ranges 1,4,6, where peak_ranges is an input text file containing seven columns that specify the 1) Probe number, 2) start and the 3) end of the wavenumber ranges of the spectral regions that are used, 4) the number of peaks in each region, and 5) the expected position, 6) intensity and 7) index of a problematic peak (e.g. a blended shoulder region of another peak). If the value of 5) is set to zero, it is assumed that there are no problmatic peaks. Incorrect specification of the any of the aforementioned parameters may cause the fitting to fail (i.e. give poor results), so it is important to have knowledge/plot of each spectral region before performing any fitting (a good practice in general for such problems).

The second argument (i.e. 1,4,6) allows you to choose the probes which you want to fit. Note that some probes take way longer to fit than others (1,4 and 5 in particular), so please be conscious of that.

# Other important notes

Plots of the first two principal components (which explain the greatest percentage variance of the distribution) for each scenario are then saved in Fit_Results folder.
