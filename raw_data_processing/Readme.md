# Overview

This script performs the processing and calibration of raw spectra, outputting the calibrated, baseline corrected, normalised and averaged spectra of the individual probes as msgpack (super fast read/write) files or (optionally) csv files, compiled according to the individual probes (and scans if QC data). This script requires the spectra and calibrations to be placed in the same folder under subfolders named 'Data' (for the spectra) and 'Calibrations' (for the calibrations) respectively. This can be done manually, or by running the script `S3_Downloader_batch.py` (currently not distributed) which both downloads and handles the files for you. Requires the mpi4py, mpipool and tqdm python packages. Note that mpi4py installation is non-trivial on windows, so please look into installing and configuring Microsoft MPI before attempting to use this script.

# How to run

This script must use parallel functionality (i.e. can make use of multiple CPU cores on multiple machines)! To execute a parallel computation on 4 cores (e.g. on a laptop), use the command `mpiexec -n 4 python raw_data_processing.py inlist`, where `inlist` is an input text file containing the following variables:

 - INPUT_FOLDER_PATH - Folder where the spectra and calibrations are stored (e.g. /home/Destop/spectra/)
 - OUTPUT_FOLDER_PATH - Folder where the output will be stored
 - SPECTROMETER - Name of the spectrometer (Zolix or Mira)
 - RESOLUTION - The step in wavelength for the output spectra
 - PROBE_ORDER - The number of the probes that are to be processed
 - DATA_TYPE - The type of input spectra for processing, either the deployment/live testing type (COVID), the multi-scan QC type (QC) or the new types (NEW and the msgpack QC_NEW). Defaults to COVID if not specified
 - INPUT_TYPE - The file type for the raw input data (JSON or MSG). Defaults to JSON if not specified
 - OUTPUT_TYPE - The file type for the processed output data (MSG or CSV). Defaults to MSG if not specified

# Other important notes

If OUTPUT_FOLDER_PATH is not specified, the output will be stored in the folder specified by INPUT_FOLDER_PATH. The msgpack spectra are placed in the subfolder 'Compiled' in the output folder.

@author: Sanjay Sekaran, with parts adapted from Khairi
