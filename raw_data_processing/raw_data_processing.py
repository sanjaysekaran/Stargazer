"""
This script performs the processing and calibration of raw spectra, outputting the
calibrated, baseline corrected, normalised and averaged spectra of the individual probes as msgpack
(super fast read/write) files or (optionally) csv files, compiled according to the individual probes
(and scans if QC data). This script requiresthe spectra and calibrations to be placed in the same folder
under subfolders named 'Data' (for the spectra) and 'Calibrations' (for the calibrations) respectively.
This can be done manually, or by running the script S3_Downloader_batch.py which
both downloads and handles the files for you. Requires the mpi4py, mpipool, tqdm and msgpack python
packages.

This script must use parallel functionality (i.e. can make use of multiple CPU cores on multiple machines)!
To execute a parallel computation on 4 cores (e.g. on a laptop), use the command mpiexec -n 4 python raw_data_processing.py inlist,
where inlist is an input text file containing the following variables:

INPUT_FOLDER_PATH - Folder where the spectra and calibrations are stored (e.g. /home/Destop/spectra/)
OUTPUT_FOLDER_PATH - Folder where the output will be stored
SPECTROMETER - Name of the spectrometer (Zolix or Mira)
RESOLUTION - The step in wavelength for the output spectra
PROBE_ORDER - The number of the probes that are to be processed
DATA_TYPE - The type of input spectra for processing, either the deployment/live testing type (COVID), the multi-scan QC type (QC)
or the new types (NEW and the msgpack QC_NEW). Defaults to COVID if not specified
INPUT_TYPE - The file type for the raw input data (JSON or MSG). Defaults to JSON.
OUTPUT_TYPE - The file type for the processed output data (MSG or CSV). Defaults to MSG.

If OUTPUT_FOLDER_PATH is not specified, the output will be stored in the folder specified by INPUT_FOLDER_PATH.
The msgpack spectra are placed in the subfolder 'Compiled' in the output folder.

@author: Sanjay Sekaran, with parts adapted from Khairi
"""

import os, sys, glob, csv
import msgpack as msg
import errno
import numpy as np
import re
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import time
import json
from scipy import interpolate
from scipy.signal import savgol_filter
from scipy.sparse import csc_matrix, eye, diags
from scipy.sparse.linalg import spsolve
from mpipool import MPIExecutor
from mpi4py import MPI
from concurrent.futures import as_completed
import functools
from tqdm import tqdm
import platform

# %% Defining functions
def WhittakerSmooth(x, w, lambda_, differences=1):
    """
    Penalized least squares algorithm for background fitting

    input
        x: input data (i.e. chromatogram of spectrum)
        w: binary masks (value of the mask is zero if a point belongs to peaks and one otherwise)
        lambda_: parameter that can be adjusted by user. The larger lambda is,  the smoother the resulting background
        differences: integer indicating the order of the difference of penalties

    output
        the fitted background vector
    """
    X = np.matrix(x)
    m = X.size
    i = np.arange(0, m)
    E = eye(m, format='csc')
    D = E[1:] - E[:-1]  # numpy.diff() does not work with sparse matrix. This is a workaround.
    W = diags(w, 0, shape=(m, m))
    A = csc_matrix(W + (lambda_ * D.T * D))
    B = csc_matrix(W * X.T)
    background = spsolve(A, B)
    return np.array(background)

def airPLS(x, lambda_=100, porder=1, itermax=15):
    """
    Adaptive iteratively reweighted penalized least squares for baseline fitting

    input
        x: input data (i.e. chromatogram of spectrum)
        lambda_: parameter that can be adjusted by user. The larger lambda is,  the smoother the resulting background, z
        porder: adaptive iteratively reweighted penalized least squares for baseline fitting

    output
        the fitted background vector
    """
    m = x.shape[0]
    w = np.ones(m)
    for i in range(1, itermax + 1):
        z = WhittakerSmooth(x, w, lambda_, porder)
        d = x - z
        dssn = np.abs(d[d < 0].sum())
        if (dssn < 0.001 * (abs(x)).sum() or i == itermax):
            if (i == itermax):
                print('WARING max iteration reached!')
            break
        w[d >= 0] = 0  # d>0 means that this point is part of a peak, so its weight is set to 0 in order to ignore it
        w[d < 0] = np.exp(i * np.abs(d[d < 0]) / dssn)
        w[0] = np.exp(i * (d[d < 0]).max() / dssn)
        w[-1] = w[0]
    return z

def calibrate_x(x, coefficients, lambda0_wavenumber):
    corrected_wavelength_nm = coefficients[0] + (x*coefficients[1]) + ((x**2)*coefficients[2])
    corrected_wavelength_cm = corrected_wavelength_nm * 1e-7
    corrected_wavenumber = 1 / corrected_wavelength_cm
    calibrated_x = lambda0_wavenumber - corrected_wavenumber
    return calibrated_x

def assert_dir_exists(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def ReadInlist(inlist):

  INPUT_FOLDER_PATH = None
  OUTPUT_FOLDER_PATH = None
  SPECTROMETER = None
  RESOLUTION = None
  PROBE_ORDER = None
  DATA_TYPE = None
  INPUT_TYPE = None
  OUTPUT_TYPE = None

  file = open(inlist,'r')
  lines = file.readlines()
  file.close()

  for line in lines:
    (par,val) = line.strip().split('=')
    if(par.strip() == 'INPUT_FOLDER_PATH'):
      INPUT_FOLDER_PATH = val.strip()
    if(par.strip() == 'OUTPUT_FOLDER_PATH'):
      OUTPUT_FOLDER_PATH = val.strip()
    if(par.strip() == 'SPECTROMETER'):
      SPECTROMETER = val.strip()
    if(par.strip() == 'RESOLUTION'):
      RESOLUTION = val.strip()
    if(par.strip() == 'PROBE_ORDER'):
      PROBE_ORDER = val.strip()
    if(par.strip() == 'DATA_TYPE'):
      DATA_TYPE = val.strip()
    if(par.strip() == 'INPUT_TYPE'):
      INPUT_TYPE = val.strip()
    if(par.strip() == 'OUTPUT_TYPE'):
      OUTPUT_TYPE = val.strip()

  #If output path not specified, default to input
  if OUTPUT_FOLDER_PATH == None:
      OUTPUT_FOLDER_PATH = INPUT_FOLDER_PATH

  #If DATA_TYPE not specified, default to COVID
  if DATA_TYPE == None:
      DATA_TYPE == 'COVID'

  #If INPUT_TYPE not specified, default to JSON
  if INPUT_TYPE == None:
      INPUT_TYPE == 'JSON'

  #If OUTPUT_TYPE not specified, default to MSG
  if OUTPUT_TYPE == None:
      OUTPUT_TYPE == 'MSG'

  #Probe order parsing
  PROBE_ORDER = PROBE_ORDER.split(",")

  if(INPUT_FOLDER_PATH == '') or not os.path.exists(INPUT_FOLDER_PATH):
    print("Error: please provide a valid path to the input folder.")
    sys.exit()
  if(OUTPUT_FOLDER_PATH == '') and not os.path.exists(OUTPUT_FOLDER_PATH):
    print("Error: please provide a valid path to the output folder.")
    sys.exit()
  if(SPECTROMETER == '') or (SPECTROMETER != 'Mira' and SPECTROMETER != 'Zolix'):
    print("Error: please input the correct spectrometer.")
    sys.exit()
  if(RESOLUTION != None and RESOLUTION == ''):
    print("Error: please provide the spectral resolution of the instrument.")
    sys.exit()
  if(PROBE_ORDER == None or PROBE_ORDER == ''):
    print("Error: please specify the number of the probes that are to be processed (e.g. 1,2,4,5,6).")
    sys.exit()
  if(DATA_TYPE == '') or (DATA_TYPE != 'COVID' and DATA_TYPE != 'QC' and DATA_TYPE != 'NEW' and DATA_TYPE != 'QC_NEW'):
    print("Error: please specify the correct data type for the input data (COVID, QC, NEW or QC_NEW).")
    sys.exit()
  if(DATA_TYPE == 'QC_NEW' and INPUT_TYPE != 'MSG'):    #If DATA_TYPE = QC_NEW, INPUT_TYPE must be MSG
    print("INPUT_TYPE = JSON is not applicable for DATA_TYPE = QC_NEW. Setting INPUT_TYPE = MSG")
    INPUT_TYPE = 'MSG'
  if(INPUT_TYPE == '') or (INPUT_TYPE != 'JSON' and INPUT_TYPE != 'MSG'):
    print("Error: please specify the correct file type for the input data (JSON or MSG).")
    sys.exit()
  if(OUTPUT_TYPE == '') or (OUTPUT_TYPE != 'MSG' and OUTPUT_TYPE != 'CSV'):
    print("Error: please specify the correct file type for the input data (MSG or CSV).")
    sys.exit()

  #If resolution is not specified, default to 2.0 per cm
  if RESOLUTION == None:
      print("Resolution set to 2.0 per cm")
      RESOLUTION = 2.0

  return Path(INPUT_FOLDER_PATH),Path(OUTPUT_FOLDER_PATH),SPECTROMETER,float(RESOLUTION),PROBE_ORDER,DATA_TYPE,INPUT_TYPE,OUTPUT_TYPE

def atoi(text):
    try:
        return int(text)
    except ValueError:
        return text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    '''
    return [ atoi(c) for c in re.split('([-]?\d+)', text) ]

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx], idx

def list_duplicates(seq):
    tally = defaultdict(list)
    for i,item in enumerate(seq):
        tally[item].append(i)
    return ((key,locs) for key,locs in tally.items()
                            if len(locs)>1)

def save_files(avg_spectrum_data, OUTPUT_FOLDER_PATH, PROBE_ORDER, OUTPUT_TYPE, arc_filepaths, subfolder_string, filestring):
    #Save msg files for each probe
    for key in avg_spectrum_data.keys():
        if key in PROBE_ORDER:
            if OUTPUT_TYPE=='MSG':
                proc_filename = 'Probe' + key + '_processed.msg'
            else:
                proc_filename = 'Probe' + key + '_processed.csv'
            if subfolder_string is None:
                proc_filepath = Path(OUTPUT_FOLDER_PATH / 'Compiled' / proc_filename)
            else:
                proc_filepath = Path(OUTPUT_FOLDER_PATH / 'Compiled' / subfolder_string / proc_filename)
            assert_dir_exists(os.path.dirname(proc_filepath))
            if OUTPUT_TYPE=='MSG':
                print('Saving ' + filestring + ' in msgpack file ' + str(proc_filepath))
                #If not created, initialise archive
                if proc_filepath not in arc_filepaths:
                    arc_filepaths.append(proc_filepath)
                    with open(proc_filepath, 'wb') as outfile:
                        pass
                avg_spectrum_dict = {'spectrum': avg_spectrum_data[key]}
                avg_spectrum_dict['x'] = avg_spectrum_data['x']
                avg_spectrum_dict = {filestring: avg_spectrum_dict}
                with open(proc_filepath, 'ab') as outfile:
                    msg.pack(avg_spectrum_dict, outfile)
            else:
                #print('Saving ' + filestring + ' in csv file ' + str(proc_filepath))
                #If not created, write empty file
                if proc_filepath not in arc_filepaths:
                    arc_filepaths.append(proc_filepath)
                    with open(proc_filepath, 'w') as outfile:
                        pass
                with open(proc_filepath, 'a') as csvfile:
                    writer = csv.writer(csvfile)
                    if os.stat(proc_filepath).st_size==0: #Write file header if not written
                        line_to_write = ['Wavenumber']
                        line_to_write += ['%.8f' % y for y in avg_spectrum_data['x']]
                        writer.writerow(line_to_write)
                    line_to_write = [filestring]
                    line_to_write += ['%.8f' % y for y in avg_spectrum_data[key]]
                    writer.writerow(line_to_write)

def probe_processing(INPUT_FOLDER_PATH, SPECTROMETER, RESOLUTION, DATA_TYPE, filepath, raw_spectrum_data):
    start = time.time()

    # #Calibrations using external json file (experimental use only!)
    # #Determine probe names (i.e. number_repeat) and number of pixels of the raw spectrum (taken from the first probe)
    # raw_probe_names = [ ]
    # if DATA_TYPE=='NEW':
    #     for val in raw_spectrum_data.keys():
    #         if val=='spectrum' or val=='data':
    #             data_val = val
    #             for key in raw_spectrum_data[val].keys():
    #                 if key!='x' and key!='fingerprint':
    #                     for i in range(1,len(raw_spectrum_data[val][key]['raw'])+1):
    #                         probestr = key + '_0' + str(i) + '_raw'
    #                         raw_probe_names.append(probestr)
    #
    #     raw_probe_names = sorted(raw_probe_names, key=natural_keys)
    #
    #     npixels = len(raw_spectrum_data[data_val]['1']['raw'][0])
    #
    #     if 'spectrometer_id' in raw_spectrum_data.keys():
    #         calibration_id = raw_spectrum_data['spectrometer_id']
    #         calibration_path = os.path.join(INPUT_FOLDER_PATH / 'Calibrations', str(calibration_id) + '.json')
    #         try:
    #             calibration_data = json.load(open(calibration_path))
    #         #    print('Using calibration file ' + os.path.basename(calibration_path))
    #         except:
    #         #    print('Calibration file ' + os.path.basename(calibration_path) + ' not found!')
    #             pass
    #     else:
    #         calibration_id = raw_spectrum_data['metadata']['spectrometer_id']
    #         calibration_path = os.path.join(INPUT_FOLDER_PATH / 'Calibrations', str(calibration_id) + '.json')
    #         try:
    #             calibration_data = json.load(open(calibration_path))
    #         #    print('Using calibration file ' + os.path.basename(calibration_path))
    #         except:
    #         #    print('Calibration file ' + os.path.basename(calibration_path) + ' not found!')
    #             pass
    # else:
    #     for key in raw_spectrum_data.keys():
    #         if key.endswith('_raw'):
    #             raw_probe_names.append(key)
    #
    #     raw_probe_names = sorted(raw_probe_names, key=natural_keys)
    #
    #     npixels = len(raw_spectrum_data[raw_probe_names[0]])
    #
    #     #Find and load corresponding calibration json file
    #     match = re.search(r'(?<=' + SPECTROMETER + '-)[0-9]+', repr(os.path.basename(filepath)), re.IGNORECASE)
    #     if match:
    #         calibration_id = match.group(0)
    #         calibration_path = os.path.join(INPUT_FOLDER_PATH / 'Calibrations', SPECTROMETER.lower() + '-' + str(calibration_id) + '.json')
    #         try:
    #             calibration_data = json.load(open(calibration_path))
    #     #        print('Using calibration file ' + os.path.basename(calibration_path))
    #         except:
    #     #        print('Calibration file ' + os.path.basename(calibration_path) + ' not found!')
    #             pass
    #     else:
    #         calibration_id = raw_spectrum_data['metadata']['spectrometer_id']
    #         calibration_path = os.path.join(INPUT_FOLDER_PATH / 'Calibrations', str(calibration_id) + '.json')
    #         try:
    #             calibration_data = json.load(open(calibration_path))
    #     #        print('Using calibration file ' + os.path.basename(calibration_path))
    #         except:
    #     #        print('Calibration file ' + os.path.basename(calibration_path) + ' not found!')
    #             pass
    #
    # #Peform calibration
    # x_calibration = calibration_data['calibration_x']
    # y_calibration = calibration_data['calibration_y']
    # laser_wavelength_nm = float(calibration_data['laser_wavelength'])   #Sometimes this field is saved as a string
    #
    # #Quadratic wavelength calibration using a ASTM standard compound (Tol/ACN) and linear correction with laser wavelength
    # #print('Calibrating X by assigning wavenumbers to pixels')
    # a = x_calibration[0]
    # b = x_calibration[1]
    # c = x_calibration[2]
    #
    # x = np.arange(npixels, dtype='float64')
    # laser_wavelength_cm = float(laser_wavelength_nm) * 1e-7
    # lambda0_wavenumber = 1 / laser_wavelength_cm
    # cal_x = calibrate_x(x, x_calibration, lambda0_wavenumber)
    #
    # # Create dictionaries for raw and calibrated probe data
    # raw_probe_data = { }
    # cal_probe_data = { }
    #
    # if DATA_TYPE=='NEW':
    #     for name in raw_probe_names:
    #         raw_probe_data[name] = [ ]
    #         cal_probe_data[name[:-4]] = [ ]
    #         key = name[:-4].split('_0')[0]
    #         repeat = int(name[:-4].split('_0')[1])-1
    #         raw_probe_data[name].append(raw_spectrum_data[data_val][key]['raw'][repeat])
    # else:
    #     for name in raw_probe_names:
    #         raw_probe_data[name] = [ ]
    #         cal_probe_data[name[:-4]] = [ ]
    #         raw_probe_data[name].append(raw_spectrum_data[str(name)])
    #
    # # #Calculate resolution of spectrometer for interpolation
    # # res = [ ]
    # # idx = 0
    # # while idx<(len(cal_x)-1):
    # #     res.append(cal_x[idx+1]-cal_x[idx])
    # #     idx = idx+1
    #
    # # #Find median resolution in effective range of calibration
    # # _, idx = find_nearest(cal_x,1350)
    # # interp_res = res[idx]
    # # print(interp_res)
    #
    # # plt.plot(cal_x[1:],res,'.')
    # # plt.xlabel(r'$\mathrm{Wavenumber \ [cm}^{-1}]$', fontsize=14)
    # # plt.ylabel(r'$\Delta\mathrm{Wavenumber \ [cm}^{-1}]$', fontsize=14)
    # # plt.show()
    #
    # #Intensity correction through comparison with a NIST standard compound SRM 2241
    # #print('Calibrating Y through interpolation on specified data resolution scale')
    # new_x = np.arange(400, 1802, step=RESOLUTION, dtype='float64')   #Effective wavenumber range is from 400 to 2300 per cm
    # #Interpolate over y_calibration based on resolution
    # interp_x = np.arange(400, 2300+1, step=1, dtype='float64')
    # f = interpolate.interp1d(interp_x, y_calibration, kind='quadratic')
    # interp_y_calibration = f(new_x)
    #
    # for name in raw_probe_data.keys():
    #     for y in raw_probe_data[name]:
    #         # plt.plot(cal_x,y,'.')
    #         # plt.xlabel(r'$\mathrm{Wavenumber \ [cm}^{-1}]$', fontsize=14)
    #         # plt.ylabel(r'$\mathrm{Intensity}$', fontsize=14)
    #         # plt.show()
    #         f = interpolate.interp1d(cal_x, y, kind='quadratic')  #Maximum order of interpolation should be quadratic and extrapolation is unnecessary and dangerous
    #         new_y = f(new_x)
    #
    #         # Y Calibration
    #         new_y = new_y * interp_y_calibration
    #
    #         # plt.plot(new_x,new_y,'.')
    #         # plt.xlabel(r'$\mathrm{Wavenumber \ [cm}^{-1}]$', fontsize=14)
    #         # plt.ylabel(r'$\mathrm{Intensity}$', fontsize=14)
    #         # plt.show()
    #
    #         cal_probe_data[name[:-4]].append(new_y)
    #
    # proc_probe_data = { }
    # proc_probe_data['x'] = new_x.tolist()

    #Use internal calibrations as default
    cal_probe_data = { }
    cal_probe_names = [ ]
    proc_probe_data = { }
    if DATA_TYPE=='NEW':
        for val in raw_spectrum_data.keys():
            if val=='spectrum' or val=='data':
                data_val = val
                for key in raw_spectrum_data[val].keys():
                    if key!='x' and key!='fingerprint':
                        for i in range(1,len(raw_spectrum_data[val][key]['calibrated'])+1):
                            probestr = key + '_0' + str(i)
                            cal_probe_names.append(probestr)

        for name in cal_probe_names:
            cal_probe_data[name] = [ ]
            key = name.split('_0')[0]
            repeat = int(name.split('_0')[1])-1
            cal_probe_data[name].append(raw_spectrum_data[data_val][key]['calibrated'][repeat])

        if 'x' in raw_spectrum_data[data_val]:
            proc_probe_data['x'] = raw_spectrum_data[data_val]['x']
        else:
            start, stop, num = raw_spectrum_data['x_range']
            proc_probe_data['x'] = list(np.linspace(start, stop, num))
    else:
        for key in raw_spectrum_data.keys():
            if not key.endswith('_raw'):
                cal_probe_names.append(key)

        cal_probe_names = sorted(cal_probe_names, key=natural_keys)

        for name in cal_probe_names:
            cal_probe_data[name] = [ ]
            cal_probe_data[name].append(raw_spectrum_data[name])

        proc_probe_data['x'] = raw_spectrum_data['x']

    #Select wavenumber ranges below 1800
    for name in cal_probe_names:
        for i, data in enumerate(cal_probe_data[name]):
            cal_probe_data[name][i] = list(np.array(data)[np.array(proc_probe_data['x'])<=1800.])

    proc_probe_data['x'] = list(np.array(proc_probe_data['x'])[np.array(proc_probe_data['x'])<=1800.])

    for name in cal_probe_data:
        proc_probe_data[name] = [ ]
        #print('Performing baseline correction on Probe ' + str(name))

        # Baseline subtraction
        for data in cal_probe_data[name]:
            # Smoothing improves background fit
            smooth_data = savgol_filter(data, 13, 2)  # window size, polynomial order (window size depends on resolution)
            # plt.plot(new_x,data)
            # plt.plot(new_x,smooth_data)
            # plt.xlabel(r'$\mathrm{Wavenumber \ [cm}^{-1}]$', fontsize=14)
            # plt.ylabel(r'$\mathrm{Intensity}$', fontsize=14)
            # plt.show()
            baseline_corr_data = data - airPLS(smooth_data)
            proc_probe_data[name].append(baseline_corr_data.tolist())
            # plt.plot(new_x,baseline_corr_data)
            # plt.xlabel(r'$\mathrm{Wavenumber \ [cm}^{-1}]$', fontsize=14)
            # plt.ylabel(r'$\mathrm{Intensity}$', fontsize=14)
            # plt.show()

    norm_spectrum_data = { }

    # Normalisation (without smoothing!)
    for name in proc_probe_data:
        if '_' in name:  #Select only probe data
            norm_spectrum_data[name] = [ ]

            for data in proc_probe_data[name]:
                norm_data = np.array(data) / np.max(data)
                norm_spectrum_data[name].append(list(norm_data))

    # Averaging probe data
    avg_spectrum_data = { }

    probe_num = [ ]
    probe_repeats = [ ]

    for name in norm_spectrum_data:
        probe_repeats.append(name)
        probe_num.append(name.split('_0')[0]) #Split probe_repeat to access probe number

    #Group all repeats of a probe together
    probe_sorting = sorted(list_duplicates(probe_num))

    probe_sets = [ ]

    for group in probe_sorting:
        probe_sets.append(np.array(probe_repeats)[group[1]])

    if probe_sets:
        #Average probe repeats for each probe
        probe_set_data = [ ]

        for group in probe_sets:
            probe = [ ]
            for id in group:
                for data in norm_spectrum_data[id]:
                    probe.append(data)
            probe_set_data.append(probe)

        for i, probe in enumerate(probe_sets):
            avg_spectrum_data[probe[0][0]] = [ ]
            avg_spectrum_data[probe[0][0]] = list(np.average(probe_set_data[i],axis=0))
    else:
        for probe in probe_num:
            for id in norm_spectrum_data:
                if f'{probe}_' in id:
                    avg_spectrum_data[probe] = norm_spectrum_data[id][0]

    # Append wavenumber data to probe data dictionary
    avg_spectrum_data['x'] = proc_probe_data['x']

    end = time.time()
    #print('Time taken to process spectrum: ' + str(end - start))

    return avg_spectrum_data

#Master running script
def run_processing(INPUT_FOLDER_PATH, OUTPUT_FOLDER_PATH, SPECTROMETER, RESOLUTION, PROBE_ORDER, DATA_TYPE, INPUT_TYPE, OUTPUT_TYPE, filepath):
    start = time.time()
    if INPUT_TYPE=='JSON':
        with open(filepath) as json_file:
            raw_spectrum_data = json.load(json_file)
    else:
        with open(filepath, 'rb') as infile:
            raw_spectrum_data = msg.unpackb(infile.read(), strict_map_key=False)

    print('Processing raw spectrum file ' + os.path.basename(filepath))

    #Find subfolder in main folder
    subfolder_string = str(filepath.parent).split('Data')[1]
    if not subfolder_string:
        subfolder_string = None
    else:
        subfolder_string = subfolder_string[1:]
    filestring = str(os.path.splitext(os.path.basename(filepath))[0])

    #for non-QC data
    if 'QC' not in DATA_TYPE:
        avg_spectrum_data = probe_processing(INPUT_FOLDER_PATH, SPECTROMETER, RESOLUTION, DATA_TYPE, filepath, raw_spectrum_data)
        return avg_spectrum_data, subfolder_string, filestring
    #for QC data
    elif 'QC' in DATA_TYPE:
        if DATA_TYPE=='QC': #json QC
            #Determine number of scans and probes from data
            scan_data = [ ]
            probe_num = [ ]
            for key in raw_spectrum_data:
                if 'probe' in key:
                    probe_num.append(key.split(':')[1])
                    scan_num = len(raw_spectrum_data[key])

            #Collate data for each scan into individual dictionaries and then append to array
            for i in range(scan_num):
                probe_scan = { }
                pid = 0
                for key in raw_spectrum_data:
                    if 'probe' in key:
                        pn = probe_num[pid]
                        pid = pid+1
                        for val in raw_spectrum_data[key][i]:
                            if '_' in val:
                                probe_scan[str(pn) + val[1:]] = raw_spectrum_data[key][i][val]
                scan_data.append(probe_scan)
        elif DATA_TYPE=='QC_NEW': #msgpack QC
            #Determine number of scans and probes from data
            scan_data = [ ]
            probe_num = [ ]
            scan_num = [ ]
            for key in raw_spectrum_data['data']:
                probe_num.append(key.split('/')[0])
                scan_num.append(int(key.split('/')[1]))

            probe_num = np.unique(probe_num)
            scan_num = np.max(scan_num) + 1

            #Collate data for each scan into individual dictionaries and then append to array
            for i in range(scan_num):
                probe_scan = { }
                for pn in probe_num:
                    for key in raw_spectrum_data['data']:
                        if key==f'{pn}/{i}':
                            for val in raw_spectrum_data['data'][key]:
                                if val=='calibrated':   #using calibrated data here as default. Change to raw to experiment.
                                    for j, rep in enumerate(raw_spectrum_data['data'][key][val]):
                                        probe_scan[f'{pn}_0{j+1}'] = rep
                start, stop, num = raw_spectrum_data['x_range']
                probe_scan['x'] = list(np.linspace(start, stop, num))
                scan_data.append(probe_scan)

        results = [ ]
        spectra = [ ]
        probe_keys = [ ]
        for ind, scan in enumerate(scan_data):
            spectra.append(dict())
            #print('Processing Scan ' + str(ind+1))
            filestring_scan = filestring + '_Scan' + str(ind+1)
            avg_spectrum_data = probe_processing(INPUT_FOLDER_PATH, SPECTROMETER, RESOLUTION, DATA_TYPE, filepath, scan)
            for key in avg_spectrum_data.keys():
                if key!='x':
                    probe_keys.append(key)
                    spectra[ind][key] = avg_spectrum_data[key].copy()
            results.append((avg_spectrum_data, subfolder_string, filestring_scan))    #If individual scan data is needed as well, uncomment

        #Also append average data
        probe_keys = np.unique(probe_keys)
        scan_avg_spectrum_data = { }
        for key in probe_keys:
            probe_avg = [ ]
            for scan in spectra:
                probe_avg.append(scan[key])
            scan_avg_spectrum_data[key] = list(np.average(probe_avg,axis=0))
        filestring_avg = filestring + '_Average'
        scan_avg_spectrum_data['x'] = avg_spectrum_data['x']
        results.append((scan_avg_spectrum_data, subfolder_string, filestring_avg))

        return results

if __name__ == '__main__':
    with MPIExecutor() as pool:
        pool.workers_exit() #Only allow master to execute startup sequence

        #Load variables from inlist
        inlist = str(sys.argv[1])
        INPUT_FOLDER_PATH, OUTPUT_FOLDER_PATH, SPECTROMETER, RESOLUTION, PROBE_ORDER, DATA_TYPE, INPUT_TYPE, OUTPUT_TYPE = ReadInlist(inlist)

        #Print important variables
        print('INPUT_FOLDER_PATH = ' + str(INPUT_FOLDER_PATH))
        print('OUTPUT_FOLDER_PATH = ' + str(OUTPUT_FOLDER_PATH))
        print('SPECTROMETER = ' + SPECTROMETER)
        print('RESOLUTION = ' + str(RESOLUTION))
        print('PROBE_ORDER = ' + str(PROBE_ORDER))
        print('DATA_TYPE = ' + str(DATA_TYPE))
        print('INPUT_TYPE = ' + str(INPUT_TYPE))
        print('OUTPUT_TYPE = ' + str(OUTPUT_TYPE))

        #Find raw spectra json files (works with any combination/depth of subfolders)
        raw_spectra_filepaths = [ ]
        if INPUT_TYPE=='JSON':
            for path in Path(INPUT_FOLDER_PATH / 'Data').rglob('*.json'):
                if path.is_file():
                    raw_spectra_filepaths.append(path)
        else:
            for path in Path(INPUT_FOLDER_PATH / 'Data').rglob('*.tr-spectra'):
                if path.is_file():
                    raw_spectra_filepaths.append(path)

        #For each raw spectrum, process and normalise
        arc_filepaths = [ ]

        #Execute parallel code
        futures = [pool.submit(functools.partial(run_processing, INPUT_FOLDER_PATH, OUTPUT_FOLDER_PATH, SPECTROMETER, RESOLUTION, PROBE_ORDER, DATA_TYPE, INPUT_TYPE, OUTPUT_TYPE), filepath) for filepath in raw_spectra_filepaths]
        #Once a result is ready, main task saves file
        for future in tqdm(as_completed(futures), total=len(raw_spectra_filepaths)):
            if 'QC' not in DATA_TYPE:
                avg_spectrum_data, subfolder_string, filestring = future.result()
                save_files(avg_spectrum_data, OUTPUT_FOLDER_PATH, PROBE_ORDER, OUTPUT_TYPE, arc_filepaths, subfolder_string, filestring)
            else:
                for avg_spectrum_data, subfolder_string, filestring in future.result():
                    save_files(avg_spectrum_data, OUTPUT_FOLDER_PATH, PROBE_ORDER, OUTPUT_TYPE, arc_filepaths, subfolder_string, filestring)

    print('All spectra have been processed!')
