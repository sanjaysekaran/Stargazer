"""
This script performs the fitting of the peaks of the spectra of individual probes
that have been saved in msgpack format. Each peak is fitted using a skewed pseudo-voigt
(a sum of a Gaussian and Lorentzian rather than a convolution) profile as per many studies in the literature,
which is described by a five parameters: amplitude, peak center, width, skewness, and
the Gaussian and Lorentzian proportion (alpha). However, for the sake of interpretability/comparability,
the output parameters include only the fraction, with the intensity, position, fwhm and asym (!) calculated
from the deconvoled model peaks individually.The spectra of the individual probes is generated using the script
raw_json_processing.py, so run that first before using this script.
Requires the lmfit, numdifftools, mpi4py, mpipool, str2bool and msgpack python packages.
These can be convieniently installed by using the command pip install -r requirements.txt, where
requirements.txt is a text file distributed along with this script. Note that mpi4py installation
is non-trivial on windows, so please look into installing and configuring Microsoft MPI before
attempting to use this script.

(!) asym  = the distance from the peak maximum position to the right wing - distance of from the peak center to the
left wing as a fraction of the FWHM (i.e. negative values for left-tailed peaks, and positive values for right-tailed peaks)

This script must be executed parallel functionality (i.e. can make use of multiple CPU cores on multiple machines)!
To execute a parallel computation on 4 cores (e.g. on a laptop), use the command mpiexec -n 4 python peak_fitter.py inlist peak_ranges,
where inlist is an input text file containing the following variables:

INPUT_FOLDER_PATH - Folder where the compiled probe spectra are stored (e.g. /home/Destop/spectra/QC_processed/)
RESULTS_FOLDER_PATH - Folder where the results will be stored
RESUME - Specifies whether the fitting should start from the beginning (False) or continue from the last saved result (True).
The default is False.

and peak_ranges is a second input text file containing seven columns that specify the 1) Probe number,
2) start and the 3) end of the wavenumber ranges of the spectral regions that are used, 4)
the number of peaks in each region, and 5) the expected position, 6) intensity and 7) index of a problematic
peak (e.g. a blended shoulder region of another peak). If the value of 5) is set to zero, it is assumed
that there are no problmatic peaks. Incorrect specification of the any of the aforementioned parameters
may cause the fitting to fail (i.e. give poor results), so it is important to have knowledge/plot
of each spectral region before performing any fitting (a good practice in general for such problems).

If RESULTS_FOLDER_PATH is not specified, the fitting results (i.e. model spectrum, parameters and errors) will be stored
in csv files (due to its tiny memory footprint) in a parallel folder named 'Fit_Results' to INPUT_FOLDER_PATH (e.g. if INPUT_FOLDER_PATH = /home/Destop/spectra/QC_processed/,
OUTPUT_FOLDER_PATH defaults to /home/Destop/spectra/Fit_Results/).

@author: Sanjay Sekaran
"""

import os, sys, glob, csv
import msgpack as msg
import errno
import numpy as np
import re
import matplotlib.pyplot as plt
from pathlib import Path
import time
from scipy.signal import find_peaks, peak_widths
from scipy.special import erf
import lmfit
from lmfit import Model, Minimizer
from lmfit.lineshapes import pvoigt
from lmfit.models import PseudoVoigtModel as PVM
from mpipool import MPIExecutor
from mpi4py import MPI
import functools
from str2bool import str2bool
from concurrent.futures import as_completed

def assert_dir_exists(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def ReadInlist(inlist,peak_range_file):

  INPUT_FOLDER_PATH = None
  RESULTS_FOLDER_PATH = None
  RESUME = None

  file = open(inlist,'r')
  lines = file.readlines()
  file.close()

  for line in lines:
    (par,val) = line.strip().split('=')
    if(par.strip() == 'INPUT_FOLDER_PATH'):
      INPUT_FOLDER_PATH = val.strip()
    if(par.strip() == 'RESULTS_FOLDER_PATH'):
      RESULTS_FOLDER_PATH = val.strip()
    if(par.strip() == 'RESUME'):
      RESUME = val.strip()

  #Load peak ranges, number of peaks, problematic peak position and intensity in each range
  PEAK_RANGES = np.genfromtxt(peak_range_file, delimiter=' ', dtype=None, names=True, encoding='utf-8')

  #If results folder path not specified, default to parallel folder 'Fit_Results'
  if RESULTS_FOLDER_PATH == None:
      RESULTS_FOLDER_PATH = str(Path(INPUT_FOLDER_PATH).parents[0] / 'Fit_Results')

  #If RESUME not specified, default to false
  if RESUME == None:
      RESUME = False
  else:
      try:
          RESUME = str2bool(RESUME)
      except:
          print("Error: please specify either True or False for RESUME.")
          sys.exit()

  if(INPUT_FOLDER_PATH == '') or not os.path.exists(INPUT_FOLDER_PATH):
    print("Error: please provide a valid path to the input folder.")
    sys.exit()
  if(RESULTS_FOLDER_PATH == '') and not os.path.exists(RESULTS_FOLDER_PATH):
    print("Error: please provide a valid path to the results folder.")
    sys.exit()
  if(RESUME == ''):
    print("Error: please specify if you want to start from the last probe spectrum (True) or the beginning (False).")
    sys.exit()

  return Path(INPUT_FOLDER_PATH),Path(RESULTS_FOLDER_PATH),RESUME, PEAK_RANGES

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

def SPVM(x, amplitude=1.0, center=0.0, sigma=1.0, fraction=0.5, skew=0.0):

    """
    Skewed pseudo-voigt model (PVM): generated by taking the product of a PVM with an
    error function, but introducing the parameter 'skew' to represent skewness
    """
    beta = skew/max(1.0e-15, (np.sqrt(2.)*sigma))
    asym = 1 + erf(beta*(x-center))
    model = asym*pvoigt(x, amplitude, center, sigma, fraction)

    return model

def param_setup(params, ind, peak_x, peak_y):
    params['V' + str(ind) + '_amplitude'].set(value=20*peak_y[ind], min=1.0e-15, max=30.)
    params['V' + str(ind) + '_center'].set(value=peak_x[ind], min=peak_x[ind]-20, max=peak_x[ind]+20)
    params['V' + str(ind) + '_sigma'].set(min=1.0e-15, max=40.)
    params['V' + str(ind) + '_fraction'].set(min=1.0e-15, max=1.0)
    fmt = ("(((1-{prefix:s}fraction)*{prefix:s}amplitude)/"
           "max({0}, ({prefix:s}sigma*sqrt(pi/log(2))))+"
           "({prefix:s}fraction*{prefix:s}amplitude)/"
           "max({0}, (pi*{prefix:s}sigma)))")
    params.add('V' + str(ind) + '_height', expr=fmt.format(1.0e-15, prefix='V' + str(ind) + '_'), vary=False)

    return params

def fit_setup(input_spectrum_data,start,end,comp,prob_peak_x,prob_peak_y,prob_peak_i):
    model = [ ]
    for i in range(comp):
        prefix = 'V' + str(i) + '_'
        mod = Model(SPVM, prefix=prefix)
        if not model:
            model = mod
            params = mod.make_params()
        else:
            model =  model + mod
            params.update(mod.make_params())

    print('Fitting wavenumber range ' + str(start) + ' to ' + str(end))

    #Find peak(s) in range and use x_coordinate/y_coordinate as first guess for center/amplitude
    x_data,y_data = map(list,zip(*[(x,y) for x,y in zip(input_spectrum_data['x'],input_spectrum_data['spectrum']) if start <= x <= end]))

    p_ind = find_peaks(y_data, height=0.02)[0]

    peak_x = [ ]
    peak_y = [ ]
    for i in p_ind:
        peak_x.append(x_data[i])
        peak_y.append(y_data[i])

    #If number of peaks detected are lower than expectation, use wavenumber information
    #of problematic peak as center and amplitude of 0.1 (assuming that peak intensity is on the low side)
    if len(p_ind)<comp:
        if prob_peak_x!=0:
            peak_x.append(prob_peak_x)
            peak_y.append(prob_peak_y)
        else:
            print('Number of peaks detected is less than expected but wavenumber position of peak not specified! Defaulting to median of fitting range.')
            peak_x.append(np.median(x_data))
            peak_y.append(0.1)

    #Add only the n highest amplitude peaks to the final selection depending on number of expected components
    peak_x = np.array(peak_x)[np.argsort(peak_y)][::-1]
    peak_y = np.sort(peak_y)[::-1]

    #If unable to find comp number of peaks (even after adding problematic peak), kill process
    if len(peak_y)<comp:
        print('Insufficient number of peaks found! Killing fitting process...')
        return None
    else:
        pass

    peak_y_selection = [ ]
    peak_x_selection = [ ]
    for i in range(comp):
        peak_y_selection.append(peak_y[i])
        peak_x_selection.append(peak_x[i])

    #For cases where the number of peaks meets expectation, check if one of the detected peaks
    #is within plus-minus 10 of prob_peak_x (because peaks have been selected in amplitude order)
    #If not, change expected index of identified peak to prob_peak
    peak_truth = [ ]
    if len(peak_x_selection) == comp:
        if prob_peak_x!=0:
            for x in peak_x_selection:
                peak_truth.append(prob_peak_x-10 <= x <= prob_peak_x+10)
            if not any(peak_truth):
                print('Misidentification of problematic peak by automated peak finder: defaulting to input value.')
                peak_x_selection[prob_peak_i] = prob_peak_x
                peak_y_selection[prob_peak_i] = prob_peak_y

    #Sort with respect to x coordinate for easy peak identification
    peak_y_selection = np.array(peak_y_selection)[np.argsort(peak_x_selection)]
    peak_x_selection = np.sort(peak_x_selection)

    #Use peak data as first guess for amplitude and peak position
    #Set limits for fraction and skewness
    for i in range(comp):
        param_setup(params, i, peak_x_selection, peak_y_selection)

    return model, params, x_data, y_data, peak_x_selection, peak_y_selection

def fit_peaks(model, params, x_data, y_data, comp):
    #Perform fitting of peak(s) using model
    #ci_stderr =  { }
    if comp == 1:
        print('Attempting least-squares optimisation of single-peak region.')
        result = model.fit(y_data, params, x=x_data, method='least_squares')
    else:
        print('Attempting stochastic optimisation for multiple-peak region.')
        result = model.fit(y_data, params, x=x_data, method='basinhopping')

    #Repeat fitting (once) if chisqr is below threshold for single peaks
    if comp == 1:
        if result.chisqr>0.05:
            print('Least-squares optimisation failed! Repeating fit with stochastic optimisation.')
            result = model.fit(y_data, params, x=x_data, method='basinhopping')

    #Get best-fit model and associated uncertainty region
    best_fit = result.best_fit
    # #Use 1-sigma confidence interval for errors
    # confidence_interval = result.conf_interval(p_names = result.params, sigmas=[1])
    # for key in confidence_interval.keys():
    #     ci_stderr[key] = (confidence_interval[key][2][1] - confidence_interval[key][0][1])/2
    uncertainty_interval = result.eval_uncertainty(sigma=1)

    #print(result.fit_report()) #For testing purposes

    return result, best_fit, uncertainty_interval

def peak_data(peak_params, peak_profiles, result, x_data, pn):
    #Use prepended peak ids to select parameters corresponding to the relevant peaks
    peak_id = [ ]
    for param in result.params:
        if 'amplitude' not in param:
            peak_id.append(param.split('_')[0])

    peak_id = np.unique(peak_id)

    #Evaluate profiles by first creating interpolated x grid at higher resolution
    #allows for more accurate determination of position, intensity and FWHM
    x_grid = np.linspace(start=min(x_data), stop=max(x_data), num=len(x_data)*20)
    x_grid_res = x_grid[1]-x_grid[0]
    profiles = result.eval_components(x=x_grid)
    pp = [ ]    #For calculation purposes
    for profile in profiles:
        pp.append([x_grid,profiles[profile]])
        peak_profiles.append([x_grid,profiles[profile]])

    no_save_list = ['amplitude', 'center', 'sigma', 'height', 'skew']

    for ind, id in enumerate(peak_id):
        pn = pn + 1 #Add 1 to peak number for every unique peak id
        #Calculate position, intensity and fwhm from component profile
        x = pp[ind][0]
        y = pp[ind][1]
        position = x[np.argmax(y)]
        intensity = np.max(y)
        pw = peak_widths(y,peaks=[np.argmax(y)])
        FWHM = pw[0][0]*x_grid_res
        fwhm_left, fwhm_right = np.abs(np.min(x) + np.concatenate(pw[2:])*x_grid_res - position)
        ASYM = (fwhm_right-fwhm_left)/FWHM #Use same convention as skewness: positive when RHS>LHS and vice versa
        result.params.add(str(id) + '_position', value=position)
        result.params.add(str(id) + '_intensity', value=intensity)
        result.params.add(str(id) + '_fwhm', value=FWHM)
        result.params.add(str(id) + '_asym', value=ASYM)
        #Propagate parameter errors from center, height and sigma
        result.params[str(id) + '_position'].stderr = result.params[str(id) + '_center'].stderr
        result.params[str(id) + '_intensity'].stderr = result.params[str(id) + '_height'].stderr
        #Catch Nonetype errors
        if result.params[str(id) + '_sigma'].stderr is not None:
            result.params[str(id) + '_fwhm'].stderr = 2.355*result.params[str(id) + '_sigma'].stderr
            result.params[str(id) + '_asym'].stderr = result.params[str(id) + '_asym']*(result.params[str(id) + '_fwhm'].stderr/result.params[str(id) + '_fwhm'])
        else:
            result.params[str(id) + '_fwhm'].stderr = result.params[str(id) + '_sigma'].stderr
            result.params[str(id) + '_asym'].stderr = result.params[str(id) + '_sigma'].stderr
        for param in result.params:
            if all([unsave not in param for unsave in no_save_list]):
                if id in param:
                    peak_params['Peak_' + str(pn) + '_' + param.split('_')[1]] = result.params[param].value
                    peak_params['Peak_' + str(pn) + '_' + param.split('_')[1] + '_std'] = result.params[param].stderr

    del(result)    #Free up memory from result array after assigning peak parameters

    return peak_params, peak_profiles, pn

def best_fit_model(model_bf, x_data, best_fit, uncertainty_interval):

    if not model_bf:
        model_bf['x'] = [ ]
        model_bf['best_fit'] = [ ]
        model_bf['uncertainty_interval'] = [ ]

    model_bf['x'].extend(x_data)
    model_bf['best_fit'].extend(list(best_fit))
    model_bf['uncertainty_interval'].extend(list(uncertainty_interval))

    del(best_fit)    #Free up memory
    del(uncertainty_interval)    #Free up memory

    return model_bf

def save_model_data(model_bf, peak_params, input_spectrum_data, key, model_filepath, params_filepath):

    print('Saving best-fit model spectrum for ' + key + ' to ' + str(model_filepath))
    with open(model_filepath, 'a') as csvfile:
        writer = csv.writer(csvfile)
        if os.stat(model_filepath).st_size==0: #Write file header if not written
            line_to_write = ['Wavenumber']
            line_to_write += ['%.8f' % y for y in model_bf['x']]
            writer.writerow(line_to_write)
        row_name = key
        line_to_write = [row_name]
        line_to_write += ['%.8f' % y for y in model_bf['best_fit']]
        writer.writerow(line_to_write)

    print('Saving best-fit model parameters for ' + key + ' to '  + str(params_filepath))
    with open(params_filepath, 'a') as csvfile:
        writer = csv.writer(csvfile)
        if os.stat(params_filepath).st_size==0: #Write file header if not written
            line_to_write = [' ']
            line_to_write += [y for y in peak_params.keys()]
            writer.writerow(line_to_write)
        row_name = key
        line_to_write = [row_name]
        #If vals are crap, output no val
        for y in peak_params.values():
            if y is not None:
                if np.isfinite(y):
                    line_to_write += ['%.8f' % y]
                else:
                    line_to_write += ['NaN']
            else:
                line_to_write += ['NaN']
        writer.writerow(line_to_write)

#Master running script
def run_fitting(INPUT_FOLDER_PATH, RESULTS_FOLDER_PATH, PEAK_RANGES, data_key):
    start_comp = time.time()
    input_spectrum_data, key, file_probe, model_filepath, params_filepath = data_key
    print('Fitting ' + file_probe + ' of spectrum ' + key)

    # #Plot input data
    # plt.plot(input_spectrum_data['x'],input_spectrum_data['spectrum'],linewidth=1.0)
    # plt.xlabel(r'$\mathrm{Wavenumber \ [cm}^{-1}]$', fontsize=14)
    # plt.ylabel(r'$\mathrm{Intensity}$', fontsize=14)
    # plt.tight_layout()
    # plt.show()

    peak_params = { }
    model_bf = { }
    pn = 0 #Keep track of peak numbers
    peak_profiles = [ ]

    #Separate the variables
    pnum = PEAK_RANGES['Probe_num']
    s = PEAK_RANGES['wavenumber_start']
    e = PEAK_RANGES['wavenumber_end']
    c = PEAK_RANGES['num_peaks']
    ppx = PEAK_RANGES['prob_peak_x']
    ppy = PEAK_RANGES['prob_peak_y']
    ppi = PEAK_RANGES['prob_peak_index']

    truth = pnum == file_probe

    pnum = pnum[truth]
    s = s[truth]
    e = e[truth]
    c = c[truth]
    ppx = ppx[truth]
    ppy = ppy[truth]
    ppi = ppi[truth]

    PEAK_RANGE_SELECTION = zip(pnum,s,e,c,ppx,ppy,ppi)

    for probe_num,start,end,comp,prob_peak_x,prob_peak_y,prob_peak_i in PEAK_RANGE_SELECTION:

        comp = int(comp)    #integer
        prob_peak_i = int(prob_peak_i)  #integer

        inputs = fit_setup(input_spectrum_data,start,end,comp,prob_peak_x,prob_peak_y,prob_peak_i)
        if inputs is not None:
            model, params, x_data, y_data, peak_x_selection, peak_y_selection = inputs
        else:
            return None

        result, best_fit, uncertainty_interval = fit_peaks(model, params, x_data, y_data, comp)

        #Collate peak data
        peak_params, peak_profiles, pn = peak_data(peak_params, peak_profiles, result, x_data, pn)

        #Collate best-fit model data
        model_bf = best_fit_model(model_bf, x_data, best_fit, uncertainty_interval)

    # #Plot fitting range and best-fit spectrum (with 1-sigma uncertainty region)
    # plt.figure()
    # plt.plot(input_spectrum_data['x'],input_spectrum_data['spectrum'],linewidth=1.0, alpha=0.5)
    # plt.plot(model_bf['x'],model_bf['best_fit'],color='k',linewidth=1.0)
    # plt.fill_between(model_bf['x'],np.array(model_bf['best_fit'])-np.array(model_bf['uncertainty_interval']),np.array(model_bf['best_fit'])+np.array(model_bf['uncertainty_interval']),color='r', alpha=0.4)
    # for profile in peak_profiles:
    #     plt.plot(profile[0],profile[1],color='k',linewidth=1.0,linestyle=':')
    # plt.xlabel(r'$\mathrm{Wavenumber \ [cm}^{-1}]$', fontsize=14)
    # plt.ylabel(r'$\mathrm{Intensity}$', fontsize=14)
    # plt.tight_layout()
    # plt.show()

    end_comp = time.time()
    print('Time taken for fitting ' + file_probe + ' of spectrum ' + key + ': ' + str(end_comp - start_comp) + ' s')

    return model_bf, peak_params, input_spectrum_data, key, file_probe, model_filepath, params_filepath

if __name__ == '__main__':
    with MPIExecutor() as pool:
        pool.workers_exit() #Only allow master to execute startup sequence

        #Load variables from inlist
        inlist = str(sys.argv[1])
        peak_range_file = str(sys.argv[2])
        INPUT_FOLDER_PATH, RESULTS_FOLDER_PATH, RESUME, PEAK_RANGES = ReadInlist(inlist,peak_range_file)

        #Print important variables
        print('INPUT_FOLDER_PATH = ' + str(INPUT_FOLDER_PATH))
        print('RESULTS_FOLDER_PATH = ' + str(RESULTS_FOLDER_PATH))
        print('RESUME = ' + str(RESUME))
        print('PEAK_RANGES = \n' + str(PEAK_RANGES))

        #Find input spectra msg files (works with any combination/depth of subfolders)
        input_spectra_filepaths = [ ]
        for path in Path(INPUT_FOLDER_PATH).rglob('*.msg'):
            if path.is_file():
                input_spectra_filepaths.append(path)

        #Get all data and arrange according to tuples with all of the relevant data (input_spectrum, probe_key, file_probe, filepaths etc.)
        data_keys = [ ]
        for filepath in input_spectra_filepaths:
            compiled_data = { }
            with open(filepath, 'rb') as infile:
                for data in msg.Unpacker(infile, raw=False):
                    for key in data:
                        compiled_data[key] = data[key].copy()

            #Find subfolder in main folder
            subfolder_string = str(filepath.parent).split(str(INPUT_FOLDER_PATH))[1]
            if not subfolder_string:
                subfolder_string = None
            else:
                subfolder_string = subfolder_string[1:]

            #Generate best-fit model and parameter filenames
            model_filename = file_probe + '_bestfit.csv'
            params_filename = file_probe + '_params.csv'

            if subfolder_string is not None:
                model_filepath = RESULTS_FOLDER_PATH / subfolder_string / model_filename
                params_filepath = RESULTS_FOLDER_PATH / subfolder_string / params_filename
            else:
                model_filepath = RESULTS_FOLDER_PATH / model_filename
                params_filepath = RESULTS_FOLDER_PATH / params_filename

            assert_dir_exists(os.path.dirname(model_filepath))

            #If RESUME is False, create new files. Else, remove completed entries from list.
            if not RESUME:
                with open(model_filepath, 'w') as csvfile:
                    pass
                with open(params_filepath, 'w') as csvfile:
                    pass
            else:
                try:
                    with open(params_filepath, 'r') as csvfile:
                        lines = csvfile.read().splitlines()
                        #find and delete completed spectra
                        for line in lines[1:]:
                            com_key = line.split(',')[0]
                            del(compiled_data[com_key])
                except:
                    print('Saved parameter file could not be loaded. Check if you have supplied the correct location to the results folder or set RESUME = False to start a new fit!')
                    sys.exit()

            #Fit individual scan/probe arrays individually by assigning each to dictionary
            input_data = { }
            probe_keys = compiled_data.keys()

            #Generate tuples for input into fitting code
            for key in probe_keys:
                input_data['x'] = compiled_data[key]['x']
                input_data['spectrum'] = compiled_data[key]['spectrum']
                data_keys.append((input_data.copy(), key, file_probe, model_filepath, params_filepath))

        del(compiled_data)    #Free data from memory after assigning to array
        del(input_data)     #Free data from memory after assigning to array

        print('Total number of spectra to fit (across all probes/scans): ' + str(len(data_keys)))

        #Now execute parallel code
        futures = [pool.submit(functools.partial(run_fitting, INPUT_FOLDER_PATH, RESULTS_FOLDER_PATH, PEAK_RANGES), data_key) for data_key in data_keys]
        #Once a result is ready, main task saves file
        for future in as_completed(futures):
            if future.result() is not None:
                model_bf, peak_params, input_spectrum_data, key, file_probe, model_filepath, params_filepath = future.result()
                if np.all([value is not None for value in peak_params.values()]):   #Catch None values in array
                    save_model_data(model_bf, peak_params, input_spectrum_data, key, model_filepath, params_filepath)
                else:
                    print('Fit for ' + file_probe + ' of ' + key + ' has failed! Best-fit model and parameters will not be saved.')
