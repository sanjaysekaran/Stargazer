"""
This script performs a principal component analysis (PCA) of the input spectra, using either the raw intensities
as input or the parameters of the major peaks in the spectrum. The peak parameters are obtained by the fitting of
the peaks of the spectra of individual probes. Each peak is fitted using a skewed pseudo-voigt (a sum of a Gaussian
and Lorentzian rather than a convolution) profile as per many studies in the literature, which is described by a five
parameters: amplitude, peak center, width, skewness, and the Gaussian and Lorentzian proportion (alpha). However, for
the sake of interpretability/comparability, the output parameters include only the fraction and skewness, with the
intensity, position and fwhm calculated from the deconvoled model peaks individually. Requires the lmfit and numdifftools
python packages.

The code requires that the data used as input (the new spectra to be fitted/decomposed) be placed in the folder 'Data',
with the training spectral data (saved as msgpack files) placed in the folder 'Train_Spectra/Compiled/'. The parameters
used for training are to be saved in the main folder under the name 'Dispensing_QC_train.csv'. Note that this script
is able to handle multiple batches of dispensed chip spectra. This requires the individual batches to be placed in appropriately
named subfolders within the 'Data' folder, which are then used to name the batches. This code also requires the subscript
pca_mod.py, which contains the classes and functions for the PCA calculations.

The code is executed using the command python Dispensing_QC_param_PCA.py peak_ranges 1,4,6, where
peak_ranges is an input text file containing seven columns that specify the 1) Probe number,
2) start and the 3) end of the wavenumber ranges of the spectral regions that are used, 4)
the number of peaks in each region, and 5) the expected position, 6) intensity and 7) index of a problematic
peak (e.g. a blended shoulder region of another peak). If the value of 5) is set to zero, it is assumed
that there are no problmatic peaks. Incorrect specification of the any of the aforementioned parameters
may cause the fitting to fail (i.e. give poor results), so it is important to have knowledge/plot
of each spectral region before performing any fitting (a good practice in general for such problems).

The second argument (i.e. 1,4,6) allows you to choose the probes which you want to fit. Note that some probes
take way longer to fit than others (1,4 and 5 in particular), so please be conscious of that.

Plots of the first two principal components (which explain the greatest percentage variance of the distribution)
for each scenario are then saved in Fit_Results folder.

@author: Sanjay Sekaran
"""

import os, sys, glob, csv, json
import msgpack as msg
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from pathlib import Path, PurePath
import time
from scipy.signal import find_peaks, peak_widths
from scipy.special import erf
from scipy import interpolate
from scipy.signal import savgol_filter
from scipy.sparse import csc_matrix, eye, diags
from scipy.sparse.linalg import spsolve
import lmfit
from lmfit import Model, Minimizer
from lmfit.lineshapes import pvoigt
from lmfit.models import PseudoVoigtModel as PVM
import functools
from str2bool import str2bool
from concurrent.futures import as_completed, ProcessPoolExecutor
from multiprocessing import freeze_support
from pca_mod import pca
from datetime import date
from collections import defaultdict
from matplotlib.font_manager import FontProperties

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
    params[f'V{ind}_amplitude'].set(value=20*peak_y[ind], min=1.0e-15, max=30.)
    params[f'V{ind}_center'].set(value=peak_x[ind], min=peak_x[ind]-20, max=peak_x[ind]+20)
    params[f'V{ind}_sigma'].set(min=1.0e-15, max=40.)
    params[f'V{ind}_fraction'].set(min=1.0e-15, max=1.0)
    fmt = ("(((1-{prefix:s}fraction)*{prefix:s}amplitude)/"
           "max({0}, ({prefix:s}sigma*sqrt(pi/log(2))))+"
           "({prefix:s}fraction*{prefix:s}amplitude)/"
           "max({0}, (pi*{prefix:s}sigma)))")
    params.add(f'V{ind}_height', expr=fmt.format(1.0e-15, prefix='V' + str(ind) + '_'), vary=False)

    return params

def fit_setup(input_spectrum_data,start,end,comp,prob_peak_x,prob_peak_y,prob_peak_i):
    model = [ ]
    for i in range(comp):
        prefix = f'V{i}_'
        mod = Model(SPVM, prefix=prefix)
        if not model:
            model = mod
            params = mod.make_params()
        else:
            model =  model + mod
            params.update(mod.make_params())

    print(f'Fitting wavenumber range {start} to {end}')

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
            peak_y.append(1000.)

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
    # confidence_interval = result.conf_interval(p_names = result, sigmas=[1])
    # for key in confidence_interval.keys():
    #     ci_stderr[key] = (confidence_interval[key][2][1] - confidence_interval[key][0][1])/2
    uncertainty_interval = result.eval_uncertainty(sigma=1)

    #print(result.fit_report()) #For testing purposes

    return result, best_fit, uncertainty_interval

def peak_data(peak_params, pp, result, x_data, x_grid_res, pn):
    #Use prepended peak ids to select parameters corresponding to the relevant peaks
    peak_id = [ ]
    for param in result:
        if 'amplitude' not in param:
            peak_id.append(param.split('_')[0])

    peak_id = np.unique(peak_id)

    no_save_list = ['amplitude', 'center', 'sigma', 'height']

    for ind, id in enumerate(peak_id):
        pn = pn + 1 #Add 1 to peak number for every unique peak id
        #Calculate position, intensity and fwhm from component profile
        x = pp[ind][0]
        y = pp[ind][1]
        position = x[np.argmax(y)]
        intensity = np.max(y)
        FWHM = peak_widths(y,peaks=[np.argmax(y)])[0][0]*x_grid_res
        result.add(f'{id}_position', value=position)
        result.add(f'{id}_intensity', value=intensity)
        result.add(f'{id}_fwhm', value=FWHM)
        #Include parameter errors from center, height and sigma
        result[f'{id}_position'].stderr = result[f'{id}_center'].stderr
        result[f'{id}_intensity'].stderr = result[f'{id}_height'].stderr
        #Catch Nonetype errors
        if result[f'{id}_sigma'].stderr is not None:
            result[f'{id}_fwhm'].stderr = 2.355*result[f'{id}_sigma'].stderr
        else:
            result[f'{id}_fwhm'].stderr = result[f'{id}_sigma'].stderr
        for param in result:
            if all([unsave not in param for unsave in no_save_list]):
                if id in param:
                    key = param.split('_')[1]
                    peak_params[f'Peak_{str(pn)}_{key}'] = result[param].value
                    peak_params[f'Peak_{str(pn)}_{key}_std'] = result[param].stderr

    del(result)    #Free up memory from result array after assigning peak parameters

    return peak_params, pn

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

def save_model_data(peak_params, spectrum_id, params_filepath):

    print(f'Saving best-fit model parameters for {spectrum_id} to {str(params_filepath)}')
    with open(params_filepath, 'a') as csvfile:
        writer = csv.writer(csvfile)
        if os.stat(params_filepath).st_size==0: #Write file header if not written
            line_to_write = [' ']
            line_to_write += [y for y in peak_params.keys()]
            writer.writerow(line_to_write)
        row_name = spectrum_id
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
def run_fitting(data_key):
    start_comp = time.time()

    input_spectrum_data,probe_num,start,end,comp,prob_peak_x,prob_peak_y,prob_peak_i = data_key

    comp = int(comp)    #integer
    prob_peak_i = int(prob_peak_i)  #integer

    inputs = fit_setup(input_spectrum_data,start,end,comp,prob_peak_x,prob_peak_y,prob_peak_i)
    if inputs is not None:
        model, params, x_data, y_data, peak_x_selection, peak_y_selection = inputs
    else:
        return None

    result, best_fit, uncertainty_interval = fit_peaks(model, params, x_data, y_data, comp)

    #Evaluate profiles by first creating interpolated x grid at higher resolution
    #allows for more accurate determination of position, intensity and FWHM
    x_grid = np.linspace(start=min(x_data), stop=max(x_data), num=len(x_data)*20)
    x_grid_res = x_grid[1]-x_grid[0]
    profiles = result.eval_components(x=x_grid)
    pp = [ ]    #For calculation purposes
    for profile in profiles:
        pp.append([x_grid,profiles[profile]])

    end_comp = time.time()
    print(f'Time taken for fitting range {str(start)} to {str(end)} of {probe_num}: {str(end_comp - start_comp)} s')

    return result.params, pp, best_fit, uncertainty_interval, x_data, x_grid_res, probe_num

def get_cov_ellipse(cov, centre, nstd, **kwargs):
    """
    Return a matplotlib Ellipse patch representing the covariance matrix
    cov centred at centre and scaled by the factor nstd.

    """

    # Find and sort eigenvalues and eigenvectors into descending order
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    # The anti-clockwise angle to rotate our ellipse by
    vx, vy = eigvecs[:,0][0], eigvecs[:,0][1]
    theta = np.arctan2(vy, vx)

    # Width and height of ellipse to draw
    width, height = 2 * nstd * np.sqrt(eigvals)
    return Ellipse(xy=centre, width=width, height=height,
                   angle=np.degrees(theta), **kwargs)

def process_input_spectrum(filepath, probe_nos):
    spectrum_id = str(os.path.splitext(os.path.basename(filepath))[0]) #spectrum_id

    if '.json' in str(filepath):
        with open(filepath) as json_file:
            raw_spectrum_data = json.load(json_file)

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
            probe_scan['x'] = raw_spectrum_data['x']
            scan_data.append(probe_scan)

    else:
        with open(filepath, 'rb') as infile:
            raw_spectrum_data = msg.unpackb(infile.read(), strict_map_key=False)

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
            start, stop, num = raw_spectrum_data['x_range']
            probe_scan['x'] = np.linspace(start,stop,num)
            for pn in probe_num:
                for key in raw_spectrum_data['data']:
                    if key==f'{pn}/{i}':
                        for val in raw_spectrum_data['data'][key]:
                            if val=='calibrated':
                                for j, rep in enumerate(raw_spectrum_data['data'][key][val]):
                                    probe_scan[f'{pn}_0{j+1}'] = rep
            scan_data.append(probe_scan)

    results = [ ]
    spectra = [ ]
    probe_keys = [ ]
    for ind, scan in enumerate(scan_data):
        spectra.append(dict())
        cal_probe_data = { }
        cal_probe_names = [ ]
        proc_probe_data = { }
        try:
            for key in scan.keys():
                if key!='x' and key!='metadata' and not key.endswith('_raw'):
                    if key[0] in probe_nos:
                        cal_probe_names.append(key)

            for name in cal_probe_names:
                cal_probe_data[name] = [ ]
                cal_probe_data[name].append(scan[str(name)])

            proc_probe_data['x'] = scan['x']
        except:
            print('Please ensure that the probe numbers inputted are correct')
            sys.exit()

        #Normalise and baseline probe data
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

        # Append wavenumber data to probe data dictionary
        avg_spectrum_data['x'] = proc_probe_data['x']

        for key in avg_spectrum_data.keys():
            if key!='x':
                probe_keys.append(key)
                spectra[ind][key] = avg_spectrum_data[key].copy()
        #results.append((avg_spectrum_data))    #If individual scan data is needed as well, uncomment

    #Also append average data
    probe_keys = np.unique(probe_keys)
    scan_avg_spectrum_data = { }
    for key in probe_keys:
        probe_avg = [ ]
        for scan in spectra:
            probe_avg.append(scan[key])
        scan_avg_spectrum_data[key] = list(np.average(probe_avg,axis=0))
    scan_avg_spectrum_data['x'] = avg_spectrum_data['x']

    return scan_avg_spectrum_data, spectrum_id

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx], idx

def list_duplicates(seq):
    tally = defaultdict(list)
    for i,item in enumerate(seq):
        tally[item].append(i)
    return ((key,locs) for key,locs in tally.items()
                            if len(locs)>1)

def train_spectra_compilation(filepath, data, train_probe_data, pnum, s, e):
    compiled_data = { }
    with open(filepath, 'rb') as infile:
        for saved_data in msg.Unpacker(infile, raw=False):
            for key in saved_data:
                compiled_data[key] = saved_data[key].copy()

    #Find probe number from filepath
    match = re.search('Probe(\d+)', repr(filepath))
    if match:
        file_probe = match.group(0)

    #Make sure that list of spectra from which QC params are derived match spectrum train set
    labels = list(np.array(data['Labels'])[np.isin(data['Labels'], list(compiled_data.keys()))])

    #trim input spectra as per peak ranges
    truth = pnum == file_probe

    pnum = pnum[truth]
    s = s[truth]
    e = e[truth]

    train_data = { }
    for key in labels:
        full_data = compiled_data[key]
        x_res = full_data['x'][1] - full_data['x'][0]
        y_data = [ ]
        for start,end in zip(s,e):
            full_data = compiled_data[key]
            x,y = map(list,zip(*[(x,y) for x,y in zip(full_data['x'],full_data['spectrum']) if start <= x <= end]))
            y_data.extend(y)
        x_data = np.linspace(0, len(y_data)*x_res, len(y_data))

        train_data[key] = dict(x = x_data, spectrum = y_data)

    if file_probe not in train_probe_data:
        train_probe_data[file_probe] = { }
        train_probe_data[file_probe].update(train_data.copy())
        train_probe_data[file_probe]['Labels'] = labels
    else:
        train_probe_data[file_probe].update(train_data.copy())
        train_probe_data[file_probe]['Labels'].extend(labels)

def test_spectra_compilation(test_probe_data, key_archive, probe_spec_archive, filepath, probe_arr, probe_nos, PEAK_RANGES):
    scan_avg_spectrum_data, spectrum_id = process_input_spectrum(filepath, probe_nos)

    #Collate input data (and trim for spectra PCA)
    data_keys = [ ]
    probe_spectra = [ ]
    for i,probe_selection in enumerate(probe_arr):
        input_spectrum_data = dict(x = scan_avg_spectrum_data['x'], spectrum = scan_avg_spectrum_data[probe_nos[i]])
        probe_spectra.append(input_spectrum_data)

        #Separate the variables
        pnum = PEAK_RANGES['Probe_num']
        s = PEAK_RANGES['wavenumber_start']
        e = PEAK_RANGES['wavenumber_end']
        c = PEAK_RANGES['num_peaks']
        ppx = PEAK_RANGES['prob_peak_x']
        ppy = PEAK_RANGES['prob_peak_y']
        ppi = PEAK_RANGES['prob_peak_index']

        truth = pnum == probe_selection

        pnum = pnum[truth]
        s = s[truth]
        e = e[truth]
        c = c[truth]
        ppx = ppx[truth]
        ppy = ppy[truth]
        ppi = ppi[truth]

        for i, val in enumerate(pnum):
            data_keys.append((input_spectrum_data.copy(),pnum[i],s[i],e[i],c[i],ppx[i],ppy[i],ppi[i]))

        y_data = [ ]
        x_res = input_spectrum_data['x'][1] - input_spectrum_data['x'][0]
        for start,end in zip(s,e):
            x,y = map(list,zip(*[(x,y) for x,y in zip(input_spectrum_data['x'],input_spectrum_data['spectrum']) if start <= x <= end]))
            y_data.extend(y)
        x_data = np.linspace(0, len(y_data)*x_res, len(y_data))
        #Interpolate to match train data resolution
        f = interpolate.interp1d(x_data, y_data, kind='quadratic')
        x_data = np.linspace(0, len(y_data)*(x_res), x_length[probe_selection])
        y_data = f(x_data)

        test_data = dict(x = x_data, spectrum = y_data)

        if probe_selection not in test_probe_data:
            test_probe_data[probe_selection] = { }
            test_probe_data[probe_selection].update({f'{spectrum_id}': test_data})
        else:
            test_probe_data[probe_selection].update({f'{spectrum_id}': test_data})
    key_archive.append(data_keys)
    probe_spec_archive.append(probe_spectra)

    return test_probe_data, key_archive, probe_spec_archive

def master_fitting(key_archive, probe_spec_archive, probelist, filepath, k):
    spectrum_id = str(os.path.splitext(os.path.basename(filepath))[0]) #spectrum_id
    data_keys = key_archive[k]
    probe_spectra = probe_spec_archive[k]
    #Now execute parallel code
    futures = [pool.submit(run_fitting,data_key) for data_key in data_keys]
    #Once a result is ready, main task appends results to array
    outputs = [ ]
    for future in as_completed(futures):
        if future.result() is not None:
            outputs.append(future.result())

    #Collate outputs and order in terms of individual probes
    order = [ ]
    output_probe = [ ]
    for probe in probe_arr:
        op = [ ]
        od = [ ]
        for output in outputs:
            result, pp, best_fit, uncertainty_interval, x_data, x_grid_res, probe_num = output
            if probe_num==probe:
                op.append(output)
                od.extend([min(x_data)])
        output_probe.append(op)
        order.append(od)

    for i, output in enumerate(output_probe):
        output_probe[i] = list(np.array(output, dtype='object')[np.argsort(order[i])])    #Sort outputs based on minimum of x_data array (to preserve peak order)

    concat_params = { }
    for i, probe_results in enumerate(output_probe):
        peak_params = { }
        model_bf = { }
        pn = 0 #Keep track of peak numbers
        peak_profiles = [ ]
        for output in probe_results:
            result, pp, best_fit, uncertainty_interval, x_data, x_grid_res, probe_num = output
            peak_profiles.extend(pp)
            #Collate peak data
            peak_params, pn = peak_data(peak_params, pp, result, x_data, x_grid_res, pn)

            #Collate best-fit model data
            model_bf = best_fit_model(model_bf, x_data, best_fit, uncertainty_interval)

        # #Plot fitting range and best-fit spectrum (with 1-sigma uncertainty region)
        # plt.figure()
        # plt.plot(probe_spectra[i]['x'],probe_spectra[i]['spectrum'],linewidth=1.0, alpha=0.5)
        # plt.plot(model_bf['x'],model_bf['best_fit'],color='k',linewidth=1.0)
        # plt.fill_between(model_bf['x'],np.array(model_bf['best_fit'])-np.array(model_bf['uncertainty_interval']),np.array(model_bf['best_fit'])+np.array(model_bf['uncertainty_interval']),color='r', alpha=0.4)
        # for profile in peak_profiles:
        #     plt.plot(profile[0],profile[1],color='k',linewidth=1.0,linestyle=':')
        # plt.xlabel(r'$\mathrm{Wavenumber \ [cm}^{-1}]$', fontsize=14)
        # plt.ylabel(r'$\mathrm{Intensity}$', fontsize=14)
        # plt.tight_layout()
        # plt.show()

        #Concat params here
        probe_array = [ ]
        for y in peak_params:
            if '_std' not in y:
                key = f'{probe_num[:-1]}_{probe_num[-1]}_{y}'
                probe_array.extend([peak_params[y]])
                concat_params[key] = peak_params[y]

        #Also append to PCA params dictionary for later
        PCA_params[probelist[i]].append(probe_array)

    if np.all([value is not None for value in concat_params.values()]):   #Catch None values in array
        save_model_data(concat_params, spectrum_id, params_filepath)
        PCA_params['Concat'].append(list(concat_params.values()))
    else:
        print(f'Fit for {spectrum_id} has failed! Best-fit parameters will not be saved.')
        sys.exit()

    return PCA_params

if __name__ == '__main__':
    freeze_support()
    with ProcessPoolExecutor() as pool:
        #Set train data folder path
        TRAIN_FOLDER_PATH = Path('Train_Spectra/')

        #Set save folder path
        RESULTS_FOLDER_PATH = Path('Fit_Results/')

        #Load variables from inlist
        peak_range_file = str(sys.argv[1])
        probe_nos = str(sys.argv[2]).split(',')
        probe_arr = [f'Probe{i}' for i in probe_nos]

        #Load peak ranges, number of peaks, problematic peak position and intensity in each range
        PEAK_RANGES = np.genfromtxt(peak_range_file, delimiter=' ', dtype=None, names=True, encoding='utf-8')

        #Print important variables
        print(f'PEAK_RANGES = \n {str(PEAK_RANGES)}')

        #sort ranges for later trimming
        pnum = PEAK_RANGES['Probe_num']
        s = PEAK_RANGES['wavenumber_start']
        e = PEAK_RANGES['wavenumber_end']

        #Find training spectra
        train_spectra_filepaths = [ ]
        for path in Path(TRAIN_FOLDER_PATH).rglob('*.msg'):
            if path.is_file():
                train_spectra_filepaths.append(path)

        #Load benchmark QC params
        param_file = 'Dispensing_QC_train.csv'
        data = np.genfromtxt(param_file, delimiter=',', dtype=None, names=True, encoding='utf-8')

        #Load training spectra
        print(f'Generating train dataset for spectra PCA. Please stand by...')
        train_probe_data = { }
        for filepath in train_spectra_filepaths:
            if any(probe in str(filepath) for probe in probe_arr):
                train_spectra_compilation(filepath, data, train_probe_data, pnum, s, e)

        #Get common labels between probes for concatenation (just in case)
        list_of_labels = [train_probe_data[probe]['Labels'] for probe in train_probe_data]
        common_labels = list(set.intersection(*map(set, list_of_labels)))

        #Also create concatenated spectra for train set for selected probes (the order depends on input!)
        train_probe_data['Concat'] = { }
        x_length = { }
        for key in common_labels:
            concat_spectrum = [ ]
            for probe in probe_arr:
                x_length[probe] = len(train_probe_data[probe][key]['x'])
                concat_spectrum.extend(train_probe_data[probe][key]['spectrum'])
            x_res = train_probe_data[probe][key]['x'][1] - train_probe_data[probe][key]['x'][0]
            concat_x = np.linspace(0,len(concat_spectrum)*x_res,len(concat_spectrum))
            train_probe_data['Concat'][key] = dict(x = concat_x, spectrum = concat_spectrum)

        #Load files
        raw_spectra_dirs = [ ]
        raw_spectra_filepaths = [ ]
        #Find raw spectra json files and corresponding subfolders (if they exist)
        for path in Path('Data/').rglob('*'):
            if path.is_file():
                raw_spectra_filepaths.append(path)
            if path.is_dir():
                raw_spectra_dirs.append(path)

        #Generate new savefile
        today = date.today()
        today = today.strftime("%Y%m%d")
        params_filename = f'concat_params_{today}.csv'
        params_filepath = RESULTS_FOLDER_PATH / params_filename
        with open(params_filepath, 'w') as csvfile:
            pass

        PCA_params = { }
        probelist = [ ]
        #Setup dictionaries for PCA
        for probe in probe_nos:
            probelist.append(f'Probe_{probe}')
            PCA_params[f'Probe_{probe}'] = [ ]
        PCA_params['Concat'] = [ ]

        #For multibatch PCA, append data to another list
        if raw_spectra_dirs:    #If subfolders exist...
            test_probe_list = [ ]
            key_arclist = [ ]
            probe_spec_arclist = [ ]
            for dir in raw_spectra_dirs:
                #Collate test data
                test_probe_data = { }
                key_archive = [ ]
                probe_spec_archive = [ ]
                for filepath in raw_spectra_filepaths:
                    if str(dir) in str(filepath):
                        test_probe_data, key_archive, probe_spec_archive = test_spectra_compilation(test_probe_data, key_archive, probe_spec_archive, filepath, probe_arr, probe_nos, PEAK_RANGES)
                #Also create concatenated spectra for test set for selected probes (the order depends on input!)
                test_probe_data['Concat'] = { }
                for key in test_probe_data[probe_arr[0]]:
                    concat_spectrum = [ ]
                    for probe in probe_arr:
                        concat_spectrum.extend(test_probe_data[probe][key]['spectrum'])
                    x_res = test_probe_data[probe][key]['x'][1] - test_probe_data[probe][key]['x'][0]
                    concat_x = np.linspace(0,len(concat_spectrum)*x_res,len(concat_spectrum))
                    test_probe_data['Concat'][key] = dict(x = concat_x, spectrum = concat_spectrum)
                test_probe_list.append(test_probe_data)
                key_arclist.append(key_archive)
                probe_spec_arclist.append(probe_spec_archive)
        else:
            test_probe_data = { }
            key_archive = [ ]
            probe_spec_archive = [ ]
            #Collate test data
            for filepath in raw_spectra_filepaths:
                test_probe_data, key_archive, probe_spec_archive = test_spectra_compilation(test_probe_data, key_archive, probe_spec_archive, filepath, probe_arr, probe_nos, PEAK_RANGES)

            #Also create concatenated spectra for test set for selected probes (the order depends on input!)
            test_probe_data['Concat'] = { }
            for key in test_probe_data[probe_arr[0]]:
                concat_spectrum = [ ]
                for probe in probe_arr:
                    concat_spectrum.extend(test_probe_data[probe][key]['spectrum'])
                x_res = test_probe_data[probe][key]['x'][1] - test_probe_data[probe][key]['x'][0]
                concat_x = np.linspace(0,len(concat_spectrum)*x_res,len(concat_spectrum))
                test_probe_data['Concat'][key] = dict(x = concat_x, spectrum = concat_spectrum)

        #Plot Spectra PCA
        train_set = { }
        test_set = { }
        classes = [ ]
        #Concatenate classes for train data and test data
        for val in data['Class']:
            classes.extend([val])
        classes = np.array(classes)
        batch_ids = [ ]
        if raw_spectra_dirs:
            for dir, test_probe_data in zip(raw_spectra_dirs, test_probe_list):
                for val in test_probe_data['Concat']:
                    batch_ids.extend([PurePath(dir).name])
        else:
            for val in test_probe_data['Concat']:
                batch_ids.extend(['New_Batch'])
        classes = np.append(classes,batch_ids)
        unique_classes = np.unique(classes)
        #Append spectra for each probe (and concatenated) to lists
        for probe in train_probe_data: #Select only spectra corresponding to selected probes!
            train_set[probe] = [ ]
            for key in train_probe_data[probe]:
                if key!='Labels':
                    train_set[probe].append(train_probe_data[probe][key]['spectrum'])
        if raw_spectra_dirs:
            for test_probe_data in test_probe_list:
                for probe in test_probe_data: #Select only spectra corresponding to selected probes!
                    if not probe in test_set: #Create array if empty
                        test_set[probe] = [ ]
                    for key in test_probe_data[probe]:
                        test_set[probe].append(test_probe_data[probe][key]['spectrum'])
        else:
            for probe in test_probe_data: #Select only spectra corresponding to selected probes!
                test_set[probe] = [ ]
                for key in test_probe_data[probe]:
                    test_set[probe].append(test_probe_data[probe][key]['spectrum'])

        for key in train_set:   #For each probe (and concat), generate PCA plot
            print(f'Generating Spectra PCA for {key} data')
            reference_data = train_set[key]
            new_data = test_set[key]

            reference_data = np.column_stack(reference_data)
            #Add new params to reference_data because otherwise normalisation may fail (no choice here)
            full_data = np.column_stack((reference_data,np.array(new_data).T)).T
            model = pca(n_components=0.95, normalize=True)    #Calculate enough principal components that explain 95% of total variance
            results = model.fit_transform(full_data, row_labels = classes)

            fontP = FontProperties()
            fontP.set_size('small')

            #Plot PCA distribution (visualise 2 components)
            fig, ax, colour_cycle  = model.scatter(label=False)
            #99.7% CI ellipse to represent acceptable range
            cov = np.cov(results['PC']['PC1'].values, results['PC']['PC2'].values)
            PC1_mean = np.mean(results['PC']['PC1'].values)
            PC2_mean = np.mean(results['PC']['PC2'].values)
            e = get_cov_ellipse(cov, (PC1_mean, PC2_mean), nstd=3, alpha=1.0, lw=1, linestyle = '--', fill=False, label = r'99.7% confidence level')
            ax.add_patch(e)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left',prop = fontP)
            ax.grid(True)
            #99% CI ellipses around each distribution
            for i,val in enumerate(unique_classes):
                if not isinstance(results['PC']['PC1'][val],float):
                    cov = np.cov(results['PC']['PC1'][val].values, results['PC']['PC2'][val].values)
                    PC1_mean = np.mean(results['PC']['PC1'][val].values)
                    PC2_mean = np.mean(results['PC']['PC2'][val].values)
                    se = get_cov_ellipse(cov, (PC1_mean, PC2_mean), nstd=3, alpha=0.1, color=colour_cycle[i,:])
                    ax.add_patch(se)
            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()
            plt.tight_layout()
            fig.savefig(RESULTS_FOLDER_PATH / f'Spectra_PCA_{key}_{today}.png', dpi=300)
            #plt.show()

        #Start fitting
        overall_start = time.time()
        if raw_spectra_dirs:
            PCA_list = [ ]
            for dir, key_archive, probe_spec_archive in zip(raw_spectra_dirs, key_arclist, probe_spec_arclist):
                #Use per-file and per-region parallelism to save time
                batchpaths = [s for s in raw_spectra_filepaths if str(dir) in str(s)]
                for k, filepath in enumerate(batchpaths):
                    if str(dir) in str(filepath):
                        PCA_params = master_fitting(key_archive, probe_spec_archive, probelist, filepath, k)
            PCA_list.append(PCA_params)
        else:
            #Use per-file and per-region parallelism to save time
            for k, filepath in enumerate(raw_spectra_filepaths):
                PCA_params = master_fitting(key_archive, probe_spec_archive, probelist, filepath, k)

        overall_end = time.time()
        print(f'Time taken for overall fitting: {str((overall_end - overall_start)/60.)} min')

        #Plot Params PCA
        param_set = { }
        cols = { }
        classes = [ ]
        #Concatenate classes for train data and test data
        for val in data['Class']:
            classes.extend([val])
        classes = np.array(classes)
        batch_ids = [ ]
        if raw_spectra_dirs:
            for dir, test_probe_data in zip(raw_spectra_dirs, test_probe_list):
                for val in test_probe_data['Concat']:
                    batch_ids.extend([PurePath(dir).name])
        else:
            for val in test_probe_data['Concat']:
                batch_ids.extend(['New_Batch'])
        classes = np.append(classes,batch_ids)
        unique_classes = np.unique(classes)
        #Append params for each probe (and concatenated) to dictionaries
        param_set['Concat'] = [ ]   #Concatenated probe params
        cols['Concat'] = [ ]
        for probe in probelist: #Select only peak params corresponding to selected probes!
            param_set[probe] = [ ]
            cols[probe] = [ ]
            for key in data.dtype.names:
                if key!='Label' and key!='Class':
                    if probe in key:
                        cols[probe].extend([key])
                        cols['Concat'].extend([key])
                        param_set[probe].append(data[key])
                        param_set['Concat'].append(data[key])

        for key in param_set:   #For each probe (and concat), generate PCA plot
            print(f'Generating Params PCA for {key} data')
            reference_data = param_set[key]
            if raw_spectra_dirs:
                new_data = [ ]
                for PCA_params in PCA_list:
                    new_data.append(PCA_params[key])
                new_data = np.vstack(new_data)
            else:
                new_data = PCA_params[key]
            columns = cols[key]

            reference_data = np.column_stack(reference_data)
            #Add new params to reference_data because otherwise normalisation may fail (no choice here)
            full_data = np.vstack((reference_data,new_data))
            model = pca(n_components=0.95, normalize=True)    #Calculate enough principal components that explain 95% of total variance
            results = model.fit_transform(full_data, row_labels = classes, col_labels = columns)

            fontP = FontProperties()
            fontP.set_size('small')

            #Plot PCA distribution (visualise 2 components)
            fig, ax, colour_cycle  = model.scatter(label=False)
            #99.7% CI ellipse to represent acceptable range
            cov = np.cov(results['PC']['PC1'].values, results['PC']['PC2'].values)
            PC1_mean = np.mean(results['PC']['PC1'].values)
            PC2_mean = np.mean(results['PC']['PC2'].values)
            e = get_cov_ellipse(cov, (PC1_mean, PC2_mean), nstd=3, alpha=1.0, lw=1, linestyle = '--', fill=False, label = r'99.7% confidence level')
            ax.add_patch(e)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left',prop = fontP)
            ax.grid(True)
            #99% CI ellipses around each distribution
            for i,val in enumerate(unique_classes):
                if not isinstance(results['PC']['PC1'][val],float):
                    cov = np.cov(results['PC']['PC1'][val].values, results['PC']['PC2'][val].values)
                    PC1_mean = np.mean(results['PC']['PC1'][val].values)
                    PC2_mean = np.mean(results['PC']['PC2'][val].values)
                    se = get_cov_ellipse(cov, (PC1_mean, PC2_mean), nstd=3, alpha=0.1, color=colour_cycle[i,:])
                    ax.add_patch(se)
            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()
            plt.tight_layout()
            fig.savefig(RESULTS_FOLDER_PATH / f'Params_PCA_{key}_{today}.png', dpi=300)
            #plt.show()
