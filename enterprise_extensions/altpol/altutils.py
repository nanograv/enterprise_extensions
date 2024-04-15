import numpy as np
from la_forge.core import Core
import pickle, os, json, glob, copy, time
from enterprise.pulsar import Pulsar
from enterprise_extensions.altpol import altorfs, altutils
from enterprise_extensions import model_utils
import matplotlib.pyplot as plt

def ml_finder(chaindir, savedir = False):
    c = Core(label = '', chaindir = chaindir)
    params = c.params[:-4]
    burn = int(.25 * c.chain.shape[0])
    chain = c.chain[burn:,:]
    ml_params  = {}
    for ii, name in enumerate(params):
        idx = np.argmax(chain[:, -4])
        max_ll_val = chain[:, ii][idx]
        ml_params[name] = max_ll_val

    ml_params_orig = copy.deepcopy(ml_params)
    keys = list(ml_params.keys())
    if not 'gw_log10_A_TT' in keys:
        ml_params['gw_log10_A_TT'] = -np.inf
    if not 'gw_log10_A_ST' in keys:
        ml_params['gw_log10_A_ST'] = -np.inf
    if not 'gw_log10_A_VL' in keys:
        ml_params['gw_log10_A_VL'] = -np.inf

    if savedir:
        os.makedirs(savedir, exist_ok = True)
        with open(savedir + '/ml_params.json', 'w' ) as fout:
            json.dump({'orig': ml_params_orig, 'added': ml_params}, fout, sort_keys=True,
                      indent=4, separators=(',', ': '))
    return ml_params_orig

def med_finder(chaindir, savedir = False):
    c = Core(label = '', chaindir = chaindir)
    params = c.params[:-4]
    burn = int(.25 * c.chain.shape[0])
    chain = c.chain[burn:,:]
    param_dict = {}
    params = list(params)
    for p in params:
        param_dict.update({p: np.median(chain[:, params.index(p)])})

    param_dict_orig = copy.deepcopy(param_dict)

    keys = list(param_dict.keys())
    if not 'gw_log10_A_TT' in keys:
        param_dict['gw_log10_A_TT'] = -np.inf
    if not 'gw_log10_A_ST' in keys:
        param_dict['gw_log10_A_ST'] = -np.inf
    if not 'gw_log10_A_VL' in keys:
        param_dict['gw_log10_A_VL'] = -np.inf

    if savedir:
        with open(savedir + '/med_params.json', 'w' ) as fout:
            json.dump({'orig': param_dict_orig, 'added': param_dict}, fout, sort_keys=True,
                      indent=4, separators=(',', ': '))
    return param_dict_orig

def crn_bins_finder(chaindir, Tspan):
    ml_params = ml_finder(chaindir)
    freqs = np.arange(1/Tspan, 30/Tspan, 1/Tspan)
    fb = 10**ml_params['gw_log10_fb']
    crn_bins = np.argmin(np.abs(freqs - fb)) + 1
    if crn_bins > 5:
        return crn_bins
    else:
        return 5

def psrs_slicer(psrs, min_year = 10, interval = 1):
    day_sec = 86400
    year_sec = 365.25 * day_sec
    a = []; sliced_psrs = {}
    max_year = int(round(model_utils.get_tspan(psrs)/(year_sec)))
    for psr in psrs: a.append(min(psr._toas)/day_sec)
    start_time = min(a)
    for bct,base in enumerate(np.arange(max_year,min_year,-interval)):
        if bct == 0:
            sliced_psrs.update({'{}'.format(max_year): copy.deepcopy(psrs)})
        else:
            psrs_moded = []
            end_time = start_time + base * (365.25)
            for psr in copy.deepcopy(psrs):
                psr.filter_data(start_time = start_time, end_time = end_time)
                if (psr.toas.size == 0) or (model_utils.get_tspan([psr]) < min_year * year_sec):
                    continue
                else:
                    psrs_moded.append(psr)
            sliced_psrs.update({'{}'.format(max_year - bct): psrs_moded})

    return sliced_psrs

def liklihood_check(pta, n = 10):
    st = time.time()
    for _ in range(n):
        x0 = np.hstack(p.sample() for p in pta.params)
        print(pta.get_lnlikelihood(x0))
    print('***Time Elapsed(s): ***', time.time() - st)

def weightedavg(rho, sig):
    weights, avg = 0., 0.
    for r,s in zip(rho,sig):
        weights += 1./(s*s)
        avg += r/(s*s)
    return avg/weights, np.sqrt(1./weights)

def bin_crosscorr(zeta, xi, rho, sig):
    rho_avg, sig_avg = np.zeros(len(zeta)), np.zeros(len(zeta))
    for i,z in enumerate(zeta[:-1]):
        myrhos, mysigs = [], []
        for x,r,s in zip(xi,rho,sig):
            if x >= z and x < (z+10.):
                myrhos.append(r)
                mysigs.append(s)
        rho_avg[i], sig_avg[i] = weightedavg(myrhos, mysigs)
    return rho_avg, sig_avg

def binned_corr_Maker(xi, rho, sig, bins):
    n_pulsars_per_bin,bin_loc = np.histogram(xi, density = False, bins = bins)
    xi_mean = []; xi_err = []; rho_avg = []; sig_avg = []
    for ii in range (len(bin_loc) - 1):
        mask = np.logical_and(xi >= bin_loc[ii] , xi < bin_loc[ii+1])
        if not rho[mask].size == 0:
            r, s = weightedavg(rho[mask], sig[mask])
            rho_avg.append(r); sig_avg.append(s)
            xi_mean.append(np.mean(xi[mask]))
            xi_err.append(np.std(xi[mask]))
    return np.array(xi_mean), np.array(xi_err), np.array(rho_avg), np.array(sig_avg)

def get_HD_curve(zeta):
    coszeta = np.cos(zeta*np.pi/180.)
    xip = (1.-coszeta) / 2.
    HD = 3.*( 1./3. + xip * ( np.log(xip) -1./6.) )
    return HD/2

def get_ST_curve(zeta):
    coszeta = np.cos(zeta*np.pi/180.)
    return 1/8 * (3 + coszeta)

def get_VL_curve(zeta):
    coszeta = np.cos(zeta*np.pi/180.)
    return 3*np.log(2/(1-coszeta)) -4*coszeta - 3

def gt( x, tau, mono ) :
    if np.all(x) == 0:
        return 1 + mono
    else:
        cos_ang = np.cos(x)
        k = 1/2*(1-cos_ang)
        return 1/8 * (3+cos_ang) + (1-tau)*3/4*k*np.log(k) + mono

def vl_search(x,a,b,c):
    cos_ang = np.cos(x)
    return a*np.log(2/(1-cos_ang)) -b*cos_ang - c

def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value


def set_axis_style(ax, labels):
    ax.xaxis.set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(labels) + 1), labels=labels)
    ax.set_xlim(0.25, len(labels) + 0.75)

def BinModelTearDrop(data, realization, xi, nbins, split_first = False, orf = altorfs.HD_ORF,
            orfname = 'HD', savedir = None, step = 'Binmodel', masterdir = None,
            facecolor = 'lightblue', barcolor = 'blue', edgecolor = 'black',
            marker_color = 'red', orf_color = 'red', alpha = 1):
    rad_to_deg = 180/np.pi
    if not data:
        data = []
        if split_first:
            pars = ['gw_orf_bin_{}'.format(_) for _ in range(nbins+1)]
        else:
            pars = ['gw_orf_bin_{}'.format(_) for _ in range(nbins)]
        if realization == 'all':
            x = altutils.chain_loader(masterdir,
                   step = step,
                   realization = 'all')
            for _ in range(len(x[step])):
                p = x[step][str(_)]
                data.append(p[0].get_param(pars))
        else:
            x = altutils.chain_loader(masterdir,
               step = step,
               realization = realization)
            p = x[step][str(realization)]
            data.append(p[0].get_param(pars))

        data = list(np.hstack([list(data[ii].T) for ii in range(len(data))]))
    bins = np.quantile(xi, np.linspace(0,1,nbins+1))
    if split_first:
        bins = np.insert(bins, 1, (bins[0] + bins[1])/2)
    widths = np.diff(bins) * rad_to_deg
    #widths = np.diff(np.insert(bins, len(bins), np.pi)) * rad_to_deg
    orfsigs = np.sqrt(.5 * (orf(xi)**2 + 4 * orf(.00001)**2))
    xi_mean, xi_err, rho_avg, sig_avg = binned_corr_Maker(xi, orf(xi), orfsigs, bins)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6), sharey=True, dpi = 170)

    parts = ax.violinplot(data, bw_method = 0.5,
                          positions = xi_mean * rad_to_deg,
                         widths = widths,
                         showmedians = True)

    for pc in parts['bodies']:
        pc.set_facecolor(facecolor)
        pc.set_edgecolor(edgecolor)
        pc.set_alpha(1)
    parts['cmedians'].set_color(barcolor)
    parts['cbars'].set_color(barcolor)
    parts['cmins'].set_color(barcolor)
    parts['cmaxes'].set_color(barcolor)
    angs = np.linspace(.001, np.pi, 100)
    plt.plot(angs * 180/np.pi, orf(angs), lw = 2, ls = '--', color = orf_color)
    plt.errorbar(xi_mean*180/np.pi, rho_avg, xerr=xi_err*180/np.pi, yerr=sig_avg, marker='o', ls='', alpha = alpha,
                                color=marker_color, capsize=4, elinewidth=1.2,
                               label = 'Theoretical {}'.format(orfname))

    plt.xlabel('Angular Separation (deg)')
    plt.ylabel('Correlations')
    plt.legend()
    if savedir:
        plt.savefig(savedir, dpi = 600)
    plt.show()
