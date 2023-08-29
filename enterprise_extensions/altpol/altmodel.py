def altpol_model(psrs,
                noise_dict,
                MG = '1000',
                crn_bins = 14,
                prior = 'log-uniform',
                kappa = None,
                pdist = None,
                gamma_list = [4.333, 5,5,5],
                amp_list = None):

    Npulsars = len(psrs)
    Tspan = model_utils.get_tspan(psrs)
    pols = ['TT', 'ST', 'VL', 'SL']
    names = [psr.name for psr in psrs]
    mask = [bool(int(m)) for m in MG]
    freqs = np.arange(1/Tspan, (crn_bins+.1)/Tspan, 1/Tspan)
    fnorm = np.repeat(freqs * kpc_to_meter/light_speed,2)

    #Amplitudes
    if not amp_list:
        amp_list = []
        if prior == 'log-uniform':
            log10_As = [parameter.Uniform(-18,-11)('gw_log10_A_{}'.format(pol)) for pol in pols]
        elif prior == 'lin-exp':
            log10_As = [parameter.LinearExp(-18,-11)('gw_log10_A_{}'.format(pol)) for pol in pols]
        else:
            raise ValueError("The prior given is invalid. Choose between 'log-uniform' and 'lin-exp'.")
        for m,l in zip(mask, log10_As):
            if m:
                amp_list.append(l)
            else:
                amp_list.append(-np.inf)

    #Spectral Indicies
    if not gamma_list:
        gamma_list = []
        sindex_list = [parameter.Uniform(0, 7)('gamma_' + pol) for pol in pols]
        for m,g in zip(mask, sindex_list):
            if m:
                gamma_list.append(g)

    #Timing model
    s = gp_signals.MarginalizingTimingModel(use_svd = True)

    #White noise
    s += blocks.white_noise_block(vary=False, inc_ecorr= True)

    #Pulsar red noise
    s +=  blocks.red_noise_block(prior='log-uniform', Tspan=Tspan, components=30)


    #ORF Functions
    orf_funcs = [model_orfs.hd_orf, model_orfs.st_orf, model_orfs.vl_orf, model_orfs.sl_orf]

    #Pulsar distances
    param_list = {'{}_p_dist'.format(names[ii]):parameter.Normal(pdist[0,ii], pdist[1,ii])('{}_gw_p_dist'.format(names[ii])) for ii in range(Npulsars)}
    param_list.update({'fnorm': fnorm})
    param_list.update({'pnames': names})

    #PSD
    kappa = parameter.Uniform(0, 10)('kappa')
    p_dist_dummy = parameter.Normal(1, .2)

    ##TT Powerlaw
    pl_tt = model_orfs.generalized_gwpol_psd(log10_A_tt = amp_list[0],
                                            log10_A_st=-np.inf,
                                            log10_A_vl=-np.inf,
                                            log10_A_sl=-np.inf,
                                            alpha_tt = (3 - gamma_list[0])/2,
                                            alpha_alt = (3 - gamma_list[1])/2,
                                            kappa = kappa,
                                            p_dist = p_dist_dummy)
    ##ST Powerlaw
    pl_st = model_orfs.generalized_gwpol_psd(log10_A_tt = -np.inf,
                                            log10_A_st= amp_list[1],
                                            log10_A_vl=-np.inf,
                                            log10_A_sl=-np.inf,
                                            alpha_tt = (3 - gamma_list[0])/2,
                                            alpha_alt = (3 - gamma_list[1])/2,
                                            kappa = kappa,
                                            p_dist = p_dist_dummy)
    ##VL Powerlaw
    pl_vl = model_orfs.generalized_gwpol_psd(log10_A_tt = -np.inf,
                                            log10_A_st=-np.inf,
                                            log10_A_vl=amp_list[2],
                                            log10_A_sl=-np.inf,
                                            alpha_tt = (3 - gamma_list[0])/2,
                                            alpha_alt = (3 - gamma_list[1])/2,
                                            kappa = kappa,
                                            p_dist = p_dist_dummy)
    ##SL Powerlaw
    pl_sl = model_orfs.generalized_gwpol_psd(log10_A_tt = -np.inf,
                                            log10_A_st=-np.inf,
                                            log10_A_vl=-np.inf,
                                            log10_A_sl=amp_list[3],
                                            alpha_tt = (3 - gamma_list[0])/2,
                                            alpha_alt = (3 - gamma_list[1])/2,
                                            kappa = kappa,
                                            p_dist = p_dist_dummy)


    #GW red noise
    s_gw = []
    s_gw.append(gp_signals.FourierBasisCommonGP(spectrum=pl_tt, components=crn_bins,Tspan=Tspan, name='{}_gw'.format(pols[0]),
                                        orf = model_orfs.hd_orf()))

    s_gw.append(gp_signals.FourierBasisCommonGP(spectrum=pl_st, components=crn_bins,Tspan=Tspan, name='{}_gw'.format(pols[1]),
                                        orf = model_orfs.st_orf()))

    s_gw.append(gp_signals.FourierBasisCommonGP(spectrum=pl_vl, components=crn_bins,Tspan=Tspan, name='{}_gw'.format(pols[2]),
                                                    orf = model_orfs.vl_orf(**param_list)))

    s_gw.append(gp_signals.FourierBasisGP(spectrum=pl_sl, components=crn_bins,Tspan=Tspan, name='{}_gw'.format(pols[3]))),

    for m,sg in zip(mask, s_gw):
        if m:
            s+=sg

    #Constructing the PTA object
    model_list = []
    for p in psrs:
        model_list.append(s(p))
    pta = signal_base.PTA(model_list)
    pta.set_default_params(noise_dict)

    #Setting the parameters to their given values
    ii = 0
    if pdist.shape[0] == 2:
        for param in pta.params:
            if 'p_dist' in param.name:
                mu = pdist[0][ii]
                sigma = pdist[1][ii]
                param._prior = parameter.Normal(mu=mu,sigma=sigma)
                param._mu = mu
                param._sigma = sigma
                ii += 1
    return pta
