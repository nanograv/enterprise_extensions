1. A deterministic signal for the solar wind that has a common mean solar electron density for all of the pulsars. One can choose one value for the entire data set, or partition into bins. 
```Python
#Here there are 11 bins
n_earth = parameter.Uniform(0,20,size=11)('n_earth')
sw = SW.solar_wind(n_earth=n_earth,n_earth_bins=11,t_init=tmin,t_final=tmax)
mean_sw = deterministic_signals.Deterministic(sw, name='mean_sw')
```
2. I have also included a astrophysical prior for the electron density:
Here is a single value example using the ACE prior:
```Python
n_earth = SW.ACE_SWEPAM_Parameter()('n_earth')
sw = SW.solar_wind(n_earth=n_earth)
mean_sw = deterministic_signals.Deterministic(sw, name='mean_sw')
```
3. There is also a function to create a Fourier design matrix for constructing a Gaussian process. This can be used as desired, but works best as a *perturbation* to the mean signal defined above.
```Python
dm_sw_basis = SW.createfourierdesignmatrix_solar_dm(nmodes=15,Tspan=Tspan)
dm_sw_prior = utils.powerlaw(log10_A=log10_A_dm_sw, gamma=gamma_dm_sw)
gp_sw = gp_signals.BasisGP(priorFunction=dm_sw_prior, basisFunction=dm_sw_basis, name='gp_sw')
```
4. There is also a convenience function for making the "best" version of the SW model. The user can choose to include a simple power-law GP for the DM as well. This is again, tuned to the best version from a lot of testing that I have done. The issue is that frequencies higher than 1/yr in the DM GP are strongly covariant with the cusps of the solar wind. If one includes these frequencies the two models give and take giving funky results, like negative electron densities. There are options to provide various priors and bases, but here is a simple example:
```Python
dm_block = SW.solar_wind_block(Tspan=Tspan, include_dmgp=True)
```

