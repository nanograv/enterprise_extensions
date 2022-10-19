import data_class
import h5py
import numpy as np
import pandas as pd

params = data_class.RunSettings()
params.update_from_file('example.ini')

pta = params.create_pta_object_from_signals()

print(pta)

# load polinas data
f = h5py.File("/data/sophie.hourihane/PTA/15yr/m2a_cw/15yr_detect_RNedv2_1e9_full_10chain.h5")
samples = f['samples_cold'][()][0]
pars = f['par_names'][()]
pars = [par.decode('UTF-8') for par in pars] # convert byte strings to strings
# change names
index = np.where(np.array(pars) == 'gwb_log10_A')[0][0]
pars[index] = 'gw_log10_A'
index = np.where(np.array(pars) == 'gwb_gamma')[0][0]
pars[index] = 'gw_gamma'

df = pd.DataFrame(samples, columns=pars)
row_dict = df[df.index == 10].to_dict('index')[10]

pta.get_lnlikelihood(row_dict)
pta.get_lnprior(row_dict)


print("done")

#pta = params.create_pta_object()

#print(pta)
#params.load_pickled_pulsars()
#pta = params.create_pta_object()
#print(pta)