import data_class
#from enterprise_extensions.pipe import data_class
import h5py
import numpy as np
import pandas as pd

params = data_class.RunSettings()
#params.update_from_file('m2a.ini')
params.update_from_file('m2a_CW.ini')

pta = params.create_pta_object_from_signals()

print(pta)


print("done")