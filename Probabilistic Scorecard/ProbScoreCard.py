#### ScoreCard functions

import os
import weatherbench2
import xarray as xr
import math
from weatherbench2.regions import SliceRegion, ExtraTropicalRegion
from weatherbench2.evaluation import evaluate_in_memory
from weatherbench2 import config
import numpy as np
import sigkernel
import torch
#from einops import rearrange
#from itertools import product
import cython
#import matplotlib.pyplot  as plt
#import tqdm
#import Functions as fu
#import line_profiler
from datetime import datetime, timedelta
#from multiprocessing import Pool, cpu_count
import time
from weatherbench2.metrics import MSE, ACC
from weatherbench2.regions import SliceRegion
#import seaborn as sns
from dateutil.relativedelta import relativedelta
import ScorecardFunctions2 as SCF
import gcsfs


import apache_beam

print("initial")

obs_path = 'gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr'
observationsera = xr.open_zarr(obs_path)
ifsens = xr.open_zarr('gs://weatherbench2/datasets/ifs_ens/2018-2022-64x32_equiangular_conservative.zarr')
print("first")
gcmens = xr.open_zarr('gs://weatherbench2/datasets/neuralgcm_ens/2020-64x32_equiangular_conservative.zarr')

print("loaded")
ifsens_12h = ifsens.isel(prediction_timedelta=slice(0, None, 2))
gcmens_cut = gcmens.isel(prediction_timedelta=slice(0, -1))

#print("yo")
#print("begun")

geo850 = np.zeros((32,6,31))
for monthval in range(12):
    geo850 = geo850 + SCF.workflowfullparallelmonthly(observationsera, ifsens_12h, 3, 31,0, month=monthval, variableval='geopotential',levelval= 850, region='no')
np.save('Signature/geo850.npy', geo850)

geo500 = np.zeros((32,6,31))
for monthval in range(12):
    geo500 = geo500 + SCF.workflowfullparallelmonthly(observationsera, ifsens_12h, 3, 31,0, month=monthval, variableval='geopotential',levelval= 500, region='no')
np.save('Signature/geo500.npy', geo500)

temp850 = np.zeros((32,6,31))
for monthval in range(12):
    temp850 = temp850 + SCF.workflowfullparallelmonthly(observationsera, ifsens_12h, 3, 31,0, month=monthval, variableval='temperature',levelval= 850, region='no')
np.save('Signature/temp850.npy', temp850)

temp500 = np.zeros((32,6,31))
for monthval in range(12):
    temp500 = temp500 + SCF.workflowfullparallelmonthly(observationsera, ifsens_12h, 3, 31,0, month=monthval, variableval='temperature',levelval= 500, region='no')
np.save('Signature/temp500.npy', temp500)

u500 = np.zeros((32,6,31))
for monthval in range(12):
    u500 = u500 + SCF.workflowfullparallelmonthly(observationsera, ifsens_12h, 3, 31,0, month=monthval, variableval='u_component_of_wind',levelval= 500, region='no')
np.save('Signature/u500.npy', u500)

v500 = np.zeros((32,6,31))
for monthval in range(12):
    v500 = v500 + SCF.workflowfullparallelmonthly(observationsera, ifsens_12h, 3, 31,0, month=monthval, variableval='v_component_of_wind',levelval= 500, region='no')
np.save('Signature/v500.npy', v500)


#ifs ens done
################################

geo850 = np.zeros((32,6,31))
for monthval in range(12):
    geo850 = geo850 + SCF.workflowfullparallelmonthly(observationsera, gcmens_cut, 3, 31,0, month=monthval, variableval='geopotential',levelval= 850, region='no', switch = True)
np.save('Signature/geo850gcm.npy', geo850)

geo500 = np.zeros((32,6,31))
for monthval in range(12):
    geo500 = geo500 + SCF.workflowfullparallelmonthly(observationsera, gcmens_cut, 3, 31,0, month=monthval, variableval='geopotential',levelval= 500, region='no', switch = True)
np.save('Signature/geo500gcm.npy', geo500)

temp850 = np.zeros((32,6,31))
for monthval in range(12):
    temp850 = temp850 + SCF.workflowfullparallelmonthly(observationsera, gcmens_cut, 3, 31,0, month=monthval, variableval='temperature',levelval= 850, region='no', switch = True)
np.save('Signature/temp850gcm.npy', temp850)

temp500 = np.zeros((32,6,31))
for monthval in range(12):
    temp500 = temp500 + SCF.workflowfullparallelmonthly(observationsera, gcmens_cut, 3, 31,0, month=monthval, variableval='temperature',levelval= 500, region='no', switch = True)
np.save('Signature/temp500gcm.npy', temp500)

u500 = np.zeros((32,6,31))
for monthval in range(12):
    u500 = u500 + SCF.workflowfullparallelmonthly(observationsera, gcmens_cut, 3, 31,0, month=monthval, variableval='u_component_of_wind',levelval= 500, region='no', switch = True)
np.save('Signature/u500gcm.npy', u500)

v500 = np.zeros((32,6,31))
for monthval in range(12):
    v500 = v500 + SCF.workflowfullparallelmonthly(observationsera, gcmens_cut, 3, 31,0, month=monthval, variableval='v_component_of_wind',levelval= 500, region='no', switch = True)
np.save('Signature/v500gcm.npy', v500)





#result = SCF.workflowfullparallelmonthly(observationsera, ifsens_12h, 3, 31,0, month=0, variableval='geopotential',levelval= 850, region='no')



