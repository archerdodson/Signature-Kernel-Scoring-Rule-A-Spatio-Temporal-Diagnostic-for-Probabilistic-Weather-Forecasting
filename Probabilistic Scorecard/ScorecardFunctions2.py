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
#import line_profiler
from datetime import datetime, timedelta
#from multiprocessing import Pool, cpu_count
#import time
from weatherbench2.metrics import MSE, ACC
from weatherbench2.regions import SliceRegion
#import seaborn as sns
from dateutil.relativedelta import relativedelta


def timecuttingmonthly(obsxarray,forsxarray, days, predtimedelta, month,variableval,levelval):
    startdate = datetime(2020,month+1,1)
    newdate = datetime(2020,month+1,days,12)
    obsnewdate = newdate + timedelta(hours = (predtimedelta-1)*12)
    
    if levelval == 0:
        return obsxarray[variableval].sel(time = slice(startdate, obsnewdate)).isel(time=slice(0, None, 2)), forsxarray[variableval].sel(time = slice(startdate, newdate)), 
    else:
        return obsxarray[variableval].sel(level = levelval, time = slice(startdate, obsnewdate)).isel(time=slice(0, None, 2)), forsxarray[variableval].sel(level = levelval, time = slice(startdate, newdate)), 

def scalebyobsadjusted(observations,forecasts,shapeparam):
    mean = np.mean(observations)
    std = np.std(observations)
    scaledobs = (observations - mean)/(std*np.sqrt(shapeparam))
    scaledfors = (forecasts - mean)/(std*np.sqrt(shapeparam))

    return scaledobs, scaledfors

weights = np.array([0.07704437, 0.23039114, 0.38151911, 0.52897285, 0.67133229,
       0.80722643, 0.93534654, 1.05445875, 1.16341595, 1.26116882,
       1.34677594, 1.41941287, 1.47838008, 1.52310968, 1.55317091,
       1.56827425, 1.56827425, 1.55317091, 1.52310968, 1.47838008,
       1.41941287, 1.34677594, 1.26116882, 1.16341595, 1.05445875,
       0.93534654, 0.80722643, 0.67133229, 0.52897285, 0.38151911,
       0.23039114, 0.07704437])

southernweights = weights[0:12]
tropicweights = weights[12:20]
northernweights = weights[20:32]

# def pkparallel_lat_split(lat_chunk, observations_chunk, forecasts_chunk, zero):
#     """
#     Function to compute results for a chunk of latitudes.
#     """
#     static_kernel = sigkernel.RBFKernel(sigma=1)
#     dyadic_order = 1
#     signature_kernel = sigkernel.SigKernel(static_kernel, dyadic_order)

#     time = forecasts_chunk.shape[0] #(time, ens, predelta, long, lat)
#     ens = forecasts_chunk.shape[1]
#     lead = forecasts_chunk.shape[2]
#     latlength = forecasts_chunk.shape[4]

#     pkarray = np.zeros((latlength, time, lead, 1))

#     for lat in range(latlength):
#         for lag in range(2, lead + 1):
#             for t in range(time):
#                 # Forecast path
#                 forpath = torch.tensor(forecasts_chunk[t,:,0:lag, :, lat], dtype=torch.double)
#                  # shape: [ens, lag, dim]

#                 # Add zero basepoint
#                 zeros = torch.zeros(ens, 1, forpath.shape[2], dtype=forpath.dtype)
#                 forpath = torch.cat([zeros, forpath], dim=1)

#                 # Time channel
#                 time_vec = torch.linspace(0, 1, steps=forpath.shape[1], dtype=forpath.dtype)
#                 time_vec = time_vec.view(1, -1, 1)
#                 time_vec = time_vec.expand(forpath.shape[0], -1, 1)
#                 forpath = torch.cat([forpath, time_vec], dim=-1)

#                 # Observation path
#                 obspath = torch.tensor(observations_chunk[t:t+lag, :, lat], dtype=torch.double)
#                 obspath = obspath.unsqueeze(0)

#                 zero_obs = torch.zeros(1, 1, obspath.shape[2], dtype=obspath.dtype)
#                 obspath = torch.cat([zero_obs, obspath], dim=1)

#                 time_vec_obs = torch.linspace(0, 1, steps=obspath.shape[1], dtype=obspath.dtype)
#                 time_vec_obs = time_vec_obs.view(1, -1, 1)
#                 obspath = torch.cat([obspath, time_vec_obs], dim=-1)


#                 # Now compute scoring rule components
#                 Score = signature_kernel.compute_scoring_rule(forpath,obspath)


#                 pkarray[lat, t, lag-1, 0] = Score.item()

#     return pkarray


# def pkparallel(observations, forecasts, zero, region, batch_size=None):
#     """
#     Main function to parallelize computation across latitudes.
#     """
#     latlength = forecasts.shape[4]

#     # Determine the number of processes (default to number of cores)
#     num_cores = cpu_count()
#     print(num_cores)
#     batch_size = batch_size or (latlength // num_cores + (latlength % num_cores > 0))

#     # Split data into chunks by latitude
#     lat_chunks = [
#         range(i, min(i + batch_size, latlength))
#         for i in range(0, latlength, batch_size)
#     ]
#     print(lat_chunks)

#     observations_chunks = [
#         observations[:, :, lat_chunk]
#         for lat_chunk in lat_chunks
#     ]

#     forecasts_chunks = [
#         forecasts[:, :, :, :, lat_chunk]
#         for lat_chunk in lat_chunks
#     ]

#     # Process chunks in parallel
#     with Pool(processes=min(num_cores, len(lat_chunks))) as pool:
#         results = pool.starmap(
#             pkparallel_lat_split,
#             [(lat_chunk, obs_chunk, for_chunk, zero)
#              for lat_chunk, obs_chunk, for_chunk in zip(
#                  lat_chunks, observations_chunks, forecasts_chunks
#              )]
#         )

    
#     if region == 'Tropics':
#         usedweights = tropicweights
#     elif region == 'Northern':
#         usedweights = northernweights
#     else:
#         usedweights = southernweights


#     # Combine results from all chunks
#     #pkarraylat = np.sum(results, axis=0)
#     pkarray = np.concatenate(results, axis=0) #Against lat chunks
#     # pkarraylat = np.sum(pkarray * usedweights[:, None, None, None], axis=0)
#     # pktime = np.mean(pkarraylat, axis=0)
#     # distance = pktime[:, 1] + pktime[:, 2] - 2 * pktime[:, 0]
#     # score = pktime[:, 1] - 2 * pktime[:, 0]

#     return pkarray #, pkarraylat, distance, score

def pkparallel_cuda(observations, forecasts, zero, region):
    """
    Computes PK scoring rule across all latitudes using GPU (CUDA).
    Assumes observations and forecasts are numpy arrays with shapes:
    - observations: [time, features, lat]
    - forecasts: [time, ens, predelta, lon, lat]
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    observations = torch.tensor(observations, dtype=torch.double, device=device)
    forecasts = torch.tensor(forecasts, dtype=torch.double, device=device)

    static_kernel = sigkernel.RBFKernel(sigma=1)
    dyadic_order = 1
    signature_kernel = sigkernel.SigKernel(static_kernel, dyadic_order)

    time, ens, lead, _, latlength = forecasts.shape

    pkarray = torch.zeros((latlength, time, lead), dtype=torch.double, device=device)

    for lat in range(latlength):
        for lag in range(2, lead + 1):
            for t in range(time):
                # Forecast path: shape [ens, lag, dim]
                forpath = forecasts[t, :, 0:lag, :, lat]

                # Add zero basepoint
                zeros = torch.zeros(ens, 1, forpath.shape[2], dtype=forpath.dtype, device=device)
                forpath = torch.cat([zeros, forpath], dim=1)

                # Add time channel
                time_vec = torch.linspace(0, 1, steps=forpath.shape[1], dtype=forpath.dtype, device=device)
                time_vec = time_vec.view(1, -1, 1).expand(forpath.shape[0], -1, 1)
                forpath = torch.cat([forpath, time_vec], dim=-1)

                # Observation path
                obspath = observations[t:t+lag, :, lat].unsqueeze(0)

                zero_obs = torch.zeros(1, 1, obspath.shape[2], dtype=obspath.dtype, device=device)
                obspath = torch.cat([zero_obs, obspath], dim=1)

                time_vec_obs = torch.linspace(0, 1, steps=obspath.shape[1], dtype=obspath.dtype, device=device)
                time_vec_obs = time_vec_obs.view(1, -1, 1)
                obspath = torch.cat([obspath, time_vec_obs], dim=-1)

                # Compute score
                Score = signature_kernel.compute_scoring_rule(forpath, obspath)
                pkarray[lat, t, lag-1] = Score.item()

    pkarray_cpu = pkarray.cpu().numpy()

    # if region == 'Tropics':
    #     usedweights = tropicweights
    # elif region == 'Northern':
    #     usedweights = northernweights
    # else:
    #     usedweights = southernweights

    return pkarray_cpu


def workflowfullparallelmonthly(observations, forecasts, days, lag,zero, month, variableval,levelval, region, switch = False):        
    ob, fo = timecuttingmonthly(observations,forecasts,days,lag+zero,month, variableval,levelval)
    ob = ob.values #(time, long, lat)
    fo = fo[:,:,0:lag,:,:].values #(time, ens, predelta, long, lat)
    if switch:
        fo = fo.swapaxes(1,0)
    ob, fo = scalebyobsadjusted(ob,fo,fo.shape[3])
    pkarray = pkparallel_cuda(ob,fo,zero,region) #, pkarraylat, distance, score

    return (pkarray)