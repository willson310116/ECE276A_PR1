from utils import *
from quaternion import *
from panorama import *

import sys
import time 
import math
import pickle
import numpy as np
import transforms3d
import matplotlib.pyplot as plt
from jax import grad, jit, vmap


# def CostFunction2(q1_T):
#     cost = 0.0
#     for t in range(T - 1):
#         wT = jnp.asarray([newImud[4][t], newImud[5][t], newImud[3][t]])
#         aT = jnp.asarray([0, newImud[0][t], newImud[1][t], newImud[2][t]])
#         deltaT = imud["ts"][0][t] - imud["ts"][0][t - 1]
#         q_exp = jnp.asarray([0, (wT[0] * deltaT) / 2, (wT[1] * deltaT) / 2, (wT[2] * deltaT) / 2])
#         g = jnp.asarray([0, 0, 0, -9.8])

#         cost += QuaternionMagnitude(2 * QuaternionLog(QuaternionMultiply(QuaternionInverse(q1_T[t + 1]), \
#         QuaternionMultiply(q1_T[t], QuaternionExp(q_exp))))) + \
#         QuaternionMagnitude(aT - jnp.asarray(QuaternionMultiply(QuaternionMultiply(QuaternionInverse(q1_T[t + 1]), g), q1_T[t + 1])))

#     return 0.5 * cost



problem = str(sys.argv[1])
dataset = str(sys.argv[2])
dataver = str(sys.argv[3])

if problem == "2":
    cfile = f"../data/{dataver}set/cam/cam{dataset}.p"
    vfile = f"../data/{dataver}set/vicon/viconRot{dataset}.p"
    camd = read_data(cfile) # dict: 'cam' : ndarray(240, 320, 3, n), 'ts' : ndarray(1, n)
    vicd = read_data(vfile) # dict: 'rots' : ndarray(3, 3, n), 'ts' : ndarray(1, n)

    save_path = f"../fig/Prob2_{dataset}.png"
    plotPanorama(vicd, camd, save_path)


elif problem == "1":
    save_path = f'../fig/Prob1_{dataset}.png'

    if dataver == "train":
        ifile = f"../data/{dataver}set/imu/imuRaw{dataset}.p"
        vfile = f"../data/{dataver}set/vicon/viconRot{dataset}.p"
        imud = read_data(ifile) # dict: 'vals' : ndarray(6, n), 'ts' : ndarray(1, n)
        vicd = read_data(vfile) # dict: 'rots' : ndarray(3, 3, n), 'ts' : ndarray(1, n)
        newImud = CalibrateIMU(imud, 100)
        PlotGraphsTrain(vicd, newImud, imud, save_path)

    elif dataver == "test":
        ifile = f"../data/{dataver}set/imu/imuRaw{dataset}.p"
        imud = read_data(ifile) # dict: 'vals' : ndarray(6, n), 'ts' : ndarray(1, n)
        newImud = CalibrateIMU(imud, 100)
        PlotGraphTest(newImud, imud, save_path)


