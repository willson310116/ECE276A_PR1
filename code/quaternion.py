import math
import numpy as np
import transforms3d
import matplotlib.pyplot as plt 
import jax.numpy as jnp

def QuaternionMagnitude(q):
    q_list = ([0] * len(q))

    for i in range(len(q)):
        q_list[i] = q[i] ** 2

    q_mag = np.sum(q_list)
    return q_mag

def QuaternionNorm(q):
    return math.sqrt(QuaternionMagnitude(q))

def QuaternionConjugate(q):
    q_conj = [0, 0, 0, 0]
    sign = [1, -1, -1, -1]

    for i in range(len(q_conj)):
        q_conj[i] = q[i] * sign[i]

    return q_conj

def QuaternionMultiply(q1, q2):
    q3 = [0, 0, 0, 0]
    q3[0] = (q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3])
    q3[1] = (q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2])
    q3[2] = (q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1])
    q3[3] = (q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0])
    return q3

def QuaternionLog(q):
    q_list = [0, 0, 0, 0]
    q_norm = QuaternionNorm(q)
    qv_norm = QuaternionNorm(q[1:4])

    if(q_norm == 0):
        q_list[0] = 0
    else:
        q_list[0] = math.log(q_norm)

    if(q[1] == 0):
        q_list[1] = 0
    else:
        q_list[1] = (q[1] / qv_norm) * math.acos(q[0] / q_norm)
        
    if(q[2] == 0):
        q_list[2] = 0
    else:
        q_list[2] = (q[2] / qv_norm) * math.acos(q[0] / q_norm)
        
    if(q[3] == 0):
        q_list[3] = 0
    else:
        q_list[3] = (q[3] / qv_norm) * math.acos(q[0] / q_norm)

    return q_list

def QuaternionExp(q):
    q_list = [0, 0, 0, 0]
    qv_norm = QuaternionNorm(q[1:4])
    exponentValue = math.exp(q[0])
    
    q_list[0] = exponentValue * math.cos(qv_norm)
    
    for i in range(1, 4):
        if q[i] == 0:
            q_list[i] = 0
        else:
            q_list[i] = exponentValue * ((q[i] / qv_norm) * math.sin(qv_norm))

    return q_list

def QuaternionRotation(R, q):
    temp = QuaternionMultiply(R, q)
    rotatedQuaternion = QuaternionMultiply(temp, QuaternionConjugate(R)) 
    return rotatedQuaternion

def QuaternionInverse(q):
    q_list = [0,0,0,0]
    q_conj = QuaternionConjugate(q)
    q_norm_square = QuaternionNorm(q) ** 2

    for i in range(len(q_list)):
        q_list[i] = q_conj[i] / q_norm_square
    
    return q_list
    
def CalibrateIMU(imud, biasRange):
    biasWX = biasWY = biasWZ = biasACCX = biasACCY = biasACCZ = 0
    # biasRange = 100
    
    for i in range(0, biasRange):
        biasWX += imud["vals"][0][i]
        biasWY += imud["vals"][1][i]
        biasWZ += imud["vals"][2][i]
        biasACCX += imud["vals"][4][i]
        biasACCY += imud["vals"][5][i]
        biasACCZ += (imud["vals"][3][i])

    biasWX = biasWX / biasRange
    biasWY = biasWY / biasRange
    biasWZ = biasWZ / biasRange
    biasACCX = biasACCX / biasRange
    biasACCY = biasACCY / biasRange
    biasACCZ = biasACCZ / biasRange

    newImud = np.empty((6, len(imud["ts"][0])))
    for i in range(0, len(imud["ts"][0])):
        newImud[0][i] = -(imud["vals"][0][i] - biasWX) * (3300 / (1023 * 300)) * (np.pi / 180.0)
        newImud[1][i] = -(imud["vals"][1][i] - biasWY) * (3300 / (1023 * 300)) * (np.pi / 180.0)
        newImud[2][i] = (imud["vals"][2][i] - biasWZ) * (3300 / (1023 * 300)) * (np.pi / 180.0)
        newImud[4][i] = (imud["vals"][4][i] - biasACCX) * (3300 / (1023 * 3.33)) * (np.pi / 180.0)
        newImud[5][i] = (imud["vals"][5][i] - biasACCY) * (3300 / (1023 * 3.33)) * (np.pi / 180.0)
        newImud[3][i] = (imud["vals"][3][i] - biasACCZ) * (3300 / (1023 * 3.33)) * (np.pi / 180.0)
    
    return newImud
    
def checkRange(val):
    return True if abs(val) > 180 else False

def PlotGraphsTrain(vicd, newImud, imud, file_name):
    vicdRoll = np.empty(len(vicd["ts"][0]))
    vicdPitch = np.empty(len(vicd["ts"][0]))
    vicdYaw = np.empty(len(vicd["ts"][0]))
    
    imudRoll = np.empty(len(vicd["ts"][0]))
    imudPitch = np.empty(len(vicd["ts"][0]))
    imudYaw = np.empty(len(vicd["ts"][0]))
    
    # q_0
    q_cur = [1, 0, 0, 0]
    n = len(vicd["ts"][0]) - 1
    
    for i in range(1, len(vicd["ts"][0])):
        if i >= newImud.shape[1]:
            n = i
            break
    # for i in range(1, newImud.shape[1]):
        # get R from vicd[i]
        # X1,Y1,Z1,X2,Y2,Z2,X3,Y3,Z3
        

        rotMatrix = np.empty((3, 3))
        for j in range(0, 3):
            for k in range(0, 3):
                rotMatrix[j][k] = float(vicd["rots"][j][k][i])

        eulerAngles = transforms3d.euler.mat2euler(rotMatrix)
        
        vicdRoll[i] = eulerAngles[0] * 180 / np.pi
        vicdPitch[i] = eulerAngles[1] * 180 / np.pi
        vicdYaw[i] = eulerAngles[2] * 180 / np.pi

        # angular velocity
        wT = (newImud[4][i], newImud[5][i], newImud[3][i])
        deltaT = imud["ts"][0][i] - imud["ts"][0][i - 1]
        
        # quaternion kinematics motion model
        q_exp = [0, (wT[0] * deltaT) / 2, (wT[1] * deltaT) / 2, (wT[2] * deltaT) / 2]
        temp = QuaternionMultiply(q_cur, QuaternionExp(q_exp))
        eulerAngles = transforms3d.euler.quat2euler(temp)
        
        imudRoll[i] = eulerAngles[0] * 180 / np.pi
        imudPitch[i] = eulerAngles[1] * 180 / np.pi
        imudYaw[i] = eulerAngles[2] * 180 / np.pi
        
        # q_t
        q_cur = temp
    
    ticks = vicd["ts"][0]
    # if checkRange(vicdRoll[0]) or checkRange(vicdPitch[0]) or checkRange(vicdYaw[0]) or \
    #     checkRange(imudRoll[0]) or checkRange(imudPitch[0]) or checkRange(imudYaw[0]):
    #     vicdRoll = vicdRoll[1:]
    #     vicdPitch = vicdPitch[1:]
    #     vicdYaw = vicdYaw[1:]
    #     imudRoll = imudRoll[1:]
    #     imudPitch = imudPitch[1:]
    #     imudYaw = imudYaw[1:]
    #     ticks = ticks[1:]

    vicdRoll = vicdRoll[1:n]
    vicdPitch = vicdPitch[1:n]
    vicdYaw = vicdYaw[1:n]
    imudRoll = imudRoll[1:n]
    imudPitch = imudPitch[1:n]
    imudYaw = imudYaw[1:n]
    ticks = ticks[1:n]
        

    # create figure
    fig = plt.figure(figsize=(10, 8))
    imgRows = 3
    imgCols = 1
    labels = ["IMU", "Ground truth"]

    fig.add_subplot(imgRows, imgCols, 1)
    plt.plot(ticks, imudRoll, color = 'red')
    plt.plot(ticks, vicdRoll, color = 'blue')
    plt.title("Roll angle")
    plt.legend(labels)
    
    fig.add_subplot(imgRows, imgCols, 2)
    plt.plot(ticks, imudPitch, color = 'red')
    plt.plot(ticks, vicdPitch, color = 'blue')
    plt.title("Pitch angle")
    plt.legend(labels)

    fig.add_subplot(imgRows, imgCols, 3)
    plt.plot(ticks, imudYaw, color = 'red')
    plt.plot(ticks, vicdYaw, color = 'blue')
    plt.title("Yaw angle")
    plt.legend(labels)
    # plt.show()
    plt.tight_layout()
    plt.savefig(file_name)


def PlotGraphTest(newImud, imud, file_name):    
    imudRoll = np.empty(len(imud["ts"][0]))
    imudPitch = np.empty(len(imud["ts"][0]))
    imudYaw = np.empty(len(imud["ts"][0]))
    
    # q_0
    q_cur = [1, 0, 0, 0]
    
    for i in range(1, len(imud["ts"][0])):
        # angular velocity
        wT = (newImud[4][i], newImud[5][i], newImud[3][i])
        deltaT = imud["ts"][0][i] - imud["ts"][0][i - 1]
        
        # quaternion kinematics motion model
        q_exp = [0, (wT[0] * deltaT) / 2, (wT[1] * deltaT) / 2, (wT[2] * deltaT) / 2]
        temp = QuaternionMultiply(q_cur, QuaternionExp(q_exp))
        eulerAngles = transforms3d.euler.quat2euler(temp)
        
        imudRoll[i] = eulerAngles[0] * 180 / np.pi
        imudPitch[i] = eulerAngles[1] * 180 / np.pi
        imudYaw[i] = eulerAngles[2] * 180 / np.pi
        
        # q_t
        q_cur = temp
    
    ticks = imud["ts"][0]

    imudRoll = imudRoll[1:]
    imudPitch = imudPitch[1:]
    imudYaw = imudYaw[1:]
    ticks = ticks[1:]
        

    # create figure
    fig = plt.figure(figsize=(10, 8))
    imgRows = 3
    imgCols = 1
    labels = ["IMU"]

    fig.add_subplot(imgRows, imgCols, 1)
    plt.plot(ticks, imudRoll, color = 'red')
    plt.title("Roll angle")
    plt.legend(labels)
    
    fig.add_subplot(imgRows, imgCols, 2)
    plt.plot(ticks, imudPitch, color = 'red')
    plt.title("Pitch angle")
    plt.legend(labels)

    fig.add_subplot(imgRows, imgCols, 3)
    plt.plot(ticks, imudYaw, color = 'red')
    plt.title("Yaw angle")
    plt.legend(labels)
    # plt.show()
    plt.tight_layout()
    plt.savefig(file_name)


def CostFunction(newImud, imud, q1_T, T):
    cost = 0
    for t in range(T - 1):
        wT = (newImud[4][t], newImud[5][t], newImud[3][t])
        aT = [newImud[0][t], newImud[1][t], newImud[2][t]]
        deltaT = imud["ts"][0][t] - imud["ts"][0][t - 1]
        q_exp = [0, (wT[0] * deltaT) / 2, (wT[1] * deltaT) / 2, (wT[2] * deltaT) / 2]
        g = [0, 0, 0, -9.8]

        cost += QuaternionMagnitude(2 * QuaternionLog(QuaternionMultiply(QuaternionInverse(q1_T[t + 1]), \
        QuaternionMultiply(q1_T[t], QuaternionExp(q_exp))))) + \
        QuaternionMagnitude(aT - QuaternionMultiply(QuaternionMultiply(QuaternionInverse(q1_T[t + 1]), g), q1_T[t + 1]))

    return 0.5 * cost


def CostFunction(newImud, imud, q1_T, T):
    cost = 0.0
    for t in range(T - 1):
        wT = jnp.asarray([newImud[4][t], newImud[5][t], newImud[3][t]])
        aT = jnp.asarray([0, newImud[0][t], newImud[1][t], newImud[2][t]])
        deltaT = imud["ts"][0][t] - imud["ts"][0][t - 1]
        q_exp = jnp.asarray([0, (wT[0] * deltaT) / 2, (wT[1] * deltaT) / 2, (wT[2] * deltaT) / 2])
        g = jnp.asarray([0, 0, 0, -9.8])

        q1_T = jnp.asarray(q1_T)

        cost += QuaternionMagnitude(2 * QuaternionLog(QuaternionMultiply(QuaternionInverse(q1_T[t + 1]), \
        QuaternionMultiply(q1_T[t], QuaternionExp(q_exp))))) + \
        QuaternionMagnitude(aT - jnp.asarray(QuaternionMultiply(QuaternionMultiply(QuaternionInverse(q1_T[t + 1]), g), q1_T[t + 1])))

    return 0.5 * cost