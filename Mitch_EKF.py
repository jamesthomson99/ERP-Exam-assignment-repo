import numpy as np
from numpy.core.numeric import identity
import matplotlib.pyplot as plt
import time

# research model = Extended KALMAN filter
import csv

accelData = np.load(r'Data\accelData.npy')
accelStationary = np.load(r'Data\accelStationary.npy')
GT = np.load(r'Data\GT.npy')
gyroData = np.load(r'Data\gyroData.npy')
gyroStationary = np.load(r'Data\gyroStationary.npy')
magData = np.load(r'Data\magData.npy')
magStationary = np.load(r'Data\magStationary.npy')

# make some changes to input data
for i in range(100):
    for j in range(4500):
        accelData[i][j][2] = - accelData[i][j][2]
        magData[i][j][1] = - magData[i][j][1]

#roll, pitch and yaw ground truths
totalRollExpected = []
totalPitchExpected = []
totalYawExpected = []
for i in range(4500):
    totalRollExpected.append(GT[i][0])
    totalPitchExpected.append(GT[i][1])
    totalYawExpected.append(GT[i][2])
# totalRollExpected now 4500 long...
# read in all data

totalRollMonte = []
totalPitchMonte = []
totalYawMonte = []

for monteRun in range(100):

    print(monteRun)

    #################################################################

    # Kalman Filter variables
    g = 9.81
    initialGyroBias = np.array([0, 0, 0])  # calibrated?
    quaternionEstimate = np.array([1, 0, 0, 0])
    xPredict = np.concatenate((quaternionEstimate, initialGyroBias)).transpose()
    xPredictPrevious = None # to be initialized later
    xPredictNext = None # to be initialized later
    yPredict = None # to be initialized later
    AMatrix = None  # to be initialized later
    BMatrix = None  # to be initialized later
    CMatrix = None  # to be initialized later
    KMatrix = None  # to be initialized later
    priorError = None
    priorErrorPrevious = np.identity(7) * 0.01 # initial estimate of error
    QMatrix = np.identity(7) * 0.001 # initial estimate of process variance
    RMatrix = np.identity(6) * 0.1 # initial estimate of variance
    accelerationReferenceFrame = np.array([0, 0, -1]).transpose()
    magnetReferenceFrame = np.array([0, 1, 0]).transpose()
    I3x3 = np.identity(3)
    I4x4 = np.identity(4)
    zero3x3 = np.zeros((3, 3))
    zero3x4 = np.zeros((3, 4))
    mag_Ainv = np.array([[ 2.06423128e-03, -1.04778851e-04, -1.09416190e-06],
                                    [-1.04778851e-04,  1.91693168e-03,  1.79409312e-05],
                                    [-1.09416190e-06,  1.79409312e-05,  1.99819154e-03]])
    mag_b = np.array([0,0,0]).transpose()

    def generate_ticks(min_num, max_num):
        ticks_return = []
        for j in range(-360, 360, 30):
            if min_num - 10 < j < max_num + 10:
                ticks_return.append(j)
        return ticks_return

    def getMagVector(m):
        magGaussRaw = np.matmul(mag_Ainv, np.array(m).transpose() - mag_b)
        magGauss_N = np.matmul(getRotationMatrix(xPredict), magGaussRaw)
        magGauss_N[2] = 0
        magGauss_N = magGauss_N / (magGauss_N[0] ** 2 + magGauss_N[1] ** 2) ** 0.5
        magGuass_B = np.matmul(getRotationMatrix(xPredict).transpose(), magGauss_N)
        return magGuass_B


    def normalizeQuaternion(quaternion):
        magnitude = (quaternion[0]**2 + quaternion[1]**2 +
                    quaternion[2]**2 + quaternion[3]**2)**0.5
        return quaternion / magnitude

    def normalizeAccelerometerVector(acc):
        accVector = np.array(acc).transpose()
        magnitude = (accVector[0] **2 + accVector[1] **2 + accVector[2] **2)**0.5
        return accVector/magnitude

    def degreesToRad(deg):
        return deg * (np.pi/180)


    def convertRadToDegrees(rad):
        return rad * (180 / np.pi)

    def getRotationMatrix(unRotatedQuart):
        matrix00 = unRotatedQuart[0] ** 2 + unRotatedQuart[1] ** 2 - unRotatedQuart[2] ** 2 - unRotatedQuart[3] ** 2
        matrix01 = 2 * (unRotatedQuart[1] * unRotatedQuart[2] - unRotatedQuart[0] * unRotatedQuart[3])
        matrix02 = 2 * (unRotatedQuart[1] * unRotatedQuart[3] + unRotatedQuart[0] * unRotatedQuart[2])
        matrix10 = 2 * (unRotatedQuart[1] * unRotatedQuart[2] + unRotatedQuart[0] * unRotatedQuart[3])
        matrix11 = unRotatedQuart[0] ** 2 - unRotatedQuart[1] ** 2 + unRotatedQuart[2] ** 2 - unRotatedQuart[3] ** 2
        matrix12 = 2 * (unRotatedQuart[2] * unRotatedQuart[3] - unRotatedQuart[0] * unRotatedQuart[1])
        matrix20 = 2 * (unRotatedQuart[1] * unRotatedQuart[3] - unRotatedQuart[0] * unRotatedQuart[2])
        matrix21 = 2 * (unRotatedQuart[2] * unRotatedQuart[3] + unRotatedQuart[0] * unRotatedQuart[1])
        matrix22 = unRotatedQuart[0] ** 2 - unRotatedQuart[1] ** 2 - unRotatedQuart[2] ** 2 + unRotatedQuart[3] ** 2
        rotatedMatrix = np.array([[matrix00, matrix01, matrix02], [matrix10, matrix11, matrix12], [matrix20, matrix21, matrix22]])
        return rotatedMatrix

    def quaternionToEulerAngles(quat):
        rotatedMatrix = getRotationMatrix(quat)
        # check for gimbal lock discontinuities
        beta = np.arcsin(-rotatedMatrix[2][0])
        pitch = np.arcsin(-rotatedMatrix[2][0])
        # answers in rad need to convert to deg
        if(np.cos(beta) != 0):
            yaw = np.arctan2(rotatedMatrix[1][0] , rotatedMatrix[0][0])
            roll = np.arctan2(rotatedMatrix[2][1] , rotatedMatrix[2][2])
        elif(pitch == np.pi/2):
            yaw = 0
            roll = np.arctan2(rotatedMatrix[0][1] , rotatedMatrix[0][2])
        elif(pitch == -np.pi/2):
            yaw = 0
            roll = np.arctan2(-rotatedMatrix[0][1] , -rotatedMatrix[0][2])
        return convertRadToDegrees(yaw), convertRadToDegrees(pitch), convertRadToDegrees(roll)

    def jMatrix(referenceFrame):
        # must change!
        qHatPrev = xPredictPrevious[0:4]
        e00 = qHatPrev[0] * referenceFrame[0] + qHatPrev[3] * referenceFrame[1] - qHatPrev[2] * referenceFrame[2]
        e01 = qHatPrev[1] * referenceFrame[0] + qHatPrev[2] * referenceFrame[1] + qHatPrev[3] * referenceFrame[2]
        e02 = -qHatPrev[2] * referenceFrame[0] + qHatPrev[1] * referenceFrame[1] - qHatPrev[0] * referenceFrame[2]
        e03 = -qHatPrev[3] * referenceFrame[0] + qHatPrev[0] * referenceFrame[1] + qHatPrev[1] * referenceFrame[2]
        e10 = -qHatPrev[3] * referenceFrame[0] + qHatPrev[0] * referenceFrame[1] + qHatPrev[1] * referenceFrame[2]
        e11 = qHatPrev[2] * referenceFrame[0] - qHatPrev[1] * referenceFrame[1] + qHatPrev[0] * referenceFrame[2]
        e12 = qHatPrev[1] * referenceFrame[0] + qHatPrev[2] * referenceFrame[1] + qHatPrev[3] * referenceFrame[2]
        e13 = -qHatPrev[0] * referenceFrame[0] - qHatPrev[3] * referenceFrame[1] + qHatPrev[2] * referenceFrame[2]
        e20 = qHatPrev[2] * referenceFrame[0] - qHatPrev[1] * referenceFrame[1] + qHatPrev[0] * referenceFrame[2]
        e21 = qHatPrev[3] * referenceFrame[0] - qHatPrev[0] * referenceFrame[1] - qHatPrev[1] * referenceFrame[2]
        e22 = qHatPrev[0] * referenceFrame[0] + qHatPrev[3] * referenceFrame[1] - qHatPrev[2] * referenceFrame[2]
        e23 = qHatPrev[1] * referenceFrame[0] + qHatPrev[2] * referenceFrame[1] + qHatPrev[3] * referenceFrame[2]
        jacobianMatrix = 2 * np.array([[e00, e01, e02, e03],[e10, e11, e12, e13],[e20, e21, e22, e23]])
        return jacobianMatrix

    def accelerometerPredict():
        global CMatrix
        h_a = jMatrix(accelerationReferenceFrame)
        accelerationPredict = np.matmul(getRotationMatrix(xPredictNext).transpose(), accelerationReferenceFrame)
        h_m = jMatrix(magnetReferenceFrame)
        magnetPredict = np.matmul(getRotationMatrix(xPredictNext).transpose(), magnetReferenceFrame)
        CMatrix = np.concatenate((np.concatenate((h_a,zero3x3),axis=1),np.concatenate((h_m,zero3x3),axis=1)))
        return np.concatenate((accelerationPredict,magnetPredict))

    #Kalman Predict and Update methods
    ###################################################################

    def kalmanPredict(delta, angularVelocityVector):
        global AMatrix, BMatrix, xPredictPrevious, xPredict, xPredictNext, yPredict, priorError,priorErrorPrevious, QMatrix
        quaternionCurrent = xPredict[0:4]
        SqRotationMatrix = np.array([[-quaternionCurrent[1], -quaternionCurrent[2], -quaternionCurrent[3]],
                                    [quaternionCurrent[0], -quaternionCurrent[3],  quaternionCurrent[2]],
                                    [quaternionCurrent[3],  quaternionCurrent[0], -quaternionCurrent[1]],
                                    [-quaternionCurrent[2],  quaternionCurrent[1],  quaternionCurrent[0]]])
        AMatrix = np.concatenate((np.concatenate((I4x4 , (-delta/2)*SqRotationMatrix) , axis=1), np.concatenate((zero3x4 , I3x3) , axis=1)) , axis=0)
        BMatrix = np.concatenate((np.array(delta/2*SqRotationMatrix), zero3x3))
        xPredictNext = np.matmul(AMatrix,xPredict) + np.matmul(BMatrix , np.array(angularVelocityVector).transpose())
        xPredictNext[0:4] = normalizeQuaternion(xPredict[0:4])
        xPredictPrevious = xPredict
        yPredict = accelerometerPredict()
        priorError = np.matmul(np.matmul(AMatrix,priorErrorPrevious) , AMatrix.transpose()) + QMatrix
        
    def kalmanUpdate(accelerometerData, magnetometerData):
        global xPredict, xPredictPrevious, CMatrix, yPredict, priorErrorPrevious, priorError
        kNumerator = np.matmul(priorError, CMatrix.transpose())
        kDenominator = np.linalg.inv(np.matmul(np.matmul(CMatrix, priorError), CMatrix.transpose()) + RMatrix)
        KMatrix = np.matmul(kNumerator, kDenominator)
        normAccVect = normalizeAccelerometerVector(accelerometerData)
        magGuass_B = getMagVector(magnetometerData)
        # don't calibrate magnetometer as received sanitized inputs already?
        sensorValues = np.concatenate((normAccVect , magGuass_B))
        xPredict = xPredictNext + np.matmul(KMatrix, sensorValues -yPredict)
        xPredict[0:4] = normalizeQuaternion(xPredict[0:4])
        I7x7 = np.identity(7)
        priorErrorPrevious = np.matmul(I7x7 - np.matmul(KMatrix, CMatrix) , priorError)

    # Run Kalman code on generated sensor measurements

    rollOutput = []
    pitchOutput = []
    yawOutput = []

    # do the predict, then update, then do getEulerAngles
    for i in range(len(accelData[monteRun])):    
         # move below line to above update method (not adding bad input)
        kalmanPredict(1/120, gyroData[monteRun][i])
        pErr = kalmanUpdate(accelData[monteRun][i], magData[monteRun][i])
        tempYaw, tempPitch, tempRoll = quaternionToEulerAngles(xPredict[0:4])
        rollOutput.append(-tempRoll)
        pitchOutput.append(-tempPitch)
        yawOutput.append(-tempYaw+92)

    totalRollMonte.append(rollOutput)
    totalPitchMonte.append(pitchOutput)
    totalYawMonte.append(yawOutput)

    # # plot roll, pitch and yaw 
    #  # max_angle = max(max(pitchOutput), max(rollOutput), max(yawOutput))
    # # min_angle = min(min(pitchOutput), min(rollOutput), min(yawOutput))
    # max_angle = 90
    # min_angle = -90

    # # plot roll, pitch and yaw 
    # timeStops = np.arange(0, len(rollOutput)*(1/120) , (1/120))
    # fig, axs = plt.subplots(3)
    # fig.suptitle("Extended Kalman Filter Orientation Estimation")
    # # for i in range(len(totalRollExpected)):
    # #         totalRollExpected[i] = float(totalRollExpected[i])*-53
    # #         totalPitchExpected[i] = float(totalPitchExpected[i])*53
    # #         totalYawExpected[i] = float(totalYawExpected[i])*53
    # axs[0].plot(timeStops, rollOutput, 'tab:orange', label="Roll")
    # axs[0].plot(timeStops, totalRollExpected, 'tab:orange', label="Roll Ground Truth", linestyle="dashed")
    # axs[1].plot(timeStops, pitchOutput, 'tab:green', label="Pitch")
    # axs[1].plot(timeStops, totalPitchExpected, 'tab:green', label="Pitch Ground Truth", linestyle="dashed")
    # axs[2].plot(timeStops, yawOutput, 'tab:blue', label="Yaw")
    # axs[2].plot(timeStops, totalYawExpected, 'tab:blue', label="Yaw Ground Truth", linestyle="dashed")
    # for ax in axs.flat:
    #     ax.set(ylabel="Angle (deg)")
    #     ax.set(xlabel="Time (s)")
    #     # ax.set_xlim(0, len(acc_x)*delta)
    #     ax.set_ylim(-90,90)
    #     ax.set_yticks(generate_ticks(min_angle, max_angle))
    #     ax.grid()
    #     ax.legend()
    # plt.show()


mseRoll = []
msePitch = []
mseYaw = []

for i in range(len(totalRollMonte)): # for each monte carlo run values
    for j in range(len(totalRollMonte[i])): # for each timestep in run
        if(i==0): # first run
            mseRoll.append(np.square(float(totalRollExpected[j])-totalRollMonte[i][j]))
            msePitch.append(np.square(float(totalPitchExpected[j])-totalPitchMonte[i][j]))
            mseYaw.append(np.square(float(totalYawExpected[j])-totalYawMonte[i][j]))
        else:
            mseRoll[j] += np.square(float(totalRollExpected[j])-totalRollMonte[i][j])
            msePitch[j] += np.square(float(totalPitchExpected[j])-totalPitchMonte[i][j])
            mseYaw[j] += np.square(float(totalYawExpected[j])-totalYawMonte[i][j])
for i in range(len(mseRoll)):
    mseRoll[i] = np.sqrt(mseRoll[i]/100)
    msePitch[i] = np.sqrt(msePitch[i]/100)
    mseYaw[i] = np.sqrt(mseYaw[i]/100)
# mseRoll etc. now has rmse for each timestep

timeStops = range(0,len(mseRoll))
fig, axs = plt.subplots(3)
fig.suptitle("Root-mean-square error of EKF for 100 Monte Carlo Runs")
axs[0].plot(timeStops, msePitch, 'crimson', label="Pitch RMSE")
axs[1].plot(timeStops, mseRoll, 'deepskyblue', label="Roll RMSE")
axs[2].plot(timeStops, mseYaw, 'lawngreen', label="Yaw RMSE")
for ax in axs.flat:
    ax.set(ylabel="Angle (°)")
    ax.set(xlabel="Samples")
    # ax.set_ylim(0,0.5)
    ax.grid(linestyle='--')
    ax.legend(loc="upper right")
plt.show()

timeStops = range(0,len(totalRollMonte[0]))
fig, axs = plt.subplots(3)
fig.suptitle("EKF orientation estimate for 1st set of data")
axs[0].plot(timeStops, totalPitchMonte[0], 'crimson', label="Pitch")
axs[0].plot(timeStops, totalPitchExpected, 'crimson', label="Ground truth", linestyle='dotted')
axs[1].plot(timeStops, totalRollMonte[0], 'deepskyblue', label="Roll")
axs[1].plot(timeStops, totalRollExpected, 'deepskyblue', label="Ground truth", linestyle='dotted')
axs[2].plot(timeStops, totalYawMonte[0], 'lawngreen', label="Yaw")
axs[2].plot(timeStops, totalYawExpected, 'lawngreen', label="Ground truth", linestyle='dotted')
for ax in axs.flat:
    ax.set(ylabel="Angle (°)")
    ax.set(xlabel="Samples")
    # ax.set_ylim(0,0.5)
    ax.grid(linestyle='--')
    ax.legend(loc="upper right")
plt.show()