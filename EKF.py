import numpy as np
from numpy.linalg import inv
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation
import csv

# Change filter variables
iterations = 1
sample_rate = 120
delta_t = 1 / sample_rate

# Define standard deviations and variances
sigma_w = 0.001
sigma_a = 0.001
sigma_m = 0.001
vw = sigma_w ** 2
va = sigma_a ** 2
vm = sigma_m ** 2
sigma_wx = 0.09371
sigma_wy = 0.06343
sigma_wz = 0.0629
vwx = sigma_wx ** 2
vwy = sigma_wy ** 2
vwz = sigma_wz ** 2

# Change plotting variables
plot_type = 0 # 0 - euler, 1 - quaternions
plot_uncertainty = 0
plot_uncertainty_zoomed = 0

# Create gravity vector and magnetic field vector based on dip angle = -62.217 °
gx = 0
gy = 0
gz = 1
rx = 0
ry = 0.4661241594
rz = 0.884719316

# Create square error lists for Monte Carlo runs
pitch_squared_error, roll_squared_error, yaw_squared_error = [], [], []

def convert_to_euler(q_input):
    rotation = Rotation.from_quat([q_input[1][0], q_input[2][0], q_input[3][0], q_input[0][0]])
    euler_angles = rotation.as_euler('xyz', degrees=True)
    return euler_angles[0] * (180 / np.pi), euler_angles[1] * (180 / np.pi), euler_angles[2] * (180 / np.pi)


# Convert ground truth data to quaternions 
def convert_gt_to_quaternion(theta, phi, psi):

    # Define return lists
    qw_gt_return, qx_gt_return, qy_gt_return, qz_gt_return = [], [], [], []

    for k in range(len(theta)):
        cos_yaw = np.cos((psi[k] * np.pi / 180) / 2)
        cos_roll = np.cos((phi[k] * np.pi / 180) / 2)
        cos_pitch = np.cos((theta[k] * np.pi / 180) / 2)
        sin_yaw = np.sin((psi[k] * np.pi / 180) / 2)
        sin_roll = np.sin((phi[k] * np.pi / 180) / 2)
        sin_pitch = np.sin((theta[k] * np.pi / 180) / 2)

        # Append ground truth quaternion values to their respective lists
        qw_gt_return.append(cos_roll * cos_pitch * cos_yaw + sin_roll * sin_pitch * sin_yaw)
        qx_gt_return.append(sin_roll * cos_pitch * cos_yaw - cos_roll * sin_pitch * sin_yaw)
        qy_gt_return.append(cos_roll * sin_pitch * cos_yaw + sin_roll * cos_pitch * sin_yaw)
        qz_gt_return.append(cos_roll * cos_pitch * sin_yaw - sin_roll * sin_pitch * cos_yaw)

    return qw_gt_return, qx_gt_return, qy_gt_return, qz_gt_return


# Predict states given gyroscope data
def predict(q_prev, P_prev, w_curr):

    # Define variables for previous quaternion states
    qw = q_prev[0][0]
    qx = q_prev[1][0]
    qy = q_prev[2][0]
    qz = q_prev[3][0]

    # Define variables for current gyro inputs
    wx = w_curr[0][0]
    wy = w_curr[1][0]
    wz = w_curr[2][0]

    # Estimate new quaternion orientation estimate
    q_new = np.array([[qw - (delta_t/2)*wx*qx - (delta_t/2)*wy*qy - (delta_t/2)*wz*qz],
                      [qx + (delta_t/2)*wx*qw - (delta_t/2)*wy*qz + (delta_t/2)*wz*qy],
                      [qy + (delta_t/2)*wx*qz + (delta_t/2)*wy*qw - (delta_t/2)*wz*qx],
                      [qz - (delta_t/2)*wx*qy + (delta_t/2)*wy*qx + (delta_t/2)*wz*qw]])

    # Calculate F Jacobian matrix to be used to calculate predicted covariance (Pt)
    F = np.array([[1, -(delta_t/2)*wx, -(delta_t/2)*wy, -(delta_t/2)*wz],
                      [(delta_t/2)*wx, 1, (delta_t/2)*wz, -(delta_t/2)*wy],
                      [(delta_t/2)*wy, -(delta_t/2)*wz, 1, (delta_t/2)*wx],
                      [(delta_t/2)*wz, (delta_t/2)*wy, -(delta_t/2)*wx, 1]])

    # Calculate Wt matrix to be used to calculate process noise covariance (Qt)
    W = (delta_t/2)*np.array([[-qx, -qy, -qz],
                              [ qw, -qz,  qy],
                              [ qz,  qw, -qx],
                              [-qy,  qx,  qw]])

    # Calculate process noise covariance (Qt)
    Q =  W @ Sigma_w @ W.T

    # Calculate predicted covariance (Pt)
    P_new = (F @ P_prev @ F.T) + Q

    return q_new, P_new


# Correct predicted states and noise matrix
def correct(q_pred, P_pred, z):

    # Define variables for predicted quaternion states
    qw, qx, qy, qz = q_pred[0][0], q_pred[1][0], q_pred[2][0], q_pred[3][0]

    # Create measurement model h(qt)
    h = 2 * np.array([[gx*(0.5 - qy**2 - qz**2) + gy*(qw*qz + qx*qy) + gz*(qx*qz - qw*qy)],
                      [gx*(qx*qy - qw*qz) + gy*(0.5 - qx**2 - qz**2) + gz*(qw*qx + qy*qz)],
                      [gx*(qw*qy + qx*qz) + gy*(qy*qz - qw*qx) + gz*(0.5 - qx**2 - qy**2)],
                      [rx*(0.5 - qy**2 - qz**2) + ry*(qw*qz + qx*qy) + rz*(qx*qz - qw*qy)],
                      [rx*(qx*qy - qw*qz) + ry*(0.5 - qx**2 - qz**2) + rz*(qw*qx + qy*qz)],
                      [rx*(qw*qy + qx*qz) + ry*(qy*qz - qw*qx) + rz*(0.5 - qx**2 - qy**2)]])

    # Create the Jacobian, H(qt), of h(qt)
    H = 2 * np.array([[gy*qz - gz*qy , gy*qy + gz*qz , -2*gx*qy + gy*qx - gz*qw , - 2*gx*qz + gy*qw + gz*qx],
                      [-gx*qz + gz*qx , gx*qy - 2*gy*qx + gz*qw , gx*qx + gz*qz , -gx*qw - 2*gy*qz + gz*qy ],
                      [gx*qy - gy*qx , gx*qz - gy*qw -2*gz*qx , gx*qw + gy*qz - 2*gz*qy , gx*qx + gy*qy    ],
                      [ry*qz - rz*qy , ry*qy + rz*qz , -2*rx*qy + ry*qx - rz*qw , - 2*rx*qz + ry*qw + rz*qx ],
                      [-rx*qz + rz*qx , rx*qy - 2*ry*qx + rz*qw , rx*qx + rz*qz , - rx*qw - 2*ry*qz + rz*qy],
                      [rx*qy - ry*qx , rx*qz - ry*qw - 2*rz*qx , rx*qw + ry*qz - 2*rz*qy , rx*qx + ry*qy   ]])

    # Calculate the innovation/measurement residual (vt)
    v = z - h

    # Calculate measurement prediction covariance (St)
    S = (H @ P_pred @ H.T) + R

    # Calculate the filter gain/ Kalman gain (Kt)
    K = P_pred @ H.T @ inv(S)

    # Correct the state (orientation estimate)
    q_corr = q_pred + K @ v

    # Correct the predicted covariance
    P_corr = (I_4 - (K @ H)) @ P_pred

    return q_corr, P_corr


# Load sensor data from the npy files
ground_truth = np.load(r'Data\GT.npy')
acc_data = np.load(r'Data\accelData.npy')
gyr_data = np.load(r'Data\gyroData.npy')
mag_data = np.load(r'Data\magData.npy')

pitch_ground_truth, roll_ground_truth, yaw_ground_truth = [], [], []

# Split ground truth data into pitch, roll, yaw components
for i in range(len(ground_truth)):
    roll_ground_truth.append(ground_truth[i][0])
    pitch_ground_truth.append(ground_truth[i][1])
    yaw_ground_truth.append(ground_truth[i][2])

# Runs *iterations* number of times (either 1 for normal or 100 for Monte Carlo)
for iterator in range(iterations):

    # Define sensor lists
    gyr_x, gyr_y, gyr_z, acc_x, acc_y, acc_z, mag_x, mag_y, mag_z = [], [], [], [], [], [], [], [], []

    # Extract the current run's accelerometer, gyroscope and magnetometer data from the npy files
    for j in range(len(acc_data[0])):
        acc_x.append(acc_data[iterator][j][0])
        acc_y.append(acc_data[iterator][j][1])
        acc_z.append(acc_data[iterator][j][2])
        gyr_x.append(gyr_data[iterator][j][0])
        gyr_y.append(gyr_data[iterator][j][1])
        gyr_z.append(gyr_data[iterator][j][2])
        mag_x.append(mag_data[iterator][j][0])
        mag_y.append(mag_data[iterator][j][1])
        mag_z.append(mag_data[iterator][j][2])

    # Convert ground truth data to quaternions
    qw_ground_truth_list, qx_ground_truth_list, qy_ground_truth_list, qz_ground_truth_list = convert_gt_to_quaternion(pitch_ground_truth, roll_ground_truth, yaw_ground_truth)

    # Define lists for storing pitch, roll, yaw angles over entire run
    pitches, rolls, yaws = [], [], []

    # Define lists for storing quaternions over entire run
    qw_list, qx_list, qy_list, qz_list = [], [], [], []

    # Define lists for storing variances for each quaternion over entire run
    qw_variances, qx_variances, qy_variances, qz_variances = [], [], [], []

    # Define length of data input
    data_len = len(gyr_x)

    # Initial quaternion orientation estimate (assuming pitch, roll, yaw = 0 deg)
    q = np.array([[1],
                  [0],
                  [0],
                  [0]])

    # Initial process noise variance matrix
    P = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])

    # Create 4x4 identity matrix
    I_4 = np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])

    # Create measurement noise covariance matrix, R
    R = np.array([[va, 0, 0, 0, 0, 0],
                  [0, va, 0, 0, 0, 0],
                  [0, 0, va, 0, 0, 0],
                  [0, 0, 0, vm, 0, 0],
                  [0, 0, 0, 0, vm, 0],
                  [0, 0, 0, 0, 0, vm]])

    # Create Sigma_w matrix to calculate Qt
    Sigma_w = np.array([[vwx, 0, 0],
                        [0, vwy, 0],
                        [0, 0, vwz]])  

    # Loop through all sensor input data
    for i in range(len(gyr_x)):

        ### PREDICTION ###

        # Get gyr input vector (u)
        w = np.array([[gyr_x[i] * (np.pi / 180)], [gyr_y[i] * (np.pi / 180)], [gyr_z[i] * (np.pi / 180)]])

        # Call predict step function to estimate states and predicted covariance
        q_predicted, P_predicted = predict(q, P, w)

        ### CORRECTION ###

        # Get acc, mag measurement vectors (z)
        a = np.array([[acc_x[i]], [acc_y[i]], [acc_z[i]]])
        m = np.array([[mag_x[i]], [mag_y[i]], [mag_z[i]]])

        # Create measurement vector zt
        z_measurements = np.concatenate((a, m))

        q_corrected, P_corrected = correct(q_predicted, P_predicted, z_measurements)

        # Update q and P variables for next loop
        q = q_corrected
        P = P_corrected

        # Store quaternions in lists for plotting
        qw_list.append(q[0][0])
        qx_list.append(q[1][0])
        qy_list.append(q[2][0])
        qz_list.append(q[3][0]) 

        # Store quaternion variances for plotting
        qw_variances.append(abs(P_predicted[0][0]))
        qx_variances.append(abs(P_predicted[1][1]))
        qy_variances.append(abs(P_predicted[2][2]))
        qz_variances.append(abs(P_predicted[3][3]))

        # Convert quaternions to euler angles and store angles in lists for plotting
        pitch, roll, yaw = convert_to_euler(q)
        pitches.append(pitch)
        rolls.append(roll)
        yaws.append(yaw)

        # Calculate square error
        # If first run of monte carlo simulation
        if iterator == 0:
            pitch_squared_error.append((pitch_ground_truth[i] - pitches[i]) ** 2)
            roll_squared_error.append((roll_ground_truth[i] - rolls[i]) ** 2)
            yaw_squared_error.append((yaw_ground_truth[i] - yaws[i]) ** 2)
        else:
            pitch_squared_error += (pitch_ground_truth[i] - pitches[i]) ** 2
            roll_squared_error += (roll_ground_truth[i] - rolls[i]) ** 2
            yaw_squared_error += (yaw_ground_truth[i] - yaws[i]) ** 2

        ### END OF LOOP ###

    # Define x axis list for plotting
    x = np.arange(0, data_len*delta_t, delta_t)

    # Only plot graphs if only 1 iteration is occurring, otherwise too many graphs are plotted
    if iterations == 1:
        # Plot either euler angles or quaternions based on value of plot_type
        if plot_type == 0:
            fig, axs = plt.subplots(3)
            fig.suptitle("Extended Kalman filter")
            axs[0].plot(x, pitches, 'tab:blue', label="Pitch")
            axs[0].plot(x, pitch_ground_truth, 'tab:blue', label="True", linestyle='dotted')
            axs[1].plot(x, rolls, 'tab:green', label="Roll")
            axs[1].plot(x, roll_ground_truth, 'tab:green', label="True", linestyle='dotted')
            axs[2].plot(x, yaws, 'tab:red', label="Yaw")
            axs[2].plot(x, yaw_ground_truth, 'tab:red', label="True", linestyle='dotted')
            axs[2].set(xlabel="Time (s)")
            for axis in axs.flat:
                axis.set_ylim(-30, 30)
                axis.set(ylabel="Angle (deg.)")
                axis.set_xlim(0, data_len*delta_t)
                axis.grid()
                axis.legend(loc=1)
            plt.show()

        elif plot_type == 1:
            fig, axs = plt.subplots(2, 2)
            fig.suptitle("Extended Kalman filter")
            axs[0, 0].plot(x, qw_list, 'tab:blue', label="qw")
            axs[0, 0].plot(x, qw_ground_truth_list, 'tab:blue', label="True", linestyle='dotted')
            axs[0, 1].plot(x, qx_list, 'tab:green', label="qx")
            axs[0, 1].plot(x, qx_ground_truth_list, 'tab:green', label="True", linestyle='dotted')
            axs[1, 0].plot(x, qy_list, 'tab:red', label="qy")
            axs[1, 0].plot(x, qy_ground_truth_list, 'tab:red', label="True", linestyle='dotted')
            axs[1, 1].plot(x, qz_list, 'tab:orange', label="qz")
            axs[1, 1].plot(x, qz_ground_truth_list, 'tab:red', label="True", linestyle='dotted')
            if plot_uncertainty:
                axs[0, 1].fill_between(x, np.subtract(qw_list, qw_variances), np.add(qw_list, qw_variances), color="green", alpha=0.2)
                axs[0, 0].fill_between(x, np.subtract(qx_list, qx_variances), np.add(qx_list, qx_variances), color="blue", alpha=0.2)
                axs[1, 1].fill_between(x, np.subtract(qy_list, qy_variances), np.add(qy_list, qy_variances), color="orange", alpha=0.2)
                axs[1, 0].fill_between(x, np.subtract(qz_list, qz_variances), np.add(qz_list, qz_variances), color="red", alpha=0.2)
            for axis in axs.flat:
                axis.set_ylim(-1, 1.2)
                if plot_uncertainty_zoomed:
                    axis.set_xlim(0, 2)
                else:
                    axis.set_xlim(0, data_len*delta_t)
                axis.grid()
                axis.legend()
            plt.show()

# Calculate RMSE and plot RMSE for Monte Carlo simulation now that all runs are complete
if iterations > 1:

    pitch_rmse, roll_rmse, yaw_rmse = [], [], []

    for j in range(len(pitch_squared_error)):
        pitch_rmse.append(np.sqrt(pitch_squared_error[j] / iterations))
        roll_rmse.append(np.sqrt(roll_squared_error[j] / iterations))
        yaw_rmse.append(np.sqrt(yaw_squared_error[j] / iterations))

    plt.plot(x, pitch_rmse, color="blue", label="Pitch")
    plt.plot(x, roll_rmse, color="green", label="Roll")
    plt.plot(x, yaw_rmse, color="red", label="Yaw")
    plt.title("Root-mean-square error")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (deg.)")
    plt.grid()
    plt.legend()
    plt.show()
