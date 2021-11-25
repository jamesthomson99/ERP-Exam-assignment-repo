import numpy as np
from matplotlib import pyplot as plt
import csv
import os

# Clear terminal at beginning of each run
os.system("cls")

# Change filter variables
iterations = 100
sample_freq = 120
weight = 0.95

pitch_squared_error = []
roll_squared_error = []
yaw_squared_error = []

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


# Runs once if do_monte_carlo == 0 and runs *iterations* number of times if do_monte_carlo == 1
for iterator in range(iterations):
    acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z, mag_x, mag_y, mag_z = [], [], [], [], [], [], [], [], []
    
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

    data_len = len(gyr_x)
    delta_t = 1/sample_freq

    pitch, roll, yaw = 0, 0, 0
    pitches, rolls, yaws = [], [], []

    # Loop through all sample points in the run
    for k in range(data_len):

        # Define variables for use in CF
        wx = gyr_x[k]
        wy = gyr_y[k]
        wz = gyr_z[k]
        ax = acc_x[k]
        ay = acc_y[k]
        az = acc_z[k]
        mx = mag_x[k]
        my = mag_y[k]
        mz = mag_z[k]

        # Actual CF algorithm
        
        # Calculate pitch, roll, due to accelerometer data
        a_pitch = np.arctan2(-ax, np.sqrt(ay**2 + az**2))
        a_roll = np.arctan2(ay, az)

        # Calculate tilt compensation for the yaw 
        Mx = mx * np.cos(a_pitch) + mz * np.sin(a_pitch)
        My = mx * np.sin(a_roll) * np.sin(a_pitch) + my * np.cos(a_roll) - mz * np.sin(a_roll) * np.cos(a_pitch)
        m_yaw = np.arctan2(-My, Mx)

        # Calculate pitch, roll, yaw due to gyroscope data
        g_pitch = wy * delta_t * (180 / np.pi)
        g_roll = wx * delta_t * (180 / np.pi)
        g_yaw = wz * delta_t * (180 / np.pi)

        # CF - combine angles with the defined weight
        pitch = weight * (pitch + g_pitch * delta_t) + (1 - weight) * a_pitch  * (180 / np.pi)
        roll = weight * (roll + g_roll * delta_t) + (1 - weight) * a_roll  * (180 / np.pi)
        yaw = weight * (yaw + g_yaw * delta_t) + (1 - weight) * m_yaw  * (180 / np.pi)
        
        # Add current angles to lists for plotting
        pitches.append(pitch)
        rolls.append(roll)
        yaws.append(yaw)

        # Calculate square error
        # If first run of monte carlo simulation
        if iterator == 0:
            pitch_squared_error.append((pitch_ground_truth[k] - pitches[k]) ** 2)
            roll_squared_error.append((roll_ground_truth[k] - rolls[k]) ** 2)
            yaw_squared_error.append((yaw_ground_truth[k] - yaws[k]) ** 2)
        else:
            pitch_squared_error[k] += (pitch_ground_truth[k] - pitches[k]) ** 2
            roll_squared_error[k] += (roll_ground_truth[k] - rolls[k]) ** 2
            yaw_squared_error[k] += (yaw_ground_truth[k] - yaws[k]) ** 2


    # Define x axis list for plotting
    x = np.arange(0, data_len * delta_t, delta_t)

    # Only plot graphs if only 1 iteration is occurring, otherwise too many graphs are plotted
    if iterations == 1:
        fig, axs = plt.subplots(3)
        fig.suptitle("Complementary filter orientation estimate")
        axs[0].plot(x, pitches, 'tab:orange', label="Pitch")
        axs[0].plot(x, pitch_ground_truth, 'tab:orange', label="True", linestyle='dotted')
        axs[1].plot(x, rolls, 'tab:blue', label="Roll")
        axs[1].plot(x, roll_ground_truth, 'tab:blue', label="True", linestyle='dotted')
        axs[2].plot(x, yaws, 'tab:green', label="Yaw")
        axs[2].plot(x, yaw_ground_truth, 'tab:green', label="True", linestyle='dotted')
        axs[2].set(xlabel="Time (s)")
        for ax in axs.flat:
            ax.set(ylabel="Angle (°)")
            ax.set_xlim(0, data_len*delta_t)
            ax.grid()
            ax.legend(loc=1)
        plt.show()

# Calculate RMSE and plot RMSE for Monte Carlo simulation now that all runs are complete
if iterations > 1:

    pitch_rmse, roll_rmse, yaw_rmse = [], [], []

    for l in range(len(pitch_squared_error)):
        pitch_rmse.append(np.sqrt(pitch_squared_error[l]/iterations))
        roll_rmse.append(np.sqrt(roll_squared_error[l]/iterations))
        yaw_rmse.append(np.sqrt(yaw_squared_error[l]/iterations))

    plt.plot(x, pitch_rmse, color="orange", label="Pitch")
    plt.plot(x, roll_rmse, color="blue", label="Roll")
    plt.plot(x, yaw_rmse, color="green", label="Yaw")
    plt.title("Complemetary filter RMSE of 100 Monte Carlo runs")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (°)")
    plt.grid()
    plt.legend()
    plt.show()
