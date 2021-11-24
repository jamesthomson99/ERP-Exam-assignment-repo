import numpy as np
from matplotlib import pyplot as plt
import csv
import os

os.system("cls")

# Choose whether to do monte carlo run
do_monte_carlo = 0

if do_monte_carlo:
    iterations = 100
else:
    iterations = 1

se_pitch = []
se_roll = []
se_yaw = []


def read_sensor_data_csv():
    csv_file = open('sensor_readings.csv', 'r')
    col_1 = []
    col_2 = []
    col_3 = []
    col_4 = []
    col_5 = []
    col_6 = []
    col_7 = []
    col_8 = []
    col_9 = []

    # Read off and discard first line, to skip headers
    csv_file.readline()

    # Split columns while reading
    for item1, item2, item3, item4, item5, item6, item7, item8, item9, item10 in csv.reader(csv_file, delimiter=','):
        # Append each variable to a separate list

        col_1.append(float(item2) + np.random.normal(0, 0.01))
        col_2.append(float(item3) + np.random.normal(0, 0.01))
        col_3.append(float(item4) + np.random.normal(0, 0.01))
        col_4.append(float(item5) + np.random.normal(0, 0.01))
        col_5.append(float(item6) + np.random.normal(0, 0.01))
        col_6.append(float(item7) + np.random.normal(0, 0.01))
        col_7.append(float(item8) + np.random.normal(0, 0.01))
        col_8.append(float(item9) + np.random.normal(0, 0.01))
        col_9.append(float(item10) + np.random.normal(0, 0.01))

    return col_1, col_2, col_3, col_4, col_5, col_6, col_7, col_8, col_9


def read_ground_truth_csv():
    csv_file = open('angles.csv', 'r')

    col_1 = []
    col_2 = []
    col_3 = []
    col_4 = []

    # Read off and discard first line, to skip headers
    csv_file.readline()

    # Split columns while reading
    for item1, item2, item3, item4, item5, item6, item7 in csv.reader(csv_file, delimiter=','):
        # Append each variable to a separate list
        col_1.append(float(item1))
        col_2.append(float(item2) * 180/np.pi)
        col_3.append(float(item3) * 180/np.pi)
        col_4.append(float(item4) * 180/np.pi)

    return col_2, col_3, col_4


# Runs once if do_monte_carlo == 0 and runs *iterations* number of times if do_monte_carlo == 1
for iterator in range(iterations):

    # Read sensor data from csv
    gyr_x, gyr_y, gyr_z, acc_x, acc_y, acc_z, mag_x, mag_y, mag_z = read_sensor_data_csv()
    pitch_gt, roll_gt, yaw_gt = read_ground_truth_csv()

    data_len = len(gyr_x)
    sample_freq = 100
    delta_t = 1/sample_freq
    weight = 0.98

    pitch = 0
    roll = 0
    yaw = 0

    pitch_list = []
    roll_list = []
    yaw_list = []

    for i in range(data_len):
        wx = gyr_x[i]
        wy = gyr_y[i]
        wz = gyr_z[i]
        ax = acc_x[i]
        ay = acc_y[i]
        az = acc_z[i]
        mx = mag_x[i]
        my = mag_y[i]
        mz = mag_z[i]

        # print("Measurements: ", wx, ", ", wy, ", ", wz, ", ", ax, ", ", ay, ", ", az, ", ", mx, ", ", my, ", ", mz)

        a_pitch = np.arctan2(-ax, np.sqrt(ay**2 + az**2)) * (180 / np.pi)
        a_roll = np.arctan2(ay, az) * (180 / np.pi)

        Mx = mx * np.cos(a_pitch) + mz * np.sin(a_pitch)
        My = mx * np.sin(a_roll) * np.sin(a_pitch) + my * np.cos(a_roll) - mz * np.sin(a_roll) * np.cos(a_pitch)
        m_yaw = np.arctan2(-My, Mx) * (180 / np.pi)

        g_pitch = wy * delta_t * (180 / np.pi)
        g_roll = wx * delta_t * (180 / np.pi)
        g_yaw = wz * delta_t * (180 / np.pi)

        pitch = weight * (pitch + g_pitch * delta_t) + (1 - weight) * a_pitch
        roll = weight * (roll + g_roll * delta_t) + (1 - weight) * a_roll
        yaw = weight * (yaw + g_yaw * delta_t) + (1 - weight) * m_yaw

        # print("Angles: ", pitch, ", ", roll, ", ", yaw)

        pitch_list.append(-pitch)
        roll_list.append(-roll)
        yaw_list.append(-yaw)

        # Calculate square error
        # If first run of monte carlo simulation
        if iterator == 0:
            se_pitch.append((pitch_gt[i] - pitch_list[i]) ** 2)
            se_roll.append((roll_gt[i] - roll_list[i]) ** 2)
            se_yaw.append((yaw_gt[i] - yaw_list[i]) ** 2)
        else:
            se_pitch[i] += (pitch_gt[i] - pitch_list[i]) ** 2
            se_roll[i] += (roll_gt[i] - roll_list[i]) ** 2
            se_yaw[i] += (yaw_gt[i] - yaw_list[i]) ** 2

    # Define x axis list for plotting
    x = np.arange(0, data_len * delta_t, delta_t)

    # Only plot graphs if only 1 iteration is occurring, otherwise too many graphs are plotted
    if not do_monte_carlo:
        fig, axs = plt.subplots(3)
        fig.suptitle("Complementary filter")
        axs[0].plot(x, pitch_list, 'tab:blue', label="Pitch")
        axs[0].plot(x, pitch_gt, 'tab:blue', label="True", linestyle='dotted')
        axs[1].plot(x, roll_list, 'tab:green', label="Roll")
        axs[1].plot(x, roll_gt, 'tab:green', label="True", linestyle='dotted')
        axs[2].plot(x, yaw_list, 'tab:red', label="Yaw")
        axs[2].plot(x, yaw_gt, 'tab:red', label="True", linestyle='dotted')
        axs[2].set(xlabel="Time (s)")
        for ax in axs.flat:
            ax.set(ylabel="Angle (deg.)")
            ax.set_xlim(0, data_len*delta_t)
            ax.grid()
            ax.legend(loc=1)
        plt.show()

# Calculate RMSE and plot RMSE for Monte Carlo simulation now that all runs are complete
if do_monte_carlo:

    rmse_pitch = []
    rmse_roll = []
    rmse_yaw = []

    for j in range(len(se_pitch)):
        rmse_pitch.append(np.sqrt(se_pitch[j]/iterations))
        rmse_roll.append(np.sqrt(se_roll[j]/iterations))
        rmse_yaw.append(np.sqrt(se_yaw)[j]/iterations)

    plt.plot(x, rmse_pitch, color="blue", label="Pitch")
    plt.plot(x, rmse_roll, color="green", label="Roll")
    plt.plot(x, rmse_yaw, color="red", label="Yaw")
    plt.title("Root-mean-square error")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (deg.)")
    plt.grid()
    plt.legend()
    plt.show()
