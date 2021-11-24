import csv
import numpy as np


# Inclination angle (magnetic dip)
dip = -62.217 * np.pi / 180
# Magnetic field vector
r = [float(np.cos(dip)), 0, -float(np.sin(dip))]
# Rotation matrix for determining sensor data
R_xyz = [[], [], []]

# Define std dev for for gyr, acc, mag noise
# gyr_sd = 0.02
# acc_sd = 0.008
# mag_sd = 0.03
gyr_sd = 0
acc_sd = 0
mag_sd = 0


# Read interpolated angle data
def get_data(file_name):
    with open(file_name, mode='r', encoding='utf-8-sig') as file:
        reader = csv.reader(file, delimiter=',')
        data_return = list(reader)
    return data_return


# Calculate angle
def get_angles(t, angle_data):
    angle_t = [float(angle_data[t][1]), float(angle_data[t][2]), float(angle_data[t][3])]
    return angle_t


# Calculate rate of change of angle
def get_rate(t, angle_data):
    rate_t = [float(angle_data[t][4]), float(angle_data[t][5]), float(angle_data[t][6])]
    return rate_t


# Gyroscope
def gyroscope(w):
    output = [w[0] * (180 / np.pi) + np.random.normal(0, gyr_sd),
              w[1] * (180 / np.pi) + np.random.normal(0, gyr_sd),
              w[2] * (180 / np.pi) + np.random.normal(0, gyr_sd)]
    return output


# Accelerometer
def accelerometer():
    output = [R_xyz[0][0] * 0 + R_xyz[1][0] * 0 + R_xyz[2][0] * 1 + np.random.normal(0, acc_sd),
              R_xyz[0][1] * 0 + R_xyz[1][1] * 0 + R_xyz[2][1] * 1 + np.random.normal(0, acc_sd),
              R_xyz[0][2] * 0 + R_xyz[1][2] * 0 + R_xyz[2][2] * 1 + np.random.normal(0, acc_sd)]
    return output


# Magnetometer
def magnetometer():
    output = [R_xyz[0][0] * r[0] + R_xyz[1][0] * r[1] + R_xyz[2][0] * r[2] + np.random.normal(0, mag_sd),
              R_xyz[0][1] * r[0] + R_xyz[1][1] * r[1] + R_xyz[2][1] * r[2] + np.random.normal(0, mag_sd),
              R_xyz[0][2] * r[0] + R_xyz[1][2] * r[1] + R_xyz[2][2] * r[2] + np.random.normal(0, mag_sd)]
    return output


# Main
filename = r'angles.csv'
data = get_data(filename)

# Output csv set up
cols = ["data_point", "gyro_x", "gyro_y", "gyro_z", "acc_x", "acc_y", "acc_z", "mag_x", "mag_y", "mag_z"]
output_file = open(r'sensor_readings.csv', 'w', newline="")
csv.writer(output_file).writerow(cols)

# Iterate through each sample in the input file and calculate the IMU values.
for i in range(1, len(data)):
    angles = get_angles(i, data)
    rate = get_rate(i, data)

    theta_t = angles[0]
    phi_t = angles[1]
    psi_t = angles[2]

    # Calculate the rotation matrix
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(phi_t), np.sin(phi_t)],
                    [0, -np.sin(phi_t), np.cos(phi_t)]])

    R_y = np.array([[np.cos(theta_t), 0, -np.sin(theta_t)],
                    [0, 1, 0],
                    [np.sin(theta_t), 0, np.cos(theta_t)]])

    R_z = np.array([[np.cos(psi_t), np.sin(psi_t), 0],
                    [-np.sin(psi_t), np.cos(psi_t), 0],
                    [0, 0, 1]])

    R_xyz = ((R_z @ R_y) @ R_x)

    gyr_data = gyroscope(rate)
    acc_data = accelerometer()
    mag_data = magnetometer()

    # Write sensor data to csv file
    output_data = [i, gyr_data[0], gyr_data[1], gyr_data[2], acc_data[0], acc_data[1], acc_data[2], mag_data[0], mag_data[1], mag_data[2]]
    writer = csv.writer(output_file)
    writer.writerow(output_data)

output_file.close()

exec(open("plotter.py").read())
