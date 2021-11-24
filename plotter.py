import csv
from matplotlib import pyplot as plt

plot = 1

# Select filter to use (Comp = 0, EKF = 1, None = 2)
filter_type = 2

if plot:
    # Read sensor values from generated output file.
    def get_data(file_name):
        with open(file_name, mode='r', encoding='utf-8-sig') as file:
            reader = csv.reader(file, delimiter=',')
            data_return = list(reader)

        return data_return


    filename = r'sensor_readings.csv'
    data = get_data(filename)

    sample_points = []
    gyr_x = []
    gyr_y = []
    gyr_z = []
    acc_x = []
    acc_y = []
    acc_z = []
    mag_x = []
    mag_y = []
    mag_z = []

    for i in range(1, len(data)):
        sample_points.append(int(data[i][0]))
        gyr_x.append(float(data[i][1]))
        gyr_y.append(float(data[i][2]))
        gyr_z.append(float(data[i][3]))
        acc_x.append(float(data[i][4]))
        acc_y.append(float(data[i][5]))
        acc_z.append(float(data[i][6]))
        mag_x.append(float(data[i][7]))
        mag_y.append(float(data[i][8]))
        mag_z.append(float(data[i][9]))

    # Plot gyroscope data
    plt.plot(sample_points, gyr_x, label="X", color="blue")
    plt.plot(sample_points, gyr_y, label="Y", color="green")
    plt.plot(sample_points, gyr_z, label="Z", color="red")
    plt.title("Gyroscope")
    plt.xlabel("Samples")
    plt.ylabel("Angular rotation rate (dps)")
    plt.legend(loc=1)
    plt.show()

    # Plot accelerometer data
    plt.plot(sample_points, acc_x, label="X", color="blue")
    plt.plot(sample_points, acc_y, label="Y", color="green")
    plt.plot(sample_points, acc_z, label="Z", color="red")
    plt.title("Accelerometer")
    plt.xlabel("Samples")
    plt.ylabel("Linear acceleration (g)")
    plt.legend(loc=1)
    plt.show()

    # Plot magnetometer data
    plt.plot(sample_points, mag_x, label="X", color="blue")
    plt.plot(sample_points, mag_y, label="Y", color="green")
    plt.plot(sample_points, mag_z, label="Z", color="red")
    plt.title("Magnetometer")
    plt.xlabel("Samples")
    plt.ylabel("Magnetic field (Gauss)")
    plt.legend(loc=1)
    plt.show()

if filter_type == 0:
    exec(open("complementary_filter.py").read())
elif filter_type == 1:
    exec(open("extended_kalman_filter.py").read())
else:
    exit()

