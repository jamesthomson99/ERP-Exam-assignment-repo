import csv
import numpy as np


class PoseGenerator:

    # filename - the name of the csv file the pose values are written to
    # sample_freq - the frequency at which sample points are generated

    def __init__(self, filename, sample_freq):
        # Initialise sampling period
        self.filename = filename
        self.sample_freq = sample_freq
        # Initialise pose and motion data
        self.sample = 0
        self.phi = 0  # Roll about X axis (rad )
        self.theta = 0  # Pitch about Y axis (rad )
        self.psi = 0  # Yaw about Z axis (rad )
        self.d_phi = 0  # Roll rate about X axis (rad / s)
        self.d_theta = 0  # Pitch rate about Y axis (rad / s)
        self.d_psi = 0  # Yaw rate about Z axis (rad / s)

        # Initialise CSV output
        self.columns = ["data_point", "theta", "phi", "psi", "d_theta", "d_phi", "d_psi"]
        with open(self.filename, "w+", newline="") as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=self.columns)
            csv_writer.writeheader()

            # Write initial pose and motion state to CSV
            self.__write_pose_to_csv()

    def __write_pose_to_csv(self):
        with open(self.filename, "a", newline="") as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=self.columns)
            data = {
                "data_point": self.sample,
                "theta": self.theta,
                "phi": self.phi,
                "psi": self.psi,
                "d_theta": self.d_theta,
                "d_phi": self.d_phi,
                "d_psi": self.d_psi
            }
            csv_writer.writerow(data)
            self.sample += 1

    # angle - new angle of phi (rad)
    def __update_phi(self, angle):
        self.d_phi = (angle - self.phi) * self.sample_freq
        self.phi = angle

    # angle - new angle of theta (rad)
    def __update_theta(self, angle):
        if angle <= - np.pi / 2 or angle >= np.pi / 2:
            raise Exception("Theta (pitch) out of bounds")
        self.d_theta = (angle - self.theta) * self.sample_freq
        self.theta = angle

    # angle - new angle of psi (rad)
    def __update_psi(self, angle):
        self.d_psi = (angle - self.psi) * self.sample_freq
        self.psi = angle

    # Set angular rates on all axes to 0
    def __reset_angular_rates(self):
        self.d_phi = 0
        self.d_theta = 0
        self.d_psi = 0

    # time - duration to remain stationary for (s)
    def remain_stationary(self, time):
        num_samples = time * self.sample_freq
        for i in range(1, num_samples + 1):
            self.__write_pose_to_csv()

    # Rolling right corresponds to a positive change in phi
    # angle - angle to roll right ( degrees )
    # time - duration to roll right (s)
    def roll_right(self, angle, time):
        num_samples = time * self.sample_freq
        angle_step = (angle * np.pi / 180) / num_samples
        initial_angle = self.phi
        for i in range(1, num_samples + 1):
            new_angle = initial_angle + angle_step * i
            self.__update_phi(new_angle)
            self.__write_pose_to_csv()
        self.__reset_angular_rates()

    # Rolling left corresponds to a negative change in phi
    # angle - angle to roll left ( degrees )
    # time - duration to roll left (s)
    def roll_left(self, angle, time):
        num_samples = time * self.sample_freq
        angle_step = (angle * np.pi / 180) / num_samples
        initial_angle = self.phi
        for i in range(1, num_samples + 1):
            new_angle = initial_angle - angle_step * i
            self.__update_phi(new_angle)
            self.__write_pose_to_csv()
        self.__reset_angular_rates()

    # Pitching up corresponds to a positive change in theta
    # angle - angle to pitch up ( degrees )
    # time - duration to pitch up (s)
    def pitch_up(self, angle, time):
        num_samples = time * self.sample_freq
        angle_step = (angle * np.pi / 180) / num_samples
        initial_angle = self.theta
        for i in range(1, num_samples + 1):
            new_angle = initial_angle + angle_step * i
            self.__update_theta(new_angle)
            self.__write_pose_to_csv()
        self.__reset_angular_rates()

    # Pitching down corresponds to a negative change in theta
    # angle - angle to pitch down ( degrees )
    # time - duration to pitch down (s)
    def pitch_down(self, angle, time):
        num_samples = time * self.sample_freq
        angle_step = (angle * np.pi / 180) / num_samples
        initial_angle = self.theta
        for i in range(1, num_samples + 1):
            new_angle = initial_angle - angle_step * i
            self.__update_theta(new_angle)
            self.__write_pose_to_csv()
        self.__reset_angular_rates()

    # Yawing left corresponds to a positive change in psi
    # angle - angle to yaw left ( degrees )
    # time - duration to yaw left (s)
    def yaw_left(self, angle, time):
        num_samples = time * self.sample_freq
        angle_step = (angle * np.pi / 180) / num_samples
        initial_angle = self.psi
        for i in range(1, num_samples + 1):
            new_angle = initial_angle + angle_step * i
            self.__update_psi(new_angle)
            self.__write_pose_to_csv()
        self.__reset_angular_rates()

    # Yawing right corresponds to a negative change in psi
    # angle - angle to yaw right ( degrees )
    # time - duration to yaw right (s)
    def yaw_right(self, angle, time):
        num_samples = time * self.sample_freq
        angle_step = (angle * np.pi / 180) / num_samples
        initial_angle = self.psi
        for i in range(1, num_samples + 1):
            new_angle = initial_angle - angle_step * i
            self.__update_psi(new_angle)
            self.__write_pose_to_csv()
        self.__reset_angular_rates()


# Generate pose data
# 1. Stationary for a period of 5 seconds , then
# 2. pitch up to 45 degrees over a period of 3 seconds , then
# 3. pitch back to 0 degrees over a period of 3 seconds , then
# 4. roll to the left 30 degrees over a period of 3 seconds , then
# 5. roll back to 0 degrees over a period of 3 seconds , then
# 6. yaw right to 120 degrees over a period of 3 seconds , then
# 7. yaw back to 0 degrees over a period of 3 seconds , then
# 8. remain stationary for a period of 5 seconds .

gen = PoseGenerator("angles.csv", 100)
gen.remain_stationary(5)
gen.pitch_up(45, 3)
gen.pitch_down(45, 3)
gen.roll_left(30, 3)
gen.roll_right(30, 3)
gen.yaw_right(120, 3)
gen.yaw_left(120, 3)
gen.remain_stationary(5)

# gen = PoseGenerator("angles.csv", 100)
# gen.remain_stationary(5)
# gen.pitch_down(20, 3)
# gen.pitch_up(45, 3)
# gen.pitch_down(25, 3)
# gen.roll_left(45, 3)
# gen.roll_right(45, 4)
# gen.yaw_left(30, 5)
# gen.yaw_right(45, 5)

exec(open("IMU_simulator.py").read())

