import numpy as np
from scipy import io
from scipy.spatial.transform import Rotation as Rot
import math
import matplotlib.pyplot as plt


# data files are numbered on the server.
# for exmaple imuRaw1.mat, imuRaw2.mat and so on.
# write a function that takes in an input number (1 through 3)
# reads in the corresponding imu Data, and estimates
# roll pitch and yaw using an unscented kalman filter


def mul_quat(q1, q2):
    q1 = q1.reshape(-1)
    q2 = q2.reshape(-1)
    t0 = q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3]
    t1 = q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2]
    t2 = q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1]
    t3 = q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0]
    return np.array([t0, t1, t2, t3])


def inv_quat(q1):
    q2 = np.zeros(4, )
    q2[0] = q1[0]
    q2[1] = -q1[1]
    q2[2] = -q1[2]
    q2[3] = -q1[3]
    q2 = q2 / (np.linalg.norm(q2) ** 2)
    return q2


def rotv2quat(rotv):
    if len(rotv.shape) == 1:
        rotv = rotv.reshape(1, 3)
    r = Rot.from_rotvec(rotv)
    q = r.as_quat()
    n, m = np.shape(rotv)
    new_q = np.zeros((n, 4))
    new_q[:, 1:4] = q[:, :-1]
    new_q[:, 0] = q[:, -1]
    return new_q


def quat2rotv(q):
    qs = q[0]
    qv = q[1:4]
    if np.linalg.norm(qv) == 0:
        v = np.transpose(np.matrix([0, 0, 0]))
    else:
        v = 2 * ((qv / np.linalg.norm(qv)) * math.acos(qs / np.linalg.norm(q)))
    return v


def euler2quat(rotv):  #########
    if len(rotv.shape) == 1:
        rotv = rotv.reshape(1, 3)
    r = Rot.from_euler('xyz', rotv, degrees=False)
    q = r.as_quat()
    n, m = np.shape(rotv)
    new_q = np.zeros((n, 4))
    new_q[:, 1:4] = q[:, :-1]
    new_q[:, 0] = q[:, -1]
    return new_q


def quat2euler(q):
    phi = math.atan2(2 * (q[0] * q[1] + q[2] * q[3]), \
                     1 - 2 * (q[1] ** 2 + q[2] ** 2))
    theta = math.asin(2 * (q[0] * q[2] - q[3] * q[1]))
    psi = math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), \
                     1 - 2 * (q[2] ** 2 + q[3] ** 2))
    return np.array([phi, theta, psi])

def normalize_quat(q):
    return q / np.linalg.norm(q)

def quat_average(Y, q0):  #q(12,7) q0(7,)
    q = Y[:,:4]
    omega = Y[:,4:]   #omega(3,)
    omega_mean = np.mean(omega,axis=0)   #omega_mean(3,)
    omega_error = omega - omega_mean
    qt = q0[:4] #(4,)
    epsilon = 0.00001
    for _ in range(1000):
        error = np.zeros((q.shape[0], 3))
        for i in range(q.shape[0]):
            error_q = normalize_quat(mul_quat(q[i, :], inv_quat(qt)))
            error_v = quat2rotv(error_q)
            if np.round(np.linalg.norm(error_v), 10) == 0:
                error[i] = np.zeros(3)
            else:
                error[i, :] = (-np.pi + np.mod(np.linalg.norm(error_v) + np.pi, 2 * np.pi)) / np.linalg.norm(
                    error_v) * error_v
        error_mean = np.mean(error, axis=0)
        qt = normalize_quat(mul_quat(rotv2quat(error_mean), qt))
        if np.linalg.norm(error_mean) < epsilon:
          qt = np.hstack((qt,omega_mean))     #qt(7,)   error(12,3)
          error = np.hstack((error,omega_error))
          return qt, error


def sigma_points(P, Q, vec):   #P(6,6), Q(6,6), vec(7,)
    n = P.shape[0]   #n=6
    S = np.linalg.cholesky(P + Q)     #(6,6)
    left = np.sqrt(n) * S
    right = -np.sqrt(n) * S
    W = np.hstack((left, right)).T     #(12,6)

    X = np.zeros((W.shape[0], 7))    #(12,7)
    for i in range(W.shape[0]):
        temp = rotv2quat(W[i,:3])
        X[i,:4] = mul_quat(vec[:4], temp)
        X[i,4:] = W[i,3:] +vec[4:]    #X(12,7)   #check later to add vec[4:] or not.
    return X


def transform_sigma(X, dt, omega_k): #X(12,7) omega_k(3,)
    q_delta = rotv2quat(omega_k*dt)
    Y = np.zeros((X.shape[0],7))
    for i in range(Y.shape[0]):
        Y[i,:4] = mul_quat(X[i,:4], q_delta)   #Y(12,7)
    return Y


def estimate_rot(data_num):
    # %% Testing the functions
    vicon = io.loadmat('vicon/viconRot' + str(data_num) + '.mat')
    vicon_euler = np.zeros((vicon["rots"].shape[2], 3))
    for i in range(vicon["rots"].shape[2]):
        r = Rot.from_matrix(vicon["rots"][:, :, i])
        vicon_euler[i] = r.as_euler('xyz')

    # imu = io.loadmat('source/imu/imuRaw' + str(data_num) + '.mat')
    imu = io.loadmat('imu/imuRaw' + str(data_num) + '.mat')

    accel = imu['vals'][0:3, :]
    gyro = imu['vals'][3:6, :]
    T = np.shape(imu['ts'])[1]
    imu_ts = imu['ts'].T

    # your code goes here

    Vref = 3300
    imu_vals = imu['vals']
    acc_x = -np.array(imu_vals[0]) # IMU Ax and Ay direction is flipped !
    acc_y = -np.array(imu_vals[1])
    acc_z = np.array(imu_vals[2])
    acc = np.array([acc_x, acc_y, acc_z]).T


    acc_sensitivity = 330.0
    acc_scale_factor = Vref/1023.0/acc_sensitivity
    acc_bias = np.mean(acc[:10], axis=0) - np.array([0,0,1])/acc_scale_factor
    print(acc_bias)
    acc = (acc-acc_bias)*acc_scale_factor

    gyro_x = np.array(imu_vals[4]) # angular rates are out of order !
    gyro_y = np.array(imu_vals[5])
    gyro_z = np.array(imu_vals[3])
    gyro = np.array([gyro_x, gyro_y, gyro_z]).T

    gyro_bias = np.mean(gyro[:10], axis=0)
    print(gyro_bias)
    gyro_sensitivity = 3.33
    gyro_scale_factor = Vref/1023/gyro_sensitivity
    gyro = (gyro-gyro_bias)*gyro_scale_factor*(np.pi/180)

    # %% Initialization of variables
    P = 0.001 * np.identity(6)     #(6,6)
    Q = 0.0001 * np.identity(6)      #(6,6)
    R = 0.00001 * np.identity(6)     #(6,6)
    qt = np.array([1, 0, 0, 0, 1, 2, 1])   #(7,)
    time = imu_vals.shape[0]  # 5000
    predicted_q = qt   #initializing
    predicted_w = np.array([1,2,1])
    cov_quat = np.array([0.001, 0.001, 0.001])
    cov_ang = np.array([0.001, 0.001, 0.001])

    for i in range(T):

        X = sigma_points(P, Q, qt)  # (12,7)
        if i == imu_ts.shape[0] - 1:
            dt = imu_ts[-1] - imu_ts[-2]
        else:
            dt = imu_ts[i + 1] - imu_ts[i]
            # Process model
            Y = transform_sigma(X, dt, gyro[i])  # (12,7)
            # compute mean
            x_k_bar, error = quat_average(Y, qt)  # 38,39 # (7,)  error:(12,6)
            # compute covariance (in vector)
            P_k_bar = np.zeros((6, 6))  # (3,3)
            for i in range(12):
                P_k_bar += np.outer(error[i, :], error[i, :])
            P_k_bar /= 24

            # measurement model
            g = np.array([0, 0, 0, 1])  # down

            Z = np.zeros((12, 6))  # vector quaternions
            for i in range(12):
                # compute predicted acceleration
                q = Y[i, :4]
                Z[i, :3] = mul_quat(mul_quat(inv_quat(q), g), q)[1:4]  # rotate from body frame to global frame   check this later
                Z[i, 3:] = Y[i, 4:]

            # measurement mean
            z_k_bar = np.mean(Z, axis=0)  # (6,)
            z_k_bar[:3] /= np.linalg.norm(z_k_bar[:3])

            # measurement cov and correlation
            Pzz = np.zeros((6, 6))
            Pxz = np.zeros((6, 6))
            Z_err = Z - z_k_bar      #(12,6)
            for i in range(12):
                Pzz += np.outer(Z_err[i, :], Z_err[i, :])
            Pxz += np.outer(error[i, :], Z_err[i, :])
            Pzz /= 24
            Pxz /= 24
            # innovation
            acc_term = acc[i] / np.linalg.norm(acc[i])

            vk = np.zeros(6,)
            vk[3:] = gyro[i] - z_k_bar[3:]
            vk[:3] = acc_term - z_k_bar[:3]  # 44    (6,)

            Pvv = Pzz + R  # 45
            # compute Kalman gain
            K = np.dot(Pxz, np.linalg.inv(Pvv))  # 72 # (6,6)
            # update
            new_x_k_bar = np.zeros(6,)
            new_x_k_bar[3:] = x_k_bar[4:]
            new_x_k_bar[:3] = quat2rotv(x_k_bar[:4])

            q_gain = new_x_k_bar + np.dot(K,vk)     #(6,)

            q_update = np.zeros(7,)
            q_update[:4] = rotv2quat(q_gain[:3])
            q_update[4:] = q_gain[3:]    #(7,)

            P_update = P_k_bar - K.dot(Pvv).dot(K.T)  # 75
            P = P_update
            qt = q_update

            temp_quat = np.zeros(3)
            temp_ang = np.zeros(3)
            for i in range(3):
                temp_quat[i] = P[i,i]
                temp_ang[i] = P[i+3,i+3]

            cov_quat = np.vstack((cov_quat, temp_quat))
            cov_ang = np.vstack((cov_ang, temp_ang))
            # print(qt[3:])
            predicted_w = np.vstack((predicted_w, qt[4:]))
            predicted_q = np.vstack((predicted_q, qt))

            roll = np.zeros(np.shape(predicted_q)[0])
            pitch = np.zeros(np.shape(predicted_q)[0])
            yaw = np.zeros(np.shape(predicted_q)[0])

            for i in range(np.shape(predicted_q)[0]):
                roll[i], pitch[i], yaw[i] = quat2euler(predicted_q[i,:4])

    # Green is the predicted values
    # Blue is Vicon values
    # print("cov_anag: ",cov_ang)
    # print(cov_quat[:,0])

    fig = plt.figure(1)
    # fig.suptitle("Mean of Quaternion")
    # fig.suptitle("Covariance of Quaternion")
    fig.suptitle("UKF and Vicon plots")
    # fig.suptitle("Mean of Angular velocity")
    # fig.suptitle("Covariance of Angular Velocity")
    plt.subplot(311)
    # plt.plot(-vicon_euler[:,1], 'b', roll, 'g')       # dataset 3
    plt.plot(vicon_euler[:, 0], 'b', roll, 'g')     # dataset 2
    # plt.plot(roll, 'r')
    # plt.plot(cov_quat[:,0],'y')
    # plt.plot(cov_ang[:,0],'y')
    # plt.plot(predicted_w[:,0],'g')
    plt.ylabel('Roll')
    plt.legend(["Vicon","UkF"], loc="upper right")
    plt.subplot(312)
    # plt.plot(-vicon_euler[:,0], 'b', pitch, 'g')     # dataset 3
    plt.plot(vicon_euler[:, 1], 'b', pitch, 'g')  # dataset 2
    # plt.plot(pitch, 'r')
    # plt.plot(cov_quat[:, 1], 'y')
    # plt.plot(cov_ang[:,1],'y')
    # plt.plot(predicted_w[:, 0],'g')
    plt.ylabel('Pitch')
    plt.legend(["Vicon","UkF"], loc="upper right")
    plt.subplot(313)
    # plt.plot(vicon_euler[:,2], 'b', yaw, 'g')      #dataset 3
    plt.plot(vicon_euler[:, 2], 'b', yaw, 'g')  # dataset 2
    # plt.plot(yaw, 'r')
    # plt.plot(cov_quat[:, 2], 'y')
    # plt.plot(cov_ang[:,2],'y')
    # plt.plot(predicted_w[:, 0],'g')
    plt.ylabel('Yaw')
    plt.legend(["Vicon","UkF"], loc="upper right")
    plt.show()
    # fig2 = plt.figure(2)
    # plt.plot(predicted_w[:, 0], 'r')
    # plt.plot(predicted_w[:, 1], 'b')
    # plt.plot(predicted_w[:, 2], 'y')
    # plt.show()



    return roll, pitch, yaw

a,b,c = estimate_rot(2)




