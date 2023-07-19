import math
import time
from collections import namedtuple
import matplotlib.pyplot as plt
import numpy as np
from filterpy.gh import GHFilter
from filterpy.kalman import predict, update
from numpy.random import randn
from kf_book.nonlinear_plots import plot_cov_ellipse_colormap
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import KalmanFilter
from kf_book.mkf_internal import plot_track
from filterpy.common import Saver
from math import sqrt
from kf_book.book_plots import plot_measurements
from scipy.linalg import block_diag
from filterpy.stats import plot_covariance_ellipse, plot_covariance
from kf_book.book_plots import plot_filter
import filterpy
from scipy.interpolate import splprep, splev, splrep
# from math import sin, cos, tan, atan2
from filterpy.kalman import ExtendedKalmanFilter as EKF
from numpy import array, sqrt
import sympy
from sympy.abc import alpha, x, y, v, w, R, theta
from sympy import Symbol, symbols, Matrix, sin, cos, tan, atan2












def TrainTrack():
    def plot_g_h_results_plt(measurements, filtered_data,
                             title='', z_label='Measurements',
                             interactive=False, **kwargs):
        plt.plot(filtered_data, **kwargs)
        plt.plot(measurements, label=z_label)
        plt.legend(loc=4)
        plt.title(title)
        plt.xlim(left=0, right=len(measurements))
        plt.grid(True)

        if interactive:
            for i in range(2, len(measurements)):
                plt.plot(filtered_data[:i], **kwargs)
                plt.plot(measurements[:i], label=z_label)
                plt.legend(loc=4)
                plt.title(title)
                plt.xlim(left=0, right=len(measurements))
                plt.grid(True)
                plt.draw()
                time.sleep(0.5)
        else:
            plt.show()

    def g_h_filter(data, x0, dx, g, h, dt=1.):
        x_est = x0
        results = []
        for z in data:
            # prediction step
            x_pred = x_est + (dx * dt)
            dx = dx

            # update step
            residual = z - x_pred
            dx = dx + h * (residual) / dt
            x_est = x_pred + g * residual
            results.append(x_est)
        return np.array(results)

    def gen_data(x0, dx, count, noise_factor, accel=0.):
        data = []
        for i in range(count):
            data.append(x0 + dx * i + (0.5) * (accel) * (i ** 2) + randn() * noise_factor)
            dx += accel
        return data

    def compute_new_position(pos, vel, dt=1.):
        """ dt is the time delta in seconds."""
        return pos + (vel * dt)

    def measure_position(pos):
        return pos + randn() * 500

    def gen_train_data(pos, vel, count):
        zs = []
        for t in range(count):
            pos = compute_new_position(pos, vel)
            zs.append(measure_position(pos))
        return np.asarray(zs)

    pos, vel = 23. * 1000, 15.
    zs = gen_train_data(pos, vel, 100)

    plt.plot(zs / 1000.)  # convert to km

    # Customize the plot
    plt.xlabel('Time')
    plt.ylabel('Position (km)')
    plt.title('Position vs. Time')
    plt.grid(True)

    # Show the plot
    plt.show()
    # book_plots.set_labels("Train pos", "time (s)", "km")

    zs = gen_train_data(pos=pos, vel=15., count=100)
    data = g_h_filter(data=zs, x0=pos, dx=15., dt=1., g=.01, h=0.0001)
    plot_g_h_results_plt(zs / 1000., data / 1000., 'g=0.01, h=0.0001')

    # Using Filterpy
    f = GHFilter(x=0., dx=0., dt=1., g=0.8, h=0.2)
    f.update(z=1.2)
    print(f.x, f.dx)

    print(f.update(z=2.1, g=0.85, h=0.15))

    print(f.batch_filter([3., 4., 5.]))

    # 3D example
    x_0 = np.array([1., 10., 100.])
    dx_0 = np.array([10., 12., .2])

    f_air = GHFilter(x=x_0, dx=dx_0, dt=1., g=.8, h=.2)
    f_air.update(z=np.array((2., 11., 102.)))
    print(' x =', f_air.x)
    print('dx =', f_air.dx)


def gaussian_track():
    # stats.plot_gaussian_pdf(mean=10., variance=1.,
    #                         xlim=(4, 16), ylim=(0, .5))
    # plt.show()

    # kf_internal.gaussian_vs_histogram()
    # plt.show()

    gaussian = namedtuple('Gaussian', ['mean', 'var'])
    gaussian.__repr__ = lambda s: f'𝒩(μ={s[0]:.3f}, 𝜎²={s[1]:.3f})'

    # g1 = gaussian(3.4, 10.1)
    # g2 = gaussian(mean=4.5, var=0.2 ** 2)
    # print(g1)
    # print(g2)
    #
    # print(g1.mean, g1[0], g1[1], g1.var)

    # def predict(pos, movement):
    #     return gaussian(pos.mean + movement.mean, pos.var + movement.var)

    # pos = gaussian(10., .2 ** 2)
    # move = gaussian(15., .7 ** 2)
    # print(predict(pos, move))

    def gaussian_multiply(g1, g2):
        mean = (g1.var * g2.mean + g2.var * g1.mean) / (g1.var + g2.var)
        variance = (g1.var * g2.var) / (g1.var + g2.var)
        return gaussian(mean, variance)

    #
    # def update(prior, likelihood):
    #     posterior = gaussian_multiply(likelihood, prior)
    #     return posterior

    # # test the update function
    # predicted_pos = gaussian(10., .2 ** 2)
    # measured_pos = gaussian(11., .1 ** 2)
    # estimated_pos = update(predicted_pos, measured_pos)
    # print(estimated_pos)
    #
    # np.random.seed(13)
    #
    # process_var = 1.
    # sensor_var = 2.
    # x = gaussian(0., 20.**2)
    # velocity = 1
    # dt = 1.
    # process_model = gaussian(velocity*dt, process_var)
    #
    # dog = DogSimulation(
    #     x0=x.mean,
    #     velocity=process_model.mean,
    #     measurement_var=sensor_var,
    #     process_var=process_model.var
    # )
    #
    # zs = [dog.move_and_sense() for _ in range(10)]
    #
    # print('\t\tPREDICT\t\t\t\tUPDATE')
    # print('     x      var\t\t  z\t    x      var')
    #
    # # perform Kalman filter on measurement z
    # for z in zs:
    #     prior = predict(x, process_model)
    #     likelihood = gaussian(z, sensor_var)
    #     x = update(prior, likelihood)
    #
    #     kf_internal.print_gh(prior, x, z)
    #
    # print()
    # print(f'final estimate:        {x.mean:10.3f}')
    # print(f'actual final position: {dog.x:10.3f}')
    #
    #
    # #save output
    # xs, prediction = [], []
    #
    # process_model = gaussian(velocity, process_var)

    # #kalman filter
    # x = gaussian(0., 20.**2)
    # for z in zs:
    #     previous = predict(x, process_model)
    #     confidence = gaussian(z, sensor_var)
    #     x = update(previous, confidence)
    #
    #     # save
    #     prediction.append(previous.mean)
    #     xs.append(x.mean)
    #
    # def plot_filter(step):
    #     plt.cla()
    #     step -= 1
    #     i = step // 3 + 1
    #
    #     book_plots.plot_predictions(prediction[:i])
    #     if step % 3 == 0:
    #         book_plots.plot_measurements(zs[:i - 1])
    #         book_plots.plot_filter(xs[:i - 1])
    #     elif step % 3 == 1:
    #         book_plots.plot_measurements(zs[:i])
    #         book_plots.plot_filter(xs[:i - 1])
    #     else:
    #         book_plots.plot_measurements(zs[:i])
    #         book_plots.plot_filter(xs[:i])
    #
    #     plt.xlim(-1, 10)
    #     plt.ylim(0, 20)
    #     plt.legend(loc=2)
    #     plt.show()

    # interact(plot_filter, step=IntSlider(value=1, min=30, max=len(prediction) * 3))
    # plt.show()
    #
    # process_var = 2.
    # sensor_var = 4.5
    # x = gaussian(0., 400.)
    # process_model = gaussian(1., process_var)
    # N = 25
    #
    # dog = DogSimulation(x.mean, process_model.mean, sensor_var, process_var)
    # zs = [dog.move_and_sense() for _ in range(N)]
    #
    # xs, prevs = np.zeros((N, 2)), np.zeros((N, 2))
    # for i, z in enumerate(zs):
    #     prev = predict(x, process_model)
    #     x = update(prev, gaussian(z, sensor_var))
    #     prevs[i] = prev
    #
    #     xs[i] = x
    #
    # book_plots.plot_measurements(zs)
    # book_plots.plot_filter(xs[:, 0], var=prevs[:, 1])
    # book_plots.plot_predictions(prevs[:, 0])
    # book_plots.show_legend()
    # kf_internal.print_variance(xs)
    # plt.show()

    def update(prior, measurement):
        x, P = prior  # mean and variance of prior
        z, R = measurement  # mean and variance of measurement

        y = z - x  # residual
        K = P / (P + R)  # Kalman gain

        x = x + K * y  # posterior
        P = (1 - K) * P  # posterior variance
        return gaussian(x, P)

    def predict(posterior, movement):
        x, P = posterior  # mean and variance of posterior
        dx, Q = movement  # mean and variance of movement
        x = x + dx
        P = P + Q
        return gaussian(x, P)
    #
    # xs = np.arange(145, 190, 0.1)
    # ys = [stats.gaussian(x, 160, 3 ** 2) for x in xs]
    # plt.plot(xs, ys, label='A', color='g')
    #
    # ys = [stats.gaussian(x, 170, 9 ** 2) for x in xs]
    # plt.plot(xs, ys, label='B', color='b')
    # plt.legend();
    # plt.errorbar(160, [0.04], xerr=[3], fmt='o', color='g', capthick=2, capsize=10)
    # plt.errorbar(170, [0.015], xerr=[9], fmt='o', color='b', capthick=2, capsize=10)
    # plt.show()
    #
    # xs = np.arange(145, 190, 0.1)
    # ys = [stats.gaussian(x, 160, 3 ** 2) for x in xs]
    # belief = np.array([random() for _ in range(40)])
    # belief = belief / sum(belief)
    #
    # x = np.linspace(155, 165, len(belief))
    # plt.gca().bar(x, belief, width=0.2)
    # plt.plot(xs, ys, label='A', color='g')
    # plt.errorbar(160, [0.04], xerr=[3], fmt='o', color='k', capthick=2, capsize=10)
    # plt.xlim(150, 170)
    # plt.show()
    #
    #
    # sensor_var = 30.
    # process_var = 2.
    # pos = gaussian(100., 500.)
    # process_model = gaussian(1., process_var)
    #
    # zs, ps = [], []
    #
    # for i in range(100):
    #     pos = predict(pos, process_model)
    #
    #     z = math.sin(i / 3.) * 2 + randn() * 1.2
    #     zs.append(z)
    #
    #     pos = update(pos, gaussian(z, sensor_var))
    #     ps.append(pos.mean)
    #
    # plt.plot(zs, c='r', linestyle='dashed', label='measurement')
    # plt.plot(ps, c='#004080', label='filter')
    # plt.legend(loc='best')
    # plt.show()

    # x, P = kf.predict(x=10., P=3., u=1., Q=2. ** 2)
    # print(f'{x:.3f}')
    #
    # x, P = kf.update(x=x, P=P, z=12., R=3.5 ** 2)
    # print(f'{x:.3f} {P:.3f}')


def multivariate_gaussians():
    # mean = [2., 17.]
    # cov = [[10., 0.],
    #        [0., 4.]]
    #
    # mkf_internal.plot_3d_covariance(mean, cov)
    # plt.show()
    #
    # x = [2.5, 7.3]
    # mu = [2.0, 7.0]
    # P = [[8.0, 0.],[
    #      0., 3.]]
    #
    # print(multivariate_gaussian(x, mu, P))
    # print(f'{multivariate_normal(mu, P).pdf(x):.4f}')

    # P = [[2, 0], [0, 6]]
    # plot_covariance_ellipse((2, 7), P, fc='g', alpha=0.2,
    #                         std=[1, 2, 3],
    #                         title='|2 0|\n|0 6|')
    # plt.gca().grid(b=False);
    # plt.show()

    plot_cov_ellipse_colormap(cov=[[2, 1.2], [1.2, 1.3]]);
    plt.show()


def multivariate_kalman_filter():
    def compute_dog_data(z_var, process_var, count=1, dt=1.):
        "returns track, measurements 1D ndarrays"
        x, vel = 0., 1.
        z_std = math.sqrt(z_var)
        p_std = math.sqrt(process_var)
        xs, zs = [], []
        for _ in range(count):
            v = vel + (randn() * p_std)
            x += v * dt
            xs.append(x)
            zs.append(x + randn() * z_std)
        return np.array(xs), np.array(zs)

    # x = np.array([10., 4.5])
    #
    # P = np.diag([500., 49.])
    #
    # dt = 0.1
    # F = np.array([[1, dt],
    #               [0, 1]])
    #
    # for i in range(4):
    #     x, P = predict(x=x, P=P, F=F, Q=0)
    #     # print('x =', x)
    #
    # # print(P)
    #
    # dt = 0.3
    # F = np.array([[1, dt], [0, 1]])
    # x = np.array([10.0, 4.5])
    # P = np.diag([500, 500])
    # plot_covariance_ellipse(x, P, edgecolor='r')
    # x, P = predict(x, P, F, Q=0)
    # plot_covariance_ellipse(x, P, edgecolor='k', ls='dashed')
    # # plt.show()
    #
    # Q = Q_discrete_white_noise(dim=2, dt=1., var=2.35)
    # # print(Q)
    #
    # B = 0.  # my dog doesn't listen to me!
    # u = 0
    # x, P = predict(x, P, F, Q, B, u)
    # # print('x =', x)
    # # print('P =', P)
    #
    # H = np.array([[1., 0.]])
    #
    # R = np.array([[5.]])
    #
    # z = 1.
    # x, P = update(x, P, z, R, H)
    # print('x = ', x)

    dog_filter = KalmanFilter(dim_x=2, dim_z=1)
    # print(f'x = {dog_filter.x.T}')
    # print(f'R = {dog_filter.R}')
    # print(f'Q = \n {dog_filter.Q}')
    def pos_vel_filter(x, P, R, Q=0., dt=1.0):
        """ Returns a KalmanFilter which implements a
        constant velocity model for a state [x dx].T
        """

        kf = KalmanFilter(dim_x=2, dim_z=1)
        kf.x = np.array([x[0], x[1]])  # location and velocity
        kf.F = np.array([[1., dt],
                         [0., 1.]])  # state transition matrix
        kf.H = np.array([[1., 0]])  # Measurement function
        kf.R *= R  # measurement uncertainty
        if np.isscalar(P):
            kf.P *= P  # covariance matrix
        else:
            kf.P[:] = P  # [:] makes deep copy
        if np.isscalar(Q):
            kf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=Q)
        else:
            kf.Q[:] = Q
        return kf

    # dt = 0.1
    # x = np.array([0., 0.])
    # kf = pos_vel_filter(x=x, P=500, R=5, Q=0.1, dt=dt)
    # print(kf)

    def run(x0=(0., 0.), P=500, R=0, Q=0, dt=1.0,
            track=None, zs=None,
            count=0, do_plot=True, **kwargs):
        """
        track is the actual position of the dog, zs are the
        corresponding measurements.
        """

        # Simulate dog if no data provided.
        if zs is None:
            track, zs = compute_dog_data(R, Q, count)

        # create the Kalman filter
        kf = pos_vel_filter(x0, R=R, P=P, Q=Q, dt=dt)

        # run the kalman filter and store the results
        xs, cov = [], []
        for z in zs:
            kf.predict()
            kf.update(z)
            xs.append(kf.x)
            cov.append(kf.P)

        xs, cov = np.array(xs), np.array(cov)
        if do_plot:
            plot_track(xs[:, 0], track, zs, cov, **kwargs)
            plt.show()
        return xs, cov

    # P = np.diag([500., 49.])
    # Ms, Ps = run(count=50, R=10, Q=0.01, P=P)
    #
    # kf = pos_vel_filter([0, .1], R=R, P=P, Q=Q, dt=1.)
    # s = Saver(kf)
    # for i in range(1, 6):
    #     kf.predict()
    #     kf.update([i])
    #     s.save()  # save the current state

def kalman_filter_design():
    # def calculate_heading(x1, y1, x2, y2):
    #     dx = x2 - x1
    #     dy = y2 - y1
    #     heading = np.arctan2(dy, dx)
    #     return np.degrees(heading)
    #
    #
    class PosSensor(object):

        def __init__(self, pos=(0, 0), vel=(0, 0), noise_std=1.):
            self.vel = vel
            self.noise_std = noise_std
            self.pos = [pos[0], pos[1]]
        def read(self):
            self.pos[0] += self.vel[0]
            self.pos[1] += self.vel[1]

            return [self.pos[0] + randn() * self.noise_std,
                    self.pos[1] + randn() * self.noise_std]

    # pos, vel = (4, 3), (2, 1)
    # sensor = PosSensor(pos, vel, noise_std=1)
    # ps = np.array([sensor.read() for i in range(50)])
    # plot_measurements(ps[:, 0], ps[:, 1])
    # plt.show()


    tracker = KalmanFilter(dim_x=4, dim_z=2)
    dt = 1.
    F = np.array([[1, dt, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, dt],
                  [0, 0, 0, 1]])

    q = Q_discrete_white_noise(dim=2, dt=dt, var=0.001)
    tracker.Q = block_diag(q, q)
    # print(tracker.Q)
    tracker.H = np.array([[1/0.3048, 0, 0, 0],
                          [0, 0, 1/0.3048, 0]])
    tracker.R = np.array([[5., 0.],
                          [0., 5.]])

    tracker.x = np.array([[0, 0, 0, 0]]).T
    tracker.P = np.eye(4)*500
    #
    R_std = 0.35
    Q_std = 0.04
    #
    def tracker1():
        tracker = KalmanFilter(dim_x=4, dim_z=2)
        dt = 1.0  # time step

        tracker.F = np.array([[1, dt, 0, 0],
                              [0, 1, 0, 0],
                              [0, 0, 1, dt],
                              [0, 0, 0, 1]])
        tracker.u = 0.
        tracker.H = np.array([[1 / 0.3048, 0, 0, 0],
                              [0, 0, 1 / 0.3048, 0]])

        tracker.R = np.eye(2) * R_std ** 2
        q = Q_discrete_white_noise(dim=2, dt=dt, var=Q_std ** 2)
        tracker.Q = block_diag(q, q)
        tracker.x = np.array([[0, 0, 0, 0]]).T
        tracker.P = np.eye(4) * 500.
        return tracker

    #simulate robot
    N = 30
    sensor = PosSensor((0, 0), (2, .2), noise_std=R_std)

    zs = np.array([sensor.read() for _ in range(N)])

    # run filter
    robot_tracker = tracker1()
    mu, cov, _, _ = robot_tracker.batch_filter(zs)

    for x, P in zip(mu, cov):
        # covariance of x and y
        cov = np.array([[P[0, 0], P[2, 0]],
                        [P[0, 2], P[2, 2]]])
        mean = (x[0, 0], x[2, 0])
        plot_covariance_ellipse(mean, cov=cov, fc='g', std=3, alpha=0.5)

    # plot results
    zs *= .3048  # convert to meters
    plot_filter(mu[:, 0], mu[:, 2])
    plot_measurements(zs[:, 0], zs[:, 1])
    plt.legend(loc=2)
    plt.xlim(0, 20)
    plt.show()


    # # Ball in circle
    #
    # R_std2 = 0.3  # Process uncertainty/noise
    # Q_std2 = 0.25  # measurement uncertainty/noise
    #
    # def tracker2():
    #     tracker = KalmanFilter(dim_x=4, dim_z=2)
    #     dt = 1.0  # time step
    #
    #     tracker.F = np.array([[1, dt, 0,  0],   # x = x + vx*dt
    #                           [0,  1, 0,  0],   # vx = vx
    #                           [0,  0, 1, dt],   # y = y + vy*dt
    #                           [0,  0, 0,  1]])  # vy = vy
    #     tracker.u = 0.  # Control input (No control in here)
    #     tracker.H = np.array([[1., 0,  0, 0],   # x = x
    #                           [0,  0, 1., 0]])  # y = y
    #
    #     # tracker.R = np.eye(2) * R_std2 ** 2
    #     tracker.R = np.array([[R_std2**2, 0.],
    #                           [0., R_std2**2]])  # Measurement uncertainty noise
    #     # q = Q_discrete_white_noise(dim=2, dt=dt, var=Q_std2 ** 2)
    #     q = np.array([[(Q_std2**2)/4, (Q_std2**2)/2],
    #                   [(Q_std2**2)/2, (Q_std2**2)]])  # White noise for model uncertainty
    #     tracker.Q = block_diag(q, q)
    #     tracker.x = np.array([[0, 0, 0, 0]]).T
    #     # tracker.P = np.eye(4) * 2.
    #     tracker.P = np.array([[2., 0,  0,  0],
    #                           [0,  2., 0,  0],
    #                           [0,  0,  2., 0],
    #                           [0,  0,  0, 2.]])  # covariance matrix
    #     return tracker
    #
    # # Parameters
    # radius = 50  # Radius of the circle
    # # speed = 10  # Angular speed (radians per frame)
    # num_frames = 25  # Number of frames to simulate
    #
    # # Generate time values
    # t = np.linspace(0, 2 * np.pi, num_frames)
    #
    # # Compute ball positions
    # x2 = radius * np.cos(t)
    # y2 = radius * np.sin(t)
    #
    # # Create a 2D array to store ball positions
    # # positions = np.column_stack((x2, y2))
    #
    # # Plot the ball's path
    # plt.plot(x2, y2, color='red')
    # plt.axis('equal')
    # plt.title("Ball's Path")
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # # plt.show()
    #
    # noise_amnt = 2
    #
    # # Generate time values
    # t2 = np.linspace(0, 2 * np.pi, num_frames)
    #
    # # Compute ball positions
    # x3 = radius * np.cos(t2) + noise_amnt*randn()
    # y3 = radius * np.sin(t2) + noise_amnt*randn()
    #
    # # Create a 2D array to store ball positions
    # measurement = np.column_stack((x3, y3))
    #
    # # run filter
    # ball_tracker = tracker2()
    #
    # mu2, cov2, _, _ = ball_tracker.batch_filter(measurement)
    #
    # for xB, PB in zip(mu2, cov2):
    #     # covariance of x and y
    #     cov = np.array([[PB[0, 0], PB[2, 0]],
    #                     [PB[0, 2], PB[2, 2]]])
    #     mean = (xB[0, 0], xB[2, 0])
    #     plot_covariance(mean, cov=cov, fc='g', std=3, alpha=0.5)
    #
    # # plot results
    # plot_filter(mu2[:, 0], mu2[:, 2])
    # plot_measurements(measurement[:, 0], measurement[:, 1])
    # plt.legend(loc=2)
    # plt.xlim(0, 20)
    # plt.show()

def ball_track():

    R_var = 0.8 ** 2  # Process uncertainty/noise
    Q_var = 0.05 ** 2  # measurement uncertainty/noise


    # Using a constant velocity model, therefore it struggles since the object (circle) is not
    # moving at a constant velocity
    def tracker1():
        tracker = KalmanFilter(dim_x=4, dim_z=2)
        dt = 1.0  # time step
                             # x  dx  y  dy
        tracker.F = np.array([[1, dt, 0, 0],  # x = x + vx*dt
                              [0, 1, 0, 0],   # vx = vx
                              [0, 0, 1, dt],  # y = y + vy*dt
                              [0, 0, 0, 1]])  # vy = vy

        tracker.H = np.array([[1., 0, 0, 0],   # x = x
                              [0, 0, 1., 0]])  # y = y

        tracker.R = np.array([[R_var, 0.],
                              [0., R_var]])  # Measurement uncertainty noise

        q = np.array([[Q_var/4, Q_var/2],
                      [Q_var/2, Q_var]])  # White noise for model uncertainty
        tracker.Q = block_diag(q, q)

        tracker.x = np.array([[0, 0, 0, 0]]).T  # State of object

        tracker.P = np.array([[2., 0, 0, 0],
                              [0, 2., 0, 0],
                              [0, 0, 2., 0],
                              [0, 0, 0, 2.]])  # covariance matrix
        return tracker

    # Ball generation
    radius = 10  # Radius of the circle
    num_frames = 50  # Number of frames to simulate

    # Generate time values
    t = np.linspace(0, 2 * np.pi, num_frames)

    # Compute ball positions
    x_true_pos = radius * np.cos(t)
    y_true_pos = radius * np.sin(t)

    # Plot the ball's true path
    plt.plot(x_true_pos, y_true_pos, color='red')
    plt.axis('equal')
    plt.title("True Path")
    plt.xlabel('X')
    plt.ylabel('Y')


    # Measured position

    input_sensor_noise = radius/6/1.75  # Add noise to measurement

    # Initialize an array to hold sensor noise values
    sensor_noise = np.random.randn(num_frames) * input_sensor_noise  # Generate random noise values

    # Calculate measured_x_pos with different sensor noise for each value
    # Compute ball positions
    measured_x_pos = radius * np.cos(t) + sensor_noise
    measured_y_pos = radius * np.sin(t) + sensor_noise

    # Create a 2D array to store ball positions
    measurement = np.column_stack((measured_x_pos, measured_y_pos))

    # run filter
    ball_tracker = tracker1()

    # Initialize means and covariances arrays
    means, covariances = np.zeros((num_frames, 4, 1)), np.zeros((num_frames, 4, 4))

    # Run through measurements
    for i, z in enumerate(measurement):
        ball_tracker.predict()
        ball_tracker.update(z)

        means[i, :] = ball_tracker.x
        covariances[i, :, :] = ball_tracker.P

    for x, P in zip(means, covariances):

        # covariance of x and y
        cov = np.array([[P[0, 0], P[2, 0]],
                        [P[0, 2], P[2, 2]]])
        mean = (x[0, 0], x[2, 0])
        plot_covariance(mean, cov=cov, fc='g', std=3, alpha=0.5)
        # plt.show()

    # plot results
    plot_filter(means[:, 0], means[:, 2])
    plot_measurements(measurement[:, 0], measurement[:, 1])
    plt.legend(loc=0)
    plt.show()



    # Straight line generation

    R_var = 0.7 ** 2  # Process uncertainty/noise
    Q_var = 0.03 ** 2  # measurement uncertainty/noise

    def tracker2():
        tracker = KalmanFilter(dim_x=4, dim_z=2)
        dt = 1.0  # time step
                             # x  dx  y  dy
        tracker.F = np.array([[1, dt, 0, 0],  # x = x + vx*dt
                              [0, 1, 0, 0],   # vx = vx
                              [0, 0, 1, dt],  # y = y + vy*dt
                              [0, 0, 0, 1]])  # vy = vy

        tracker.H = np.array([[1., 0, 0, 0],   # x = x
                              [0, 0, 1., 0]])  # y = y

        tracker.R = np.array([[R_var, 0.],
                              [0., R_var]])  # Measurement uncertainty noise

        q = np.array([[Q_var/4, Q_var/2],
                      [Q_var/2, Q_var]])  # White noise for model uncertainty
        tracker.Q = block_diag(q, q)

        tracker.x = np.array([[0, 0, 0, 0]]).T  # State of object

                            #  x   dx y  dy
        tracker.P = np.array([[2., 0, 0, 0],
                              [0, 2., 0, 0],
                              [0, 0, 2., 0],
                              [0, 0, 0, 2.]])  # covariance matrix
        return tracker



    length = 100  # length of the Track
    num_frames = int(length/5)  # Number of frames to simulate

    # Generate time values
    t = np.linspace(0, 1, num_frames)

    true_x_pos = length*t
    true_y_pos = length*t

    # Plot x and y positions
    plt.plot(true_x_pos, true_y_pos, color='red')
    plt.xlabel('X position')
    plt.ylabel('Y position')
    plt.title('Ball Movement')

    # Measured position
    input_sensor_noise = math.sqrt(length)/3  # Add noise to measurement

    # Initialize an array to hold sensor noise values
    sensor_noise = np.random.randn(num_frames) * input_sensor_noise  # Generate random noise values

    # Calculate measured_x_pos with different sensor noise for each value
    # Compute ball positions
    measured_x_pos = true_x_pos + sensor_noise
    measured_y_pos = true_y_pos - sensor_noise

    # Create a 2D array to store ball positions
    measurement = np.column_stack((measured_x_pos, measured_y_pos))

    # run filter
    line_tracker = tracker2()

    # Initialize means and covariances arrays
    means, covariances = np.zeros((num_frames, 4, 1)), np.zeros((num_frames, 4, 4))

    # Run through measurements
    for i, z in enumerate(measurement):
        line_tracker.predict()
        line_tracker.update(z)

        means[i, :] = line_tracker.x
        covariances[i, :, :] = line_tracker.P

    for x, P in zip(means, covariances):
        # covariance of x and y
        cov = np.array([[P[0, 0], P[2, 0]],
                        [P[0, 2], P[2, 2]]])
        mean = (x[0, 0], x[2, 0])
        plot_covariance(mean, cov=cov, fc='g', std=3, alpha=0.5)
        # plt.show()

    # plot results
    plot_filter(means[:, 0], means[:, 2])
    plot_measurements(measurement[:, 0], measurement[:, 1])
    plt.legend(loc=0)
    plt.show()






    # Adding in Constant Acceleration

    R_var = 9.0 ** 2  # measurement uncertainty/noise
    Q_var = 0.5 ** 2  # Process uncertainty/noise

    # Using a constant velocity model, therefore it struggles since the object (circle) is not
    # moving at a constant velocity
    def tracker3():
        tracker = KalmanFilter(dim_x=6, dim_z=2)
        dt = 1.0  # time step
#                              x  dx            ax,  y  dy              ay
        tracker.F = np.array([[1, dt, (0.5)*(dt**2), 0, 0,              0],
                              [0,  1,            dt, 0, 0,              0],
                              [0,  0,             1, 0, 0,              0],
                              [0,  0,             0, 1, dt, (0.5)*(dt**2)],
                              [0,  0,             0, 0, 1,             dt],
                              [0,  0,             0, 0, 0,              1]])

        tracker.H = np.array([[1., 0, 0,  0, 0, 0],  # x = x
                              [0,  0, 0, 1., 0, 0]])  # y = y

        q = np.array([[Q_var/16, Q_var/8, Q_var/4],
                      [Q_var/8, Q_var/4, Q_var/2],
                      [Q_var/4, Q_var/2, Q_var]])
        tracker.Q = block_diag(q, q)

        tracker.R *= R_var

        tracker.x = np.array([[0, 0, 0, 0, 0, 0]]).T  # State of object

        tracker.P *= 5.

        return tracker

    # Ball generation
    radius = 500  # Radius of the circle
    num_frames = int(radius/5)  # Number of frames to simulate

    # Generate time values
    t = np.linspace(0, 2 * np.pi, num_frames)

    # Compute ball positions
    x_true_pos = radius * np.cos(t)
    y_true_pos = radius * np.sin(t)

    # Plot the ball's true path
    plt.plot(x_true_pos, y_true_pos, color='red')
    plt.axis('equal')
    plt.title("True Path")
    plt.xlabel('X')
    plt.ylabel('Y')

    # Measured position

    input_sensor_noise = radius / 6 / 1.75  # Add noise to measurement
    # input_sensor_noise = 0.0

    # Initialize an array to hold sensor noise values
    sensor_noise = np.random.randn(num_frames) * input_sensor_noise  # Generate random noise values

    # Calculate measured_x_pos with different sensor noise for each value
    # Compute ball positions
    measured_x_pos = radius * np.cos(t) + sensor_noise
    measured_y_pos = radius * np.sin(t) + sensor_noise

    # Create a 2D array to store ball positions
    measurement = np.column_stack((measured_x_pos, measured_y_pos))

    # run filter
    ball_tracker = tracker3()

    # Initialize means and covariances arrays
    means, covariances = np.zeros((num_frames, 6, 1)), np.zeros((num_frames, 6, 6))

    # Run through measurements
    for i, z in enumerate(measurement):
        ball_tracker.predict()
        ball_tracker.update(z)

        means[i, :] = ball_tracker.x
        covariances[i, :, :] = ball_tracker.P

    for x, P in zip(means, covariances):
        # covariance of x and y
        cov = np.array([[P[0, 0], P[3, 0]],
                        [P[0, 3], P[3, 3]]])
        mean = (x[0, 0], x[3, 0])
        plot_covariance(mean, cov=cov, fc='g', std=3, alpha=0.5)

    # plot results
    plot_filter(means[:, 0], means[:, 3])
    plot_measurements(measurement[:, 0], measurement[:, 1])
    plt.legend(loc=0)
    plt.show()


def constant_steering_constant_velocity():
    # Adding in Constant Acceleration



    # Using a constant velocity model, therefore it struggles since the object (circle) is not
    # moving at a constant velocity
    def tracker1(R_var, Q_var):
        tracker = KalmanFilter(dim_x=6, dim_z=2)
        dt = 1.0  # time step
        #                              x  dx            ax,  y  dy              ay
        tracker.F = np.array([[1, dt, (0.5) * (dt ** 2), 0,  0,                0],
                              [0,  1,                dt, 0,  0,                0],
                              [0,  0,                 1, 0,  0,                0],
                              [0,  0,                 0, 1, dt, (0.5) * (dt ** 2)],
                              [0,  0,                 0, 0,  1,                dt],
                              [0,  0,                 0, 0,  0,                 1]])

        tracker.H = np.array([[1.0, 0, 0, 0, 0, 0],  # x = x
                              [0, 0, 0, 1.0, 0, 0]])  # y = y

        q = np.array([[Q_var / 16, Q_var / 8, Q_var / 4],
                      [Q_var / 8, Q_var / 4, Q_var / 2],
                      [Q_var / 4, Q_var / 2, Q_var]])
        tracker.Q = block_diag(q, q)

        tracker.R *= R_var

        tracker.x = np.array([[0, 0, 0, 0, 0, 0]]).T  # State of object

        tracker.P *= 5.

        return tracker

    def tracker2(R_var, Q_var):
        tracker = KalmanFilter(dim_x=6, dim_z=2)
        dt = 1.0  # time step

        x = 1
        vx = 1
        ax = 0
        y = 0
        vy = 0
        ay = 0
        yaw = 1
        heading = yaw*dt
        c1 = (sin(heading))/heading
        c2 = (cos(heading) - 1)/heading
        c3 = cos(heading)
        c4 = sin(heading)
        d1 = (heading*c3 - c4)/(yaw**2)
        d2 = (dt*c4 - c2)/yaw
        d3 = (yaw*(dt**2)*c3 - 2*dt*c4 - 2*c2)/(yaw**2)
        d4 = ((2-(heading**2))*c4 - 2*heading*c3)/(yaw**3)
        d5 = -d2
        d6 = d1
        d7 = -d4
        d8 = d3
        f13 = (dt*(c2 + c4))/yaw
        f43 = (c4 - heading*c3)/(yaw**2)
        f16 = -f43
        f46 = f13
        f17 = vx*d1+vy*d2+ax*d3+ay*d4
        f47 = vx*d5+vy*d6*ax*d7+ay*d8
        f27 = -dt*(ax*c3 - ay*c4)
        f57 = dt*(vx*c3)
        # Got stuck afterwards, see here for reference: https://diglib.tugraz.at/download.php?id=576a84b9336f5&location=browse




        #                      x  vx                 ax, y  vy                  ay
        tracker.F = np.array([[1, dt, (0.5) * (dt ** 2), 0,  0,                 0],
                              [0, 1,                 dt, 0,  0,                 0],
                              [0, 0,                  1, 0,  0,                 0],
                              [0, 0,                  0, 1, dt, (0.5) * (dt ** 2)],
                              [0, 0,                  0, 0,  1,                dt],
                              [0, 0,                  0, 0,  0,                 1]])

        tracker.H = np.array([[1., 0, 0, 0, 0, 0],  # x = x
                              [0, 0, 0, 1., 0, 0]])  # y = y

        q = np.array([[Q_var / 16, Q_var / 8, Q_var / 4],
                      [Q_var / 8, Q_var / 4, Q_var / 2],
                      [Q_var / 4, Q_var / 2, Q_var]])
        tracker.Q = block_diag(q, q)

        tracker.R *= R_var

        tracker.x = np.array([[0, 0, 0, 0, 0, 0, 0]]).T  # State of object

        tracker.P *= 5.

        return tracker

    def tracker3(R_var, Q_var):
        # CTRV model
        # State vector is [x, y, v, theta, acceleration, yaw]
        tracker = KalmanFilter(dim_x=6, dim_z=2)
        dt = 1.0  # time step

        x = tracker.x[0]
        y = tracker.x[1]
        v = 1
        theta = tracker.x[3]
        a = tracker.x[4]
        yaw = 1
        heading = yaw*dt

        a1 = (sin(heading + theta) - sin(theta)) / yaw
        a2 = (v * (cos(heading + theta) - cos(theta))) / yaw
        a3 = (v * ((heading * cos(heading + theta)) - (sin(heading + theta)) + sin(theta))) / (yaw**2)
        a4 = (cos(theta) - cos(heading + theta)) / yaw
        a5 = (v * (sin(heading + theta) - sin(theta))) / yaw
        a6 = (v * ((heading * sin(heading + theta)) + (cos(heading + theta)) - cos(theta))) / (yaw**2)


        #                      x  y            ax,  y  dy              ay
        tracker.F = np.array([[1, 0, a1, a2, 0, a3],
                              [0, 1, a4, a5, 0, a6],
                              [0, 0,  1,  0, 0,  0],
                              [0, 0,  0,  1, 0, dt],
                              [0, 0,  0,  0, 0,  0],
                              [0, 0,  0,  0, 0,  1]])

        tracker.H = np.array([[1.0,   0, 0, 0, 0, 0],  # x = x
                              [  0, 1.0, 0, 0, 0, 0]])  # y = y

        q = np.array([[Q_var / 16, Q_var / 8, Q_var / 4],
                      [Q_var / 8, Q_var / 4, Q_var / 2],
                      [Q_var / 4, Q_var / 2, Q_var]])
        tracker.Q = block_diag(q, q)

        tracker.R *= R_var

        # tracker.x = np.array([[0, 0, 0, 0, 0, 0]]).T  # State of object

        tracker.P *= 5.

        return tracker

    class ekf_tracker(EKF):
        def __init__(self, dt, wheelbase, std_vel, std_steer):
            EKF.__init__(self, dim_x=3, dim_z=3,dim_u=2)
            self.dt = dt
            self.wheelbase = wheelbase
            self.std_vel = std_vel
            self.std_steer = std_steer

            a, x, y, v, w, theta, time = symbols('a, x, y, v, w, theta, t')
            d = v*time
            beta = (d/w)*sympy.tan(a)
            r = w/sympy.tan(a)

            self.fxu = Matrix([
                [x-r*sin(theta)+r*sin(theta+beta)],
                [y+r*cos(theta)-r*cos(theta+beta)],
                [theta+beta]
            ])
            self.F_j = self.fxu.jacobian(Matrix([x, y, theta]))
            self.V_J = self.fxu.jacobian(Matrix([v, a]))

            self.subs = {x: 0, y: 0, v: 0, a: 0, time: dt, w: wheelbase, theta: 0}
            self.x_x, self.x_y = x, y
            self.v, self.a, self.theta = v, a, theta

            def predict(self, u):
                self.x = self.move(self.x, u, self.dt)
                self.subs[self.x_x] = self.x[0, 0]
                self.subs[self.x_y] = self.x[1, 0]

                self.subs[self.theta] = self.x[2, 0]
                self.subs[self.v] = u[0]
                self.subs[self.a] = u[1]

                F = array(self.F_j.evalf(subs=self.subs)).astype(float)
                V = array(self.V_j.evalf(subs=self.subs)).astype(float)

                # covariance of motion noise
                M = array([[self.std_vel**2, 0],
                           [0, self.std_steer**2]])

                self.P = F @ self.P @ F.T + V @ M @ V.T

            def move(self, x, u, dt):
                hdg = x[2, 0]
                vel = u[0]
                steering_angle = u[1]
                dist = vel * dt

                if abs(steering_angle) > 0.001:  # is robot turning?
                    beta = (dist / self.wheelbase) * tan(steering_angle)
                    r = self.wheelbase / tan(steering_angle)  # radius

                    dx = np.array([[-r * sin(hdg) + r * sin(hdg + beta)],
                                   [r * cos(hdg) - r * cos(hdg + beta)],
                                   [beta]])
                else:  # moving in straight line
                    dx = np.array([[dist * cos(hdg)],
                                   [dist * sin(hdg)],
                                   [0]])
                return x + dx

            def residual(a, b):
                """ compute residual (a-b) between measurements containing
                [range, bearing]. Bearing is normalized to [-pi, pi)"""
                y = a - b
                y[1] = y[1] % (2 * np.pi)  # force in range [0, 2 pi)
                if y[1] > np.pi:  # move to [-pi, pi)
                    y[1] -= 2 * np.pi
                return y

    class ekf_tracker2():
        #  [x, y, v, yaw/heading, yaw rate]
        print('f')

    # Track Generation

    # Predefined points that define the shape of the race track
    track_points = np.array([[0, 0], [0.2, 0.3], [0.4, 0.35], [0.5, 0.5], [0.8, 0.3],
                             [0.7, 0.1], [0.8, 0], [0.8, -0.3], [0.6, -0.4], [0.4, -0.3],
                             [0.2, -0.05], [0, 0]]).T
    points = track_points

    points *= 10000
    # Interpolate a smooth curve using splines
    tck, u = splprep(points, s=0.01, per=1)

    num_frames = 50

    # Define the resolution of the curve
    u_new = np.linspace(u.min(), u.max(), num=num_frames)

    # Evaluate the curve at the given resolution
    true_x, true_y = splev(u_new, tck)

    plt.plot(true_x, true_y, color='Red')
    plt.axis('equal')
    plt.title("Path")
    plt.xlabel('X')
    plt.ylabel('Y')

    # plt.show()


    # Measured position


    input_sensor_noise = 250  # Add noise to measurement
    # input_sensor_noise = 0.0

    # Initialize an array to hold sensor noise values
    sensor_noise = np.random.randn(num_frames) * input_sensor_noise  # Generate random noise values

    # Calculate measured_x_pos with different sensor noise for each value
    # Compute ball positions
    measured_x_pos = true_x + sensor_noise
    measured_y_pos = true_y - sensor_noise

    # Create a 2D array to store ball positions
    measurement = np.column_stack((measured_x_pos, measured_y_pos))

    # run filter
    R_var = 5.0 ** 2  # measurement uncertainty/noise
    Q_var = 2.0 ** 2  # Process uncertainty/noise
    ball_tracker = tracker1(R_var=R_var, Q_var=Q_var)

    # Initialize means and covariances arrays
    means, covariances = np.zeros((num_frames, 6, 1)), np.zeros((num_frames, 6, 6))

    # Run through measurements
    for i, z in enumerate(measurement):
        ball_tracker.predict()
        ball_tracker.update(z)

        means[i, :] = ball_tracker.x
        covariances[i, :, :] = ball_tracker.P

    for x, P in zip(means, covariances):
        # covariance of x and y
        cov = np.array([[P[0, 0], P[3, 0]],
                        [P[0, 3], P[3, 3]]])
        mean = (x[0, 0], x[3, 0])
        plot_covariance(mean, cov=cov, fc='g', std=3, alpha=0.5)

    # plot results
    plot_filter(means[:, 0], means[:, 3])
    plot_measurements(measurement[:, 0], measurement[:, 1])
    plt.legend(loc=0)
    plt.show()


def extended_kalman_filter_robot():

    time = symbols('t')
    d = v * time
    beta = (d / w) * sympy.tan(alpha)
    r = w / sympy.tan(alpha)

    fxu = Matrix([[x - r * sympy.sin(theta) + r * sympy.sin(theta + beta)],
                  [y + r * sympy.cos(theta) - r * sympy.cos(theta + beta)],
                  [theta + beta]])
    F = fxu.jacobian(Matrix([x, y, theta]))

    # reduce common expressions
    B, R = symbols('beta, R')
    F = F.subs((d / w) * sympy.tan(alpha), B)
    F.subs(w / sympy.tan(alpha), R)

    V = fxu.jacobian(Matrix([v, alpha]))
    V = V.subs(sympy.tan(alpha) / w, 1 / R)
    V = V.subs(time * v / R, B)
    V = V.subs(time * v, 'd')

    px, py = symbols('p_x, p_y')
    z = Matrix([[sympy.sqrt((px - x) ** 2 + (py - y) ** 2)],
                [sympy.atan2(py - y, px - x) - theta]])
    z.jacobian(Matrix([x, y, theta]))

    def H_of(x, landmark_pos):
        """ compute Jacobian of H matrix where h(x) computes
        the range and bearing to a landmark for state x """

        px = landmark_pos[0]
        py = landmark_pos[1]
        hyp = (px - x[0, 0]) ** 2 + (py - x[1, 0]) ** 2
        dist = sqrt(hyp)

        H = array(
            [[-(px - x[0, 0]) / dist, -(py - x[1, 0]) / dist, 0],
             [(py - x[1, 0]) / hyp, -(px - x[0, 0]) / hyp, -1]])
        return H

    def Hx(x, landmark_pos):
        """ takes a state variable and returns the measurement
        that would correspond to that state.
        """
        px = landmark_pos[0]
        py = landmark_pos[1]
        dist = sqrt((px - x[0, 0]) ** 2 + (py - x[1, 0]) ** 2)

        Hx = array([[dist],
                    [atan2(py - x[1, 0], px - x[0, 0]) - x[2, 0]]])
        return Hx

    class RobotEKF(EKF):
        def __init__(self, dt, wheelbase, std_vel, std_steer):
            EKF.__init__(self, 3, 2, 2)
            self.dt = dt
            self.wheelbase = wheelbase
            self.std_vel = std_vel
            self.std_steer = std_steer

            a, x, y, v, w, theta, time = symbols(
                'a, x, y, v, w, theta, t')
            d = v * time
            beta = (d / w) * sympy.tan(a)
            r = w / sympy.tan(a)

            self.fxu = Matrix(
                [[x - r * sympy.sin(theta) + r * sympy.sin(theta + beta)],
                 [y + r * sympy.cos(theta) - r * sympy.cos(theta + beta)],
                 [theta + beta]])

            self.F_j = self.fxu.jacobian(Matrix([x, y, theta]))
            self.V_j = self.fxu.jacobian(Matrix([v, a]))

            # save dictionary and it's variables for later use
            self.subs = {x: 0, y: 0, v: 0, a: 0,
                         time: dt, w: wheelbase, theta: 0}
            self.x_x, self.x_y, = x, y
            self.v, self.a, self.theta = v, a, theta

        def predict(self, u):
            self.x = self.move(self.x, u, self.dt)
            self.subs[self.x_x] = self.x[0, 0]
            self.subs[self.x_y] = self.x[1, 0]

            self.subs[self.theta] = self.x[2, 0]
            self.subs[self.v] = u[0]
            self.subs[self.a] = u[1]

            F = array(self.F_j.evalf(subs=self.subs)).astype(float)
            V = array(self.V_j.evalf(subs=self.subs)).astype(float)

            # covariance of motion noise in control space
            M = array([[self.std_vel ** 2, 0],
                       [0, self.std_steer ** 2]])

            self.P = F @ self.P @ F.T + V @ M @ V.T

        def move(self, x, u, dt):
            hdg = x[2, 0]
            vel = u[0]
            steering_angle = u[1]
            dist = vel * dt

            if abs(steering_angle) > 0.001:  # is robot turning?
                beta = (dist / self.wheelbase) * tan(steering_angle)
                r = self.wheelbase / tan(steering_angle)  # radius

                dx = np.array([[-r * sin(hdg) + r * sin(hdg + beta)],
                               [r * cos(hdg) - r * cos(hdg + beta)],
                               [beta]])
            else:  # moving in straight line
                dx = np.array([[dist * cos(hdg)],
                               [dist * sin(hdg)],
                               [0]])
            return x + dx

    def residual(a, b):
        """ compute residual (a-b) between measurements containing
            [range, bearing]. Bearing is normalized to [-pi, pi)"""
        y = a - b
        y[1] = y[1] % (2 * np.pi)  # force in range [0, 2 pi)
        if y[1] > np.pi:  # move to [-pi, pi)
            y[1] -= 2 * np.pi
        return y

    dt = 1.0

    def z_landmark(lmark, sim_pos, std_rng, std_brg):
        x, y = sim_pos[0, 0], sim_pos[1, 0]
        d = np.sqrt((lmark[0] - x) ** 2 + (lmark[1] - y) ** 2)
        a = atan2(lmark[1] - y, lmark[0] - x) - sim_pos[2, 0]
        z = np.array([[d + randn() * std_rng],
                      [a + randn() * std_brg]])
        return z

    def ekf_update(ekf, z, landmark):
        ekf.update(z, HJacobian=H_of, Hx=Hx,
                   residual=residual,
                   args=(landmark), hx_args=(landmark))

    def run_localization(landmarks, std_vel, std_steer,
                         std_range, std_bearing,
                         step=10, ellipse_step=20, ylim=None):
        ekf = RobotEKF(dt, wheelbase=0.5, std_vel=std_vel,
                       std_steer=std_steer)
        ekf.x = array([[2, 6, .3]]).T  # x, y, steer angle
        ekf.P = np.diag([.1, .1, .1])
        ekf.R = np.diag([std_range ** 2, std_bearing ** 2])

        sim_pos = ekf.x.copy()  # simulated position
        # steering command (vel, steering angle radians)
        u = array([1.1, .01])

        plt.figure()
        plt.scatter(landmarks[:, 0], landmarks[:, 1],
                    marker='s', s=60)

        track = []
        for i in range(200):
            sim_pos = ekf.move(sim_pos, u, dt / 10.)  # simulate robot
            track.append(sim_pos)

            if i % step == 0:
                ekf.predict(u=u)

                if i % ellipse_step == 0:
                    plot_covariance_ellipse(
                        (ekf.x[0, 0], ekf.x[1, 0]), ekf.P[0:2, 0:2],
                        std=6, facecolor='k', alpha=0.3)

                x, y = sim_pos[0, 0], sim_pos[1, 0]
                for lmark in landmarks:
                    z = z_landmark(lmark, sim_pos,
                                   std_range, std_bearing)
                    ekf_update(ekf, z, lmark)

                if i % ellipse_step == 0:
                    plot_covariance_ellipse(
                        (ekf.x[0, 0], ekf.x[1, 0]), ekf.P[0:2, 0:2],
                        std=6, facecolor='g', alpha=0.8)
        track = np.array(track)
        plt.plot(track[:, 0], track[:, 1], color='k', lw=2)
        plt.axis('equal')
        plt.title("EKF Robot localization")
        if ylim is not None: plt.ylim(*ylim)
        plt.show()
        return ekf

    landmarks = array([[5, 10], [10, 5], [15, 15]])

    ekf = run_localization(
        landmarks, std_vel=0.1, std_steer=np.radians(1),
        std_range=0.3, std_bearing=0.1)
    print('Final P:', ekf.P.diagonal())

    landmarks = array([[5, 10], [10, 5], [15, 15], [20, 5]])

    ekf = run_localization(
        landmarks, std_vel=0.1, std_steer=np.radians(1),
        std_range=0.3, std_bearing=0.1)
    plt.show()
    print('Final P:', ekf.P.diagonal())

    ekf = run_localization(
        landmarks[0:2], std_vel=1.e-10, std_steer=1.e-10,
        std_range=1.4, std_bearing=.05)
    print('Final P:', ekf.P.diagonal())

    ekf = run_localization(
        landmarks[0:1], std_vel=1.e-10, std_steer=1.e-10,
        std_range=1.4, std_bearing=.05)
    print('Final P:', ekf.P.diagonal())

    landmarks = array([[5, 10], [10, 5], [15, 15], [20, 5], [15, 10],
                       [10, 14], [23, 14], [25, 20], [10, 20]])

    ekf = run_localization(
        landmarks, std_vel=0.1, std_steer=np.radians(1),
        std_range=0.3, std_bearing=0.1, ylim=(0, 21))
    print('Final P:', ekf.P.diagonal())


    # class ekf_tracker2():
    #     #  [x, y, v, yaw/heading, yaw rate]
    #     print('f')
    #
    # # Track Generation
    #
    # # Predefined points that define the shape of the race track
    # track_points = np.array([[0, 0], [0.2, 0.3], [0.4, 0.35], [0.5, 0.5], [0.8, 0.3],
    #                          [0.7, 0.1], [0.8, 0], [0.8, -0.3], [0.6, -0.4], [0.4, -0.3],
    #                          [0.2, -0.05], [0, 0]]).T
    # points = track_points
    #
    # points *= 10000
    # # Interpolate a smooth curve using splines
    # tck, u = splprep(points, s=0.01, per=1)
    #
    # num_frames = 50
    #
    # # Define the resolution of the curve
    # u_new = np.linspace(u.min(), u.max(), num=num_frames)
    #
    # # Evaluate the curve at the given resolution
    # true_x, true_y = splev(u_new, tck)
    #
    # plt.plot(true_x, true_y, color='Red')
    # plt.axis('equal')
    # plt.title("Path")
    # plt.xlabel('X')
    # plt.ylabel('Y')
    #
    # # plt.show()
    #
    # # Measured position
    #
    # input_sensor_noise = 250  # Add noise to measurement
    # # input_sensor_noise = 0.0
    #
    # # Initialize an array to hold sensor noise values
    # sensor_noise = np.random.randn(num_frames) * input_sensor_noise  # Generate random noise values
    #
    # # Calculate measured_x_pos with different sensor noise for each value
    # # Compute ball positions
    # measured_x_pos = true_x + sensor_noise
    # measured_y_pos = true_y - sensor_noise
    #
    # # Create a 2D array to store ball positions
    # measurement = np.column_stack((measured_x_pos, measured_y_pos))
    #
    # # run filter
    # R_var = 5.0 ** 2  # measurement uncertainty/noise
    # Q_var = 2.0 ** 2  # Process uncertainty/noise
    # ball_tracker = ekf_tracker(R_var=R_var, Q_var=Q_var)
    #
    # # Initialize means and covariances arrays
    # means, covariances = np.zeros((num_frames, 6, 1)), np.zeros((num_frames, 6, 6))
    #
    # # Run through measurements
    # for i, z in enumerate(measurement):
    #     ball_tracker.predict()
    #     ball_tracker.update(z)
    #
    #     means[i, :] = ball_tracker.x
    #     covariances[i, :, :] = ball_tracker.P
    #
    # for x, P in zip(means, covariances):
    #     # covariance of x and y
    #     cov = np.array([[P[0, 0], P[3, 0]],
    #                     [P[0, 3], P[3, 3]]])
    #     mean = (x[0, 0], x[3, 0])
    #     plot_covariance(mean, cov=cov, fc='g', std=3, alpha=0.5)
    #
    # # plot results
    # plot_filter(means[:, 0], means[:, 3])
    # plot_measurements(measurement[:, 0], measurement[:, 1])
    # plt.legend(loc=0)
    # plt.show()

def ekf_cvtr():
    # Track Generation

    # Predefined points that define the shape of the race track
    # track_points = np.array([[0, 0], [0.2, 0.3], [0.4, 0.35], [0.5, 0.5], [0.8, 0.3],
    #                          [0.7, 0.1], [0.8, 0], [0.8, -0.3], [0.6, -0.4], [0.4, -0.3],
    #                          [0.2, -0.05], [0, 0]]).T
    #
    track_points = np.array([[0, 0], [0.2, 0.3], [0.4, 0.35], [0.5, 0.5], [0.8, 0.3],
                             [0.7, 0.1], [0.8, 0]]).T

    # track_points = np.array([[0, 0], [0.2, 0.2], [0.3, 0.15], [0.4, 0.3], [0.5, 0.4]]).T
    points = track_points * 10000

    # points *= 10000
    # Interpolate a smooth curve using splines
    tck, u = splprep(points, s=0.01, per=0)

    num_frames = 100

    # Define the resolution of the curve
    u_new = np.linspace(u.min(), u.max(), num=num_frames*5)

    # Evaluate the curve at the given resolution
    true_x, true_y = splev(u_new, tck)

    # plt.plot(true_x, true_y, linewidth=2, color='Red')
    # plt.axis('equal')
    # plt.title("Path")
    # plt.xlabel('X')
    # plt.ylabel('Y')

    # plt.show()

    # Measured position


    input_sensor_noise = 50  # Add noise to measurement
    # input_sensor_noise = 0.0

    # Initialize an array to hold sensor noise values
    # num_samples = int(num_frames/5.0)
    sensor_noise = np.random.randn(num_frames) * input_sensor_noise  # Generate random noise values

    # Calculate measured_x_pos with different sensor noise for each value
    # Compute positions


    # Randomly sample 20% of the points
    random_indices = np.random.choice(np.arange(len(true_x)), size=num_frames, replace=False)
    sorted_idx = np.sort(random_indices)
    sampled_x = true_x[sorted_idx] + sensor_noise
    sampled_y = true_y[sorted_idx] - sensor_noise

    # measured_x_pos = true_x[::5] + sensor_noise
    # measured_y_pos = true_y[::5] - sensor_noise

    measured_x_pos = sampled_x
    measured_y_pos = sampled_y

    # Calculate velocities using numerical differentiation
    dx = np.diff(measured_x_pos)
    dy = np.diff(measured_y_pos)
    velocities = np.sqrt(dx ** 2 + dy ** 2)

    # Calculate headings using trigonometry
    headings = np.arctan2(dy, dx)  # in radians
    headings = np.degrees(headings)  # convert to degrees

    # Create a 2D array to store ball positions
    measurement = np.column_stack((measured_x_pos, measured_y_pos))

    # Time step
    dt = 1.0/20.0





    # state = [x, y, heading, v, yaw_rate]

    vs, psis, dpsis, dts, xs, ys, lats, lons = symbols('v \psi \dot\psi T x y lat lon')


    # Dynamic model function
    gs = Matrix([[xs + (vs / dpsis) * (sin(psis + dpsis * dts) - sin(psis))],
                 [ys + (vs / dpsis) * (-cos(psis + dpsis * dts) + cos(psis))],
                 [psis + dpsis * dts],
                 [vs],
                 [dpsis]])

    state = Matrix([xs, ys, psis, vs, dpsis])

    Jgs = gs.jacobian(state)

    # P = np.diag([1000.0, 1000.0, 1000.0, 1000.0, 1000.0])
    # Set the initial uncertainties for each state variable
    initial_uncertainties = [10.0, 10.0, np.radians(5), 20.0, np.radians(2)]

    # Initialize the P matrix
    P = np.diag(initial_uncertainties) ** 2

    # process covariance
    # state = [x, y, heading, v, yaw_rate]

    # # turn_var -> maximum of 3.5 G's
    # turn_var = 3.5 * 9.81 * dt**2
    # # val var = max speed is 55 metres per second
    # vel_var = 20.0*dt
    # pos_var = input_sensor_noise/5.0
    # turn_rate_var = 2 * 9.81*dt**2
    #
    # Q = np.diag([pos_var**2, pos_var**2, turn_var**2, vel_var**2, turn_rate_var**2])

    # Define maximum centripetal acceleration
    max_centripetal_acceleration = 3 * 9.8  # Maximum lateral acceleration in m/s^2

    # Define variances for each state variable
    var_x = 20.0 ** 2  # Variance for x position
    var_y = 20.0 ** 2  # Variance for y position
    var_heading = np.radians(5) ** 2  # Variance for heading angle
    var_velocity = (0.05 * max_centripetal_acceleration) ** 2  # Variance for velocity
    var_yaw_rate = np.radians(2) ** 2  # Variance for yaw rate

    # Create the Q matrix
    Q = np.diag([var_x, var_y, var_heading, var_velocity, var_yaw_rate])

    # Measurment function h

    hs = Matrix([[xs],
                 [ys]])

    JHs = hs.jacobian(state)

    # Measurement Noise Covariance
    # m_noise = input_sensor_noise**2
    # m_noise = input_sensor_noise**2
    # R = np.matrix([[m_noise, 0.0],
    #                [0.0, m_noise]])
    R = np.diag([input_sensor_noise**2, input_sensor_noise**2])

    I = np.eye(5)

    # Initial state
    x = np.matrix([[0.0, 0.0, 3.14, 6.001, 0.05]]).T

    # Preallocation for Plotting
    x0 = []
    x1 = []
    x2 = []
    x3 = []
    x4 = []
    x5 = []
    Zx = []
    Zy = []
    Px = []
    Py = []
    Pdx = []
    Pdy = []
    Pddx = []
    Pddy = []
    Kx = []
    Ky = []
    Kdx = []
    Kdy = []
    Kddx = []
    dstate = []

    def savestates(x, Z, P, K):
        # x0.append(float(x[0]))
        x0.append(x[0].item())
        # x1.append(float(x[1]))
        x1.append(x[1].item())
        # x2.append(float(x[2]))
        x2.append(x[2].item())
        # x3.append(float(x[3]))
        x2.append(x[3].item())
        # x4.append(float(x[4]))
        x4.append(x[4].item())
        # Zx.append(float(Z[0]))
        Zx.append(Z[0].item())
        # Zy.append(float(Z[1]))
        Zy.append(Z[1].item())
        Px.append(float(P[0, 0]))
        Py.append(float(P[1, 1]))
        Pdx.append(float(P[2, 2]))
        Pdy.append(float(P[3, 3]))
        Pddx.append(float(P[4, 4]))
        Kx.append(float(K[0, 0]))
        Ky.append(float(K[1, 0]))
        Kdx.append(float(K[2, 0]))
        Kdy.append(float(K[3, 0]))
        Kddx.append(float(K[4, 0]))


    for filterstep in range(measurement.shape[0]):
        # Predict using dynamic model
        # [x, y, heading, velocity, yaw rate] = x[0, 1, 2, 3, 4]

        if np.abs(x[4]) < 0.0001: # If yawrate < 0.001 therefore driving straight
            x[0] = x[0] + x[3] * dt * np.cos(x[2])
            x[1] = x[1] + x[3] * dt * np.sin(x[2])
            x[2] = x[2]
            x[3] = x[3]
            x[4] = 0.0000001  # avoid numerical issues in Jacobians
            dstate.append(1)
        else:
            x[0] = x[0] + (x[3] / x[4]) * (np.sin(x[4] * dt + x[2]) - np.sin(x[2]))
            x[1] = x[1] + (x[3] / x[4]) * (-np.cos(x[4] * dt + x[2]) + np.cos(x[2]))
            x[2] = (x[2] + x[4] * dt + np.pi) % (2.0 * np.pi) - np.pi
            x[3] = x[3]
            x[4] = x[4]
            dstate.append(1)


        # Calculate jacobian WRT state vector
        a13 = (x[3] / x[4]) * (np.cos(x[4] * dt + x[2]) - np.cos(x[2]))
        a14 = (1.0 / x[4]) * (np.sin(x[4] * dt + x[2]) - np.sin(x[2]))
        a15 = (dt * x[3] / x[4]) * np.cos(x[4] * dt + x[2]) - (x[3] / x[4] ** 2) * (
                    np.sin(x[4] * dt + x[2]) - np.sin(x[2]))
        a23 = (x[3] / x[4]) * (np.sin(x[4] * dt + x[2]) - np.sin(x[2]))
        a24 = (1.0 / x[4]) * (-np.cos(x[4] * dt + x[2]) + np.cos(x[2]))
        a25 = (dt * x[3] / x[4]) * np.sin(x[4] * dt + x[2]) - (x[3] / x[4] ** 2) * (
                    -np.cos(x[4] * dt + x[2]) + np.cos(x[2]))

        # Convert to floats
        a13 = a13.item()
        a14 = a14.item()
        a15 = a15.item()
        a23 = a23.item()
        a24 = a24.item()
        a25 = a25.item()

        JA = np.matrix([[1.0, 0.0, a13, a14, a15],
                        [0.0, 1.0, a23, a24, a25],
                        [0.0, 0.0, 1.0, 0.0, dt],
                        [0.0, 0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 1.0]])

        # Project the error covariance ahead
        P = JA * P * JA.T + Q

        # Measurement Update (Correction)
        # ===============================
        # Measurement Function
        measure_x = x[0].item()
        measure_y = x[1].item()
        hx = np.matrix([[measure_x],
                        [measure_y]])

        if filterstep % 2 == 0:  # every 5th step
            JH = np.matrix([[1.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0, 0.0]])
        else:  # every other step
            JH = np.matrix([[0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0]])

        S = JH * P * JH.T + R
        K = (P * JH.T) * np.linalg.inv(S)

        # Update the estimate via
        # Z = measurement[:, filterstep].reshape(JH.shape[0], 1)
        Z = measurement[filterstep].reshape(JH.shape[0], 1)
        y = Z - (hx)  # Innovation or Residual
        x = x + (K * y)
        # Update the error covariance
        P = (I - (K * JH)) * P
        # Save states for Plotting
        savestates(x, Z, P, K)

    def plotxy():

        fig = plt.figure(figsize=(16, 9))

        # U = cos(x2)
        # V = sin(x2)

        # EKF State
        # plt.quiver(x0, x1, color='#94C600', units='xy', width=0.05, scale=0.5)
        plt.plot(x0, x1, label='EKF Position', c='k', lw=5)

        # Measurements
        # plt.scatter(measured_x_pos[::5], measured_y_pos[::5], s=50, label='Measurements', marker='+')
        plt.scatter(measured_x_pos, measured_y_pos, s=50, label='Measurements', marker='+')

        # cbar=plt.colorbar(ticks=np.arange(20))
        # cbar.ax.set_ylabel(u'EPE', rotation=270)
        # cbar.ax.set_xlabel(u'm')

        # Start / Goal
        plt.scatter(x0[0], x1[0], s=250, label='Start', c='g')
        plt.scatter(x0[-1], x1[-1], s=250, label='Goal', c='r')

        # Plot true path
        plt.plot(true_x, true_y, color='Red', linewidth=2)

        plt.xlabel('X [m]')
        plt.ylabel('Y [m]')
        plt.title('Position')
        plt.legend(loc='best')
        # plt.axis('equal')
        # plt.show()
        # plt.tight_layout()
        plt.savefig('EKF-CTRV.png', dpi=72, transparent=False, bbox_inches='tight')

    plotxy()
    plt.show()



if __name__ == '__main__':
    # TrainTrack()
    # gaussian_track()
    # multivariate_gaussians()
    # multivariate_kalman_filter()
    # extended_kalman_filter_robot()
    # kalman_filter_design()
    # ball_track()
    # constant_steering_constant_velocity()
    ekf_cvtr()