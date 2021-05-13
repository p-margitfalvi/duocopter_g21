import numpy as np
import matplotlib.pyplot as plt

g = 9.81
thrust_N = 2*9.81 # max thrust approx 2kgf 
copter_weight_KG = 0.5

start_y = 0
start_v = 0

accel_f = lambda thrust_fraction: (thrust_N*thrust_fraction -
        g*copter_weight_KG)/copter_weight_KG # TODO: Add friction effects

def main():
    t = 0
    dt = 1E-3
    t_final = 5

    # Waypoint array, formatted as [height, time]
    waypts = [[1, 1], [2, 3], [6, 10]]

    y_setpoint = 10
    cur_wpt = 0

    v = start_v
    y = start_y
    plot = True

    times = []
    y_history = []
    v_history = []
    thrust_history = []
    err_history = []


    # if plot:
    #     plt.ion()
    #     fig = plt.figure()
    #     copter_pt = plt.scatter(0, y)
    #     plt.draw()

    while t < t_final:
        # breakpoint()
        if t > waypts[cur_wpt][1]:
            cur_wpt += 1
            cur_wpt = min(len(waypts) - 1, cur_wpt)

        v_req = (waypts[cur_wpt][0] - y) / (waypts[cur_wpt][1] - t)
        err = v_req - v

        thrust = 0.5 * err
        thrust = min(max(thrust, -1), 1)
        accel = accel_f(thrust)

        y += v*dt
        v += accel*dt

        t += dt

        print(f't: {t}, \ty: {y}, \t v: {v}, \t thrust: {thrust}\n')

        times.append(t)
        y_history.append(y)
        v_history.append(v)
        thrust_history.append(thrust)
        err_history.append(err)

        # if plot:
        #     copter_pt.set_offsets([0, y])
        #     fig.canvas.draw_idle()
        #     plt.pause(dt)
    fig, axs = plt.subplots(1, 4)
    axs[0].plot(times, y_history, label='y')
    axs[1].plot(times, v_history, label='v')
    axs[2].plot(times, thrust_history, label='thrust')
    axs[3].plot(times, err_history, label='err')
    plt.show()


if __name__ == '__main__':
    main()
