import zmq
import matplotlib
matplotlib.use("TkAgg")  # Avoid Qt issues
import matplotlib.pyplot as plt
import numpy as np
import math

# FOV and limits
x_min, x_max = 0, 3
y_min, y_max = -2,2

def main():
    # ZeroMQ subscriber
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.CONFLATE, 1)  # Keep only the last message
    socket.connect("tcp://192.168.2.42:8080")  # Replace with actual IP
    socket.setsockopt_string(zmq.SUBSCRIBE, "")

    plt.ion()
    fig, ax = plt.subplots()

    while True:
        try:
            obstacle_points = socket.recv_pyobj()

            if isinstance(obstacle_points, list):
                obstacle_points = np.array(obstacle_points)

            if obstacle_points.ndim == 2 and obstacle_points.shape[1] >= 2:

                x = obstacle_points[:, 0]
                y = obstacle_points[:, 1]
                print(x,y)
            else:
                continue

            ax.clear()
            ax.scatter(x, y, c="red", s=100)
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("X (forward)")
            ax.set_ylabel("Y (lateral)")
            ax.set_title("Obstacle Points (FoV 54°)")
            ax.grid(True)
            plt.pause(0.01)

        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()
