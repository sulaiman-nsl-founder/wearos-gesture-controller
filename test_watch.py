import collections
import threading
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from touch_sdk import Watch

# 1. Set up data buffers (keeps the last 100 data points on screen)
MAX_POINTS = 100
x_data = collections.deque([0.0] * MAX_POINTS, maxlen=MAX_POINTS)
y_data = collections.deque([0.0] * MAX_POINTS, maxlen=MAX_POINTS)
z_data = collections.deque([0.0] * MAX_POINTS, maxlen=MAX_POINTS)

# 2. Define the Watch listener using your working documentation
class MyWatchGraph(Watch):
    def on_sensors(self, sensors):
        # The documentation states this returns (x, y, z)
        accel = sensors.acceleration
        if accel:
            x_data.append(accel[0])
            y_data.append(accel[1])
            z_data.append(accel[2])
            
    def on_tap(self):
        print("🟢 TAP DETECTED on watch!")

# 3. Set up the Matplotlib Graph interface
fig, ax = plt.subplots(figsize=(10, 5))
line_x, = ax.plot([], [], label="X Axis", color="red", lw=2)
line_y, = ax.plot([], [], label="Y Axis", color="green", lw=2)
line_z, = ax.plot([], [], label="Z Axis", color="blue", lw=2)

# Set the scale of the graph (Gravity is ~9.8, so +/- 20 m/s^2 is a safe viewing range)
ax.set_ylim(-20, 20) 
ax.set_xlim(0, MAX_POINTS)
ax.set_title("Live Touch SDK Accelerometer Stream")
ax.legend(loc="upper right")
ax.grid(True)

# 4. Define the function that updates the lines on the graph
def update_plot(frame):
    # Convert the deques to lists to draw them
    line_x.set_data(range(MAX_POINTS), list(x_data))
    line_y.set_data(range(MAX_POINTS), list(y_data))
    line_z.set_data(range(MAX_POINTS), list(z_data))
    return line_x, line_y, line_z

if __name__ == "__main__":
    print("Connecting to Watch...")
    watch = MyWatchGraph()
    
    # 5. Run the watch listener in a background thread so the graph doesn't freeze
    watch_thread = threading.Thread(target=watch.start, daemon=True)
    watch_thread.start()

    # 6. Start the live graph animation loop (updates every 50 milliseconds)
    ani = animation.FuncAnimation(fig, update_plot, interval=50, blit=True, cache_frame_data=False)
    
    # Show the graph window (This must run on the main thread!)
    plt.show()