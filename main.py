import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import SGD
from rocketpy import Rocket, SolidMotor, Environment, Flight
from IPython.display import HTML

# ====================== ROCKETPY SETUP ======================
env = Environment(latitude=32.99, longitude=-106.97, elevation=1400)
env.set_date(date=(2025, 6, 15, 12), timezone="UTC")

motor = SolidMotor(
    thrust_source=2800, burn_time=6.0, dry_mass=1.8,
    dry_inertia=(0.2, 0.2, 0.02),
    center_of_dry_mass_position=0.3,
    grain_number=1, grain_density=1700,
    grain_outer_radius=0.045, grain_initial_inner_radius=0.015,
    grain_initial_height=0.35, grain_separation=0.0,
    grains_center_of_mass_position=0.397,
    nozzle_radius=0.035, nozzle_position=0.0
)

rocket = Rocket(
    radius=0.085, mass=12.0, inertia=(12, 12, 0.6),
    power_off_drag=0.5, power_on_drag=0.8,
    center_of_mass_without_motor=1.35,
    coordinate_system_orientation="tail_to_nose"
)
rocket.add_motor(motor, position=0.0)
rocket.add_nose(length=0.7, kind="vonkarman", position=2.1)
rocket.add_trapezoidal_fins(n=4, root_chord=0.32, tip_chord=0.15, span=0.18, sweep_length=0.12, position=0.4)
rocket.add_tail(top_radius=0.085, bottom_radius=0.06, length=0.25, position=0.1)

test_flight = Flight(rocket=rocket, environment=env, rail_length=5.0, inclination=89, heading=0)

# ====================== HYBRID AI SETUP ======================
nn = Sequential([
    Input(shape=(1,)), 
    Dense(32, activation='relu'), 
    Dense(16, activation='relu'), 
    Dense(1)
])
nn.compile(optimizer=SGD(learning_rate=0.01), loss='mse')
nn.fit(np.linspace(-0.6,0.6,1000).reshape(-1,1), 
       np.linspace(-0.6,0.6,1000).reshape(-1,1)*0.15, 
       epochs=10, verbose=0)

# ====================== ANIMATION ENGINE ======================
fig, ax = plt.subplots(figsize=(12, 7))
ax.set_xlim(-300, 300)
ax.set_ylim(0, 600)
ax.set_aspect('equal')
ax.set_facecolor('#050520')
ax.set_title('RocketPy Physics Engine + Hybrid AI Digital Twin', color='white')

rocket_line, = ax.plot([], [], 'w-', linewidth=18, solid_capstyle='round')
nose_cone, = ax.plot([], [], 'red', linewidth=24)
flame, = ax.plot([], [], color='orange', linewidth=15, alpha=0.8)
angle_text = ax.text(-280, 550, '', fontsize=12, color='white', bbox=dict(facecolor='black', alpha=0.7))

# Physics Parameters
dt = 0.02
theta, omega = np.deg2rad(5.0), 0.0
integral, last_error = 0.0, 0.0
Kp, Ki, Kd = 150.0, 40.0, 60.0

def update(frame):
    global theta, omega, integral, last_error
    t = frame * dt
    
    current_thrust = motor.thrust(t)
    current_mass = rocket.total_mass(t)
    inertia = rocket.I_22(t)
    
    error = -theta
    integral += error * dt
    derivative = (error - last_error) / dt
    last_error = error
    gimbal = np.clip(Kp*error + Ki*integral + Kd*derivative, -0.4, 0.4)

    torque = (current_thrust * 1.85 * np.sin(gimbal)) + (-9.81 * current_mass * np.sin(theta) * 0.8)
    alpha = torque / inertia
    omega += alpha * dt
    theta += omega * dt

    ai_bias = nn.predict(np.array([[theta]]), verbose=0)[0][0]
    theta_vis = theta + (ai_bias * 0.2)

    cx, cy = 0, 150
    L_vis = 200
    x2, y2 = cx + L_vis * np.sin(theta_vis), cy - L_vis * np.cos(theta_vis)
    
    rocket_line.set_data([cx, x2], [cy, y2])
    nose_cone.set_data([x2, x2 + 30*np.sin(theta_vis)], [y2, y2 - 30*np.cos(theta_vis)])
    
    if current_thrust > 0:
        fx = cx + 80 * np.sin(theta_vis + gimbal)
        fy = cy + 80 * np.cos(theta_vis + gimbal)
        flame.set_data([cx, fx], [cy, fy])
    else:
        flame.set_data([], [])

    angle_text.set_text(f"T+{t:.2f}s | Thrust: {current_thrust:.0f}N\nAngle: {np.rad2deg(theta_vis):.2f}°")
    return rocket_line, nose_cone, flame, angle_text

ani = FuncAnimation(fig, update, frames=300, interval=20, blit=True)
plt.close(fig) # Prevents an extra static plot from appearing
HTML(ani.to_jshtml())
