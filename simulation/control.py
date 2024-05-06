import time
import numpy as np
import mujoco
import mujoco.viewer


m = mujoco.MjModel.from_xml_path("C:\\Users\\arshs\\OneDrive\\Documents\\GitHub\\mujoco_menagerie\\hello_robot_stretch\\scene.xml")
d = mujoco.MjData(m)

# print(dir(m))
with mujoco.viewer.launch_passive(m, d) as viewer:
  # Close the viewer automatically after 30 wall-seconds.
  start = time.time()
  while True: #:
    step_start = time.time()

    # print(m.nu)
    action = np.array([1, 0, 0, 0, 0, 0, 0, 0])
    # action[0] = right/left lateral movement
    # action[1] = spinning cw/ccw
    # action[2] = raises arm up/down
    # action[3] = arm goes out/in
    # action[4] = wrist rotates cw/ccw
    # action[5] = hand open/close
    # action[6] = camera spin
    # action[7] = camera pitch


    # action = np.random.uniform(-1, 1, size=m.nu)
    d.ctrl[:] = action
    
    # mj_step can be replaced with code that also evaluates
    # a policy and applies a control signal before stepping the physics.
    mujoco.mj_step(m, d)

    # Example modification of a viewer option: toggle contact points every two seconds.
    with viewer.lock():
      viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)

    # Pick up changes to the physics state, apply perturbations, update options from GUI.
    viewer.sync()

    # Rudimentary time keeping, will drift relative to wall clock.
    time_until_next_step = m.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)