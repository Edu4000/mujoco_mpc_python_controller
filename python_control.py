# Copyright 2022 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# %%
import time as time_

import mujoco
import mujoco.viewer as viewer
from mujoco_mpc import agent as agent_lib
import numpy as np

import pathlib

m = mujoco.MjModel.from_xml_path("mjpc/tasks/quadruped/task_flat.xml")
d = mujoco.MjData(m)

# %%
print(pathlib.Path(agent_lib.__file__).parent / "mjpc" / "ui_agent_server")

# Initializing our agent
agent = agent_lib.Agent(
        # This is to enable the ui
        server_binary_path=pathlib.Path(agent_lib.__file__).parent
        / "mjpc"
        / "ui_agent_server",
        task_id="Quadruped Flat", 
        model=m)

####################################
# Data needed for the model to run #
####################################

# weights
agent.set_cost_weights({"Position": 0.15})
print("Cost weights:", agent.get_cost_weights())

# parameters
agent.set_task_parameter("Walk speed", 1.0)
print("Parameters:", agent.get_task_parameters())

# rollout horizon
T = 1500

# trajectories
qpos = np.zeros((m.nq, T))
qvel = np.zeros((m.nv, T))
ctrl = np.zeros((m.nu, T - 1))
time = np.zeros(T) 

# costs
cost_total = np.zeros(T - 1)
cost_terms = np.zeros((len(agent.get_cost_term_values()), T - 1))

# rollout
mujoco.mj_resetData(m, d)

# cache initial state
qpos[:, 0] = d.qpos
qvel[:, 0] = d.qvel
time[0] = d.time
# %%
# Debugging
print(m.body('goal').pos)

print(m.body('trunk'))

print(np.linalg.norm(m.body('goal').pos - m.body('trunk').pos))
# %%

############################################
# Executing simulation with multiple goals #
############################################
goals = [
    [5, 0, 0.26],
    [0, 5, 0.26],
    [-5, 0, 0.26],
    [0, -5, 0.26]
]
i = 0

with mujoco.viewer.launch_passive(agent.model, d) as viewer:
    # Close the viewer automatically after 30 wall-seconds.

    start = time_.time()
    while viewer.is_running() and time_.time():

        # set planner state
        agent.set_state(
            time=d.time,
            qpos=d.qpos,
            qvel=d.qvel,
            act=d.act,
            mocap_pos=d.mocap_pos,
            mocap_quat=d.mocap_quat,
            userdata=d.userdata,
        )

        # run planner for num_steps
        num_steps = 8
        for _ in range(num_steps):
            agent.planner_step()

        # set ctrl from agent policy
        d.ctrl = agent.get_action()

        print(agent.model.geom(31).pos)
        print(m.geom(31).pos)
        print()
        
        # time_.sleep(0.1)
        if (np.linalg.norm(m.body('goal').pos - m.body('trunk').pos) < 1):
            m.body('goal').pos = goals[i]
            i += 1
            i /= len(goals)

        mujoco.mj_step(agent.model,d)
        mujoco.mj_step(m,d)

        # Example modification of a viewer option: toggle contact points every two seconds.
        with viewer.lock():
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)

        # Pick up changes to the physics state, apply perturbations, update options from GUI.
        viewer.sync()
# %%
