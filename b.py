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
from mujoco_mpc.proto import agent_pb2

m = mujoco.MjModel.from_xml_path("mjpc/tasks/quadruped/task_flat.xml")
d = mujoco.MjData(m)

# %%
print(pathlib.Path(agent_lib.__file__).parent / "mjpc" / "ui_agent_server")

# Initializing our agent
agent = agent_lib.Agent(
        real_time_speed=0.33,
        # This is to enable the ui
        server_binary_path=pathlib.Path(agent_lib.__file__).parent
        / "mjpc"
        / "ui_agent_server",
        task_id="Quadruped Flat", 
        model=m)

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

with mujoco.viewer.launch_passive(m, d) as viewer:
    # Close the viewer automatically after 30 wall-seconds.
    start = time_.time()

    while viewer.is_running() and time_.time():
        i += 1

        if i > 70:
            print(d.body('goal'))
            print()

            d.body('goal').xpos += [0, -0.1, 0]