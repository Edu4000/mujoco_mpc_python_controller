
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
import numpy as np

import mujoco
import mujoco.viewer as viewer

from mujoco_mpc import agent as agent_lib
import xml.etree.ElementTree as ET

import pathlib

agents_pos = ['0 0 0',
              '0 0 0',
              '0 0 0',
              '0 0 0']

# XML Functions for Preparing the Agents

def spawn_agents (agent_pos):
    agents = []

    for i in range(len(agents_pos)):
        # Generating each agent's xml
        xml_string = ""

        # Creating agents and adding them to the list
        agents.append(
            agent_lib.Agent(
                # This is to enable the ui
                server_binary_path=pathlib.Path(agent_lib.__file__).parent
                / "mjpc"
                / "ui_agent_server",
                task_id="Quadruped Flat",
                model=mujoco.MjModel.from_xml_string(xml_string),
                real_time_speed=0.2,)
        )

        # Setting cost weights and task parameters
        agents[i].set_cost_weights({"Position": 0.15})
        agents[i].set_task_parameter("Walk speed", 1.0)

    return agents

agent_body = ET.parse("./mjpc/tasks/quadruped/a1_body.xml").getroot().find("worldbody")

# Creating model and data
m_agent = mujoco.MjModel.from_xml_path("./mjpc/tasks/quadruped/task_flat copy.xml")

m = mujoco.MjModel.from_xml_path("./mjpc/tasks/quadruped/task_flat_1.xml")
d = mujoco.MjData(m)

# %%


print(pathlib.Path(agent_lib.__file__).parent / "mjpc" / "ui_agent_server")

# # Initializing our agent (agent server/executable)
agent = agent_lib.Agent(
        # This is to enable the ui
        server_binary_path=pathlib.Path(agent_lib.__file__).parent
        / "mjpc"
        / "ui_agent_server",
        task_id="Quadruped Flat", 
        model=m_agent,
        real_time_speed=0.2,)

agent2 = agent_lib.Agent(
        # This is to enable the ui
        server_binary_path=pathlib.Path(agent_lib.__file__).parent
        / "mjpc"
        / "ui_agent_server",
        task_id="Quadruped Flat", 
        model=m,
        real_time_speed=0.2)

# ####################################
# # Data needed for the model to run #
# ####################################

# weights
agent.set_cost_weights({"Position": 0.15})
print("Cost weights:", agent.get_cost_weights())
agent2.set_cost_weights({"Position": 0.15})
print("Parameters:", agent2.get_cost_weights())

# parameters
agent2.set_task_parameter("Walk speed", 1.0)
print("Parameters:", agent2.get_task_parameters())
agent.set_task_parameter("Walk speed", 1.0)
print("Parameters:", agent.get_task_parameters())


# %%

############################################
# Executing simulation with multiple goals #
############################################
goals = [
    [5, 0, 0.26],
    [0, 2, 0.26],
    [-2, 0, 0.26],
    [0, -5, 0.26]
]
i = 0

print("\n################")
# print(agent.get_task_parameters())
print(d.mocap_pos[1:])
print(d.mocap_pos[[0,2]])
print("\n################")

# %%
with mujoco.viewer.launch_passive(m, d) as viewer:
    # Close the viewer automatically after 30 wall-seconds.

    start = time_.time()
    while viewer.is_running() and time_.time():

        # run planner for num_steps
        num_steps = 8
        for _ in range(num_steps):
            agent.planner_step()
            agent2.planner_step()

        # get action from planner
        agent_ctrl = agent.get_action()
        agent_ctrl2 = agent2.get_action()
        agent_ctrl = np.append(agent_ctrl, agent_ctrl2, axis=0)

        # update joints with action
        d.ctrl = agent_ctrl#agent.get_action()
        
        # update state of each agent
        agent.set_state(
            time=d.time,
            qpos=d.qpos[:19],
            qvel=d.qvel[:18],
            act=d.act,
            mocap_pos=d.mocap_pos[1:],
            mocap_quat=d.mocap_quat[1:],
            userdata=d.userdata,
        )
        agent2.set_state(
            time=d.time,
            qpos=d.qpos[19:],
            qvel=d.qvel[18:],
            act=d.act,
            mocap_pos=d.mocap_pos[[0,2]],
            mocap_quat=d.mocap_quat[[0,2]],
            userdata=d.userdata,
        )


        # Check if arrived to a goal
        if (np.linalg.norm(d.mocap_pos[0] - d.body('trunk').xpos) < 1 or 
            np.linalg.norm(d.mocap_pos[0] - d.body('trunk').xpos) < 1) :
            print("\nARRIVED!")
            d.mocap_pos[0] = goals[i]
            print(d.mocap_pos)
            i = (i + 1) % len(goals)

        # Step the simulation forward.
        mujoco.mj_step(m,d)

        # Example modification of a viewer option: toggle contact points every two seconds.
        with viewer.lock():
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)

        # Pick up changes to the physics state, apply perturbations, update options from GUI.
        viewer.sync()
# %%
# import xml.etree.ElementTree as ET

# agent_num = 2
# agents_pos = ['0 0 0',
#               '0 1 0']

# # Loading Templates as XML and String
# agent_root = ET.parse("./mjpc/tasks/quadruped/a1_template.xml").getroot()
# tasks_root = ET.parse("./mjpc/tasks/quadruped/task_flat_template.xml").getroot()

# body_template     = open("./mjpc/tasks/quadruped/a1_body.xml", "r").read()      # For a1.xml
# actuator_template = open("./mjpc/tasks/quadruped/a1_actuator.xml", "r").read()  # For task.xml
# sensor_template   = open("./mjpc/tasks/quadruped/a1_sensor.xml", "r").read()    # For task.

# # Array of individual agent xmls
# agent_xml = []
# tasks_xml = []

# # XML used in the visualizer
# sim_agent_xml = ""
# sim_tasks_actuator_xml = ""
# sim_tasks_sensor_xml = ""

# # Editing templates to get individual XMLs and General XML
# for i in range(agent_num):
#     # Getig a copy of the templates
#     body = body_template
#     actuator = actuator_template
#     sensor = sensor_template

#     # Replace the substring in agent
#     body_xml = body.replace("TEMP", f"{i}")
#     body_xml = body_xml.replace("POSITION", f"{agents_pos[i]}")
#     sim_agent_xml += body_xml
#     agent_root.find("worldbody").append(ET.fromstring(body_xml))

#     # Replace the substring in tasks
#     actuator_xml = actuator.replace("TEMP", f"{i}");  sim_tasks_actuator_xml += actuator_xml
#     sensor_xml = sensor.replace("TEMP", f"{i}");      sim_tasks_sensor_xml += sensor_xml
#     tasks_root.find("actuator").append(ET.fromstring(actuator_xml))
#     tasks_root.find("sensor").append(ET.fromstring(sensor_xml))
#     tasks_root.find("home").text

#     # Save individual xmls for each agent
#     worldbody_body.append(body_xml)

#     # Create a new element, set its text, and add it to the "worldbody" element
#     new_element = ET.fromstring(body_xml)
#     worldbody_agent.append(new_element)
#     agents_xml.append(ET.tostring(root, encoding='unicode', method='xml'))
#     worldbody_agent.clear()


#     actuator = actuator_template
#     sensor = sensor_template

# # %%
# model_xml = ""
# for xml in agents_xml:
#     print('\n\n')
#     print(xml)
#     model_xml += xml
#     model_xml += '\n'

# for body in worldbody_body:
#     new_element = ET.fromstring(body)
#     worldbody_agent.append(new_element)
# print(ET.tostring(root, encoding='unicode', method='xml'))
# %%
