# Author:   Andy Edmondson
# Email:    andrew.edmondson@gmail.com
# Date:     3 May 2023
# Purpose:  EPuck robot controller for use with RL algorithms
#
# Includes:
import copy
import gc
import math
import time
from copy import deepcopy
import os
from dataclasses import dataclass
from datetime import datetime as dt
import typing
import inspect
import pickle

import numpy
import numpy as np
import torch
import torch.multiprocessing as mp
from multiprocessing.managers import BaseManager
from torch.multiprocessing import Manager, Queue, Process
from controller import Robot, Receiver, Emitter, GPS, Compass, Motor, DistanceSensor, LightSensor, Supervisor
from collections import deque, namedtuple

from network_model import MLPActorCritic
from replay_buffer import ReplayBuffer, PrioritizedReplayBuffer, DWR
import ddpg_optimise
from hyperparameters import DDPGHyperParameters, SimulationHyperParameters, OptimiseHyperparams
import data_collector
from data_collector import AgentData, OptimiseData
import application_log
import kmeans_funcs

from sklearn.cluster import KMeans

torch.cuda.empty_cache()
gc.collect()


# Define a multiprocessing manager for the replay buffer which will be shared.
class ReplayBufferManager(BaseManager):
    pass


class Controller:
    def __init__(self, robot, model_hyperparameters: DDPGHyperParameters,
                 simulation_hyperparameters: SimulationHyperParameters):

        # Hyperparameters
        self.model_name = "DDPG-"
        self.model_version = "1.3"
        self.model_hyperparameters = model_hyperparameters
        self.simulation_hyperparameters = simulation_hyperparameters

        # Robot Parameters
        self.robot = robot
        self.time_step = self.simulation_hyperparameters.time_step
        self.max_speed = self.simulation_hyperparameters.max_speed

        # Enable the robot systems
        self.enable_motors()
        self.enable_proximity_sensors()
        self.enable_gps()
        self.enable_compass()
        self.enable_emitter_and_receiver()

        # List for input sensor data
        self.inputs = []
        self.no_noise_inputs = []

        # Fitness value (initialization fitness parameters once)
        self.calculated_rewards = []
        self.episode_rewards = []
        self.episode_lengths = []
        self.mean_step_rewards = 0.
        self.__end_of_episode_reward = 0.
        self.goal_achieved = False
        self.has_crashed = False
        self.previous_action = [0, 0]  # Will be part of the state
        self.current_bearing = 0.  # Will be part of the state
        self.current_position = [0., 0.]  # Will be part of the state

        # Starting position
        self.startPosition = [0., 0.]
        self.target_position = [0., 0.]  # Target position on the ground plane. Ignoring any y offset.

        self.__episode_status = "end_of_episode"
        self.__current_episode = 0

        # Set up the DDPG agent
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.agent = self.setup_agent_network(self.model_hyperparameters.num_inputs,
                                              self.model_hyperparameters.num_actions,
                                              self.model_hyperparameters.action_max)
        self.agent.to(self.device)
        self.agent.share_memory()  # Share the agent network parameters across processes

        # Create the target network here and pass it.
        self.target_network = deepcopy(self.agent)
        self.target_network.to(self.device)
        self.target_network.share_memory()
        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for params in self.target_network.parameters():
            params.requires_grad = False

        # Logging and data collection
        self.agent_data_object = AgentData()
        self.agent_data_queue = Queue(maxsize=10_000)
        self.log_interference = False

        # Array for information I want passed between agent and optimiser
        # 1. Current run id {"run_id": string}
        # 2. Current episode number {"episode": int}
        # 3. Status - 1 = running, 0 = pause, -1 = stop {"status": int}
        self.agent_info = Manager().dict({"run_id": "", "episode": 0, "status": 0})

        # Create the replay buffer as a shared memory object, choosing standard or DWR
        if self.model_hyperparameters.use_dwr:
            ReplayBufferManager.register('ReplayBuffer', DWR,
                                         exposed=['add', 'sample',
                                                  '__len__', 'update_priorities', 'load', 'save', 'get_buffer'])
            self.replay_buffer_manager = ReplayBufferManager()
            self.replay_buffer_manager.start()
            self.replay_buffer = self.replay_buffer_manager.ReplayBuffer(size=self.model_hyperparameters.replay_size,
                                                                         wsize=self.model_hyperparameters.sec_win_size,
                                                                         wstep=self.model_hyperparameters.step_size)

        else:
            ReplayBufferManager.register('ReplayBuffer', PrioritizedReplayBuffer,
                                         exposed=['add', 'sample',
                                                  '__len__', 'update_priorities', 'load', 'save', 'get_buffer'])
            self.replay_buffer_manager = ReplayBufferManager()
            self.replay_buffer_manager.start()
            self.replay_buffer = self.replay_buffer_manager.ReplayBuffer(size=self.model_hyperparameters.replay_size)

        self.experience_buffer = deque()  # Buffer for storing n-step experiences before adding to replay buffer
        self.load_replay()

        # Member variables for IQ (not used)
        self.kmeans = KMeans(n_clusters=self.model_hyperparameters.n_contexts, random_state=0)
        self.kmeans_centroids = [0] * self.model_hyperparameters.n_contexts
        self.kmeans_cluster_counters = [0] * self.model_hyperparameters.n_contexts

    def setup_agent_network(self, inputs: int, outputs: int, action_high: float) -> MLPActorCritic:
        """Set up and return the agent actor-critic network."""
        print(f'Creating agent network with {inputs} inputs and {outputs} outputs')
        agent = MLPActorCritic(inputs, outputs, action_high)

        if self.simulation_hyperparameters.load_model:
            # Check the ./models/load directory and get the name of the most recently modified model
            path = './models/load'
            files = sorted(os.listdir(path), key=lambda x: os.stat(os.path.join(path, x)).st_mtime)
            if files:  # If there are files in the directory
                latest_file = files[-1]
                print(f'Loading model from {latest_file}')
                agent = torch.load(os.path.join(path, latest_file))

        return agent

    def load_replay(self):
        """Load the replay buffer from the ./replays/load directory."""
        if self.simulation_hyperparameters.load_model:
            path = './replays/load'
            files = sorted(os.listdir(path), key=lambda x: os.stat(os.path.join(path, x)).st_mtime)
            if files:
                latest_file = files[-1]
                print(f'Loading replay buffer from {latest_file}')
                with open(os.path.join(path, latest_file), "rb") as fp:  # Unpickling
                    experiences, priorities, self.kmeans_centroids, self.kmeans_cluster_counters = pickle.load(fp)
                    self.replay_buffer.load([experiences, priorities])

    def save_model_info(self):
        """Save the model and replay for later use"""
        torch.save(self.agent, f"./models/{self.agent_info['run_id']}.pt")
        with open(f"./replays/{self.agent_info['run_id']}.npy", "wb") as fp:
            experiences, priorities = self.replay_buffer.save()
            pickle.dump([experiences, priorities, self.kmeans_centroids,
                         self.kmeans_cluster_counters], fp)

    def enable_motors(self):
        """Enable Motors"""
        self.left_motor = self.robot.getDevice('left wheel motor')
        self.right_motor = self.robot.getDevice('right wheel motor')
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)
        self.velocity_left = 0
        self.velocity_right = 0

    def enable_proximity_sensors(self):
        """Enable the 8 Proximity Sensors"""
        self.proximity_sensors = []
        for i in range(8):
            sensor_name = 'ps' + str(i)
            self.proximity_sensors.append(self.robot.getDevice(sensor_name))
            self.proximity_sensors[i].enable(self.time_step)
            print(self.proximity_sensors[i].getLookupTable())

    def enable_gps(self):
        """Enable the GPS"""
        self.gps = self.robot.getDevice('gps')
        self.gps.enable(self.time_step)

    def enable_compass(self):
        """Enable the Compass"""
        self.compass = self.robot.getDevice('compass')
        self.compass.enable(self.time_step)

    def enable_emitter_and_receiver(self):
        """Enable the Emitter and Receiver for communication with the supervisor"""
        self.emitter = self.robot.getDevice("emitter")
        self.receiver = self.robot.getDevice("receiver")
        self.receiver.enable(self.time_step)
        self.__previous_message = ""

    def compute_and_actuate(self, t: int, state: torch.Tensor, testing_flag: bool) -> list:
        """Get motor drive values from the network"""

        # Get random actions for the first n steps as long as we are optimising, otherwise use the network
        if self.simulation_hyperparameters.optimise and t < self.model_hyperparameters.start_steps:
            output = np.random.uniform(0.0, 1.0, [2])
        else:
            if testing_flag:
                output = self.agent.act(state).cpu().numpy()
            else:
                output = self.agent.get_action(state, self.model_hyperparameters.act_noise).cpu().numpy()
                self.model_hyperparameters.act_noise *= self.model_hyperparameters.noise_decay

        # raw_action: np.ndarray = output.copy()  # Save raw action for state construction
        output[0] *= self.simulation_hyperparameters.left_motor_factor
        output[1] *= self.simulation_hyperparameters.right_motor_factor
        self.simulation_hyperparameters.left_motor_factor *= self.simulation_hyperparameters.motor_decay
        # print(np.concatenate([raw_action, output]).tolist())
        self.previous_action = output  # Save previous action for state construction

        # Multiply the motor values my the max speed
        self.left_motor.setVelocity(output[0] * self.max_speed)
        self.right_motor.setVelocity(output[1] * self.max_speed)

        return [output[0] * self.max_speed, output[1] * self.max_speed]

    def calculate_step_reward(self, step: int, action: list) -> float:
        """Calculate the step reward."""
        # Distance rewards.
        current_distance = np.linalg.norm(np.subtract(self.target_position, self.current_position))
        distance_reward = 1.0 - (current_distance / 5.6569)

        # Forward drive reward
        nv = np.divide(action, self.max_speed)
        forward_reward = (nv[0] + nv[1]) / 2

        # Orientation reward, towards target is better
        orientation_reward = 1.0 - abs(self.inputs[-1])

        # Total step reward
        calculated_value = orientation_reward * distance_reward * forward_reward

        # Modify when find goal or crash
        self.goal_achieved = False
        self.has_crashed = False
        if self.detect_success():
            self.goal_achieved = True
            calculated_value = 15.0 * self.simulation_hyperparameters.n_steps * (1.0 - (step / 1200)) + \
                               2 * self.simulation_hyperparameters.n_steps  # 60 seconds * 20 Hz = 1200
        elif self.detect_crash():
            self.has_crashed = True
            calculated_value = -2.

        self.calculated_rewards.append(calculated_value)
        self.mean_step_rewards = np.mean(self.calculated_rewards[-400:])

        # print(f"distance: {distance_reward} orientation: {orientation_reward}, {self.inputs[-1]} "
        #       f"forward_reward: {forward_reward} crash: {self.has_crashed} total {calculated_value}")

        return calculated_value

    def send_data_to_supervisor(self, data: str = ""):
        """Emit data to the supervisor."""
        self.emitter.send(data)

    def get_data_from_supervisor(self):
        """Get data from the supervisor. Expected in the format: '0 4.2' indicating do not continue and a
        final score of 4.2."""
        if self.receiver.getQueueLength() > 0:
            text = self.receiver.getString()
            self.receiver.nextPacket()
            if text == self.__previous_message:
                return
            self.__previous_message = text
            o, r, e, i, p = text.split()
            self.__current_episode = int(e)
            self.__episode_status = o
            self.target_position = eval(p)  # Will be a 2D vector in list form.
            self.log_interference = True if i == 'True' else False
            # print(self.__episode_status)
            self.__end_of_episode_reward = float(r)

    def read_distance_sensors(self) -> None:
        """Read the distance sensor data and give it a value in the range (0,1)."""
        # Set clip range
        min_ds = self.simulation_hyperparameters.sensor_min
        max_ds = self.simulation_hyperparameters.sensor_max

        # Reset no noise sensor readings
        self.no_noise_inputs = []

        for i in range(8):
            temp = self.proximity_sensors[i].getValue()
            val = (temp - min_ds) / (max_ds - min_ds)
            if self.simulation_hyperparameters.negative_environment:
                val *= -1
            self.inputs.append(val)
            self.no_noise_inputs.append(val)

        # Add gaussian noise
        self.inputs = np.add(self.inputs,
                             np.random.normal(0.0, self.simulation_hyperparameters.sensor_noise_std,
                                              len(self.inputs))).tolist()

        # Add salt and pepper noise
        self.inputs = self.add_noise(self.inputs, p=self.simulation_hyperparameters.salt_and_pepper_noise_prob,
                                     max_value=self.simulation_hyperparameters.salt_and_pepper_noise_magnitude)

        # Add bias to sensor reading.
        # self.inputs = np.add(self.inputs, self.simulation_hyperparameters.sensor_bias_constant).tolist()
        self.inputs = np.add(self.inputs, self.simulation_hyperparameters.sensor_bias).tolist()
        self.simulation_hyperparameters.sensor_bias += self.simulation_hyperparameters.sensor_bias_step

    def add_noise(self, arr, p=0.5, max_value=100.) -> list:
        """Add salt and pepper noise to the sensor array."""
        return [x + np.random.normal(0, max_value) if np.random.uniform(0, 1) < p else x for x in arr]

    def get_state(self) -> typing.Tuple[torch.Tensor, int]:
        """Get the current state of the robot. This is the input to the neural network.
        :return: state, a vector of 14 elements.
        State = [distance sensor data [8],
                distance to target [1],
                previous action [2],
                local vector to target [2],
                local bearing to target [1]]"""
        self.inputs = []
        self.read_distance_sensors()
        # print("-" * 30)

        # Current position relative to the world
        self.current_position = self.gps.getValues()[0:2]  # X Y position
        current_distance = np.linalg.norm(np.subtract(self.target_position, self.current_position))
        self.inputs.append(current_distance)

        # Get the agent direction and bearing in degrees relative to the world
        agent_direction = self.get_agent_vector_direction()
        agent_bearing = self.vector_to_bearing(agent_direction[0], agent_direction[1])

        self.inputs.extend(self.previous_action)

        # Get the direction vector and bearing in degrees to the target position relative to world North
        # Calculate the vector to the target position
        vect2target = self.get_vector_to_target(self.current_position)
        agent_bearing_rad = agent_bearing * math.pi / 180.
        relative_vector = [math.cos(agent_bearing_rad) * vect2target[0] - math.sin(agent_bearing_rad) * vect2target[1],
                           math.sin(agent_bearing_rad) * vect2target[0] + math.cos(agent_bearing_rad) * vect2target[1]]
        agent_bear2target = self.vector_to_bearing(relative_vector[0], relative_vector[1])

        # Make bearing symmetrical and normalise
        if agent_bear2target > 180:
            agent_bear2target -= 360
        # print(f"bearing offset {agent_bear2target} vector offset {relative_vector}.")
        agent_bear2target /= 180

        self.inputs.extend(relative_vector)
        self.inputs.append(agent_bear2target)

        # Get the kmeans cluster
        cluster = kmeans_funcs.calculate_centroid(self.kmeans_centroids, self.inputs)
        self.kmeans_cluster_counters[cluster] += 1
        self.kmeans_centroids = kmeans_funcs.update_centroids(self.kmeans_centroids,
                                                              self.kmeans_cluster_counters, self.inputs, cluster)

        return torch.Tensor(self.inputs), cluster

    def get_vector_to_target(self, position: list) -> list:
        """Get the direction vector to the target position."""
        vector_to_target = np.subtract(self.target_position, position)
        distance = np.linalg.norm(vector_to_target)
        return vector_to_target / distance

    def vector_to_bearing(self, x: float, y: float) -> float:
        """Convert a vector to a bearing."""
        rad = math.atan2(x, y)
        bearing = rad / np.pi * 180.0
        if bearing < 0.0:
            bearing += 360.0
        return bearing

    def get_agent_vector_direction(self) -> list:
        """Get the direction vector of the agent."""
        north = self.compass.getValues()
        rad = math.atan2(north[1], north[0])
        vector = [math.sin(rad), math.cos(rad)]
        return vector / np.linalg.norm(vector)

    def detect_crash(self) -> bool:
        """Detect a crash. If any of the ground sensors are triggered, then the robot has crashed."""
        if min(np.abs(self.no_noise_inputs)) < 0.025:
            # print("Crash detected.")
            return True

    def detect_success(self) -> bool:
        """Detect a successful run. If the robot is within 0.2 m of the target, then the run is successful."""
        if np.linalg.norm(np.subtract(self.gps.getValues()[0:2], self.target_position)) < 0.2:
            return True

    def calculate_state_clusters(self):
        """Calculate the state clusters."""
        # Use the save and load functions to avoid adding more, very similar, methods.
        experiences, priorities = self.replay_buffer.save()
        states, state_clusters, actions, rewards, next_states, next_state_clusters, done_flags = zip(*experiences)

        # We need to cluster all the states, not just the current state, because we want to calculate
        # the next_state_clusters as well. I'm not 100% sure this is needed, because this must introduce
        # a lot of repetition but the IQ paper does this, so I will do this for now.

        all_state = list(states)
        all_state.extend(next_states)

        all_state_clusters = list(state_clusters)
        all_state_clusters.extend(next_state_clusters)

        self.kmeans.fit_predict(all_state)
        self.kmeans_centroids = self.kmeans.cluster_centers_
        _, self.kmeans_cluster_counters = np.unique(self.kmeans.labels_, return_counts=True)

        new_state_clusters = self.kmeans.labels_[0:len(states)]
        new_next_state_clusters = self.kmeans.labels_[len(states):]

        new_experiences = list(zip(states, new_state_clusters, actions, rewards, next_states, new_next_state_clusters,
                                   done_flags))
        self.replay_buffer.load([new_experiences, priorities])

    def robot_step(self, step_count: int, state: torch.Tensor, episode_step: int, testing_flag: bool) \
            -> typing.Tuple[list, float, torch.Tensor, int]:
        """Perform a step of the robot. Take an action, calculate the reward and capture the new observation.
        :return: action, reward, new_state"""
        action = self.compute_and_actuate(step_count, state, testing_flag)
        new_state, new_cluster = self.get_state()
        reward = self.calculate_step_reward(episode_step, action)
        return action, reward, new_state, new_cluster

    # -----------------------------------------------------------------------------------------------
    def run_robot(self):
        """Main loop"""

        application_log.log(level="INFO", function=f"{inspect.stack()[0][3]}", message="Starting robot run")

        self.agent_info['run_id'] = f"{self.model_name}_{self.model_version}_" \
                                    f"{time.gmtime().tm_year}_{time.gmtime().tm_mon}_{time.gmtime().tm_mday}_" \
                                    f"{time.gmtime().tm_hour}_{time.gmtime().tm_min}_{time.gmtime().tm_sec}"

        torch.manual_seed(42)
        np.random.seed(42)

        cumulative_step_rewards = 0
        max_episode_reward = 0
        episode_step_count = 0
        total_step_count = 0
        self.__current_episode = 0
        training_flag = False
        testing_flag = False
        crashed_flag = False
        success_flag = False
        self.__epsilon = 1.0

        # DATABASE PROCESS FOR AGENT DATA ############################################################
        print("Creating database process for agent data")
        agent_database_process = Process(target=data_collector.agent_data_database_writer,
                                         args=(self.agent_data_queue,))
        agent_database_process.start()
        application_log.log(level="INFO", function=f"{inspect.stack()[0][3]}", message="Created db process for agent")
        ################################################################################################
        # OPTIMISE PROCESS FOR THE MODEL ###############################################################
        if self.simulation_hyperparameters.optimise:
            print("Creating optimise process for the model")
            optimise_process = torch.multiprocessing.spawn(ddpg_optimise.optimise_loop,
                                                           args=(self.replay_buffer, self.agent, self.target_network,
                                                                 self.agent_info),
                                                           nprocs=1, join=False, daemon=False, start_method='spawn')
            application_log.log(level="INFO", function=f"{inspect.stack()[0][3]}", message="Created opt process")
        ################################################################################################

        self.robot.step(self.time_step)  # Do 1 time step to get the sensors to update
        state, cluster = self.get_state()  # Initial sensor observation as starting state

        # MAIN LOOP ###################################################################################
        while self.robot.step(self.time_step) != -1:

            self.get_data_from_supervisor()

            if self.__episode_status == "end_of_run":
                print("Saving end of run checkpoint.")
                if self.simulation_hyperparameters.save_model:
                    self.save_model_info()
                print("End of training, exiting.")
                application_log.log(level='INFO', function=f"{inspect.stack()[0][3]}",
                                    message=f"Agent has received signal to end run {self.agent_info['run_id']}.")
                break

            if self.__episode_status == "run_training" and training_flag is False:
                # application_log.log(level='DEBUG', function=f"{inspect.stack()[0][3]}",
                #                     message=f"Detected run training episode.")
                training_flag = True
                self.agent_info['status'] = 3 if self.log_interference else 1

            elif self.__episode_status == "run_test" and testing_flag is False:
                # application_log.log(level='DEBUG', function=f"{inspect.stack()[0][3]}",
                #                     message=f"Detected run test.")
                testing_flag = True
                self.agent_info['status'] = 1  # Start optimiser

            self.agent_info['episode'] = self.__current_episode

            # Need the if because we can be waiting for the next episode to start
            # and, during that time, we don't want to do anything.
            # Can't use self.__episode_status because it then gets stuck, unable to write terminal states.
            if (training_flag or testing_flag) and crashed_flag is False and success_flag is False:
                self.send_data_to_supervisor("running episode")  # Tell supervisor we're running an episode
                # application_log.log(level='DEBUG', function=f"{inspect.stack()[0][3]}",
                #                     message=f"Running episode {self.agent_info['episode']}.")

                # GATHER EXPERIENCES AND PUT IN BUFFER ##########################################
                # application_log.log(level='DEBUG', function=f"{inspect.stack()[0][3]}",
                #                     message=f"Getting an experience. tr: {training_flag} te: {testing_flag}")
                action, step_reward, new_state, new_cluster = self.robot_step(total_step_count, state,
                                                                              episode_step_count, testing_flag)

                # Check for crash - Uncomment this to end episode on collision
                # if self.has_crashed:
                #     crashed_flag = True
                #     self.send_data_to_supervisor("crashed")
                # application_log.log(level='DEBUG', function=f"{inspect.stack()[0][3]}", message="Crashed")

                if self.goal_achieved:
                    success_flag = True
                    self.send_data_to_supervisor("success")
                    # application_log.log(level='DEBUG', function=f"{inspect.stack()[0][3]}", message="Success")

                episode_step_count += 1
                total_step_count += 1
                cumulative_step_rewards += step_reward

                # Add experience tuple to the buffer
                self.experience_buffer.append((state.cpu().numpy(), cluster, action, step_reward,
                                               new_state.cpu().numpy(), new_cluster,
                                               self.__episode_status == "end_of_episode"))

                # Don't record experiences if we are not optimising
                if self.simulation_hyperparameters.optimise:
                    # Calculate the n-step experience and add to the replay buffer.
                    if len(self.experience_buffer) >= self.simulation_hyperparameters.n_steps:
                        n_state, n_state_cluster, n_action, n_reward, \
                            n_new_state, n_new_state_cluster, n_done = self.experience_buffer.popleft()
                        g = self.simulation_hyperparameters.gamma
                        discounted_reward = n_reward
                        for _, _, _, n_r, _, _, _ in list(self.experience_buffer):
                            discounted_reward += (g * n_r)
                            g *= self.simulation_hyperparameters.gamma

                        self.replay_buffer.add(n_state, n_state_cluster, n_action, discounted_reward,
                                               new_state.cpu().numpy(), new_cluster,
                                               self.__episode_status == "end_of_episode")

                state, cluster = [new_state, new_cluster]

                # RECORD TO DATABASE ##########################################################
                self.agent_data_object.run_id = self.agent_info['run_id']
                self.agent_data_object.episode = self.__current_episode
                self.agent_data_object.step = episode_step_count
                self.agent_data_object.step_reward = step_reward
                self.agent_data_object.x = self.current_position[0]
                self.agent_data_object.y = self.current_position[1]
                self.agent_data_object.episode_end = self.__episode_status == "end_of_episode"
                self.agent_data_object.goal_achieved = self.goal_achieved
                self.agent_data_object.run_type = "training" if training_flag else "testing"
                self.agent_data_queue.put([True, self.agent_data_object])

            # END OF EPISODE ##############################################################
            # If we've been told it's the end of the episode, do a couple of things and then tell
            # the supervisor we are ready for the next one
            if self.__episode_status == "end_of_episode":
                # There will be an 'end of episode' score, so now we calculate the final reward
                # Only do these things once at the end of the episode, not while waiting for the next one.
                if episode_step_count > 0:
                    total_episode_reward = self.mean_step_rewards
                    print(f"Completed episode {self.__current_episode} with mean reward {total_episode_reward:.4f} "
                          f"in {episode_step_count} steps.")

                    # application_log.log(level='DEBUG', function=f"{inspect.stack()[0][3]}",
                    #                     message=f"Detected end of episode {self.__current_episode}")
                    if (total_episode_reward > max_episode_reward and
                            total_step_count > (self.model_hyperparameters.start_steps * 4) and
                            # testing_flag is True and
                            success_flag is True and
                            episode_step_count > 450):  # Short episodes can score very high
                        max_episode_reward = total_episode_reward
                        print(f"New high mean reward: {max_episode_reward:.4f}")
                        application_log.log(level='INFO', function=f"{inspect.stack()[0][3]}",
                                            message=f"New max reward: {max_episode_reward:.4f}")
                        if self.simulation_hyperparameters.save_model:
                            self.save_model_info()

                self.send_data_to_supervisor("end of update")  # Tell the supervisor we are ready for the next episode

                episode_step_count = 0
                cumulative_step_rewards = 0
                self.mean_step_rewards = 0
                self.calculated_rewards = []
                self.agent.reset()  # Reset the agent's noise process at the end of each episode
                # Plot episode_reward etc. to track learning - remember this loops per step so latch outputs

                # Reset flags at the end of the end of episode section
                training_flag = False
                testing_flag = False
                crashed_flag = False
                success_flag = False

                self.simulation_hyperparameters.left_motor_factor = \
                    self.simulation_hyperparameters.left_motor_factor_start  # Reset motor factor
                self.simulation_hyperparameters.sensor_bias = \
                    self.simulation_hyperparameters.sensor_bias_start  # Reset sensor bias

        # Join processes after while exits.
        # application_log.log(level='DEBUG', function=f"{inspect.stack()[0][3]}",
        #                     message="Optimiser being signalled to stop.")
        self.agent_info['status'] = -1  # Signal optimiser to stop
        # while True:                     # Wait for optimiser to acknowledge to avoid race condition
        #     if self.agent_info['status'] == -2:
        #         break
        if self.simulation_hyperparameters.optimise:
            optimise_process.join()
            application_log.log(level='INFO', function=f"{inspect.stack()[0][3]}",
                                message="Optimiser process has joined.")

        self.agent_data_queue.put([False, None])  # Signal agent db process to stop
        agent_database_process.join()
        application_log.log(level='INFO', function=f"{inspect.stack()[0][3]}",
                            message="Agent database process has joined.")

        self.replay_buffer_manager.shutdown()


if __name__ == "__main__":
    my_robot = Robot()
    controller = Controller(my_robot, DDPGHyperParameters(), SimulationHyperParameters())
    controller.run_robot()
