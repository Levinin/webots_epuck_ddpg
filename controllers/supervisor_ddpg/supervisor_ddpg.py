import inspect
import sqlite3
from sqlite3 import OperationalError

from controller import Supervisor
from controller import Keyboard
from controller import Display

import numpy as np
import sys


def log(level: str, function: str, message: str):
    """Log to the database. Level is one of DEBUG, INFO, WARNING, ERROR, CRITICAL.
    Duplicated method from application_log.py to avoid distant import."""

    conn = sqlite3.connect("../logs/app_log.db")
    cursor: sqlite3.Cursor = conn.cursor()

    while True:  # Keep trying until successful, otherwise may occasionally fail due to database lock
        try:
            cursor.execute("INSERT INTO app_log (id, level, function, message) VALUES (NULL, ?, ?, ?)",
                           (level, function, message))
            conn.commit()
            break
        except OperationalError as e:
            if "SQLITE_BUSY" not in str(e):
                raise e

    conn.close()


class SimpleSupervisor:
    def __init__(self):
        self.time_step = 50    # ~ 20 Hz        #32  # ms  ~ 30 Hz

        self.supervisor = Supervisor()

        # Get the robot
        self.robot_node = self.supervisor.getFromDef("EPUCK")
        if self.robot_node is None:
            sys.stderr.write("No DEF Controller node found in the current world file\n")
            sys.exit(1)

        # Get the robot translation and rotation
        self.trans_field = self.robot_node.getField("translation")
        self.rot_field = self.robot_node.getField("rotation")

        # Get the target
        self.target_node = self.supervisor.getFromDef("TARGET")
        if self.target_node is None:
            sys.stderr.write("No DEF node 'TARGET' found in the current world file\n")
            sys.exit(1)

        # Get the 10 obstacles
        self.obstacle_node1 = self.supervisor.getFromDef("OBS1")
        if self.obstacle_node1 is None:
            sys.stderr.write("No DEF node 'OBS1' found in the current world file\n")
            sys.exit(1)
        self.obstacle_node2 = self.supervisor.getFromDef("OBS2")
        if self.obstacle_node2 is None:
            sys.stderr.write("No DEF node 'OBS2' found in the current world file\n")
            sys.exit(1)
        self.obstacle_node3 = self.supervisor.getFromDef("OBS3")
        if self.obstacle_node3 is None:
            sys.stderr.write("No DEF node 'OBS3' found in the current world file\n")
            sys.exit(1)
        self.obstacle_node4 = self.supervisor.getFromDef("OBS4")
        if self.obstacle_node4 is None:
            sys.stderr.write("No DEF node 'OBS4' found in the current world file\n")
            sys.exit(1)
        self.obstacle_node5 = self.supervisor.getFromDef("OBS5")
        if self.obstacle_node5 is None:
            sys.stderr.write("No DEF node 'OBS5' found in the current world file\n")
            sys.exit(1)
        self.obstacle_node6 = self.supervisor.getFromDef("OBS6")
        if self.obstacle_node6 is None:
            sys.stderr.write("No DEF node 'OBS6' found in the current world file\n")
            sys.exit(1)
        self.obstacle_node7 = self.supervisor.getFromDef("OBS7")
        if self.obstacle_node7 is None:
            sys.stderr.write("No DEF node 'OBS7' found in the current world file\n")
            sys.exit(1)
        self.obstacle_node8 = self.supervisor.getFromDef("OBS8")
        if self.obstacle_node8 is None:
            sys.stderr.write("No DEF node 'OBS8' found in the current world file\n")
            sys.exit(1)
        self.obstacle_node9 = self.supervisor.getFromDef("OBS9")
        if self.obstacle_node9 is None:
            sys.stderr.write("No DEF node 'OBS9' found in the current world file\n")
            sys.exit(1)
        self.obstacle_node10 = self.supervisor.getFromDef("OBS10")
        if self.obstacle_node10 is None:
            sys.stderr.write("No DEF node 'OBS10' found in the current world file\n")
            sys.exit(1)

        # Get the target translation
        self.target_trans_field = self.target_node.getField("translation")

        # Enable Receiver and Emitter
        self.emitter = self.supervisor.getDevice("emitter")
        self.receiver = self.supervisor.getDevice("receiver")
        self.receiver.enable(self.time_step)
        self.previous_message = ""

        self.__waiting_for_end = True

        self.__crashed = False
        self.__success = False

    def get_message_from_agent(self):
        """Get a message from the agent message queue."""
        if self.receiver.getQueueLength() > 0:
            text = self.receiver.getString()
            self.receiver.nextPacket()
            if text == self.previous_message:
                return
            self.previous_message = text
            if text == "end of update":
                self.__waiting_for_end = False
            elif text == "crashed":
                self.__crashed = True
            elif text == "success":
                self.__success = True

    def send_message_to_agent(self, status: str, end_of_episode_reward: float = 0.0,
                              episode: int = 0, interference: bool = False):
        """Send a message to the agent.
        A message consists of the following parts:
        :param status: The status of the robot. One of "run_test", "run_train", "run_demo", "end_of_episode"
        :param end_of_episode_reward: The reward for the episode
        :param episode: The episode number
        :param interference: Whether the episode was run with interference or not
        Additionally, the message contains the current target position.
        """
        self.emitter.send(f"{status} {end_of_episode_reward} {episode} {interference} "
                          f"[{self.target_trans_field.getSFVec3f()[0]},{self.target_trans_field.getSFVec3f()[1]}]")

    def reset_robot(self, episode: int, test: bool = False):
        """Set the robot at the start of an episode to a random location within the arena."""
        self.trans_field.setSFVec3f([-1.7, -1.7, 0.0])
        self.rot_field.setSFRotation([0.0, 0.0, 1, 0.75])
        self.robot_node.resetPhysics()

    def reset_target(self, episode: int, test: bool = False):
        """Set the target at the start of an episode to a random location within the arena. Make sure it is
        at least 1 m away from the robot."""
        success_flag = False
        for _ in range(100):        # Try to place 100 times before failing
            # Make it likely the agent finds the target for the first few episodes
            if episode < 10 and not test:
                self.target_trans_field.setSFVec3f([-1.5, -1.5, -0.515])
                success_flag = True
                break
            else:
                self.target_trans_field.setSFVec3f([np.random.uniform(-1.8, 1.8), np.random.uniform(-1.8, 1.8), -0.515])

                vector = np.subtract(self.target_trans_field.getSFVec3f(), self.trans_field.getSFVec3f())
                if np.linalg.norm(vector[0:2]) > 2.0:
                    success_flag = True
                    break
        self.target_node.resetPhysics()
        if not success_flag:
            raise RuntimeError("Could not place target at least 1 m away from robot")

    def reset_obstacles(self, episode: int):
        """Set the 10 obstacles at the start of an episode to random locations within the arena. Make sure they are
        at least 0.6 m away from the robot, target and each other."""

        locations = [self.target_trans_field.getSFVec3f()[0:2], self.trans_field.getSFVec3f()[0:2]]

        # Place the obstacles. If it fails to place an obstacle, just move on without it.
        self.find_location(self.obstacle_node1, locations)
        self.find_location(self.obstacle_node2, locations)
        self.find_location(self.obstacle_node3, locations)
        self.find_location(self.obstacle_node4, locations)
        self.find_location(self.obstacle_node5, locations)
        self.find_location(self.obstacle_node6, locations)
        self.find_location(self.obstacle_node7, locations)
        self.find_location(self.obstacle_node8, locations)
        self.find_location(self.obstacle_node9, locations)
        self.find_location(self.obstacle_node10, locations)

    def find_location(self, node, locations: list):
        """Find the location of a node in the arena."""
        too_close = False
        for _ in range(100):        # Try to place 100 times before failing
            new_location = [np.random.uniform(-1.7, 1.7), np.random.uniform(-1.7, 1.7), 0.2]
            too_close = False
            for loc in locations:
                vector = np.subtract(new_location[0:2], loc)
                if np.linalg.norm(vector) < 0.7:
                    too_close = True
                    break
            if too_close:
                continue
            node.getField("translation").setSFVec3f(new_location)
            locations.append(new_location[0:2])
            break
        if too_close:
            node.getField("translation").setSFVec3f([-5, -5, -5])     # Move the obstacle out of the way
            log(level="INFO", function=f"{inspect.stack()[0][3]}", message=f"Could not place obstacle {node} "
                                                                           f"at least 0.6 m away from robot, target "
                                                                           f"and other obstacles")

    def run_episode(self, seconds: int = 15) -> float:
        """Run episode for passed number of seconds."""
        stop = int((seconds * 1000) / self.time_step)
        iterations = 0

        while self.supervisor.step(self.time_step) != -1:
            if stop == iterations:
                break
            iterations += 1

            self.get_message_from_agent()
            if self.__crashed or self.__success:
                break

        end_of_episode_reward = 0.0
        return end_of_episode_reward

    def run_demo(self):
        """Run the trained policy"""
        pass

    def run_test_episodes(self, num_episodes: int = 5, seconds: int = 20):
        """Run test episodes to check the current policy performance."""
        # log(level="DEBUG", function=f"{inspect.stack()[0][3]}", message=f"About to start test episodes")

        for episode in range(num_episodes):
            print(f"Supervisor setting test episode: {episode}")
            # log(level="DEBUG", function=f"{inspect.stack()[0][3]}", message=f"About to run test episode {episode}")
            self.complete_episode(episode, "run_test", seconds, test=True)

    def complete_episode(self, episode: int, status: str, seconds: int, interference: bool = False, test: bool = False):
        self.reset_robot(episode, test)
        self.reset_target(episode, test)         # Must happen after reset_robot()
        self.reset_obstacles(episode)            # Must happen after reset_target()

        # Reset flags ahead of running the next episode
        self.__waiting_for_end = True
        self.__crashed = False
        self.__success = False

        # Tell the robot a new episode is starting.
        self.send_message_to_agent(status=status, end_of_episode_reward=0.0, episode=episode, interference=interference)
        end_of_episode_reward = self.run_episode(seconds)

        self.send_message_to_agent("end_of_episode", end_of_episode_reward, episode, interference)
        # log(level="DEBUG", function=f"{inspect.stack()[0][3]}",
        #     message=f"Sent end of episode  {episode} message to agent")

        # Wait for signal from agent to make sure we don't interrupt an update.
        while self.__waiting_for_end:
            self.get_message_from_agent()
        # log(level="DEBUG", function=f"{inspect.stack()[0][3]}", message=f"Agent ack for end of ep {episode}")

    def run_training_episodes(self, num_episodes: int = 50, seconds: int = 15,
                              test_every: int = 5000, test_eps: int = 5,
                              interference: bool = False, test: bool = False):
        """Runs the training episodes. The main processing is completed in the epuck."""
        log(level="INFO", function=f"{inspect.stack()[0][3]}", message="Starting training episodes")
        for episode in range(num_episodes):

            if episode % test_every == 0 and episode != 0:
                # log(level="DEBUG", function=f"{inspect.stack()[0][3]}", message="Running test episodes")
                self.run_test_episodes(num_episodes=test_eps, seconds=seconds)

            print(f"Supervisor setting training episode: {episode}")
            # log(level="DEBUG", function=f"{inspect.stack()[0][3]}", message=f"About to run training ep {episode}")
            self.complete_episode(episode, "run_training", seconds, interference, test)

        # log(level="DEBUG", function=f"{inspect.stack()[0][3]}", message=f"Completed training episodes")
        # After all training is complete, run a final test.
        self.run_test_episodes(num_episodes=test_eps, seconds=seconds)
        # log(level="DEBUG", function=f"{inspect.stack()[0][3]}", message=f"Completed final test episodes")

        # Once episodes have been completed, tell the controller to exit this run.
        self.send_message_to_agent("end_of_run", 0.0, 0, False)
        # log(level="DEBUG", function=f"{inspect.stack()[0][3]}", message=f"Sent message to agent to end run")


if __name__ == "__main__":
    # Call Supervisor function to initiate the supervisor module   
    supervisor = SimpleSupervisor()

    keyboard = Keyboard()
    keyboard.enable(50)
    print_flag = True

    NUM_EPISODES = 400
    SECONDS = 60
    TEST_EPS = 250

    log(level="INFO", function=f"{inspect.stack()[0][3]}", message="Starting supervisor")

    while supervisor.supervisor.step(supervisor.time_step) != -1:
        if print_flag:
            print("\n*\n********\n***************\nEnter 't' to train a policy, 'i' to train with interference, "
                  "'r' to run a policy, "
                  "'q' to quit: \n***************\n********\n*\n")
            print_flag = False
        key = keyboard.getKey()
        if key == ord('T'):
            supervisor.run_training_episodes(num_episodes=NUM_EPISODES, seconds=SECONDS,
                                             test_eps=0, interference=False)
            break
        if key == ord('I'):
            supervisor.run_training_episodes(num_episodes=NUM_EPISODES*2, seconds=SECONDS,
                                             test_eps=0, interference=True)
            break
        elif key == ord('R'):
            supervisor.run_training_episodes(num_episodes=0, seconds=SECONDS,
                                             test_eps=TEST_EPS, interference=False, test=True)
            break
        elif key == ord('Q'):
            break


