# Author:   Andy Edmondson
# Email:    andrew.edmondson@gmail.com
# Date:     3 March 2023
# Purpose:  Optimisation functions for DDPG
#
# Includes: compute_q_loss
#           compute_policy_loss
#           one_step_update
#
# References
# ----------
# DDPG paper:
#       Lillicrap, Timothy P., Jonathan J. Hunt, Alexander Pritzel, Nicolas Manfred Otto Heess, Tom Erez,
#       Yuval Tassa, David Silver, and Daan Wierstra. ‘Continuous Control with Deep Reinforcement Learning’.
#       CoRR abs/1509.02971 (2016).
#
# This implementation based on:
#       Sanghi, Nimish. Deep Reinforcement Learning with Python: With PyTorch, TensorFlow and OpenAI Gym.
#       New York: Apress, 2021. https://doi.org/10.1007/978-1-4842-6809-4.
#       This code is licenced under the Freeware Licence as described here:
#       https://github.com/Apress/deep-reinforcement-learning-python/blob/main/LICENSE.txt
#
# ----------
import torch
import torch.multiprocessing
from torch.multiprocessing import Process, Queue
from time import sleep
import numpy as np
from copy import deepcopy
import inspect

import application_log
from network_model import MLPActorCritic
from replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from hyperparameters import OptimiseHyperparams
from data_collector import OptimiseData
import data_collector


def compute_q_loss(agent, target_network, batch, weights, idxs, gamma=0.99):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    states, state_clusters, actions, rewards, next_states, next_state_clusters, done_flags = zip(*batch)

    # convert numpy array to torch tensors
    states = torch.tensor(np.array(states), dtype=torch.float).to(device)
    actions = torch.tensor(np.array(actions), dtype=torch.float).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float).to(device)
    next_states = torch.tensor(np.array(next_states), dtype=torch.float).to(device)
    done_flags = torch.tensor(done_flags, dtype=torch.float).to(device)
    weights = torch.tensor(weights, dtype=torch.float).to(device)

    # get q-values for all actions in current states
    # use agent network
    predicted_qvalues = agent.q(states, actions)
    # Bellman backup for Q function
    with torch.no_grad():
        q__next_state_values = target_network.q(next_states, target_network.policy(next_states))
        target = rewards + gamma * (1 - done_flags) * q__next_state_values

    # MSE loss against weighted Bellman backup
    loss_q = (((predicted_qvalues - target)**2) * weights).mean()
    # calculate new priorities and update buffer
    with torch.no_grad():
        new_priorities = predicted_qvalues.detach() - target.detach()
        new_priorities = np.absolute(new_priorities.detach().cpu().numpy())
        # replay_buffer.update_priorities(buffer_idxs, new_priorities)
    # application_log.log(level='DEBUG', function=f"{inspect.stack()[0][3]}", message="Q-loss 0.8")
    return loss_q, predicted_qvalues.mean().item(), new_priorities, idxs


def compute_policy_loss(agent, batch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    states, _, _, _, _, _, _ = zip(*batch)

    # convert numpy array to torch tensors
    states = torch.tensor(np.array(states), dtype=torch.float).to(device)
    predicted_qvalues = agent.q(states, agent.policy(states))
    loss_policy = - predicted_qvalues.mean()
    return loss_policy


def one_step_update(agent, target_network, hyperparameters: OptimiseHyperparams,
                    replay: PrioritizedReplayBuffer, agent_info: dict, step: int):
    # one step gradient for q-values
    r_id = agent_info['run_id']
    ep = agent_info['episode']
    batch, weights, idxs = replay.sample(hyperparameters.batch_size)
    data = OptimiseData(run_id=r_id, episode=ep,
                        step=step, q_loss=0, a_loss=0, inter=0, q_val=0)

    hyperparameters.q_optimizer.zero_grad()
    loss_q, data.q_val, new_priorities, index = compute_q_loss(agent, target_network, batch, weights, idxs,
                                                               hyperparameters.gamma)
    loss_q.backward()
    data.q_loss = loss_q.item()
    torch.nn.utils.clip_grad_norm_(agent.q.parameters(), 1.0)               # clip gradients
    hyperparameters.q_optimizer.step()
    with torch.no_grad():
        replay.update_priorities(index, new_priorities)

    # Freeze Q-network
    for params in agent.q.parameters():
        params.requires_grad = False

    # One-step gradient for policy network
    hyperparameters.policy_optimizer.zero_grad()
    loss_policy = compute_policy_loss(agent, batch)
    data.a_loss = loss_policy.item()
    loss_policy.backward()
    torch.nn.utils.clip_grad_norm_(agent.policy.parameters(), 1.0)         # clip gradients
    hyperparameters.policy_optimizer.step()

    # Un-freeze Q-network
    for params in agent.q.parameters():
        params.requires_grad = True

    # update target networks with polyak averaging
    with torch.no_grad():
        for params, params_target in zip(agent.parameters(), target_network.parameters()):
            params_target.data.mul_(1 - hyperparameters.polyak)
            params_target.data.add_(hyperparameters.polyak * params.data)

    return data


def optimise_loop(i, replay: PrioritizedReplayBuffer, agent: MLPActorCritic, target: MLPActorCritic, agent_info: dict):
    """Optimise the agent's policy and Q-networks using a batch of experiences from the replay buffer"""
    print(f"Optimiser {i} starting")
    application_log.log(level='INFO', function=f"{inspect.stack()[0][3]}",
                        message=f"Optimiser process started successfully with status {agent_info['status']}.")

    # DATABASE PROCESS FOR OPTIMISER DATA ############################################################
    print("Starting optimiser database process")
    optimise_queue = Queue(maxsize=50_000)
    optimise_database_process = Process(target=data_collector.optimiser_data_database_writer, args=(optimise_queue,))
    optimise_database_process.start()
    application_log.log(level='INFO', function=f"{inspect.stack()[0][3]}",
                        message="Started optimiser database process.")
    #################################################################################################
    hyperparams = OptimiseHyperparams()
    hyperparams.q_optimizer = torch.optim.Adam(agent.q.parameters(), lr=hyperparams.q_lr)
    hyperparams.policy_optimizer = torch.optim.Adam(agent.policy.parameters(), lr=hyperparams.policy_lr)

    step = 0
    while True:
        step += 1
        if agent_info['status'] == -1:
            application_log.log(level='INFO', function=f"{inspect.stack()[0][3]}",
                                message="Received signal to stop optimisation process.")
            optimise_queue.put([False, None])
            agent_info['status'] = -2           # Send an ack to the agent process.
            break
        elif agent_info['status'] == 1 and len(replay) > hyperparams.update_after:         # run
            # Do the optimisation step

            data = one_step_update(agent, target, hyperparams, replay, agent_info, step)
            optimise_queue.put([True, data])

            # Check if the agent has been updated
        elif agent_info['status'] == 3 and len(replay) > hyperparams.aei_batch_size:   # AEI measurement
            aei_batch, weights, idxs = replay.sample(hyperparams.aei_batch_size)
            with torch.no_grad():
                aei_q_loss, _, _, _ = compute_q_loss(agent, target, aei_batch, weights, idxs, hyperparams.gamma)

            data = one_step_update(agent, target, hyperparams, replay, agent_info, step)

            # Now the updates have been made, measure interference of the batch.
            # Interference is the mean squared difference of the q loss before and after the update.
            # Catastrophic Interference in Reinforcement Learning:
            # A Solution Based on Context Division and Knowledge Distillation
            # https://arxiv.org/pdf/2109.00525.pdf - Page 5 - Equation 5
            # In the paper, these expectation values are normalised prior to plotting - see plotted values are [0,1]
            with torch.no_grad():
                updated_aei_q_loss, _, _, _ = compute_q_loss(agent, target, aei_batch, weights, idxs, hyperparams.gamma)
                data.inter = updated_aei_q_loss.item()**2 - aei_q_loss.item()**2

            optimise_queue.put([True, data])
            # Check if the agent has been updated
        else:
            sleep(0.1)

    optimise_database_process.join()
    application_log.log(level='INFO', function=f"{inspect.stack()[0][3]}",
                        message="Optimiser database process has joined.")


# TEST FUNCTION ##########################################################################################
def test_function(i, _test_model, _test_target, _test_batch):
    # test_loss, test_value, _, _ = compute_q_loss(test_model, test_target, test_batch, 0.99)
    # print(test_loss.item(), test_value)
    rb = ReplayBuffer(1000)
    rb.add(_test_batch[0][0], _test_batch[0][1], _test_batch[0][2], _test_batch[0][3], _test_batch[0][4])
    rb.add(_test_batch[1][0], _test_batch[1][1], _test_batch[1][2], _test_batch[1][3], _test_batch[1][4])
    rb.add(_test_batch[3][0], _test_batch[3][1], _test_batch[3][2], _test_batch[3][3], _test_batch[3][4])
    rb.add(_test_batch[4][0], _test_batch[4][1], _test_batch[4][2], _test_batch[4][3], _test_batch[4][4])
    rb.add(_test_batch[2][0], _test_batch[2][1], _test_batch[2][2], _test_batch[2][3], _test_batch[2][4])
    optimise_loop(i, PrioritizedReplayBuffer(1000), _test_model, _test_target,
                  {"run_id": "id", "episode": 0, "status": -1})

    # optimise_queue = Queue(maxsize=10_000)
    # optimise_database_process = Process(target=data_collector.optimiser_data_database_writer, args=(optimise_queue,))
    # optimise_database_process.start()
    # sleep(5)
    # optimise_queue.put([False, None])
    # optimise_database_process.join()


if __name__ == "__main__":
    test_model = MLPActorCritic(2, 2, 1.).cuda()
    test_target = deepcopy(test_model).cuda()
    test_model.share_memory()
    test_target.share_memory()
    test_batch = [[np.array([0.5, 0.5]), [0.5, 0.5], 0.1, np.array([0.5, 0.5]), False],
                  [np.array([0.5, 0.5]), [0.5, 0.5], 0.2, np.array([0.5, 0.5]), False],
                  [np.array([0.5, 0.5]), [0.5, 0.5], 0.4, np.array([0.5, 0.5]), False],
                  [np.array([0.5, 0.5]), [0.5, 0.5], 0.2, np.array([0.5, 0.5]), True],
                  [np.array([0.5, 0.5]), [0.5, 0.5], 0.1, np.array([0.5, 0.5]), False]]

    #test_process = Process(target=test_function, args=(test_model, test_target, test_batch))
    # Need to use spawn!
    # https://pytorch.org/docs/stable/multiprocessing.html
    test_proc = torch.multiprocessing.spawn(test_function, args=(test_model, test_target, test_batch), nprocs=1,
                                            join=False, daemon=False, start_method='spawn')

    print("Tell me this isn't blocking!")
    # test_loss, test_value, _, _ = compute_q_loss(test_model, test_target, test_batch, 0.99)
    # print(test_loss.item(), test_value)
    test_proc.join()
