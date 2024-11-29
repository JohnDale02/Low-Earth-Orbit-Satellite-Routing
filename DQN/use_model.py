import tensorflow as tf
import os 
from train_DQN import DQNAgent
import gym
import numpy as np
import networkx as nx
from random import random

ENV_NAME = 'GraphEnv-v1'
graph_topology = 0 # 0==NSFNET, 1==GEANT2, 2==Small Topology, 3==GBN
SEED = 37
ITERATIONS = 10000
TRAINING_EPISODES = 20
EVALUATION_EPISODES = 40
FIRST_WORK_TRAIN_EPISODE = 60
listofDemands = [8, 32, 64]

MULTI_FACTOR_BATCH = 6 # Number of batches used in training
TAU = 0.08 # Only used in soft weights copy
env = gym.make(ENV_NAME)
np.random.seed(SEED)
env.seed(SEED)
env.generate_environment(graph_topology, listofDemands)

batch_size = 32
agent = DQNAgent(batch_size, env.numEdges, env.listofDemands)

eval_ep = 0
train_ep = 0
max_reward = 0
reward_id = 0

# Load the trained model
full_path = os.path.realpath(__file__)
checkpoint_path = os.path.join(os.path.dirname(full_path), "modelssample_DQN_agent", "ckpt-349")  # HIGHEST SCORED MODEL << HARDCODED 
checkpoint = tf.train.Checkpoint(model=agent.primary_network, optimizer=agent.optimizer)
status = checkpoint.restore(checkpoint_path)
print(f"Checkpoint loaded: {checkpoint_path}") 
checkpoint.restore(checkpoint_path)
print("Model Summary:")
agent.primary_network.summary()

# Simulate dynamic conditions
num_scenarios = 100
model_metrics = []
baseline_metrics = []


def route_data(env, path, demand):
    """
    Simulates routing data along a chosen path.
    Deducts capacity from each edge in the path if the demand can be routed.
    Returns the amount of data successfully routed (0 if the demand cannot be routed).
    """
    # Check if the path can handle the demand
    for i in range(len(path) - 1):
        edge = env.edgesDict[f"{path[i]}:{path[i + 1]}"]
        if env.graph_state[edge][0] < demand:
            # If any edge can't handle the demand, routing fails
            return 0

    # If all edges can handle the demand, route it
    for i in range(len(path) - 1):
        edge = env.edgesDict[f"{path[i]}:{path[i + 1]}"]
        env.graph_state[edge][0] -= demand

    # Successfully routed the full demand
    return demand

def main():
    for scenario in range(num_scenarios):
        env.reset()  # Reset the environment for each scenario

        model_throughput, baseline_throughput = 0, 0
        for demand, source, destination in env.traffic:
            #print(f"Demand: {demand}, Source: {source}, Destination: {destination}")
            try:
                # Trained model
                action, _ = agent.act(env, env.graph_state, demand, source, destination, True)
                chosen_path = env.allPaths[str(source) + ':' + str(destination)][action]
                model_throughput += route_data(env, chosen_path, demand)

                # Baseline (Shortest Path)
                static_path = nx.shortest_path(env.graph, source, target=destination, weight='capacity')
                baseline_throughput += route_data(env, static_path, demand)
            except Exception as e:
                print(f"Error during routing: {e}")
                continue

        model_metrics.append(model_throughput)
        baseline_metrics.append(baseline_throughput)
        print(f"Scenario {scenario} Completed: Model Throughput = {model_throughput}, Baseline Throughput = {baseline_throughput}")

    print(f"Model Avg Throughput: {np.mean(model_metrics)}")
    print(f"Baseline Avg Throughput: {np.mean(baseline_metrics)}")

if __name__ =="__main__":
    main()