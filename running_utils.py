import numpy as np
import time

from env import ToyEnv
from agents import UCBVI, MVP


def trial(S, A, H, K, run_lst, trial_id=1):

    env = ToyEnv(S, A, H)
    opt = env.computeOptimalValueFunction()
    opt_rewards = opt * np.ones(K)

    results_all = {}

    for alg in run_lst:
        start_time = time.time()
        print("Starting trial " + str(trial_id+1) + " for " + alg + ".")
        if alg == "MVP":   
            agent = MVP(S, A, H, K, env.getRewards(), 1/K)
        elif alg == "CHI":
            agent = UCBVI(S, A, H, K, env.getRewards(), 'CH', improved=True)
        elif alg == "CH":
            agent = UCBVI(S, A, H, K, env.getRewards(), 'CH', improved=False)
        elif alg == "BFI":
            agent = UCBVI(S, A, H, K, env.getRewards(), 'BF', improved=True)
        elif alg == "BF":
            agent = UCBVI(S, A, H, K, env.getRewards(), 'BF', improved=False)
        else:
            raise ValueError("Error in input")
        rewards = np.zeros((K, H))
        for k in range(K):
            env.reset()
            agent.newEpisode()
            state = env.getCurrentState()
            for h in range(H):
                action = agent.choose(state, h)
                state, reward = env.step(action)
                agent.update(state, reward)
                rewards[k, h] = reward
        ep_rewards = rewards.sum(axis=1)
        ep_regret = opt_rewards - ep_rewards
        ep_regret = ep_regret.cumsum()
        print("Terminating trial " + str(trial_id+1) + " for " + alg + ". Elapsed time: " + str(int(time.time() - start_time)) + " sec.")
        results_all[alg] = ep_regret
    
    return results_all