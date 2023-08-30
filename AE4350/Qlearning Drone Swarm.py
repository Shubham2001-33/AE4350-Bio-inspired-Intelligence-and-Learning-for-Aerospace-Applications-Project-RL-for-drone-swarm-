import gym, gym_minigrid
from gym_minigrid.plot import Plotter
import numpy as np
import time
from optparse import OptionParser
from collections import defaultdict
import csv
import pickle
import yaml


def main():
    parser = OptionParser()
    parser.add_option(
        "-e",
        "--env-name",
        dest="env_name",
        help="gym environment to load",
        default='MiniGrid-Empty-8x8-v0'
    )
    (options, args) = parser.parse_args()

    # gym environment is loaded here
    env = gym.make(options.env_name)

    # Action is converted from numerical value to directional action for the grid
    def get_action(t_action):
        act = None

        if t_action == 0:
            act = env.actions.left
        elif t_action == 1:
            act = env.actions.up
        elif t_action == 2:
            act = env.actions.right
        elif t_action == 3:
            act = env.actions.down
        else:
            print("unknown key")
            return

        return act

    plotter = Plotter()

    with open("config.yml", 'r') as ymlfile:
        configuration= yaml.load(ymlfile,Loader=yaml.FullLoader)

    #rendering the boolean values
    grid_render = configuration['rnd']['grid_render']
    grid_obs_render = configuration['rnd']['grid_obs_render']
    obs_render = configuration['rnd']['obs_render']
    gray = configuration['rnd']['grayscale']
    sleep = configuration['rnd']['sleep']

    #parameters are pulled from the config file and can be adjusted there  
    episodes = configuration['ql']['episodes']
    epsilon = configuration['ql']['epsilon']
    decay = configuration['ql']['decay']
    alpha = configuration['ql']['alpha']
    gamma = configuration['ql']['gamma']

   
    steps_to_completion = []

    # Initialization of q-table
   
    q_table = defaultdict(lambda: np.zeros(shape=(len(env.actions),)))

    run_ep = 0
    for e in range(episodes):

        # new exploration rate (epsilon) value is computed each time. It decays with each iteration, when multiplied by a decay factor.
        
        epsilon = epsilon*decay
        cumulative_reward = [0] # Initialize cumulative reward for each agent

        # agents/drones are initialized
        obs = env.reset()

        states = {}
        for a_id in obs:
            temp_obs = ''
            for list in obs[a_id]:
                temp = ','.join(map(str, list))
                temp_obs += ',' + temp
            states[a_id]  = temp_obs

        while True:
            if obs_render:
                env.get_obs_render(obs, grayscale=gray)
            if grid_render:
                env.render('human', highlight=grid_obs_render, grayscale=gray, info="Episode: %s \tStep: %s" % (str(e),str(env.step_count)))

            time.sleep(sleep)

            # Here it is determined if in the current iteration all agents will explore or exploit the grid
            if np.random.uniform(0, 1) < epsilon:
                exploit = False 
            else:
                exploit = True 

            # Action for each agent is determined
            actions = {}
            for a_id in obs:
                if exploit is False:
                    temp_action = env.action_space.sample() 
                else:
                    temp_action = np.argmax(q_table[states[a_id]]) 

            # Action is converted from numerical value to directional action for the grid
                actions[a_id] = get_action(temp_action)

            # action is executed
            obs, reward, done, agents, info = env.step(actions)
            #cumulative_reward += reward
            
            # q-table values for each agent are calculated
            for a_id in obs:
                # agents new position returned from the environment are used to convert grid coordinates to states
              
                next_state = ''
                for list in obs[a_id]:
                    temp = ','.join(map(str, list))
                    next_state += ',' + temp
                q_old = q_table[states[a_id]][actions[a_id]]
                
                
                next_max = np.max(q_table[next_state])

                # new q values are calculated
                q_new = (1-alpha*decay) * q_old + alpha*decay * (reward[a_id] + gamma * next_max)
                print(str(a_id) + ':' + 'episode=%s, step=%s, reward=%.2f, q_new=%.2f, state=%s, action=%s' \
                            % (e, env.step_count, reward[a_id], q_new, states[a_id], actions[a_id]))
                
                q_table[states[a_id]][actions[a_id]] = q_new

                states[a_id] = next_state
           
            if done:
                print('done!')

                # plot steps by episode
                steps_to_completion.append(env.step_count)
                break

             
    print(f"Episode {e + 1} - Cumulative Reward: {sum(cumulative_reward)}")
    print("Training finished.\n")

    # steps_to_completion stored in excel file
    filename = "steps_to_completion.csv".format(env.grid_size, env.grid_size, configuration['env']['obstacles'], env.n_agents, env.obs_radius, env.reward_type)
    w = csv.writer(open(filename, "w+"))
    for i in range(len(steps_to_completion)):
        w.writerow([i, steps_to_completion[i]])

    # png save plot/show
    plotter.plot_steps(steps_to_completion)

    # q_table
    w = csv.writer(open("qtable.csv", "w+"))
    for key, val in q_table.items():
        w.writerow([key, val])


    # pkl q_table
    f = open("qt.pkl","wb+")
    for key in q_table:
        print(key)
    pickle.dump(dict(q_table), f)
    f.close()

if __name__ == "__main__":
    main()
