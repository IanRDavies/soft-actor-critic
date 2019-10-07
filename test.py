import gym
import argparse
from utils import initialise_environment
from soft_actor_critic import SAC


def test(agent, env, num_episodes, max_episode_length, render=False):
    '''
    Run several test episodes for a trained agent.

    Arguments
    agent - SAC instance
        The agent to be tested
    env - gym.Env
        The environment in which the agent is to act
    num_episodes - int
        The number of episodes to run
    max_episode_length - int
        The maximal number of timesteps to allow an episode to run for.
        If the episode is ongoing at this point terminate it.
    render - bool
        Boolean determining whether or not to render the environment
        for the user to view.
    '''
    # Loop through the required number of episodes
    for i in range(num_episodes):
        # Attain initial observation
        obs = env.reset()
        # Reset end of episode boolean and time and reward trackers
        terminal = False
        t = 0
        ep_rew = 0
        # Play an episode
        while not terminal:
            # Agent selects action and the environment processes it
            action = agent.action(obs)
            next_obs, reward, terminal, info = env.step(action)
            # Increment the timestep and if the episode has gone on
            # too long end it
            t += 1
            if t == max_episode_length:
                terminal = True
            # Update the full episode reward
            ep_rew += reward
            # Update the observation ready for the next step
            obs = next_obs
            # If required render the environment for the user
            if render:
                env.render()
        # Log some statistics at the end of each episode
        print('Episode {}:\t\tReward: {:.2f}\t\tEpisode Length: {}'.format(i, ep_rew, t))


def parse_args():
    '''Parse arguments passed from the command line'''
    # Set up parser
    parser = argparse.ArgumentParser('Test Arguments for Soft Actor-Critic.')
    # Environment Arguments
    parser.add_argument('--env', type=str, default='LunarLanderContinuous-v2')
    parser.add_argument('-r', '--render', default=True, action='store_true')
    parser.add_argument('-l', '--load_path', type=str, default='./saved_model/model')
    parser.add_argument('-n', '--num_episodes', type=int, default=10)
    parser.add_argument('-T', '--max_ep_len', type=int, default=1000)
    # Arguments should match those used to train the agent.
    # Other hyperparameters are not important as there is no training
    parser.add_argument('-pns', '--policy_net_shape', type=tuple, default=(100, 30))
    parser.add_argument('-qns', '--q_net_shape', type=tuple, default=(100, 30))
    parser.add_argument('-vns', '--value_net_shape', type=tuple, default=(100, 30))
    return parser.parse_args()


def main(args):
    '''Load a soft actor-critic agent and let them play'''
    # Set up environment
    env = initialise_environment(args.env)
    # Initialise agent with defaults in place of unused
    # hyperparameters
    agent = SAC(env, 0.99, 0.2, 1e-3, 1e-3,
                0.995, args.policy_net_shape, args.q_net_shape,
                args.value_net_shape, 256, int(1e6))
    # Load in the weights from a previous saved model
    agent.load(args.load_path)
    # Run the test function
    test(agent, env, args.num_episodes,
         args.max_ep_len, args.render)


if __name__ == '__main__':
    # Parse the arguments and then run the main test function.
    args = parse_args()
    main(args)
