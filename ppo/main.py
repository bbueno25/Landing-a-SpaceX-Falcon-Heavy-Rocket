import argparse
import logging
import os
import task

logging.basicConfig(level=logging.INFO)
DIR_PATH = os.path.dirname(os.path.realpath(__file__))
DIR_PATH = os.path.abspath(os.path.join(DIR_PATH, os.pardir))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size',
                        default=256,
                        help='How many experiences per gradient descent update step [default:256].')
    parser.add_argument('--beta',
                        default=2.5e-3,
                        help='Strength of entropy regularization [default:2.5e-3].')
    parser.add_argument('--buffer-size',
                        default=2.5e-3 * 16,
                        help='How large the experience buffer should be before gradient descent [default:2048].')
    parser.add_argument('--dir_path',
                        default=DIR_PATH,
                        help='Name of directory.')
    parser.add_argument('--env_name',
                        default='VerticalLanding-v0',
                        help='Name of the environment.')
    parser.add_argument('--epsilon',
                        default=0.2,
                        help='Acceptable threshold around ratio of old and new policy probabilities [default:0.2].')
    parser.add_argument('--gamma',
                        default=0.99,
                        help='Reward discount rate [default:0.99].')
    parser.add_argument('--hidden_units',
                        default=128,
                        help='Number of units in hidden layer [default:128].')
    parser.add_argument('--keep_checkpoints',
                        default=5,
                        help='How many model checkpoints to keep [default:5].')
    parser.add_argument('--lambd',
                        default=0.95,
                        help='Lambda parameter for GAE [default:0.95].')
    parser.add_argument('--learning_rate',
                        default=1e-4,
                        help='Model learning rate [default:1e-4].')
    parser.add_argument('--load_model',
                        default=False,
                        help='Whether to load the model or randomly initialize [default:False].')
    parser.add_argument('--max_steps',
                        default=20e6,
                        help='Maximum number of steps to run environment [default:20e6].')
    parser.add_argument('--model_name',
                        default='v1',
                        help='The sub-directory name for model and summary statistics.')
    parser.add_argument('--normalize_steps',
                        default=10e6,
                        help='Activate state normalization for this many steps and freeze statistics afterwards.')
    parser.add_argument('--num_epoch',
                        default=10,
                        help='Number of gradient descent steps per batch of experiences [default:10].')
    parser.add_argument('--num_layers',
                        default=1,
                        help='Number of hidden layers between state/observation and outputs [default:1].')
    parser.add_argument('--record',
                        default=False,
                        help='Save recordings of episodes.')
    parser.add_argument('--render',
                        default=True,
                        help='Render environment to display progress.')
    parser.add_argument('--save_freq',
                        default=2.5e-3 * 5,
                        help='Frequency at which to save training statistics [default:50000].')
    parser.add_argument('--summary_freq',
                        default=2.5e-3 * 5,
                        help='Frequency at which to save training statistics [default:10000].')
    parser.add_argument('--summary_path',
                        default='models',
                        help='The sub-directory name for model and summary statistics.')
    parser.add_argument('--time_horizon',
                        default=2048,
                        help='How many steps to collect per agent before adding to buffer [default: 2048].')
    parser.add_argument('--train_model',
                        default=True,
                        help='Whether to train model, or only run inference [default:True].')
    args = parser.parse_args()
    task.ppo(args)
