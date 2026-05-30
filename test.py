import os

os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import torch
from src.env import create_train_env
from src.model import PPO
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
from torch.distributions import Categorical
import torch.nn.functional as F


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of model described in the paper: Proximal Policy Optimization Algorithms for Contra Nes""")
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--action_type", type=str, default="simple")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--output_path", type=str, default="output")
    parser.add_argument("--test_episodes", type=int, default=10, help="Number of attempts before stopping")
    parser.add_argument("--max_steps", type=int, default=5000, help="Maximum environment steps per attempt")
    parser.add_argument("--deterministic", action="store_true", help="Use argmax action instead of sampling")
    args = parser.parse_args()
    return args


def test(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    if opt.action_type == "right":
        actions = RIGHT_ONLY
    elif opt.action_type == "simple":
        actions = SIMPLE_MOVEMENT
    else:
        actions = COMPLEX_MOVEMENT
    os.makedirs(opt.output_path, exist_ok=True)
    output_file = "{}/video_{}_{}.mp4".format(opt.output_path, opt.world, opt.stage)
    env = create_train_env(opt.world, opt.stage, actions, output_file)
    completed = False
    completed_attempt = None
    try:
        model = PPO(env.observation_space.shape[0], len(actions))
        if torch.cuda.is_available():
            model.load_state_dict(torch.load("{}/ppo_super_mario_bros_{}_{}".format(opt.saved_path, opt.world, opt.stage)))
            model.cuda()
        else:
            model.load_state_dict(torch.load("{}/ppo_super_mario_bros_{}_{}".format(opt.saved_path, opt.world, opt.stage),
                                             map_location=lambda storage, loc: storage))
        model.eval()
        state = torch.from_numpy(env.reset())
        episode = 1
        step = 0
        while episode <= opt.test_episodes:
            if torch.cuda.is_available():
                state = state.cuda()
            logits, value = model(state)
            policy = F.softmax(logits, dim=1)
            if opt.deterministic:
                action = torch.argmax(policy).item()
            else:
                action = Categorical(policy).sample().item()
            state, reward, done, info = env.step(action)
            state = torch.from_numpy(state)
            env.render()
            step += 1
            if info["flag_get"]:
                completed = True
                completed_attempt = episode
                break
            if done or step >= opt.max_steps:
                print(
                    "Attempt {} failed. x_pos: {}. reward: {:.4f}".format(
                        episode,
                        info.get("x_pos", 0),
                        float(reward),
                    )
                )
                episode += 1
                step = 0
                if episode <= opt.test_episodes:
                    state = torch.from_numpy(env.reset())
        else:
            print("World {} stage {} not completed after {} attempts".format(
                opt.world,
                opt.stage,
                opt.test_episodes,
            ))
    finally:
        env.close()
    if completed:
        print("World {} stage {} completed on attempt {}".format(opt.world, opt.stage, completed_attempt))
        if os.path.isfile(output_file):
            print("Saved video: {}".format(os.path.abspath(output_file)))
        else:
            print("Video was not saved: {}".format(os.path.abspath(output_file)))


if __name__ == "__main__":
    opt = get_args()
    test(opt)
