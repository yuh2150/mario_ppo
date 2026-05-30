import os

os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
from src.env import MultipleEnvironments
from src.model import PPO
from src.process import eval
import torch.multiprocessing as _mp
from torch.distributions import Categorical
import torch.nn.functional as F
import numpy as np
import shutil


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of model described in the paper: Proximal Policy Optimization Algorithms for Super Mario Bros""")
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--action_type", type=str, default="simple")
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.9, help='discount factor for rewards')
    parser.add_argument('--tau', type=float, default=1.0, help='parameter for GAE')
    parser.add_argument('--beta', type=float, default=0.01, help='entropy coefficient')
    parser.add_argument('--epsilon', type=float, default=0.2, help='parameter for Clipped Surrogate Objective')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument("--num_local_steps", type=int, default=512)
    parser.add_argument("--num_global_steps", type=int, default=5e6)
    parser.add_argument("--num_processes", type=int, default=8)
    parser.add_argument("--save_interval", type=int, default=50, help="Number of steps between savings")
    parser.add_argument("--max_actions", type=int, default=200, help="Maximum repetition steps in test phase")
    parser.add_argument("--log_path", type=str, default="tensorboard", help="Base TensorBoard log directory")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--max_episodes", type=int, default=1000, help="Stop after this many training episodes")
    parser.add_argument("--eval_during_train", action="store_true", help="Run the extra evaluation process while training")
    parser.add_argument("--render_eval", action="store_true", help="Render the extra evaluation process")
    parser.add_argument("--reset_log", action="store_true", help="Delete only this world/stage run log before training")
    args = parser.parse_args()
    return args


def train(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    run_name = "world_{}_stage_{}_{}".format(opt.world, opt.stage, opt.action_type)
    run_log_path = os.path.join(opt.log_path, run_name)
    if opt.reset_log and os.path.isdir(run_log_path):
        shutil.rmtree(run_log_path)
    os.makedirs(run_log_path, exist_ok=True)
    writer = SummaryWriter(run_log_path)
    print("TensorBoard log path: {}".format(run_log_path))
    if not os.path.isdir(opt.saved_path):
        os.makedirs(opt.saved_path)
    mp = _mp.get_context("spawn")
    envs = MultipleEnvironments(opt.world, opt.stage, opt.action_type, opt.num_processes)
    model = PPO(envs.num_states, envs.num_actions)
    if torch.cuda.is_available():
        model.cuda()
    model.share_memory()
    process = None
    if opt.eval_during_train:
        process = mp.Process(target=eval, args=(opt, model, envs.num_states, envs.num_actions))
        process.daemon = True
        process.start()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    [agent_conn.send(("reset", None)) for agent_conn in envs.agent_conns]
    curr_states = [agent_conn.recv() for agent_conn in envs.agent_conns]
    curr_states = torch.from_numpy(np.concatenate(curr_states, 0))
    if torch.cuda.is_available():
        curr_states = curr_states.cuda()
    curr_episode = 0
    episode_rewards = np.zeros(opt.num_processes, dtype=np.float32)
    completed_episode_rewards = []
    while True:
        # if curr_episode % opt.save_interval == 0 and curr_episode > 0:
        #     torch.save(model.state_dict(),
        #                "{}/ppo_super_mario_bros_{}_{}".format(opt.saved_path, opt.world, opt.stage))
        #     torch.save(model.state_dict(),
        #                "{}/ppo_super_mario_bros_{}_{}_{}".format(opt.saved_path, opt.world, opt.stage, curr_episode))
        if curr_episode % opt.save_interval == 0 and curr_episode > 0:
            torch.save(
                model.state_dict(),
                "{}/ppo_super_mario_bros_{}_{}".format(opt.saved_path, opt.world, opt.stage)
            )
        curr_episode += 1
        old_log_policies = []
        actions = []
        values = []
        states = []
        rewards = []
        dones = []
        rollout_rewards = []
        completed_rewards_this_rollout = []
        for _ in range(opt.num_local_steps):
            states.append(curr_states)
            logits, value = model(curr_states)
            values.append(value.squeeze(1))
            policy = F.softmax(logits, dim=1)
            old_m = Categorical(policy)
            action = old_m.sample()
            actions.append(action)
            old_log_policy = old_m.log_prob(action)
            old_log_policies.append(old_log_policy)
            if torch.cuda.is_available():
                [agent_conn.send(("step", act)) for agent_conn, act in zip(envs.agent_conns, action.cpu())]
            else:
                [agent_conn.send(("step", act)) for agent_conn, act in zip(envs.agent_conns, action)]

            state, reward, done, info = zip(*[agent_conn.recv() for agent_conn in envs.agent_conns])
            reward_np = np.array(reward, dtype=np.float32)
            done_np = np.array(done, dtype=np.bool_)
            rollout_rewards.extend(reward_np.tolist())
            episode_rewards += reward_np
            for env_idx, is_done in enumerate(done_np):
                if is_done:
                    completed_reward = float(episode_rewards[env_idx])
                    completed_rewards_this_rollout.append(completed_reward)
                    completed_episode_rewards.append(completed_reward)
                    episode_rewards[env_idx] = 0.0
            state = torch.from_numpy(np.concatenate(state, 0))
            if torch.cuda.is_available():
                state = state.cuda()
                reward = torch.cuda.FloatTensor(reward)
                done = torch.cuda.FloatTensor(done)
            else:
                reward = torch.FloatTensor(reward)
                done = torch.FloatTensor(done)
            rewards.append(reward)
            dones.append(done)
            curr_states = state

        _, next_value, = model(curr_states)
        next_value = next_value.squeeze(1)
        old_log_policies = torch.cat(old_log_policies).detach()
        actions = torch.cat(actions)
        values = torch.cat(values).detach()
        states = torch.cat(states)
        gae = 0
        R = []
        for value, reward, done in list(zip(values, rewards, dones))[::-1]:
            gae = gae * opt.gamma * opt.tau
            gae = gae + reward + opt.gamma * next_value.detach() * (1 - done) - value.detach()
            next_value = value
            R.append(gae + value)
        R = R[::-1]
        R = torch.cat(R).detach()
        advantages = R - values
        total_losses = []
        actor_losses = []
        critic_losses = []
        entropy_losses = []
        kl_divs = []
        clip_fractions = []
        for i in range(opt.num_epochs):
            indice = torch.randperm(opt.num_local_steps * opt.num_processes)
            for j in range(opt.batch_size):
                batch_indices = indice[
                                int(j * (opt.num_local_steps * opt.num_processes / opt.batch_size)): int((j + 1) * (
                                        opt.num_local_steps * opt.num_processes / opt.batch_size))]
                logits, value = model(states[batch_indices])
                new_policy = F.softmax(logits, dim=1)
                new_m = Categorical(new_policy)
                new_log_policy = new_m.log_prob(actions[batch_indices])
                ratio = torch.exp(new_log_policy - old_log_policies[batch_indices])
                actor_loss = -torch.mean(torch.min(ratio * advantages[batch_indices],
                                                   torch.clamp(ratio, 1.0 - opt.epsilon, 1.0 + opt.epsilon) *
                                                   advantages[
                                                       batch_indices]))
                # critic_loss = torch.mean((R[batch_indices] - value) ** 2) / 2
                critic_loss = F.smooth_l1_loss(R[batch_indices], value.squeeze(1))
                entropy_loss = torch.mean(new_m.entropy())
                total_loss = actor_loss + critic_loss - opt.beta * entropy_loss
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

                # Record metrics
                total_losses.append(total_loss.item())
                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                entropy_losses.append(entropy_loss.item())
                with torch.no_grad():
                    log_ratio = new_log_policy - old_log_policies[batch_indices]
                    approx_kl = ((torch.exp(log_ratio) - 1.0) - log_ratio).mean().item()
                    kl_divs.append(approx_kl)
                    
                    clipped = (ratio < 1.0 - opt.epsilon) | (ratio > 1.0 + opt.epsilon)
                    clip_fraction = clipped.float().mean().item()
                    clip_fractions.append(clip_fraction)

        avg_total_loss = float(np.mean(total_losses)) if total_losses else 0.0
        avg_actor_loss = float(np.mean(actor_losses)) if actor_losses else 0.0
        avg_critic_loss = float(np.mean(critic_losses)) if critic_losses else 0.0
        avg_entropy = float(np.mean(entropy_losses)) if entropy_losses else 0.0
        avg_kl = float(np.mean(kl_divs)) if kl_divs else 0.0
        avg_clip_fraction = float(np.mean(clip_fractions)) if clip_fractions else 0.0

        rollout_reward_mean = float(np.mean(rollout_rewards)) if rollout_rewards else 0.0
        rollout_reward_sum = float(np.sum(rollout_rewards)) if rollout_rewards else 0.0
        completed_reward_mean = (
            float(np.mean(completed_rewards_this_rollout))
            if completed_rewards_this_rollout
            else 0.0
        )
        completed_reward_max = (
            float(np.max(completed_rewards_this_rollout))
            if completed_rewards_this_rollout
            else 0.0
        )
        completed_mean_100 = (
            float(np.mean(completed_episode_rewards[-100:]))
            if completed_episode_rewards
            else 0.0
        )

        writer.add_scalar("Loss/total", avg_total_loss, curr_episode)
        writer.add_scalar("Loss/actor", avg_actor_loss, curr_episode)
        writer.add_scalar("Loss/critic", avg_critic_loss, curr_episode)
        writer.add_scalar("Loss/entropy", avg_entropy, curr_episode)
        writer.add_scalar("Diagnostics/approx_kl", avg_kl, curr_episode)
        writer.add_scalar("Diagnostics/clip_fraction", avg_clip_fraction, curr_episode)
        writer.add_scalar("Reward/rollout_mean", rollout_reward_mean, curr_episode)
        writer.add_scalar("Reward/rollout_sum", rollout_reward_sum, curr_episode)
        writer.add_scalar("Reward/completed_mean", completed_reward_mean, curr_episode)
        writer.add_scalar("Reward/completed_mean_100", completed_mean_100, curr_episode)
        writer.add_scalar("Reward/completed_max", completed_reward_max, curr_episode)
        writer.add_scalar("Train/completed_episodes", len(completed_episode_rewards), curr_episode)
        writer.flush()
        print(
            "Episode: {}. Total loss: {:.6f}. Actor loss: {:.6f}. Critic loss: {:.6f}. Entropy: {:.6f}. KL: {:.6f}. Clip frac: {:.4f}. Reward mean: {:.4f}. Reward sum: {:.4f}. Completed mean (100 eps): {:.4f}. Completed episodes: {}".format(
                curr_episode,
                avg_total_loss,
                avg_actor_loss,
                avg_critic_loss,
                avg_entropy,
                avg_kl,
                avg_clip_fraction,
                rollout_reward_mean,
                rollout_reward_sum,
                completed_mean_100,
                len(completed_episode_rewards),
            )
        )
        if curr_episode >= opt.max_episodes:
            torch.save(
                model.state_dict(),
                "{}/ppo_super_mario_bros_{}_{}".format(opt.saved_path, opt.world, opt.stage)
            )
            print("Reached max episodes. Saved checkpoint: {}/ppo_super_mario_bros_{}_{}".format(
                opt.saved_path, opt.world, opt.stage
            ))
            break
    if process is not None and process.is_alive():
        process.terminate()
        process.join()
    envs.close()
    writer.close()


if __name__ == "__main__":
    opt = get_args()
    train(opt)
