import gym_super_mario_bros
import warnings
from gym.spaces import Box
from gym import Wrapper
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
import cv2
import numpy as np
import subprocess as sp
import torch.multiprocessing as mp

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*out of date.*")


class Monitor:
    def __init__(self, width, height, saved_path):
        self.pipe = None
        self.command = ["ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo", "-s", "{}X{}".format(width, height),
                        "-pix_fmt", "rgb24", "-r", "15", "-i", "-", "-an", "-vcodec", "mpeg4", saved_path]
        try:
            self.pipe = sp.Popen(self.command, stdin=sp.PIPE, stderr=sp.PIPE)
        except FileNotFoundError:
            pass

    def record(self, image_array):
        if self.pipe is not None:
            self.pipe.stdin.write(image_array.tobytes())

    def close(self):
        if self.pipe is None:
            return
        self.pipe.stdin.close()
        self.pipe.wait()
        self.pipe = None


def process_frame(frame):
    if frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (84, 84))[None, :, :] / 255.
        return frame
    else:
        return np.zeros((1, 84, 84))


class CustomReward(Wrapper):
    def __init__(self, env=None, world=None, stage=None, monitor=None):
        super(CustomReward, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(1, 84, 84))
        self.curr_score = 0
        self.current_x = 40
        self.world = world
        self.stage = stage
        if monitor:
            self.monitor = monitor
        else:
            self.monitor = None

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        if self.monitor:
            self.monitor.record(state)
        state = process_frame(state)
        reward += (info["score"] - self.curr_score) / 40.
        self.curr_score = info["score"]
        if done:
            if info["flag_get"]:
                reward += 50
            else:
                reward -= 50
        if self.world == 7 and self.stage == 4:
            if (506 <= info["x_pos"] <= 832 and info["y_pos"] > 127) or (
                    832 < info["x_pos"] <= 1064 and info["y_pos"] < 80) or (
                    1113 < info["x_pos"] <= 1464 and info["y_pos"] < 191) or (
                    1579 < info["x_pos"] <= 1943 and info["y_pos"] < 191) or (
                    1946 < info["x_pos"] <= 1964 and info["y_pos"] >= 191) or (
                    1984 < info["x_pos"] <= 2060 and (info["y_pos"] >= 191 or info["y_pos"] < 127)) or (
                    2114 < info["x_pos"] < 2440 and info["y_pos"] < 191) or info["x_pos"] < self.current_x - 500:
                reward -= 50
                done = True
        if self.world == 4 and self.stage == 4:
            if (info["x_pos"] <= 1500 and info["y_pos"] < 127) or (
                    1588 <= info["x_pos"] < 2380 and info["y_pos"] >= 127):
                reward = -50
                done = True

        self.current_x = info["x_pos"]
        return state, reward / 10., done, info

    def reset(self):
        self.curr_score = 0
        self.current_x = 40
        return process_frame(self.env.reset())

    def close(self):
        if self.monitor:
            self.monitor.close()
        self.env.close()


class CustomSkipFrame(Wrapper):
    def __init__(self, env, skip=4):
        super(CustomSkipFrame, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(skip, 84, 84))
        self.skip = skip
        self.states = np.zeros((skip, 84, 84), dtype=np.float32)

    def step(self, action):
        total_reward = 0
        last_states = []
        for i in range(self.skip):
            state, reward, done, info = self.env.step(action)
            total_reward += reward
            if i >= self.skip / 2:
                last_states.append(state)
            if done:
                self.reset()
                return self.states[None, :, :, :].astype(np.float32), total_reward, done, info
        max_state = np.max(np.concatenate(last_states, 0), 0)
        self.states[:-1] = self.states[1:]
        self.states[-1] = max_state
        return self.states[None, :, :, :].astype(np.float32), total_reward, done, info

    def reset(self):
        state = self.env.reset()
        self.states = np.concatenate([state for _ in range(self.skip)], 0)
        return self.states[None, :, :, :].astype(np.float32)

    def close(self):
        self.env.close()


def create_train_env(world, stage, actions, output_path=None):
    env_id = "SuperMarioBros-{}-{}-v0".format(world, stage)
    try:
        env = gym_super_mario_bros.make(env_id, disable_env_checker=True)
    except TypeError:
        env = gym_super_mario_bros.make(env_id)
    if output_path:
        monitor = Monitor(256, 240, output_path)
    else:
        monitor = None

    env = JoypadSpace(env, actions)
    env = CustomReward(env, world, stage, monitor)
    env = CustomSkipFrame(env)
    return env


def get_actions(action_type):
    if action_type == "right":
        return RIGHT_ONLY
    if action_type == "simple":
        return SIMPLE_MOVEMENT
    return COMPLEX_MOVEMENT


def run_env_worker(env_conn, world, stage, action_type, output_path):
    actions = get_actions(action_type)
    env = create_train_env(world, stage, actions, output_path=output_path)
    try:
        while True:
            try:
                request, action = env_conn.recv()
            except (EOFError, BrokenPipeError, OSError):
                break
            if request == "step":
                env_conn.send(env.step(action.item()))
            elif request == "reset":
                env_conn.send(env.reset())
            elif request == "close":
                break
            else:
                raise NotImplementedError
    finally:
        env.close()
        env_conn.close()


class MultipleEnvironments:
    def __init__(self, world, stage, action_type, num_envs, output_path=None):
        self.agent_conns, self.env_conns = zip(*[mp.Pipe() for _ in range(num_envs)])
        actions = get_actions(action_type)
        self.num_states = 4
        self.num_actions = len(actions)
        self.processes = []
        for env_conn in self.env_conns:
            process = mp.Process(target=run_env_worker, args=(env_conn, world, stage, action_type, output_path))
            process.daemon = True
            process.start()
            self.processes.append(process)
            env_conn.close()

    def close(self):
        for agent_conn in self.agent_conns:
            try:
                agent_conn.send(("close", None))
                agent_conn.close()
            except (BrokenPipeError, EOFError, OSError):
                pass
        for process in self.processes:
            process.join(timeout=1)
            if process.is_alive():
                process.terminate()
                process.join()
