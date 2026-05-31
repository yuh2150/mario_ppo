import torch
from src.env import create_train_env
from src.model import PPO
import torch.nn.functional as F
from collections import deque
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY


def eval(opt, global_model, num_states, num_actions):
    # Process eval doc model global va choi thu bang policy hien tai.
    torch.manual_seed(123)

    # Chon action set giong luc train.
    if opt.action_type == "right":
        actions = RIGHT_ONLY
    elif opt.action_type == "simple":
        actions = SIMPLE_MOVEMENT
    else:
        actions = COMPLEX_MOVEMENT
    env = create_train_env(opt.world, opt.stage, actions)

    # Local model dung de copy weight tu global model.
    local_model = PPO(num_states, num_actions)
    if torch.cuda.is_available():
        local_model.cuda()
    local_model.eval()
    state = torch.from_numpy(env.reset())
    if torch.cuda.is_available():
        state = state.cuda()
    done = True
    curr_step = 0
    actions = deque(maxlen=opt.max_actions)
    while True:
        curr_step += 1
        # Moi episode moi thi dong bo weight moi nhat.
        if done:
            local_model.load_state_dict(global_model.state_dict())
        logits, value = local_model(state)
        policy = F.softmax(logits, dim=1)
        # Eval dung action co xac suat cao nhat.
        action = torch.argmax(policy).item()
        state, reward, done, info = env.step(action)

        # Luu model neu agent qua man.
        if info["flag_get"]:
            print("Finished")
            torch.save(local_model.state_dict(),
                       "{}/ppo_super_mario_bros_{}_{}_{}".format(opt.saved_path, opt.world, opt.stage, curr_step))

        if getattr(opt, "render_eval", False):
            env.render()
        actions.append(action)
        # Reset neu qua gioi han step hoac bi lap action qua lau.
        if curr_step > opt.num_global_steps or actions.count(actions[0]) == actions.maxlen:
            done = True
        if done:
            curr_step = 0
            actions.clear()
            state = env.reset()
        state = torch.from_numpy(state)
        if torch.cuda.is_available():
            state = state.cuda()
