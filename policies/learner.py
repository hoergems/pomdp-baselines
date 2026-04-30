# -*- coding: future_fstrings -*-
import os, sys
import time

import math
import numpy as np
import torch
from torch.nn import functional as F
import gym

from .models import AGENT_CLASSES, AGENT_ARCHS
from torchkit.networks import ImageEncoder, ImageEncoder32

# Markov policy
from buffers.simple_replay_buffer import SimpleReplayBuffer

# RNN policy on vector-based task
from buffers.seq_replay_buffer_vanilla import SeqReplayBuffer

# RNN policy on image/vector-based task
from buffers.seq_replay_buffer_efficient import RAMEfficient_SeqReplayBuffer

from utils import helpers as utl
from torchkit import pytorch_utils as ptu
from utils import evaluation as utl_eval
from utils import logger


class Learner:
    def __init__(self, env_args, train_args, eval_args, policy_args, seed, **kwargs):
        self.seed = seed

        self.init_env(**env_args)

        self.init_agent(**policy_args)

        self.init_train(**train_args)

        self.init_eval(**eval_args)

    def init_env(
        self,
        env_type,
        env_name,
        max_rollouts_per_task=None,
        num_tasks=None,
        num_train_tasks=None,
        num_eval_tasks=None,
        eval_envs=None,
        worst_percentile=None,
        **kwargs
    ):

        # initialize environment
        assert env_type in [
            "meta",
            "pomdp",
            "credit",
            "rmdp",
            "generalize",
            "atari",
        ]
        self.env_type = env_type

        if self.env_type == "meta":  # meta tasks: using varibad wrapper
            from envs.meta.make_env import make_env

            self.train_env = make_env(
                env_name,
                max_rollouts_per_task,
                seed=self.seed,
                n_tasks=num_tasks,
                **kwargs,
            )  # oracle in kwargs
            self.eval_env = self.train_env
            self.eval_env.seed(self.seed + 1)

            if self.train_env.n_tasks is not None:
                # NOTE: This is off-policy varibad's setting, i.e. limited training tasks
                # split to train/eval tasks
                assert num_train_tasks >= num_eval_tasks > 0
                shuffled_tasks = np.random.permutation(
                    self.train_env.unwrapped.get_all_task_idx()
                )
                self.train_tasks = shuffled_tasks[:num_train_tasks]
                self.eval_tasks = shuffled_tasks[-num_eval_tasks:]
            else:
                # NOTE: This is on-policy varibad's setting, i.e. unlimited training tasks
                assert num_tasks == num_train_tasks == None
                assert (
                    num_eval_tasks > 0
                )  # to specify how many tasks to be evaluated each time
                self.train_tasks = []
                self.eval_tasks = num_eval_tasks * [None]

            # calculate what the maximum length of the trajectories is
            self.max_rollouts_per_task = max_rollouts_per_task
            self.max_trajectory_len = self.train_env.horizon_bamdp  # H^+ = k * H

        elif self.env_type in [
            "pomdp",
            "credit",
        ]:  # pomdp/mdp task, using pomdp wrapper
            import envs.pomdp
            import envs.credit_assign

            assert num_eval_tasks > 0

            env_kwargs = kwargs.copy() if kwargs is not None else {}
            print(f"[DEBUG] Creating POMDP env '{env_name}' with kwargs: {env_kwargs}")

            # --- Register custom envs (safe to call multiple times) ---
            try:
                from pomdp_problems.legged_locomotion_generative_model.register_gym_env_batched import (
                    register_legged_locomotion_env,
                )

                register_legged_locomotion_env()
                print("[DEBUG] Registered LeggedLocomotionPOMDP-v0")

            except Exception as e:
                print(f"[WARNING] Could not register custom env: {e}")
            
            self.train_env = gym.make(env_name, **env_kwargs)

            self.vectorized_env = kwargs.get("vectorized", False)
            self.num_envs = getattr(self.train_env, "num_envs", 1)            

            #self.train_env = gym.make(env_name)
            self.train_env.seed(self.seed)
            self.train_env.action_space.np_random.seed(self.seed)  # crucial
            self.train_env.unwrapped.set_use_reward_shaping(True)

            self.eval_env = self.train_env
            self.eval_env.seed(self.seed + 1)

            self.train_tasks = []
            self.eval_tasks = num_eval_tasks * [None]

            self.max_rollouts_per_task = 1
            self.max_trajectory_len = getattr(
                self.train_env.unwrapped, "_max_episode_steps", 60
            )

        elif self.env_type == "atari":
            from envs.atari import create_env

            assert num_eval_tasks > 0
            self.train_env = create_env(env_name)
            self.train_env.seed(self.seed)
            self.train_env.action_space.np_random.seed(self.seed)  # crucial

            self.eval_env = self.train_env
            self.eval_env.seed(self.seed + 1)

            self.train_tasks = []
            self.eval_tasks = num_eval_tasks * [None]

            self.max_rollouts_per_task = 1
            self.max_trajectory_len = self.train_env._max_episode_steps

        elif self.env_type == "rmdp":  # robust mdp task, using robust mdp wrapper
            sys.path.append("envs/rl-generalization")
            import sunblaze_envs

            assert (
                num_eval_tasks > 0 and worst_percentile > 0.0 and worst_percentile < 1.0
            )
            self.train_env = sunblaze_envs.make(env_name, **kwargs)  # oracle
            self.train_env.seed(self.seed)
            assert np.all(self.train_env.action_space.low == -1)
            assert np.all(self.train_env.action_space.high == 1)

            self.eval_env = self.train_env
            self.eval_env.seed(self.seed + 1)

            self.worst_percentile = worst_percentile

            self.train_tasks = []
            self.eval_tasks = num_eval_tasks * [None]

            self.max_rollouts_per_task = 1
            self.max_trajectory_len = self.train_env._max_episode_steps

        elif self.env_type == "generalize":
            sys.path.append("envs/rl-generalization")
            import sunblaze_envs

            self.train_env = sunblaze_envs.make(env_name, **kwargs)  # oracle in kwargs
            self.train_env.seed(self.seed)
            assert np.all(self.train_env.action_space.low == -1)
            assert np.all(self.train_env.action_space.high == 1)

            def check_env_class(env_name):
                if "Normal" in env_name:
                    return "R"
                if "Extreme" in env_name:
                    return "E"
                return "D"

            self.train_env_name = check_env_class(env_name)

            self.eval_envs = {}
            for env_name, num_eval_task in eval_envs.items():
                eval_env = sunblaze_envs.make(env_name, **kwargs)  # oracle in kwargs
                eval_env.seed(self.seed + 1)
                self.eval_envs[eval_env] = (
                    check_env_class(env_name),
                    num_eval_task,
                )  # several types of evaluation envs

            logger.log(self.train_env_name, self.train_env)
            logger.log(self.eval_envs)

            self.train_tasks = []
            self.max_rollouts_per_task = 1
            self.max_trajectory_len = self.train_env._max_episode_steps

        else:
            raise ValueError

        # get action / observation dimensions
        if self.train_env.action_space.__class__.__name__ == "Box":
            # continuous action space
            self.act_dim = self.train_env.action_space.shape[0]
            self.act_continuous = True
        else:
            assert self.train_env.action_space.__class__.__name__ == "Discrete"
            self.act_dim = self.train_env.action_space.n
            self.act_continuous = False
        self.obs_dim = self.train_env.observation_space.shape[0]  # include 1-dim done
        logger.log("obs_dim", self.obs_dim, "act_dim", self.act_dim)

    def init_agent(
        self,
        seq_model,
        separate: bool = True,
        image_encoder=None,
        reward_clip=False,
        **kwargs
    ):
        # initialize agent
        if seq_model == "mlp":
            agent_class = AGENT_CLASSES["Policy_MLP"]
            rnn_encoder_type = None
            assert separate == True
        elif "-mlp" in seq_model:
            agent_class = AGENT_CLASSES["Policy_RNN_MLP"]
            rnn_encoder_type = seq_model.split("-")[0]
            assert separate == True
        else:
            rnn_encoder_type = seq_model
            if separate == True:
                agent_class = AGENT_CLASSES["Policy_Separate_RNN"]
            else:
                agent_class = AGENT_CLASSES["Policy_Shared_RNN"]

        self.agent_arch = agent_class.ARCH
        logger.log(agent_class, self.agent_arch)

        if image_encoder is not None:
            image_shape = self.train_env.image_space.shape
            encoder_cfg = dict(image_encoder)

            encoder_type = encoder_cfg.pop("type", "large")

            if encoder_type == "small":
                encoder_cls = ImageEncoder
            elif encoder_type == "large":
                encoder_cls = ImageEncoder32            
            else:
                raise ValueError(f"Unknown image encoder type: {encoder_type}")

            image_encoder_fn = lambda: encoder_cls(
                image_shape=image_shape,
                **encoder_cfg,
            )
        else:
            image_encoder_fn = lambda: None

        self.agent = agent_class(
            encoder=rnn_encoder_type,
            obs_dim=self.obs_dim,
            action_dim=self.act_dim,
            image_encoder_fn=image_encoder_fn,
            **kwargs,
        ).to(ptu.device)
        logger.log(self.agent)

        self.reward_clip = reward_clip  # for atari

    def init_train(
        self,
        buffer_size,
        batch_size,
        num_iters=None,
        num_init_rollouts_pool=0,
        num_rollouts_per_iter=1,
        num_updates_per_iter=None,
        sampled_seq_len=None,
        sample_weight_baseline=None,
        buffer_type=None,
        target_env_steps_total=None,
        target_initial_env_steps=None,
        collect_env_steps_per_iter=None,
        **kwargs
    ):

        if num_updates_per_iter is None:
            num_updates_per_iter = 1.0
        assert isinstance(num_updates_per_iter, int) or isinstance(
            num_updates_per_iter, float
        )
        # if int, it means absolute value; if float, it means the multiplier of collected env steps
        self.num_updates_per_iter = num_updates_per_iter
        self.save_chkpt = kwargs.get('save_chkpt', False)

        if self.agent_arch == AGENT_ARCHS.Markov:
            self.policy_storage = SimpleReplayBuffer(
                max_replay_buffer_size=int(buffer_size),
                observation_dim=self.obs_dim,
                action_dim=self.act_dim if self.act_continuous else 1,  # save memory
                max_trajectory_len=self.max_trajectory_len,
                add_timeout=False,  # no timeout storage
            )

        else:  # memory, memory-markov
            if sampled_seq_len == -1:
                sampled_seq_len = self.max_trajectory_len

            if buffer_type is None or buffer_type == SeqReplayBuffer.buffer_type:
                buffer_class = SeqReplayBuffer
            elif buffer_type == RAMEfficient_SeqReplayBuffer.buffer_type:
                buffer_class = RAMEfficient_SeqReplayBuffer
            logger.log(buffer_class)

            self.policy_storage = buffer_class(
                max_replay_buffer_size=int(buffer_size),
                observation_dim=self.obs_dim,
                action_dim=self.act_dim if self.act_continuous else 1,  # save memory
                sampled_seq_len=sampled_seq_len,
                sample_weight_baseline=sample_weight_baseline,
                observation_type=self.train_env.observation_space.dtype,
            )

        self.batch_size = batch_size
        self.num_iters = num_iters
        self.num_init_rollouts_pool = num_init_rollouts_pool
        self.num_rollouts_per_iter = num_rollouts_per_iter

        # Backwards-compatible defaults:
        # old semantics: total_env_steps = (init_rollouts + iters * rollouts_per_iter) * horizon
        if target_env_steps_total is None:
            assert num_iters is not None
            total_rollouts = num_init_rollouts_pool + num_iters * num_rollouts_per_iter
            target_env_steps_total = self.max_trajectory_len * total_rollouts

        if target_initial_env_steps is None:
            target_initial_env_steps = num_init_rollouts_pool * self.max_trajectory_len

        if collect_env_steps_per_iter is None:
            collect_env_steps_per_iter = num_rollouts_per_iter * self.max_trajectory_len

        self.n_env_steps_total = int(target_env_steps_total)
        self.target_initial_env_steps = int(target_initial_env_steps)
        self.collect_env_steps_per_iter = int(collect_env_steps_per_iter)

        logger.log(
            "*** target env steps",
            self.n_env_steps_total,
            "target initial env steps",
            self.target_initial_env_steps,
            "collect env steps per iter",
            self.collect_env_steps_per_iter,
        )

    def init_eval(
        self,
        log_interval=None,
        save_interval=-1,
        log_tensorboard=True,
        eval_stochastic=False,
        num_episodes_per_task=1,
        eval_every_env_steps=None,
        save_every_env_steps=None,
        **kwargs
    ):

        self.log_interval = log_interval
        self.save_interval = save_interval

        # Backwards-compatible defaults.
        # Old log_interval was in "iterations", where one iter was roughly num_rollouts_per_iter * horizon steps.
        if eval_every_env_steps is None:
            if log_interval is None:
                eval_every_env_steps = self.collect_env_steps_per_iter
            else:
                eval_every_env_steps = log_interval * self.collect_env_steps_per_iter

        if save_every_env_steps is None:
            if save_interval is not None and save_interval > 0:
                save_every_env_steps = save_interval * self.collect_env_steps_per_iter
            else:
                save_every_env_steps = -1

        self.eval_every_env_steps = int(eval_every_env_steps)
        self.save_every_env_steps = int(save_every_env_steps)
        self.log_tensorboard = log_tensorboard
        self.eval_stochastic = eval_stochastic
        self.eval_num_episodes_per_task = num_episodes_per_task

    def _start_training(self):
        self._n_env_steps_total = 0
        self._n_env_steps_total_last = 0
        self._n_rl_update_steps_total = 0
        self._n_rollouts_total = 0
        self._successes_in_buffer = 0

        self._best_eval_return = -np.inf

        self._start_time = time.time()
        self._start_time_last = time.time()

    @torch.no_grad()
    def collect_rollouts_for_env_steps(
        self,
        target_env_steps,
        random_actions=False,
        reset_envs=True,
    ):
        before_env_steps = self._n_env_steps_total

        if getattr(self, "vectorized_env", False) and reset_envs:
            self.reset_vectorized_rollout_state(random_actions=random_actions)

        while self._n_env_steps_total - before_env_steps < target_env_steps:
            print(f"[DEBUG]: Keep collecting collouts. {self._n_env_steps_total - before_env_steps}/{target_env_steps} steps collected")
            self.collect_rollouts(
                num_rollouts=1,
                random_actions=random_actions,
            )

        return self._n_env_steps_total - before_env_steps

    @torch.no_grad()
    def reset_vectorized_rollout_state(self, random_actions=False):
        obs = self.train_env.reset()
        obs = ptu.from_numpy(obs) if isinstance(obs, np.ndarray) else obs.to(ptu.device)
        obs = obs.view(obs.shape[0], -1)

        num_envs = obs.shape[0]

        self._vec_obs = obs
        self._vec_obs_lists = [[] for _ in range(num_envs)]
        self._vec_act_lists = [[] for _ in range(num_envs)]
        self._vec_rew_lists = [[] for _ in range(num_envs)]
        self._vec_next_obs_lists = [[] for _ in range(num_envs)]
        self._vec_term_lists = [[] for _ in range(num_envs)]
        self._vec_ep_returns = np.zeros(num_envs, dtype=np.float32)
        self._vec_ep_lengths = np.zeros(num_envs, dtype=np.int64)

        if self.agent_arch == AGENT_ARCHS.Memory and not random_actions:
            action, reward, internal_state = self.expand_initial_info(num_envs)
            self._vec_action = action
            self._vec_reward = reward
            self._vec_internal_state = internal_state
        else:
            self._vec_action = None
            self._vec_reward = None
            self._vec_internal_state = None

    def train(self):
        """
        training loop
        """
        if not hasattr(self, "_n_env_steps_total") or self._n_env_steps_total == 0:
            self._start_training()
        else:
            logger.log("[Resume] Skipping _start_training() (resuming state)")

        if self.target_initial_env_steps > 0 and self._n_env_steps_total == 0:
            logger.log("Collecting initial pool of data..")

            if getattr(self, "vectorized_env", False):
                self.reset_vectorized_rollout_state(random_actions=True)

            while self.policy_storage.size() < self.target_initial_env_steps:
                self.collect_rollouts_for_env_steps(
                    target_env_steps=self.collect_env_steps_per_iter,
                    random_actions=True,
                    reset_envs=False,
                )
            logger.log(
                "Done! env steps",
                self._n_env_steps_total,
                "rollouts",
                self._n_rollouts_total,
            )

            if isinstance(self.num_updates_per_iter, float):
                # update: pomdp task updates more for the first iter_
                train_stats = self.update(
                    int(self._n_env_steps_total * self.num_updates_per_iter),                    
                )
                self.log_train_stats(train_stats)

        next_eval_env_steps = self.eval_every_env_steps
        next_save_env_steps = self.save_every_env_steps if self.save_every_env_steps > 0 else None
        next_model_save_steps = 20000
        perf = -np.inf

        if getattr(self, "vectorized_env", False):
            self.reset_vectorized_rollout_state(random_actions=False)

        while self._n_env_steps_total < self.n_env_steps_total:
            print(
                f"[DEBUG]: Collect policy data for about "
                f"{self.collect_env_steps_per_iter} env steps"
            )

            env_steps = self.collect_rollouts_for_env_steps(
                target_env_steps=self.collect_env_steps_per_iter,
                random_actions=False,
                reset_envs=False,
            )

            print(f"[DEBUG]: Collected {env_steps} steps")
            logger.log(f"[DEBUG]: env steps {self._n_env_steps_total}/{self.n_env_steps_total}")

            train_stats = self.update(
                self.num_updates_per_iter
                if isinstance(self.num_updates_per_iter, int)
                else int(math.ceil(self.num_updates_per_iter * env_steps))
            )
            self.log_train_stats(train_stats)

            if self._n_env_steps_total >= next_eval_env_steps:
                perf = self.log()

                if getattr(self, "vectorized_env", False):
                    self.reset_vectorized_rollout_state(random_actions=False)

                while next_eval_env_steps <= self._n_env_steps_total:
                    next_eval_env_steps += self.eval_every_env_steps

            if (
                next_save_env_steps is not None
                and self._n_env_steps_total > 0.75 * self.n_env_steps_total
                and self._n_env_steps_total >= next_save_env_steps
            ):
                self.save_model(self._n_env_steps_total, perf)            
                while next_save_env_steps <= self._n_env_steps_total:
                    next_save_env_steps += self.save_every_env_steps
            if self._n_env_steps_total >= next_model_save_steps and self.save_chkpt == True:                
                self.save_training_checkpoint(
                    os.path.join(logger.get_dir(), "save", "training_latest.pt"),
                )                
                self.save_replay_buffer(os.path.join(logger.get_dir(), "save", "replay_buffer.npz"))
                next_model_save_steps += 20000

        self.save_model(self._n_env_steps_total, perf)
        if self.save_chkpt == True:
            self.save_training_checkpoint(
                os.path.join(logger.get_dir(), "save", "training_latest.pt"),
            )
            self.save_replay_buffer(os.path.join(logger.get_dir(), "save", "replay_buffer.npz"))

        print("[DEBUG]: Close env")
        self.train_env.close()
        print("[DEBUG]: Done")

    def expand_initial_info(self, num_envs):
        action, reward, internal_state = self.agent.get_initial_info()

        action = action.expand(num_envs, -1).clone()
        reward = reward.expand(num_envs, -1).clone()

        if isinstance(internal_state, tuple):  # LSTM
            h, c = internal_state
            h = h.expand(-1, num_envs, -1).clone()
            c = c.expand(-1, num_envs, -1).clone()
            internal_state = (h, c)
        else:  # GRU
            internal_state = internal_state.expand(-1, num_envs, -1).clone()

        return action, reward, internal_state

    @torch.no_grad()
    def collect_rollouts_vectorized(self, num_rollouts, random_actions=False):
        before_env_steps = self._n_env_steps_total
        if getattr(self, "_vec_obs", None) is None:
            self.reset_vectorized_rollout_state(random_actions=random_actions)

        obs = self._vec_obs
        num_envs = obs.shape[0]
        obs = ptu.from_numpy(obs) if isinstance(obs, np.ndarray) else obs.to(ptu.device)
        obs = obs.view(obs.shape[0], -1)

        obs_lists = self._vec_obs_lists
        act_lists = self._vec_act_lists
        rew_lists = self._vec_rew_lists
        next_obs_lists = self._vec_next_obs_lists
        term_lists = self._vec_term_lists
        ep_returns = self._vec_ep_returns
        ep_lengths = self._vec_ep_lengths

        num_batched_rollouts = max(1, int(num_rollouts))
        completed_usable_rollouts = 0
        while completed_usable_rollouts < num_batched_rollouts:
            next_internal_state = None

            if random_actions:
                if self.act_continuous:
                    action_np = np.stack(
                        [self.train_env.action_space.sample() for _ in range(num_envs)]
                    )
                    action = ptu.from_numpy(action_np).float()
                else:
                    act_idx = torch.randint(
                        low=0,
                        high=self.act_dim,
                        size=(num_envs,),
                        device=ptu.device,
                    )
                    action = F.one_hot(act_idx, num_classes=self.act_dim).float()
            else:
                if self.agent_arch == AGENT_ARCHS.Memory:
                    (action, _, _, _), next_internal_state = self.agent.act(
                        prev_internal_state=self._vec_internal_state,
                        prev_action=self._vec_action,
                        reward=self._vec_reward,
                        obs=obs,
                        deterministic=False,
                    )
                else:
                    action, _, _, _ = self.agent.act(obs, deterministic=False)

            next_obs, reward_new, done, info = utl.env_step_batched(
                self.train_env, action
            )

            next_obs = next_obs.view(next_obs.shape[0], -1)

            if self.reward_clip and self.env_type == "atari":
                reward_for_policy = torch.tanh(reward_new)
            else:
                reward_for_policy = reward_new

            if self.agent_arch == AGENT_ARCHS.Memory and not random_actions:
                self._vec_reward = reward_for_policy.clone()
                self._vec_internal_state = next_internal_state
                self._vec_action = action.clone()

            done_bool = done.squeeze(-1).bool()

            ep_lengths += 1
            ep_returns += ptu.get_numpy(reward_new.squeeze(-1))

            finished_env_ids = []

            for i in range(num_envs):
                done_i = bool(done_bool[i].item())
                cutoff_i = ep_lengths[i] >= self.max_trajectory_len

                #term_for_buffer = done_i and not cutoff_i
                timeout_i = cutoff_i and not done_i
                term_for_buffer = done_i and not timeout_i
                episode_finished = done_i or cutoff_i

                obs_lists[i].append(obs[i : i + 1])
                act_lists[i].append(action[i : i + 1])
                rew_lists[i].append(reward_for_policy[i : i + 1])
                next_obs_lists[i].append(next_obs[i : i + 1])
                term_lists[i].append(term_for_buffer)

                if episode_finished:
                    episode_len = ep_lengths[i]
                    usable_rollout = episode_len >= 2

                    if usable_rollout:
                        act_buffer = torch.cat(act_lists[i], dim=0)

                        if not self.act_continuous:
                            act_buffer = torch.argmax(
                                act_buffer,
                                dim=-1,
                                keepdim=True,
                            )

                        self.policy_storage.add_episode(
                            observations=ptu.get_numpy(torch.cat(obs_lists[i], dim=0)),
                            actions=ptu.get_numpy(act_buffer),
                            rewards=ptu.get_numpy(torch.cat(rew_lists[i], dim=0)),
                            terminals=np.array(term_lists[i], dtype=np.float32).reshape(-1, 1),
                            next_observations=ptu.get_numpy(
                                torch.cat(next_obs_lists[i], dim=0)
                            ),
                        )
                        completed_usable_rollouts += 1
                    
                    self._n_rollouts_total += 1

                    '''print(
                        f"env {i} steps: {ep_lengths[i]} "
                        f"done: {done_i} "
                        f"cutoff: {cutoff_i} "
                        f"term: {term_for_buffer} "
                        f"ret: {ep_returns[i]:.2f}"
                    )'''

                    obs_lists[i].clear()
                    act_lists[i].clear()
                    rew_lists[i].clear()
                    next_obs_lists[i].clear()
                    term_lists[i].clear()

                    ep_returns[i] = 0.0
                    ep_lengths[i] = 0

                    finished_env_ids.append(i)

                    if self.agent_arch == AGENT_ARCHS.Memory and not random_actions:
                        if isinstance(next_internal_state, tuple):
                            next_internal_state[0][:, i, :] = 0.0
                            next_internal_state[1][:, i, :] = 0.0
                        else:
                            next_internal_state[:, i, :] = 0.0

                        action[i] = 0.0
                        reward_for_policy[i] = 0.0

            self._n_env_steps_total += num_envs

            # Important:
            # reset only the envs whose episodes finished, then overwrite their
            # next_obs entries with fresh initial observations.
            if len(finished_env_ids) > 0:
                finished_env_ids_t = torch.as_tensor(
                    finished_env_ids,
                    device=ptu.device,
                    dtype=torch.long,
                )

                reset_obs = self.train_env.reset(env_ids=finished_env_ids_t)
                reset_obs = (
                    ptu.from_numpy(reset_obs)
                    if isinstance(reset_obs, np.ndarray)
                    else reset_obs.to(ptu.device)
                )
                reset_obs = reset_obs.view(reset_obs.shape[0], -1)

                next_obs[finished_env_ids_t] = reset_obs

            obs = next_obs.clone()
            self._vec_obs = obs

            if self.agent_arch == AGENT_ARCHS.Memory and not random_actions:
                self._vec_action = action.clone()
                self._vec_reward = reward_for_policy.clone()
                self._vec_internal_state = next_internal_state

        print(
            f"[DEBUG] collected usable rollouts: "
            f"{completed_usable_rollouts}/{num_batched_rollouts}, "
            f"buffer size: {self.policy_storage.size()}"
        )

        return self._n_env_steps_total - before_env_steps

    @torch.no_grad()
    def collect_rollouts(self, num_rollouts, random_actions=False):
        """collect num_rollouts of trajectories in task and save into policy buffer
        :param random_actions: whether to use policy to sample actions, or randomly sample action space
        """
        if getattr(self, "vectorized_env", False):
            return self.collect_rollouts_vectorized(num_rollouts, random_actions)

        before_env_steps = self._n_env_steps_total
        for idx in range(num_rollouts):
            steps = 0

            if self.env_type == "meta" and self.train_env.n_tasks is not None:
                task = self.train_tasks[np.random.randint(len(self.train_tasks))]
                obs = ptu.from_numpy(self.train_env.reset(task=task))  # reset task
            else:
                obs = ptu.from_numpy(self.train_env.reset())  # reset

            obs = obs.reshape(1, obs.shape[-1])
            done_rollout = False

            if self.agent_arch in [AGENT_ARCHS.Memory, AGENT_ARCHS.Memory_Markov]:
                # temporary storage
                obs_list, act_list, rew_list, next_obs_list, term_list = (
                    [],
                    [],
                    [],
                    [],
                    [],
                )

            if self.agent_arch == AGENT_ARCHS.Memory:
                # get hidden state at timestep=0, None for markov
                # NOTE: assume initial reward = 0.0 (no need to clip)
                action, reward, internal_state = self.agent.get_initial_info()

            while not done_rollout:
                if random_actions:
                    action = ptu.FloatTensor(
                        [self.train_env.action_space.sample()]
                    )  # (1, A) for continuous action, (1) for discrete action
                    if not self.act_continuous:
                        action = F.one_hot(
                            action.long(), num_classes=self.act_dim
                        ).float()  # (1, A)
                else:
                    # policy takes hidden state as input for memory-based actor,
                    # while takes obs for markov actor
                    if self.agent_arch == AGENT_ARCHS.Memory:
                        (action, _, _, _), internal_state = self.agent.act(
                            prev_internal_state=internal_state,
                            prev_action=action,
                            reward=reward,
                            obs=obs,
                            deterministic=False,
                        )
                    else:
                        action, _, _, _ = self.agent.act(obs, deterministic=False)

                # observe reward and next obs (B=1, dim)
                next_obs, reward, done, info = utl.env_step(
                    self.train_env, action.squeeze(dim=0)
                )

                for name, value in {
                    "obs": obs,
                    "action": action,
                    "next_obs": next_obs,
                    "reward": reward,
                    "done": done,
                }.items():
                    if value is None:
                        raise RuntimeError(f"[collect_rollouts] {name} is None")

                    if not torch.is_tensor(value):
                        raise RuntimeError(
                            f"[collect_rollouts] {name} is not a tensor: {type(value)}"
                        )

                    if not torch.isfinite(value).all():
                        raise RuntimeError(
                            f"[collect_rollouts] {name} has non-finite values at "
                            f"rollout={idx}, step={steps}. value={value}"
                        )

                if self.reward_clip and self.env_type == "atari":
                    reward = torch.tanh(reward)

                done_rollout = False if ptu.get_numpy(done[0][0]) == 0.0 else True
                # update statistics
                steps += 1

                ## determine terminal flag per environment
                if self.env_type == "meta" and "is_goal_state" in dir(
                    self.train_env.unwrapped
                ):
                    # NOTE: following varibad practice: for meta env, even if reaching the goal (term=True),
                    # the episode still continues.
                    term = self.train_env.unwrapped.is_goal_state()
                    self._successes_in_buffer += int(term)
                elif self.env_type == "credit":  # delayed rewards
                    term = done_rollout
                else:
                    # term ignore time-out scenarios, but record early stopping
                    term = (
                        False
                        if "TimeLimit.truncated" in info
                        or steps >= self.max_trajectory_len
                        else done_rollout
                    )

                # add data to policy buffer
                if self.agent_arch == AGENT_ARCHS.Markov:
                    self.policy_storage.add_sample(
                        observation=ptu.get_numpy(obs.squeeze(dim=0)),
                        action=ptu.get_numpy(
                            action.squeeze(dim=0)
                            if self.act_continuous
                            else torch.argmax(
                                action.squeeze(dim=0), dim=-1, keepdims=True
                            )  # (1,)
                        ),
                        reward=ptu.get_numpy(reward.squeeze(dim=0)),
                        terminal=np.array([term], dtype=float),
                        next_observation=ptu.get_numpy(next_obs.squeeze(dim=0)),
                    )
                else:  # append tensors to temporary storage
                    obs_list.append(obs)  # (1, dim)
                    act_list.append(action)  # (1, dim)
                    rew_list.append(reward)  # (1, dim)
                    term_list.append(term)  # bool
                    next_obs_list.append(next_obs)  # (1, dim)

                # set: obs <- next_obs
                obs = next_obs.clone()

            if self.agent_arch in [AGENT_ARCHS.Memory, AGENT_ARCHS.Memory_Markov]:
                # add collected sequence to buffer
                act_buffer = torch.cat(act_list, dim=0)  # (L, dim)
                if not self.act_continuous:
                    act_buffer = torch.argmax(
                        act_buffer, dim=-1, keepdims=True
                    )  # (L, 1)

                self.policy_storage.add_episode(
                    observations=ptu.get_numpy(torch.cat(obs_list, dim=0)),  # (L, dim)
                    actions=ptu.get_numpy(act_buffer),  # (L, dim)
                    rewards=ptu.get_numpy(torch.cat(rew_list, dim=0)),  # (L, dim)
                    terminals=np.array(term_list).reshape(-1, 1),  # (L, 1)
                    next_observations=ptu.get_numpy(
                        torch.cat(next_obs_list, dim=0)
                    ),  # (L, dim)
                )
                print(
                    f"steps: {steps} term: {term} ret: {torch.cat(rew_list, dim=0).sum().item():.2f}"
                )
            self._n_env_steps_total += steps
            self._n_rollouts_total += 1
        return self._n_env_steps_total - before_env_steps

    def sample_rl_batch(self, batch_size):
        """sample batch of episodes for vae training"""
        if self.agent_arch == AGENT_ARCHS.Markov:
            batch = self.policy_storage.random_batch(batch_size)
        else:  # rnn: all items are (sampled_seq_len, B, dim)
            batch = self.policy_storage.random_episodes(batch_size)
        return ptu.np_to_pytorch_batch(batch)

    def update(self, num_updates):
        rl_losses_agg = {}
        print(f"[DEBUG]: Execute {num_updates} policy updates")
        for update in range(num_updates):            
            # sample random RL batch: in transitions
            batch = self.sample_rl_batch(self.batch_size)

            # RL update
            rl_losses = self.agent.update(batch)

            for k, v in rl_losses.items():
                if update == 0:  # first iterate - create list
                    rl_losses_agg[k] = [v]
                else:  # append values
                    rl_losses_agg[k].append(v)
        # statistics
        for k in rl_losses_agg:
            rl_losses_agg[k] = np.mean(rl_losses_agg[k])
        self._n_rl_update_steps_total += num_updates

        return rl_losses_agg

    @torch.no_grad()
    def evaluate_batched(self, tasks, deterministic=True, save_best=True, eval_shaped_reward=True):
        num_eval_episodes = len(tasks)
        print(f"[DEBUG]: Evaluate using {num_eval_episodes} episodes")

        num_episodes = self.max_rollouts_per_task
        assert num_episodes == 1, "evaluate_batched currently assumes one episode per task."

        num_steps_per_episode = self.max_trajectory_len
        observations = None

        use_reward_shaping = self.eval_env.unwrapped.get_use_reward_shaping()
        print(f"[DEBUG]: Use reward shaping: {use_reward_shaping}")
        # self.eval_env.unwrapped.set_use_reward_shaping(False)

        try:
            obs = self.eval_env.reset()
            obs = ptu.from_numpy(obs) if isinstance(obs, np.ndarray) else obs.to(ptu.device)
            obs = obs.view(obs.shape[0], -1)

            num_envs = obs.shape[0]
            assert num_eval_episodes <= num_envs, "num_eval_episodes must be <= num_envs"

            num_tracked_envs = num_eval_episodes
            tracked_env_ids = np.arange(num_tracked_envs)
            tracked_done = np.zeros(num_tracked_envs, dtype=bool)

            returns_flat = np.zeros(num_eval_episodes, dtype=np.float32)
            success_flat = np.zeros(num_eval_episodes, dtype=np.float32)
            steps_flat = np.zeros(num_eval_episodes, dtype=np.float32)

            running_returns = np.zeros(num_envs, dtype=np.float32)
            running_steps = np.zeros(num_envs, dtype=np.int64)
            running_success = np.zeros(num_envs, dtype=bool)

            if self.agent_arch == AGENT_ARCHS.Memory:
                action, reward, internal_state = self.expand_initial_info(num_envs)

            completed = 0

            while completed < num_tracked_envs:
                if self.agent_arch == AGENT_ARCHS.Memory:
                    (action, _, _, _), next_internal_state = self.agent.act(
                        prev_internal_state=internal_state,
                        prev_action=action,
                        reward=reward,
                        obs=obs,
                        deterministic=deterministic,
                    )
                else:
                    action, _, _, _ = self.agent.act(obs, deterministic=deterministic)
                    next_internal_state = None

                next_obs, reward_new, done, info = utl.env_step_batched(
                    self.eval_env, action
                )
                next_obs = next_obs.view(next_obs.shape[0], -1)

                done_np = ptu.get_numpy(done.squeeze(-1).bool())
                if eval_shaped_reward:
                    reward_np = ptu.get_numpy(reward_new.squeeze(-1))
                else:
                    if "raw_reward" not in info or info["raw_reward"] is None:
                        raise RuntimeError("Couldn't find raw_reward in info")
                    reward_np = ptu.get_numpy(info['raw_reward'].squeeze(-1))

                # Active for metric accumulation means:
                # - untracked envs: always active, since they are just filler envs
                # - tracked envs: active only until their first eval episode finishes
                active_mask = np.ones(num_envs, dtype=bool)
                active_mask[tracked_env_ids[tracked_done]] = False

                running_returns[active_mask] += reward_np[active_mask]
                running_steps[active_mask] += 1

                if isinstance(info, dict) and "success" in info:
                    success_info = info["success"]

                    if isinstance(success_info, torch.Tensor):
                        success_np = ptu.get_numpy(success_info).astype(bool).reshape(-1)
                        if success_np.size == 1:
                            if bool(success_np.item()):
                                running_success[active_mask] = True
                        else:
                            assert success_np.size == num_envs
                            running_success[active_mask] |= success_np[active_mask]

                    elif isinstance(success_info, np.ndarray):
                        success_np = success_info.astype(bool).reshape(-1)
                        if success_np.size == 1:
                            if bool(success_np.item()):
                                running_success[active_mask] = True
                        else:
                            assert success_np.size == num_envs
                            running_success[active_mask] |= success_np[active_mask]

                    elif bool(success_info):
                        running_success[active_mask] = True

                cutoff = running_steps >= num_steps_per_episode
                episode_finished = done_np | cutoff

                for local_idx, env_id in enumerate(tracked_env_ids):
                    if tracked_done[local_idx]:
                        continue

                    if episode_finished[env_id]:
                        returns_flat[local_idx] = running_returns[env_id]
                        success_flat[local_idx] = float(running_success[env_id])
                        steps_flat[local_idx] = float(running_steps[env_id])

                        tracked_done[local_idx] = True
                        completed += 1

                # Reset all finished envs, including tracked ones.
                # For tracked envs that are already done, metrics are frozen by active_mask.
                reset_env_ids = np.where(episode_finished)[0].tolist()

                if len(reset_env_ids) > 0:
                    reset_env_ids_t = torch.as_tensor(
                        reset_env_ids,
                        device=ptu.device,
                        dtype=torch.long,
                    )

                    reset_obs = self.eval_env.reset(env_ids=reset_env_ids_t)
                    reset_obs = (
                        ptu.from_numpy(reset_obs)
                        if isinstance(reset_obs, np.ndarray)
                        else reset_obs.to(ptu.device)
                    )
                    reset_obs = reset_obs.view(reset_obs.shape[0], -1)

                    next_obs[reset_env_ids_t] = reset_obs

                    running_returns[reset_env_ids] = 0.0
                    running_steps[reset_env_ids] = 0
                    running_success[reset_env_ids] = False

                    if self.agent_arch == AGENT_ARCHS.Memory:
                        if isinstance(next_internal_state, tuple):
                            next_internal_state[0][:, reset_env_ids_t, :] = 0.0
                            next_internal_state[1][:, reset_env_ids_t, :] = 0.0
                        else:
                            next_internal_state[:, reset_env_ids_t, :] = 0.0

                        action[reset_env_ids_t] = 0.0
                        reward_new[reset_env_ids_t] = 0.0

                obs = next_obs.clone()

                if self.agent_arch == AGENT_ARCHS.Memory:
                    reward = reward_new.clone()
                    internal_state = next_internal_state

        finally:
            pass
            # self.eval_env.unwrapped.set_use_reward_shaping(use_reward_shaping)

        returns_per_episode = returns_flat.reshape(num_eval_episodes, num_episodes)
        success_rate = success_flat
        total_steps = steps_flat

        avg_return = float(np.mean(returns_per_episode))
        if not hasattr(self, "_best_eval_return"):
            self._best_eval_return = -np.inf

        if avg_return >= self._best_eval_return:
            self._best_eval_return = avg_return

            if save_best:
                self.save_model(
                    step=self._n_env_steps_total,
                    perf=avg_return,
                    filename="best_agent.pt",
                )

                print(
                    f"[Best Model] New best eval return: {avg_return:.3f} "
                    f"at env step {self._n_env_steps_total}"
                )

        return returns_per_episode, success_rate, observations, total_steps

    @torch.no_grad()
    def evaluate(self, tasks, deterministic=True):

        num_episodes = self.max_rollouts_per_task  # k
        # max_trajectory_len = k*H
        returns_per_episode = np.zeros((len(tasks), num_episodes))
        success_rate = np.zeros(len(tasks))
        total_steps = np.zeros(len(tasks))

        if self.env_type == "meta":
            num_steps_per_episode = self.max_trajectory_len  # H
            obs_size = self.eval_env.unwrapped.observation_space.shape[
                0
            ]  # original size
            observations = np.zeros((len(tasks), self.max_trajectory_len + 1, obs_size))
        else:  # pomdp, rmdp, generalize
            num_steps_per_episode = self.max_trajectory_len
            observations = None

        for task_idx, task in enumerate(tasks):
            step = 0

            if self.env_type == "meta" and self.eval_env.n_tasks is not None:
                obs = ptu.from_numpy(self.eval_env.reset(task=task))  # reset task
                observations[task_idx, step, :] = ptu.get_numpy(obs[:obs_size])
            else:
                obs = ptu.from_numpy(self.eval_env.reset())  # reset

            obs = obs.reshape(1, obs.shape[-1])

            if self.agent_arch == AGENT_ARCHS.Memory:
                # assume initial reward = 0.0
                action, reward, internal_state = self.agent.get_initial_info()

            # Cache current use_reward_shaping                
            use_reward_shaping = self.eval_env.unwrapped.get_use_reward_shaping()
            self.eval_env.unwrapped.set_use_reward_shaping(False)

            try:
                for episode_idx in range(num_episodes):
                    running_reward = 0.0
                    for _ in range(num_steps_per_episode):
                        if self.agent_arch == AGENT_ARCHS.Memory:
                            (action, _, _, _), internal_state = self.agent.act(
                                prev_internal_state=internal_state,
                                prev_action=action,
                                reward=reward,
                                obs=obs,
                                deterministic=deterministic,
                            )
                        else:
                            action, _, _, _ = self.agent.act(
                                obs, deterministic=deterministic
                            )

                

                        # observe reward and next obs
                        next_obs, reward, done, info = utl.env_step(
                            self.eval_env, action.squeeze(dim=0)
                        )

                        # add raw reward
                        running_reward += reward.item()
                        # clip reward if necessary for policy inputs
                        if self.reward_clip and self.env_type == "atari":
                            reward = torch.tanh(reward)

                        step += 1
                        done_rollout = False if ptu.get_numpy(done[0][0]) == 0.0 else True

                        if self.env_type == "meta":
                            observations[task_idx, step, :] = ptu.get_numpy(
                                next_obs[0, :obs_size]
                            )

                        # set: obs <- next_obs
                        obs = next_obs.clone()

                        if (
                            self.env_type == "meta"
                            and "is_goal_state" in dir(self.eval_env.unwrapped)
                            and self.eval_env.unwrapped.is_goal_state()
                        ):
                            success_rate[task_idx] = 1.0  # ever once reach
                        elif (
                            self.env_type == "generalize"
                            and self.eval_env.unwrapped.is_success()
                        ):
                            success_rate[task_idx] = 1.0  # ever once reach
                        elif "success" in info and info["success"] == True:  # keytodoor
                            success_rate[task_idx] = 1.0

                        if done_rollout:
                            # for all env types, same
                            break
                        if self.env_type == "meta" and info["done_mdp"] == True:
                            # for early stopping meta episode like Ant-Dir
                            break

                    returns_per_episode[task_idx, episode_idx] = running_reward
                total_steps[task_idx] = step
            finally:
                # Set cached use_reward_shaping
                self.eval_env.unwrapped.set_use_reward_shaping(use_reward_shaping)

        return returns_per_episode, success_rate, observations, total_steps

    def log_train_stats(self, train_stats):
        logger.record_step(self._n_env_steps_total)
        ## log losses
        for k, v in train_stats.items():
            logger.record_tabular("rl_loss/" + k, v)
        ## gradient norms
        if self.agent_arch in [AGENT_ARCHS.Memory, AGENT_ARCHS.Memory_Markov]:
            results = self.agent.report_grad_norm()
            for k, v in results.items():
                logger.record_tabular("rl_loss/" + k, v)
        logger.dump_tabular()

    def log(self):
        # --- log training  ---
        ## set env steps for tensorboard: z is for lowest order
        logger.record_step(self._n_env_steps_total)
        logger.record_tabular("z/env_steps", self._n_env_steps_total)
        logger.record_tabular("z/rollouts", self._n_rollouts_total)
        logger.record_tabular("z/rl_steps", self._n_rl_update_steps_total)

        # --- evaluation ----
        if self.env_type == "meta":
            if self.train_env.n_tasks is not None:
                (
                    returns_train,
                    success_rate_train,
                    observations,
                    total_steps_train,
                ) = self.evaluate(self.train_tasks[: len(self.eval_tasks)])
            (
                returns_eval,
                success_rate_eval,
                observations_eval,
                total_steps_eval,
            ) = self.evaluate(self.eval_tasks)
            if self.eval_stochastic:
                (
                    returns_eval_sto,
                    success_rate_eval_sto,
                    observations_eval_sto,
                    total_steps_eval_sto,
                ) = self.evaluate(self.eval_tasks, deterministic=False)

            if self.train_env.n_tasks is not None and "plot_behavior" in dir(
                self.eval_env.unwrapped
            ):
                # plot goal-reaching trajs
                for i, task in enumerate(
                    self.train_tasks[: min(5, len(self.eval_tasks))]
                ):
                    self.eval_env.reset(task=task)  # must have task argument
                    logger.add_figure(
                        "trajectory/train_task_{}".format(i),
                        utl_eval.plot_rollouts(observations[i, :], self.eval_env),
                    )

                for i, task in enumerate(
                    self.eval_tasks[: min(5, len(self.eval_tasks))]
                ):
                    self.eval_env.reset(task=task)
                    logger.add_figure(
                        "trajectory/eval_task_{}".format(i),
                        utl_eval.plot_rollouts(observations_eval[i, :], self.eval_env),
                    )
                    if self.eval_stochastic:
                        logger.add_figure(
                            "trajectory/eval_task_{}_sto".format(i),
                            utl_eval.plot_rollouts(
                                observations_eval_sto[i, :], self.eval_env
                            ),
                        )

            if "is_goal_state" in dir(
                self.eval_env.unwrapped
            ):  # goal-reaching success rates
                # some metrics
                logger.record_tabular(
                    "metrics/successes_in_buffer",
                    self._successes_in_buffer / self._n_env_steps_total,
                )
                if self.train_env.n_tasks is not None:
                    logger.record_tabular(
                        "metrics/success_rate_train", np.mean(success_rate_train)
                    )
                logger.record_tabular(
                    "metrics/success_rate_eval", np.mean(success_rate_eval)
                )
                if self.eval_stochastic:
                    logger.record_tabular(
                        "metrics/success_rate_eval_sto", np.mean(success_rate_eval_sto)
                    )

            for episode_idx in range(self.max_rollouts_per_task):
                if self.train_env.n_tasks is not None:
                    logger.record_tabular(
                        "metrics/return_train_episode_{}".format(episode_idx + 1),
                        np.mean(returns_train[:, episode_idx]),
                    )
                logger.record_tabular(
                    "metrics/return_eval_episode_{}".format(episode_idx + 1),
                    np.mean(returns_eval[:, episode_idx]),
                )
                if self.eval_stochastic:
                    logger.record_tabular(
                        "metrics/return_eval_episode_{}_sto".format(episode_idx + 1),
                        np.mean(returns_eval_sto[:, episode_idx]),
                    )

            if self.train_env.n_tasks is not None:
                logger.record_tabular(
                    "metrics/total_steps_train", np.mean(total_steps_train)
                )
                logger.record_tabular(
                    "metrics/return_train_total",
                    np.mean(np.sum(returns_train, axis=-1)),
                )
            logger.record_tabular("metrics/total_steps_eval", np.mean(total_steps_eval))
            logger.record_tabular(
                "metrics/return_eval_total", np.mean(np.sum(returns_eval, axis=-1))
            )
            if self.eval_stochastic:
                logger.record_tabular(
                    "metrics/total_steps_eval_sto", np.mean(total_steps_eval_sto)
                )
                logger.record_tabular(
                    "metrics/return_eval_total_sto",
                    np.mean(np.sum(returns_eval_sto, axis=-1)),
                )

        elif self.env_type == "generalize":
            returns_eval, success_rate_eval, total_steps_eval = {}, {}, {}
            for env, (env_name, eval_num_episodes_per_task) in self.eval_envs.items():
                self.eval_env = env  # assign eval_env, not train_env
                for suffix, deterministic in zip(["", "_sto"], [True, False]):
                    if deterministic == False and self.eval_stochastic == False:
                        continue
                    return_eval, success_eval, _, total_step_eval = self.evaluate(
                        eval_num_episodes_per_task * [None],
                        deterministic=deterministic,
                    )
                    returns_eval[
                        self.train_env_name + env_name + suffix
                    ] = return_eval.squeeze(-1)
                    success_rate_eval[
                        self.train_env_name + env_name + suffix
                    ] = success_eval
                    total_steps_eval[
                        self.train_env_name + env_name + suffix
                    ] = total_step_eval

            for k, v in returns_eval.items():
                logger.record_tabular(f"metrics/return_eval_{k}", np.mean(v))
            for k, v in success_rate_eval.items():
                logger.record_tabular(f"metrics/succ_eval_{k}", np.mean(v))
            for k, v in total_steps_eval.items():
                logger.record_tabular(f"metrics/total_steps_eval_{k}", np.mean(v))

        elif self.env_type == "rmdp":
            returns_eval, _, _, total_steps_eval = self.evaluate(self.eval_tasks)
            returns_eval = returns_eval.squeeze(-1)
            # np.quantile is introduced in np v1.15, so we have to use np.percentile
            cutoff = np.percentile(returns_eval, 100 * self.worst_percentile)
            worst_indices = np.where(
                returns_eval <= cutoff
            )  # must be "<=" to avoid empty set
            returns_eval_worst, total_steps_eval_worst = (
                returns_eval[worst_indices],
                total_steps_eval[worst_indices],
            )

            logger.record_tabular("metrics/return_eval_avg", returns_eval.mean())
            logger.record_tabular(
                "metrics/return_eval_worst", returns_eval_worst.mean()
            )
            logger.record_tabular(
                "metrics/total_steps_eval_avg", total_steps_eval.mean()
            )
            logger.record_tabular(
                "metrics/total_steps_eval_worst", total_steps_eval_worst.mean()
            )

        elif self.env_type in ["pomdp", "credit", "atari"]:
            eval_fn = self.evaluate_batched if getattr(self, "vectorized_env", False) else self.evaluate

            returns_eval, success_rate_eval, _, total_steps_eval = eval_fn(
                self.eval_tasks
            )

            if self.eval_stochastic:
                (
                    returns_eval_sto,
                    success_rate_eval_sto,
                    _,
                    total_steps_eval_sto,
                ) = eval_fn(self.eval_tasks, deterministic=False)

            logger.record_tabular("metrics/total_steps_eval", np.mean(total_steps_eval))
            logger.record_tabular(
                "metrics/return_eval_total", np.mean(np.sum(returns_eval, axis=-1))
            )
            logger.record_tabular(
                "metrics/success_rate_eval", np.mean(success_rate_eval)
            )

            if self.eval_stochastic:
                logger.record_tabular(
                    "metrics/total_steps_eval_sto", np.mean(total_steps_eval_sto)
                )
                logger.record_tabular(
                    "metrics/return_eval_total_sto",
                    np.mean(np.sum(returns_eval_sto, axis=-1)),
                )
                logger.record_tabular(
                    "metrics/success_rate_eval_sto", np.mean(success_rate_eval_sto)
                )

        else:
            raise ValueError

        logger.record_tabular("z/time_cost", int(time.time() - self._start_time))
        logger.record_tabular(
            "z/fps",
            (self._n_env_steps_total - self._n_env_steps_total_last)
            / (time.time() - self._start_time_last),
        )
        self._n_env_steps_total_last = self._n_env_steps_total
        self._start_time_last = time.time()

        logger.dump_tabular()

        if self.env_type == "generalize":
            return sum([v.mean() for v in success_rate_eval.values()]) / len(
                success_rate_eval
            )
        else:
            return np.mean(np.sum(returns_eval, axis=-1))

    def save_model(self, step, perf, filename=None):
        if filename is None:
            filename = f"agent_{step}_perf{perf:.3f}.pt"

        save_dir = os.path.join(logger.get_dir(), "save")
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, filename)
        torch.save(self.agent.state_dict(), save_path)

        print(f"[Model] Saved to {save_path}")

    def load_model(self, ckpt_path):
        self.agent.load_state_dict(torch.load(ckpt_path, map_location=ptu.device))
        print("load successfully from", ckpt_path)

    def _collect_training_state(self):
        import random

        algo = self.agent.algo

        ckpt = {
            "agent_state_dict": self.agent.state_dict(),

            "actor_optimizer_state_dict": self.agent.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.agent.critic_optimizer.state_dict(),

            "n_env_steps_total": self._n_env_steps_total,
            "n_env_steps_total_last": self._n_env_steps_total_last,
            "n_rl_update_steps_total": self._n_rl_update_steps_total,
            "n_rollouts_total": self._n_rollouts_total,
            "successes_in_buffer": self._successes_in_buffer,
            "best_eval_return": self._best_eval_return,

            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None,
            },
        }

        if getattr(algo, "automatic_entropy_tuning", False):
            ckpt["sacd"] = {
                "automatic_entropy_tuning": True,
                "log_alpha_entropy": algo.log_alpha_entropy.detach().cpu(),
                "alpha_entropy": float(algo.alpha_entropy),
                "alpha_entropy_optim_state_dict": algo.alpha_entropy_optim.state_dict(),
            }
        else:
            ckpt["sacd"] = {
                "automatic_entropy_tuning": False,
                "alpha_entropy": float(algo.alpha_entropy),
            }        

        return ckpt

    def save_replay_buffer(self, path):
        buf = self.policy_storage

        np.savez_compressed(
            path,
            observations=buf._observations,
            actions=buf._actions,
            rewards=buf._rewards,
            terminals=buf._terminals,
            ends=buf._ends,
            valid_starts=buf._valid_starts,
            top=buf._top,
            size=buf._size,
            num_eps=buf._num_total_episodes_seen,
            num_skipped=buf._num_skipped_short_episodes,
        )


    def save_training_checkpoint(
        self,
        path=None,        
    ):
        """
        Save full-ish training state for ModelFreeOffPolicy_Separate_RNN + SACD.

        This saves:
          - actor/critic/target network weights
          - actor/critic optimizer states
          - SAC-D entropy alpha state
          - SAC-D alpha optimizer state
          - learner counters
          - RNG states
          - optionally replay/sequence buffer
        """
        if path is None:
            path = os.path.join(logger.get_dir(), "save", "training_latest.pt")

        os.makedirs(os.path.dirname(path), exist_ok=True)

        ckpt = self._collect_training_state()
        torch.save(ckpt, path)

        print(f"[Checkpoint] Saved training checkpoint to: {path}")

    def load_replay_buffer(self, path):
        data = np.load(path)

        buf = self.policy_storage

        buf._observations[:] = data["observations"]
        buf._actions[:] = data["actions"]
        buf._rewards[:] = data["rewards"]
        buf._terminals[:] = data["terminals"]
        buf._ends[:] = data["ends"]
        buf._valid_starts[:] = data["valid_starts"]

        buf._top = int(data["top"])
        buf._size = int(data["size"])
        buf._num_total_episodes_seen = int(data["num_eps"])
        buf._num_skipped_short_episodes = int(data["num_skipped"])


    def load_training_checkpoint(
        self,
        path,        
        restore_rng=True,
    ):
        """
        Load checkpoint saved by save_training_checkpoint().
        Assumes the Learner/agent/env have already been constructed with matching config.
        """
        import random

        ckpt = torch.load(path, map_location=ptu.device)

        self.agent.load_state_dict(ckpt["agent_state_dict"])

        self.agent.actor_optimizer.load_state_dict(
            ckpt["actor_optimizer_state_dict"]
        )
        self.agent.critic_optimizer.load_state_dict(
            ckpt["critic_optimizer_state_dict"]
        )

        self._n_env_steps_total = int(ckpt.get("n_env_steps_total", 0))
        self._n_env_steps_total_last = int(
            ckpt.get("n_env_steps_total_last", self._n_env_steps_total)
        )
        self._n_rl_update_steps_total = int(
            ckpt.get("n_rl_update_steps_total", 0)
        )
        self._n_rollouts_total = int(ckpt.get("n_rollouts_total", 0))
        self._successes_in_buffer = int(ckpt.get("successes_in_buffer", 0))  
        self._best_eval_return = float(ckpt.get("best_eval_return", -np.inf))      

        sacd_state = ckpt.get("sacd", None)
        algo = self.agent.algo

        if sacd_state is not None:
            if sacd_state.get("automatic_entropy_tuning", False):
                if not getattr(algo, "automatic_entropy_tuning", False):
                    raise RuntimeError(
                        "Checkpoint uses SAC-D automatic entropy tuning, "
                        "but current algo has automatic_entropy_tuning=False."
                    )

                algo.log_alpha_entropy.data.copy_(
                    sacd_state["log_alpha_entropy"].to(ptu.device)
                )
                algo.alpha_entropy = float(algo.log_alpha_entropy.exp().item())

                algo.alpha_entropy_optim.load_state_dict(
                    sacd_state["alpha_entropy_optim_state_dict"]
                )
            else:
                algo.alpha_entropy = float(sacd_state["alpha_entropy"])        

        if restore_rng and "rng_state" in ckpt:
            rng = ckpt["rng_state"]

            if rng.get("python", None) is not None:
                random.setstate(rng["python"])

            if rng.get("numpy", None) is not None:
                np.random.set_state(rng["numpy"])

            if rng.get("torch", None) is not None:
                torch.set_rng_state(rng["torch"])

            if torch.cuda.is_available() and rng.get("cuda", None) is not None:
                torch.cuda.set_rng_state_all(rng["cuda"])

        print(
            f"[Checkpoint] Loaded training checkpoint from: {path} | "
            f"env_steps={self._n_env_steps_total}, "
            f"rl_updates={self._n_rl_update_steps_total}, "
            f"rollouts={self._n_rollouts_total}, "
            f"alpha={getattr(self.agent.algo, 'alpha_entropy', None)}"
        )
