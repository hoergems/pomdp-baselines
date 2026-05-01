import numpy as np


class RAMEfficient_SeqReplayBuffer:
    buffer_type = "seq_efficient"

    def __init__(
        self,
        max_replay_buffer_size,
        observation_dim,
        action_dim,
        sampled_seq_len: int,
        sample_weight_baseline: float,
        observation_type,
        **kwargs
    ):
        """
        Sequence replay buffer.

        This version explicitly touches all allocated arrays once during construction
        so RSS reflects the true buffer footprint immediately, instead of growing
        gradually as pages are written during training.
        """
        self._max_replay_buffer_size = max_replay_buffer_size
        self._observation_dim = observation_dim
        self._action_dim = action_dim

        if observation_type == np.uint8:  # pixel
            observation_type = np.uint8
        else:  # treat all as float32
            observation_type = np.float32

        self._observations = np.empty(
            (max_replay_buffer_size, observation_dim),
            dtype=observation_type,
        )
        self._actions = np.empty(
            (max_replay_buffer_size, action_dim),
            dtype=np.float32,
        )
        self._rewards = np.empty(
            (max_replay_buffer_size, 1),
            dtype=np.float32,
        )

        # terminals are "done" signals, useful for policy training
        self._terminals = np.empty(
            (max_replay_buffer_size, 1),
            dtype=np.uint8,
        )

        # episode boundary markers
        self._ends = np.empty(
            (max_replay_buffer_size,),
            dtype=np.uint8,
        )

        # valid sequence starts / sampling weights
        self._valid_starts = np.empty(
            (max_replay_buffer_size,),
            dtype=np.float32,
        )

        # Force physical page commitment now.
        # This makes RSS jump at construction instead of slowly rising as the buffer fills.
        self._observations.fill(1)
        self._actions.fill(1.0)
        self._rewards.fill(1.0)
        self._terminals.fill(1)
        self._ends.fill(1)
        self._valid_starts.fill(1.0)

        # Reset to the real initial contents.
        self._observations.fill(0)
        self._actions.fill(0.0)
        self._rewards.fill(0.0)
        self._terminals.fill(0)
        self._ends.fill(0)
        self._valid_starts.fill(0.0)

        assert sampled_seq_len >= 2
        assert sample_weight_baseline >= 0.0
        self._sampled_seq_len = sampled_seq_len
        self._sample_weight_baseline = sample_weight_baseline

        self.clear()

        RAM = 0.0
        for name, var in vars(self).items():
            if isinstance(var, np.ndarray):
                RAM += var.nbytes
        print(f"buffer RAM usage: {RAM / 1024 ** 3 :.2f} GB")

    def size(self):
        return self._size

    def clear(self):
        self._top = 0  # trajectory level (first dim in 3D buffer)
        self._size = 0  # trajectory level (first dim in 3D buffer)
        self._num_total_episodes_seen = 0
        self._num_skipped_short_episodes = 0

    def add_episode(self, observations, actions, rewards, terminals, next_observations):
        """
        Add one full episode/sequence/trajectory.

        All inputs must have shape (L, dim). For this sequence replay buffer,
        only episodes with L >= 2 are stored. Shorter episodes are skipped.
        """

        if not (
            observations.shape[0]
            == actions.shape[0]
            == rewards.shape[0]
            == terminals.shape[0]
            == next_observations.shape[0]
        ):
            raise ValueError(
                f"Mismatched episode lengths: "
                f"obs={observations.shape}, "
                f"act={actions.shape}, "
                f"rew={rewards.shape}, "
                f"term={terminals.shape}, "
                f"next_obs={next_observations.shape}"
            )

        seq_len = observations.shape[0]

        # --- bookkeeping for diagnostics ---
        self._num_total_episodes_seen += 1

        if seq_len < 2:
            self._num_skipped_short_episodes += 1

            if (
                self._num_skipped_short_episodes <= 10
                or self._num_skipped_short_episodes % 100 == 0
            ):
                skipped_frac = (
                    self._num_skipped_short_episodes / self._num_total_episodes_seen
                )
                print(
                    f"[DEBUG] Skipping short episode of length {seq_len} "
                    f"(skipped={self._num_skipped_short_episodes}, "
                    f"total={self._num_total_episodes_seen}, "
                    f"frac={skipped_frac:.3f})"
                )
            return

        write_start = self._top
        write_len = seq_len + 1
        self._invalidate_overwritten_starts(write_start, write_len)

        indices = list(
            np.arange(self._top, self._top + seq_len) % self._max_replay_buffer_size
        )

        self._observations[indices] = observations
        self._actions[indices] = actions
        self._rewards[indices] = rewards
        self._terminals[indices] = terminals
        self._valid_starts[indices] = self._compute_valid_starts(seq_len)
        self._ends[indices] = 0

        self._top = (self._top + seq_len) % self._max_replay_buffer_size

        # add final transition: obs is useful but the others are just padding
        self._observations[self._top] = next_observations[-1]  # final obs
        self._actions[self._top] = 0.0
        self._rewards[self._top] = 0.0
        self._terminals[self._top] = 1
        self._valid_starts[self._top] = 0.0  # never be sampled as starts
        self._ends[self._top] = 1  # the end of one episode

        self._top = (self._top + 1) % self._max_replay_buffer_size
        self._size = min(self._size + seq_len + 1, self._max_replay_buffer_size)

    def _compute_valid_starts(self, seq_len):
        valid_starts = np.ones((seq_len), dtype=float)

        num_valid_starts = float(max(1.0, seq_len - self._sampled_seq_len + 1.0))

        # compute weights: baseline + num_of_can_sampled_indices
        total_weights = self._sample_weight_baseline + num_valid_starts

        # now each item has even weight, if baseline is 0.0, then it's 1s
        valid_starts *= total_weights / num_valid_starts

        # set the num_valid_starts: indices are zeros
        valid_starts[int(num_valid_starts) :] = 0.0

        return valid_starts

    def random_episodes(self, batch_size):
        """
        return each item has 3D shape (sampled_seq_len, batch_size, dim)
        """
        sampled_episode_starts = self._sample_indices(batch_size)  # (B,)

        # get sequential indices
        indices = []
        next_indices = []  # for next obs
        for start in sampled_episode_starts:  # small loop
            end = start + self._sampled_seq_len  # continuous + T
            indices += list(np.arange(start, end) % self._max_replay_buffer_size)
            next_indices += list(
                np.arange(start + 1, end + 1) % self._max_replay_buffer_size
            )

        # extract data
        batch = self._sample_data(indices, next_indices)
        # each item has 2D shape (num_episodes * sampled_seq_len, dim)

        # generate masks (B, T)
        masks = self._generate_masks(indices, batch_size)
        batch["mask"] = masks

        for k in batch.keys():
            batch[k] = (
                batch[k]
                .reshape(batch_size, self._sampled_seq_len, -1)
                .transpose(1, 0, 2)
            )

        return batch

    def _sample_indices(self, batch_size):
        # self._top points at the start of a new sequence
        # self._top - 1 is the end of the recently stored sequence
        valid_starts_indices = np.where(self._valid_starts > 0.0)[0]

        sample_weights = np.copy(self._valid_starts[valid_starts_indices])
        # normalize to probability distribution
        sample_weights /= sample_weights.sum()

        return np.random.choice(valid_starts_indices, size=batch_size, p=sample_weights)

    def _sample_data(self, indices, next_indices):
        return dict(
            obs=self._observations[indices],
            act=self._actions[indices],
            rew=self._rewards[indices],
            term=self._terminals[indices],
            obs2=self._observations[next_indices],
        )

    def _generate_masks(self, indices, batch_size):
        """
        input: sampled_indices list of len B*T
        output: masks (B, T)
        """

        # get ends of sampled sequences (B, T)
        # each row starts with 0, like 0000000 or 0000010001
        sampled_seq_ends = (
            np.copy(self._ends[indices])
            .reshape(batch_size, self._sampled_seq_len)
            .astype(np.float32)
        )

        # build masks
        masks = np.ones_like(sampled_seq_ends)  # (B, T), default is 1

        # we want to find the boundary (ending) of sampled sequences
        # 	i.e. **the FIRST 1 after 0** (if exists)
        # 	this is important for varying length episodes
        # the boundary (ending) appears at the FIRST -1 in diff
        diff = sampled_seq_ends[:, :-1] - sampled_seq_ends[:, 1:]  # (B, T-1)
        # add 0s into the first column
        diff = np.concatenate([np.zeros((batch_size, 1)), diff], axis=1)  # (B, T)

        # now the start of next episode appears at the FIRST -1 in diff
        invalid_starts_b, invalid_starts_t = np.where(
            diff == -1.0
        )  # (1D array in batch dim, 1D array in seq dim)
        invalid_indices_b = []
        invalid_indices_t = []
        last_batch_index = -1

        for batch_index, start_index in zip(invalid_starts_b, invalid_starts_t):
            if batch_index == last_batch_index:
                # for same batch_idx, we only care the first appearance of -1
                continue
            last_batch_index = batch_index

            invalid_indices = list(
                np.arange(start_index, self._sampled_seq_len)
            )  # to the end
            # extend to the list
            invalid_indices_b += [batch_index] * len(invalid_indices)
            invalid_indices_t += invalid_indices

        # set invalids in the masks
        masks[invalid_indices_b, invalid_indices_t] = 0.0

        return masks

    def _invalidate_overwritten_starts(self, write_start, write_len):
        """
        Invalidate any old sequence start whose sampled window may overlap
        the storage region about to be overwritten.
        """
        invalidate_indices = (
            np.arange(
                write_start - self._sampled_seq_len,
                write_start + write_len,
            )
            % self._max_replay_buffer_size
        )
        self._valid_starts[invalidate_indices] = 0.0


if __name__ == "__main__":
    buffer_size = 100
    obs_dim = act_dim = 1
    sampled_seq_len = 7
    baseline = 0.0
    buffer = RAMEfficient_SeqReplayBuffer(
        buffer_size, obs_dim, act_dim, sampled_seq_len, baseline, np.uint8
    )
    for l in range(sampled_seq_len - 1, sampled_seq_len + 5):
        print(l)
        assert buffer._compute_valid_starts(l)[0] > 0.0
        print(buffer._compute_valid_starts(l))
    for _ in range(200):
        e = np.random.randint(3, 10)
        buffer.add_episode(
            np.arange(e).reshape(e, 1),
            np.zeros((e, 1)),
            np.zeros((e, 1)),
            np.zeros((e, 1)),
            np.arange(1, e + 1).reshape(e, 1),
        )
    print(buffer._size, buffer._top)
    print(
        np.concatenate(
            [buffer._observations, buffer._valid_starts[:, np.newaxis]], axis=-1
        )
    )

    for _ in range(10):
        batch = buffer.random_episodes(1)  # (T, B, dim)
        print(batch["obs"][:, 0, 0])
        print(batch["obs2"][:, 0, 0])
        print(batch["mask"][:, 0, 0].astype(np.int32))
        print("\n")
