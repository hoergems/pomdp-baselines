# -*- coding: future_fstrings -*-
import sys, os, time
import socket
import numpy as np
import torch
import json
from ruamel.yaml import YAML
from absl import flags
from pathlib import Path

from utils import system, logger
from torchkit.pytorch_utils import set_gpu_mode
from policies.learner import Learner


FLAGS = flags.FLAGS

flags.DEFINE_string("cfg", None, "path to configuration file")
flags.DEFINE_string("env", None, "env_name")
flags.DEFINE_string("algo", None, '["td3", "sac", "sacd"]')
flags.DEFINE_integer("seed", None, "seed")
flags.DEFINE_integer("cuda", None, "cuda device id")
flags.DEFINE_boolean("oracle", False, "whether observe privileged POMDP info")
flags.DEFINE_string("agent_weights", None, "path to best_agent.pt")
flags.DEFINE_boolean("deterministic", True, "run deterministic policy")
flags.DEFINE_integer("num_episodes", 1, "number of eval episodes to run")


def main():
    t0 = time.time()

    flags.FLAGS(sys.argv)

    yaml = YAML()
    v = yaml.load(open(FLAGS.cfg))

    # overwrite config params
    if FLAGS.env is not None:
        v["env"]["env_name"] = FLAGS.env
    if FLAGS.algo is not None:
        v["policy"]["algo_name"] = FLAGS.algo
    if FLAGS.seed is not None:
        v["seed"] = FLAGS.seed
    if FLAGS.cuda is not None:
        v["cuda"] = FLAGS.cuda
    if FLAGS.oracle:
        v["env"]["oracle"] = True

    seq_model = v["policy"]["seq_model"]
    algo = v["policy"]["algo_name"]

    assert seq_model in ["mlp", "lstm", "gru", "lstm-mlp", "gru-mlp"]
    assert algo in ["td3", "sac", "sacd"]

    if FLAGS.agent_weights is None:
        raise ValueError("Please provide --agent_weights /path/to/best_agent.pt")

    # system: device, threads, seed
    seed = v["seed"]
    system.reproduce(seed)

    torch.set_num_threads(1)
    np.set_printoptions(precision=3, suppress=True)
    torch.set_printoptions(precision=3, sci_mode=False)

    set_gpu_mode(torch.cuda.is_available() and v["cuda"] >= 0, v["cuda"])

    # Minimal logging dir
    log_folder = os.path.join("logs", "run_agent", system.now_str())
    logger.configure(
        dir=log_folder,
        format_strs=["stdout", "log"],
        precision=4,
    )

    logger.log(f"preload cost {time.time() - t0:.2f}s")
    logger.log("pid", os.getpid(), socket.gethostname())
    yaml.dump(v, Path(f"{log_folder}/variant_run.yml"))

    learner = Learner(
        env_args=v["env"],
        train_args=v["train"],
        eval_args=v["eval"],
        policy_args=v["policy"],
        seed=seed,
    )

    learner.load_model(FLAGS.agent_weights)

    learner.eval_tasks = FLAGS.num_episodes * [None]

    if getattr(learner, "vectorized_env", False):
        assert FLAGS.num_episodes <= learner.num_envs, (
            f"num_episodes={FLAGS.num_episodes} must be <= "
            f"num_envs={learner.num_envs} for evaluate_batched()."
        )
        eval_fn = learner.evaluate_batched
    else:
        eval_fn = learner.evaluate

    returns, success_rate, _, total_steps = eval_fn(
        learner.eval_tasks,
        deterministic=FLAGS.deterministic,
        save_best=False,
    )

    returns_total = np.sum(returns, axis=-1)

    logger.log("==== Run Agent Results ====")
    logger.log("returns:", returns_total)
    logger.log("mean return:", float(np.mean(returns_total)))
    logger.log("success rate:", float(np.mean(success_rate)))
    logger.log("steps:", total_steps)
    logger.log("mean steps:", float(np.mean(total_steps)))

    results = {
        "returns": returns.tolist(),
        "returns_total": returns_total.tolist(),
        "success_rate": success_rate.tolist(),
        "total_steps": total_steps.tolist(),
        "mean_return": float(np.mean(returns_total)),
        "std_return": float(np.std(returns_total)),
        "mean_success_rate": float(np.mean(success_rate)),
        "mean_steps": float(np.mean(total_steps)),
        "num_episodes": int(FLAGS.num_episodes),
        "deterministic": bool(FLAGS.deterministic),        
    }

    results_path = os.path.join(log_folder, "results.json")

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.log(f"Saved machine-readable results to {results_path}")

    learner.train_env.close()
    learner.eval_env.close()


if __name__ == "__main__":
    main()