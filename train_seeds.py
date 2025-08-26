import argparse
import os
from sb3_contrib import CrossQ
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from env.navigation_env import NavigationEnv
from typing import Callable
import torch

class SaveModelCallback(BaseCallback):
    def __init__(self, save_freq, save_path, verbose=1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            self.model.save(self.save_path)  # filename includes seed
        return True


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    return lambda progress_remaining: progress_remaining * initial_value


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save_every", type=int, default=10000)
    parser.add_argument("--eval_every", type=int, default=10000)
    parser.add_argument("--training_steps", type=int, default=200000)
    args = parser.parse_args()
    import numpy as np

    seeds = np.random.randint(0, 1_000_000, size=10)

    for seed in seeds:
        seed = int(seed)
        print(f"\n=== Training with seed {seed} ===")
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        env = NavigationEnv(window=False)
        eval_env = NavigationEnv(window=False, eval=True)

        save_file = f"CrossQ_navigation_seed{seed}.zip"
        save_callback = SaveModelCallback(save_freq=args.save_every, save_path=save_file)
        eval_callback = EvalCallback(eval_env, eval_freq=args.eval_every, deterministic=True)

        if args.resume and os.path.exists(save_file):
            print(f"Resuming training from {save_file}...")
            model = CrossQ.load(save_file, env=env, seed=seed)
        else:
            print("Starting fresh training...")
            model = CrossQ(
                "MlpPolicy",
                env,
                batch_size=512,
                verbose=1,
                gamma=0.95,
                tensorboard_log=f"./CrossQ_nav_tensorboard_seed{seed}/",
                learning_rate=linear_schedule(1e-3),
                seed=seed
            )

        model.learn(
            total_timesteps=args.training_steps,
            callback=[save_callback, eval_callback],
            progress_bar=True,
        )

        model.save(save_file)  # final save
        env.close()
        eval_env.close()
