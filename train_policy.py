import argparse
import os

from sb3_contrib import CrossQ
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.env_checker import check_env

from env.navigation_env import NavigationEnv


class SaveModelCallback(BaseCallback):
    def __init__(self, save_freq, verbose=1):
        super(SaveModelCallback, self).__init__(verbose)
        self.save_freq = save_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            save_file = "CrossQ_navigation.zip"
            self.model.save(save_file)
        return True  # Continue training


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the last saved model",
    )
    parser.add_argument(
        "--save_every", type=int, default=10000, help="Save model every n steps"
    )
    parser.add_argument(
        "--eval_every", type=int, default=10000, help="Evaluate model every n steps"
    )
    parser.add_argument(
        "--training_steps", type=int, default=500000, help="Number of training steps"
    )
    args = parser.parse_args()
    env = NavigationEnv(window=False)
    eval_env = NavigationEnv(window=False, eval=True)

    check_env(env, warn=True)
    save_callback = SaveModelCallback(save_freq=args.save_every)
    eval_callback = EvalCallback(
        eval_env, eval_freq=args.eval_every, deterministic=True
    )

    if args.resume and os.path.exists("CrossQ_navigation.zip"):
        print("Resuming training from saved model...")
        model = CrossQ.load("CrossQ_navigation", env=env)
    else:
        print("Starting fresh training...")
        model = CrossQ(
            "MlpPolicy",
            env,
            batch_size=512,
            verbose=1,
            tensorboard_log="./CrossQ_nav_tensorboard/",
        )

    model.learn(
        total_timesteps=args.training_steps,
        callback=[save_callback, eval_callback],
        progress_bar=True,
    )
    model.save("CrossQ_navigation")
