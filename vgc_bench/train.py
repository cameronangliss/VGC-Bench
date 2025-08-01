import argparse
import os

from src.callback import Callback
from src.env import ShowdownEnv
from src.policy import MaskedActorCriticPolicy
from src.utils import LearningStyle, allow_mirror_match, chooses_on_teampreview, num_envs, steps
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv


def train(
    teams: list[int],
    port: int,
    device: str,
    learning_style: LearningStyle,
    behavior_clone: bool,
    num_frames: int,
):
    env = (
        ShowdownEnv.create_env(teams, port, device, learning_style, num_frames)
        if learning_style == LearningStyle.PURE_SELF_PLAY
        else SubprocVecEnv(
            [
                lambda: ShowdownEnv.create_env(
                    [0] if learning_style == LearningStyle.EXPLOITER else teams,
                    port,
                    device,
                    learning_style,
                    num_frames,
                )
                for _ in range(num_envs)
            ]
        )
    )
    run_ident = "".join(
        [
            "-bc" if behavior_clone else "",
            f"-fs{num_frames}" if num_frames > 1 else "",
            "-" + learning_style.abbrev,
            "-xm" if not allow_mirror_match else "",
        ]
    )[1:]
    ppo = PPO(
        MaskedActorCriticPolicy,
        env,
        learning_rate=1e-5,
        n_steps=(
            3072 // (2 * num_envs)
            if learning_style == LearningStyle.PURE_SELF_PLAY
            else 3072 // num_envs
        ),
        batch_size=64,
        gamma=1,
        ent_coef=1e-3,
        tensorboard_log=f"results/logs-{run_ident}",
        policy_kwargs={"num_frames": num_frames, "chooses_on_teampreview": chooses_on_teampreview},
        device=device,
    )
    num_saved_timesteps = 0
    if (
        os.path.exists(f"results/saves-{run_ident}/{','.join([str(t) for t in teams])}-teams")
        and len(os.listdir(f"results/saves-{run_ident}/{','.join([str(t) for t in teams])}-teams"))
        > 0
    ):
        saved_policy_timesteps = [
            int(file[:-4])
            for file in os.listdir(
                f"results/saves-{run_ident}/{','.join([str(t) for t in teams])}-teams"
            )
            if int(file[:-4]) >= 0
        ]
        if saved_policy_timesteps:
            num_saved_timesteps = max(saved_policy_timesteps)
            ppo.set_parameters(
                f"results/saves-{run_ident}/{','.join([str(t) for t in teams])}-teams/{num_saved_timesteps}.zip",
                device=ppo.device,
            )
            if num_saved_timesteps < steps:
                num_saved_timesteps = 0
            ppo.num_timesteps = num_saved_timesteps
    if num_saved_timesteps >= 5_013_504:
        print(f"Skipping training, already trained for {num_saved_timesteps} steps")
        return
    ppo.learn(
        steps,
        callback=Callback(teams, port, device, learning_style, behavior_clone, num_frames),
        tb_log_name=f"{','.join([str(t) for t in teams])}-teams",
        reset_num_timesteps=False,
    )
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Pokémon AI model")
    parser.add_argument("--teams", nargs="+", type=int, help="Indices of teams to train with")
    parser.add_argument("--num_teams", type=int, help="Number of teams to train with")
    parser.add_argument("--port", type=int, default=8000, help="Port to run showdown server on")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        choices=["cuda:0", "cuda:1", "cuda:2", "cuda:3"],
        help="CUDA device to use for training",
    )
    parser.add_argument("--exploiter", action="store_true", help="play against fixed bot")
    parser.add_argument("--self_play", action="store_true", help="do pure self-play")
    parser.add_argument("--last_self", action="store_true", help="do last-self play")
    parser.add_argument("--fictitious_play", action="store_true", help="do fictitious play")
    parser.add_argument("--double_oracle", action="store_true", help="do double oracle")
    parser.add_argument(
        "--behavior_clone", action="store_true", help="Warm up with behavior cloning"
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=1,
        help="number of frames to use for frame stacking. default is 1",
    )
    args = parser.parse_args()
    assert (args.teams is None) != (
        args.num_teams is None
    ), "Only pass one of --teams and --num_teams in"
    assert (
        int(args.exploiter)
        + int(args.self_play)
        + int(args.last_self)
        + int(args.fictitious_play)
        + int(args.double_oracle)
        == 1
    )
    teams = args.teams if args.teams is not None else list(range(args.num_teams))
    if args.exploiter:
        style = LearningStyle.EXPLOITER
    elif args.self_play:
        style = LearningStyle.PURE_SELF_PLAY
    elif args.last_self:
        style = LearningStyle.LAST_SELF
    elif args.fictitious_play:
        style = LearningStyle.FICTITIOUS_PLAY
    elif args.double_oracle:
        style = LearningStyle.DOUBLE_ORACLE
    else:
        raise TypeError()
    train(teams, args.port, args.device, style, args.behavior_clone, args.num_frames)
