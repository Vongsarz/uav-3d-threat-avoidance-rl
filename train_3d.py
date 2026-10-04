import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback
from envs.uav_env_3d import UAVThreatAvoidance3DEnv

def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    env = UAVThreatAvoidance3DEnv()
    check_env(env)
    print("✓ 3B Gymnasium ortamı doğrulandı.")

    checkpoint_callback = CheckpointCallback(
        save_freq=30_000,
        save_path="models/",
        name_prefix="ppo_uav_3d"
    )

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        tensorboard_log="logs/",
        verbose=1
    )

    print("3B PPO Eğitimi Başlıyor (250.000 Adım)...")
    model.learn(total_timesteps=250_000, callback=checkpoint_callback)

    model.save("models/ppo_uav_3d_final")
    print("✓ Eğitim tamamlandı. Model 'models/ppo_uav_3d_final.zip' olarak kaydedildi.")

if __name__ == "__main__":
    main()