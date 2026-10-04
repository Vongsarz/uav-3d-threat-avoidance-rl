import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback
from envs.uav_env import UAVThreatAvoidanceEnv
def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # 1. Ortamı başlat ve doğrula
    env = UAVThreatAvoidanceEnv()
    check_env(env)
    print("✓ Gymnasium ortamı standartlara uygun.")

    # 2. Her 20.000 adımda bir modeli yedekleyen callback
    checkpoint_callback = CheckpointCallback(
        save_freq=20_000,
        save_path="models/",
        name_prefix="ppo_uav"
    )

    # 3. PPO Algoritması Yapılandırması
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,          # Keşfi teşvik eden entropi katsayısı
        tensorboard_log="logs/",
        verbose=1
    )

    # 4. Eğitimi Başlat (150.000 adım)
    print("Eğitim başlıyor...")
    model.learn(total_timesteps=300_000, callback=checkpoint_callback)

    # 5. Nihai Modeli Kaydet
    model.save("models/ppo_uav_final")
    print("✓ Eğitim tamamlandı. Model 'models/ppo_uav_final.zip' olarak kaydedildi.")

if __name__ == "__main__":
    main()