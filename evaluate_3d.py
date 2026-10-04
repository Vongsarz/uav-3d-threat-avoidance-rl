import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from stable_baselines3 import PPO
from envs.uav_env_3d import UAVThreatAvoidance3DEnv

def run_evaluation(episodes=3):
    env = UAVThreatAvoidance3DEnv()
    model = PPO.load("models/ppo_uav_3d_final")

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        uav_x, uav_y, uav_z = [], [], []
        th_x, th_y, th_z = [], [], []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            uav_x.append(env.uav_pos[0])
            uav_y.append(env.uav_pos[1])
            uav_z.append(env.uav_pos[2])

            th_x.append(env.threat_pos[0])
            th_y.append(env.threat_pos[1])
            th_z.append(env.threat_pos[2])

        # 3B Grafik Çizimi
        fig = plt.figure(figsize=(11, 7))
        ax = fig.add_subplot(111, projection='3d')

        ax.plot(uav_x, uav_y, uav_z, label="İHA 3B Rotası", color="blue", linewidth=2.5)
        ax.plot(th_x, th_y, th_z, label="Tehdit 3B Rotası", color="red", linestyle="--", linewidth=1.8)
        ax.scatter([env.goal_pos[0]], [env.goal_pos[1]], [env.goal_pos[2]], color="green", marker="*", s=250, label="Hedef")

        status = "Başarılı (Hedefe Ulaştı)" if info.get("is_success") else "Başarısız/Vuruldu"
        ax.set_title(f"3B Test Bölümü {ep+1} - Sonuç: {status}", fontsize=13)
        ax.set_xlabel("X (Metre)")
        ax.set_ylabel("Y (Metre)")
        ax.set_zlabel("İrtifa / Z (Metre)")
        ax.legend()
        plt.show()

if __name__ == "__main__":
    run_evaluation()