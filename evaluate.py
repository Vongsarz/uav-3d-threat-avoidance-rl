import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from envs.uav_env import UAVThreatAvoidanceEnv

def run_evaluation(episodes=3):
    env = UAVThreatAvoidanceEnv()
    model = PPO.load("models/ppo_uav_final")

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        uav_traj_x, uav_traj_y = [], []
        threat_traj_x, threat_traj_y = [], []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            uav_traj_x.append(env.uav_pos[0])
            uav_traj_y.append(env.uav_pos[1])
            threat_traj_x.append(env.threat_pos[0])
            threat_traj_y.append(env.threat_pos[1])

        # Yörünge Grafiğini Çizdir
        plt.figure(figsize=(10, 5))
        plt.plot(uav_traj_x, uav_traj_y, label="İHA Rotası", color="blue", linewidth=2)
        plt.plot(threat_traj_x, threat_traj_y, label="Tehdit Rotası", color="red", linestyle="--")
        plt.scatter([env.goal_pos[0]], [env.goal_pos[1]], color="green", marker="*", s=200, label="Hedef")
        
        status = "Başarılı" if info.get("is_success") else "Başarısız/Vuruldu"
        plt.title(f"Test Bölümü {ep+1} - Sonuç: {status}")
        plt.xlabel("X (metre)")
        plt.ylabel("Y (metre)")
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    run_evaluation()