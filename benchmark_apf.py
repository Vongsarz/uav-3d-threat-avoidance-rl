import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from envs.uav_env_3d import UAVThreatAvoidance3DEnv


class DynamicAPFController:
    """
    Gelişmiş Kural Tabanlı Yöntem: Dinamik Yapay Potansiyel Alanlar (D-APF)
    - Konum farkının yanında bağıl hız vektörünü (damping) hesaba katar.
    - Algılama menzili 250 metreye çıkarılmıştır.
    """
    def __init__(self, k_att=1.2, k_rep=850.0, k_damp=3.5, d_trigger=250.0):
        self.k_att = k_att
        self.k_rep = k_rep
        self.k_damp = k_damp
        self.d_trigger = d_trigger

    def compute_action(self, env):
        # 1. Hedefe Çekici Kuvvet
        vec_to_goal = env.goal_pos - env.uav_pos
        dist_to_goal = np.linalg.norm(vec_to_goal)
        f_att = (vec_to_goal / (dist_to_goal + 1e-6)) * self.k_att

        # 2. Tehditten İtici Kuvvet (Konum + Bağıl Hız Sönümlemesi)
        f_rep = np.array([0.0, 0.0, 0.0])
        if not env.threat_evaded:
            vec_to_threat = env.threat_pos - env.uav_pos
            dist_threat = np.linalg.norm(vec_to_threat)

            if dist_threat < self.d_trigger:
                # Konum tabanlı itme
                rep_magnitude = self.k_rep * (1.0 / (dist_threat + 1e-6) - 1.0 / self.d_trigger) / (dist_threat**2 + 1e-6)
                unit_vec = vec_to_threat / (dist_threat + 1e-6)
                f_pos = -unit_vec * rep_magnitude

                # Bağıl hız bileşeni (Tehdidin geliş doğrultusuna dik kaçış teşviki)
                v_threat_vec = np.array([
                    env.threat_vel * np.cos(env.threat_heading),
                    env.threat_vel * np.sin(env.threat_heading),
                    0.0
                ])
                v_uav_vec = np.array([
                    env.uav_vel * np.cos(env.uav_heading),
                    env.uav_vel * np.sin(env.uav_heading),
                    0.0
                ])
                rel_v = v_threat_vec - v_uav_vec
                f_damp = -self.k_damp * (np.dot(rel_v, unit_vec)) * unit_vec

                f_rep = f_pos + f_damp

        # 3. Bileşke Kuvvet ve Eylem Eşleme
        f_total = f_att + f_rep

        desired_yaw = np.arctan2(f_total[1], f_total[0])
        yaw_err = (desired_yaw - env.uav_heading + np.pi) % (2 * np.pi) - np.pi
        turn_rate = np.clip(yaw_err / env.dt, -env.max_turn_rate, env.max_turn_rate)

        # Hız ve Dikey Manevra
        accel = 2.5 if (not env.threat_evaded and np.linalg.norm(env.threat_pos - env.uav_pos) < 180.0) else 0.5
        vz = np.clip(f_total[2] * 2.0, -env.max_climb_rate, env.max_climb_rate)

        return np.array([accel, turn_rate, vz], dtype=np.float32)


def apply_stress_scenario(env, seed_val):
    """
    Stres Testi Koşulları:
    - Tehdit Hızı: 38 - 46 m/s (Daha agresif)
    - Geliş Açısı Sapması: ±45 dereceye kadar geniş lateral sapma
    - Rastgele Rüzgar Bozuntusu
    """
    obs, info = env.reset(seed=seed_val)
    np.random.seed(seed_val)

    # Tehdit parametrelerini zorlaştır
    th_x = float(np.random.uniform(580.0, 680.0))
    th_y = float(np.random.uniform(-140.0, 140.0))  # Geniş açı (±45 dereceye yakın)
    th_z = float(np.random.uniform(70.0, 130.0))
    env.threat_pos = np.array([th_x, th_y, th_z], dtype=np.float32)
    env.threat_vel = float(np.random.uniform(38.0, 46.0))

    vec_to_uav = env.uav_pos - env.threat_pos
    env.threat_heading = float(np.arctan2(vec_to_uav[1], vec_to_uav[0]))

    return env._get_obs(), info


def run_benchmark(num_episodes=100):
    os.makedirs("figures", exist_ok=True)
    env = UAVThreatAvoidance3DEnv()
    rl_model = PPO.load("models/ppo_uav_3d_final")
    apf_controller = DynamicAPFController()

    metrics = {
        "PPO": {"success": 0, "collision": 0, "timeout": 0, "min_distance": []},
        "D-APF": {"success": 0, "collision": 0, "timeout": 0, "min_distance": []}
    }

    print(f"[*] 100 Bölümlük Stres ve Dayanıklılık Testi Başlıyor (PPO vs. D-APF)...")

    # --- PPO TESTİ ---
    for ep in range(num_episodes):
        obs, _ = apply_stress_scenario(env, seed_val=ep)
        done = False
        min_dist = float("inf")

        while not done:
            action, _ = rl_model.predict(obs, deterministic=True)
            
            # Rastgele rüzgar gürültüsü enjeksiyonu
            wind = np.random.uniform(-0.4, 0.4, size=3)
            env.uav_pos += wind * env.dt

            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            dist_th = float(np.linalg.norm(env.threat_pos - env.uav_pos))
            if dist_th < min_dist:
                min_dist = dist_th

        metrics["PPO"]["min_distance"].append(min_dist)
        if info.get("is_success", False):
            metrics["PPO"]["success"] += 1
        elif terminated:
            metrics["PPO"]["collision"] += 1
        else:
            metrics["PPO"]["timeout"] += 1

    # --- D-APF TESTİ ---
    for ep in range(num_episodes):
        apply_stress_scenario(env, seed_val=ep)
        done = False
        min_dist = float("inf")

        while not done:
            action = apf_controller.compute_action(env)
            
            wind = np.random.uniform(-0.4, 0.4, size=3)
            env.uav_pos += wind * env.dt

            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            dist_th = float(np.linalg.norm(env.threat_pos - env.uav_pos))
            if dist_th < min_dist:
                min_dist = dist_th

        metrics["D-APF"]["min_distance"].append(min_dist)
        if info.get("is_success", False):
            metrics["D-APF"]["success"] += 1
        elif terminated:
            metrics["D-APF"]["collision"] += 1
        else:
            metrics["D-APF"]["timeout"] += 1

    # --- SONUÇLARI YAZDIR ---
    print("\n" + "="*55)
    print("      100 BÖLÜM STRES TESTİ MONTE CARLO SONUÇLARI")
    print("="*55)
    print(f"Metrik                      | PPO (Derin RL) | D-APF (Dinamik APF)")
    print("-" * 55)
    print(f"Başarı Oranı (%)            | %{metrics['PPO']['success']:<14} | %{metrics['D-APF']['success']}")
    print(f"Çarpışma / İsabet Oranı (%) | %{metrics['PPO']['collision']:<14} | %{metrics['D-APF']['collision']}")
    print(f"Zaman Aşımı / Sınır Aşımı   | %{metrics['PPO']['timeout']:<14} | %{metrics['D-APF']['timeout']}")
    print(f"Ort. Min Tehdit Mesafesi (m)| {np.mean(metrics['PPO']['min_distance']):.2f} m        | {np.mean(metrics['D-APF']['min_distance']):.2f} m")
    print("="*55)

    # --- GRAFİK GÜNCELLEME ---
    labels = ['Başarı Oranı (%)', 'Çarpışma Oranı (%)']
    ppo_vals = [metrics['PPO']['success'], metrics['PPO']['collision']]
    apf_vals = [metrics['D-APF']['success'], metrics['D-APF']['collision']]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, ppo_vals, width, label='PPO (Öğrenilmiş Politika)', color='#1f77b4')
    ax.bar(x + width/2, apf_vals, width, label='D-APF (Dinamik Kural Tabanlı)', color='#d62728')

    ax.set_ylabel('Yüzde (%)')
    ax.set_title('Zorlu Koşullarda Karşılaştırmalı Performans (100 Monte Carlo Senaryosu)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 110)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    chart_path = os.path.join("figures", "comparison_benchmark.png")
    plt.savefig(chart_path, dpi=300)
    print(f"\n[✓] Gerçekçi karşılaştırma grafiği '{chart_path}' konumuna güncellendi.")


if __name__ == "__main__":
    run_benchmark(num_episodes=100)