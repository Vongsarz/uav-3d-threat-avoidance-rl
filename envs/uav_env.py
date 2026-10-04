import gymnasium as gym
from gymnasium import spaces
import numpy as np


class UAVThreatAvoidanceEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        self.dt = 0.1
        self.max_steps = 500
        self.current_step = 0

        # İHA Uçuş Zarfı Sınırları
        self.max_turn_rate = float(np.radians(35.0))  # 35 deg/s
        self.v_min, self.v_max = 18.0, 32.0          # Uçuş hız zarfı (m/s)
        self.danger_radius = 35.0                    # Vurulma yarıçapı (m)
        self.goal_radius = 30.0                      # Hedefe varış yarıçapı (m)

        # Tehdit Kinematik Sınırları (Sonsuz manevra yapamaz, İHA kaçabilsin)
        self.threat_max_turn_rate = float(np.radians(20.0))  # Tehdit dönüş sınırı: 20 deg/s

        # Eylem Uzayı: [0] İvme (-5..+5 m/s^2), [1] Dönüş Hızı (-35..+35 deg/s)
        self.action_space = spaces.Box(
            low=np.array([-5.0, -self.max_turn_rate], dtype=np.float32),
            high=np.array([5.0, self.max_turn_rate], dtype=np.float32),
            dtype=np.float32
        )

        # Gözlem Uzayı
        obs_low = np.array([self.v_min, 0.0, -np.pi, 0.0, -np.pi, 0.0], dtype=np.float32)
        obs_high = np.array([self.v_max, 3000.0, np.pi, 3000.0, np.pi, 100.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        self.uav_pos = None
        self.uav_vel = None
        self.uav_heading = None
        self.threat_pos = None
        self.threat_vel = None
        self.threat_heading = None
        self.goal_pos = None
        self.prev_dist_to_goal = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        # İHA Başlangıcı
        self.uav_pos = np.array([0.0, 0.0], dtype=np.float32)
        self.uav_vel = 22.0
        self.uav_heading = 0.0  # Hedefe dönük başla

        self.goal_pos = np.array([1000.0, 0.0], dtype=np.float32)
        self.prev_dist_to_goal = float(np.linalg.norm(self.goal_pos - self.uav_pos))

        # --- DİNAMİK TEHDİT RASTGELELEŞTİRMESİ ---
        # 1. Başlangıç Konumu: 650-800m arası mesafe, [-250, 250]m dikey sapma
        threat_x = float(np.random.uniform(650.0, 800.0))
        threat_y = float(np.random.uniform(-250.0, 250.0))
        self.threat_pos = np.array([threat_x, threat_y], dtype=np.float32)

        # 2. Tehdit Hızı: 30 - 45 m/s arası rastgele
        self.threat_vel = float(np.random.uniform(30.0, 45.0))

        # 3. Tehdidin İlk Geliş Açısı: İHA'ya doğru dönük başlasın
        vec_to_uav = self.uav_pos - self.threat_pos
        self.threat_heading = float(np.arctan2(vec_to_uav[1], vec_to_uav[0]))

        obs = self._get_obs()
        return obs, {}

    def _get_obs(self):
        vec_to_goal = self.goal_pos - self.uav_pos
        dist_to_goal = float(np.linalg.norm(vec_to_goal))
        angle_to_goal = float(np.arctan2(vec_to_goal[1], vec_to_goal[0]) - self.uav_heading)
        angle_to_goal = (angle_to_goal + np.pi) % (2 * np.pi) - np.pi

        vec_to_threat = self.threat_pos - self.uav_pos
        dist_to_threat = float(np.linalg.norm(vec_to_threat))
        angle_to_threat = float(np.arctan2(vec_to_threat[1], vec_to_threat[0]) - self.uav_heading)
        angle_to_threat = (angle_to_threat + np.pi) % (2 * np.pi) - np.pi

        return np.array([
            self.uav_vel,
            dist_to_goal,
            angle_to_goal,
            dist_to_threat,
            angle_to_threat,
            self.threat_vel
        ], dtype=np.float32)

    def step(self, action):
        self.current_step += 1
        accel, turn_rate = float(action[0]), float(action[1])

        # 1. İHA Hareketi
        self.uav_vel = float(np.clip(self.uav_vel + accel * self.dt, self.v_min, self.v_max))
        self.uav_heading += float(np.clip(turn_rate, -self.max_turn_rate, self.max_turn_rate) * self.dt)
        self.uav_heading = (self.uav_heading + np.pi) % (2 * np.pi) - np.pi

        self.uav_pos[0] += self.uav_vel * np.cos(self.uav_heading) * self.dt
        self.uav_pos[1] += self.uav_vel * np.sin(self.uav_heading) * self.dt

        # 2. Gerçekçi Tehdit Güdümü (Sınırlı dönüş oranıyla İHA'yı takip)
        vec_to_uav = self.uav_pos - self.threat_pos
        desired_threat_heading = float(np.arctan2(vec_to_uav[1], vec_to_uav[0]))
        heading_err = (desired_threat_heading - self.threat_heading + np.pi) % (2 * np.pi) - np.pi
        
        # Tehdit dönüş hız kısıtı
        threat_turn = np.clip(heading_err, -self.threat_max_turn_rate * self.dt, self.threat_max_turn_rate * self.dt)
        self.threat_heading += float(threat_turn)
        self.threat_heading = (self.threat_heading + np.pi) % (2 * np.pi) - np.pi

        self.threat_pos[0] += self.threat_vel * np.cos(self.threat_heading) * self.dt
        self.threat_pos[1] += self.threat_vel * np.sin(self.threat_heading) * self.dt

        # 3. Mesafeler
        dist_to_goal = float(np.linalg.norm(self.goal_pos - self.uav_pos))
        dist_to_threat = float(np.linalg.norm(self.threat_pos - self.uav_pos))

        # 4. Ödül Tasarımı
        progress = self.prev_dist_to_goal - dist_to_goal
        reward = progress * 2.5
        self.prev_dist_to_goal = dist_to_goal

        # Süre cezası
        reward -= 0.1

        terminated = False
        truncated = False

        # Tehdit Kaçınma Bölgesi: Sadece kritik 100m altında ceza ver
        if dist_to_threat < 100.0:
            reward -= (100.0 - dist_to_threat) * 0.15

        # Çarpışma / Vurulma
        if dist_to_threat < self.danger_radius:
            reward -= 400.0
            terminated = True

        # Hedefe Ulaşma
        if dist_to_goal < self.goal_radius:
            reward += 600.0
            terminated = True

        if self.current_step >= self.max_steps:
            truncated = True

        obs = self._get_obs()
        info = {
            "dist_to_goal": dist_to_goal,
            "dist_to_threat": dist_to_threat,
            "is_success": (dist_to_goal < self.goal_radius)
        }

        return obs, reward, terminated, truncated, info