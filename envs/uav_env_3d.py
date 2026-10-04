import gymnasium as gym
from gymnasium import spaces
import numpy as np


class UAVThreatAvoidance3DEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super().__init__()
        self.dt = 0.1
        self.max_steps = 450
        self.current_step = 0

        # Uçuş Zarfı ve Dinamik Sınırlar
        self.v_min, self.v_max = 20.0, 30.0
        self.max_turn_rate = float(np.radians(30.0))
        self.max_climb_rate = 6.0
        self.min_alt, self.max_alt = 30.0, 250.0
        self.y_corridor_limit = 250.0  # Haritadan sonsuza kaçmayı engelleyen koridor limiti

        self.danger_radius = 30.0      # Vurulma yarıçapı
        self.goal_radius = 40.0        # Hedefe varış yarıçapı

        # Tehdit Kinematiği (İHA'dan daha az manevra kabiliyetine sahip)
        self.threat_max_turn = float(np.radians(15.0))
        self.threat_max_climb = 3.0

        # Eylem Uzayı: [0] İvme (-3..+3), [1] Yaw Hızı (-30°..+30°), [2] Dikey Hız (-6..+6 m/s)
        self.action_space = spaces.Box(
            low=np.array([-3.0, -self.max_turn_rate, -self.max_climb_rate], dtype=np.float32),
            high=np.array([3.0, self.max_turn_rate, self.max_climb_rate], dtype=np.float32),
            dtype=np.float32
        )

        # Gözlem Uzayı: TAMAMEN [-1, 1] veya [0, 1] ARASINDA NORMALİZE EDİLMİŞTİR
        # [0]: Hız oranı, [1]: Normalize İrtifa, [2]: Hedef Mesafe/1000, [3]: Hedef Yaw/pi, [4]: Hedef Pitch/(pi/2)
        # [5]: Tehdit Mesafe/1000, [6]: Tehdit Yaw/pi, [7]: Tehdit Pitch/(pi/2)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)

        self.uav_pos = None
        self.uav_vel = None
        self.uav_heading = None
        self.threat_pos = None
        self.threat_vel = None
        self.threat_heading = None
        self.goal_pos = None
        self.prev_dist_to_goal = None
        self.threat_evaded = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.threat_evaded = False

        # İHA Başlangıcı: Orijin, 100m irtifa, Doğuya dönük
        self.uav_pos = np.array([0.0, 0.0, 100.0], dtype=np.float32)
        self.uav_vel = 24.0
        self.uav_heading = 0.0

        # Hedef: 1000m ileride 100m irtifada
        self.goal_pos = np.array([1000.0, 0.0, 100.0], dtype=np.float32)
        self.prev_dist_to_goal = float(np.linalg.norm(self.goal_pos - self.uav_pos))

        # Tehdit: 600m - 700m mesafeden, hafif açı farkıyla yaklaşır
        th_x = float(np.random.uniform(620.0, 700.0))
        th_y = float(np.random.uniform(-50.0, 50.0))
        th_z = float(np.random.uniform(90.0, 110.0))
        self.threat_pos = np.array([th_x, th_y, th_z], dtype=np.float32)
        self.threat_vel = float(np.random.uniform(32.0, 36.0))

        vec_to_uav = self.uav_pos - self.threat_pos
        self.threat_heading = float(np.arctan2(vec_to_uav[1], vec_to_uav[0]))

        return self._get_obs(), {}

    def _get_obs(self):
        # Hedef Bağıl Koordinatlar
        vec_goal = self.goal_pos - self.uav_pos
        dist_goal = float(np.linalg.norm(vec_goal))
        yaw_goal = (np.arctan2(vec_goal[1], vec_goal[0]) - self.uav_heading + np.pi) % (2 * np.pi) - np.pi
        pitch_goal = float(np.arctan2(vec_goal[2] - self.uav_pos[2], np.linalg.norm(vec_goal[:2])))

        # Tehdit Bağıl Koordinatlar
        vec_threat = self.threat_pos - self.uav_pos
        dist_threat = float(np.linalg.norm(vec_threat))
        yaw_threat = (np.arctan2(vec_threat[1], vec_threat[0]) - self.uav_heading + np.pi) % (2 * np.pi) - np.pi
        pitch_threat = float(np.arctan2(vec_threat[2] - self.uav_pos[2], np.linalg.norm(vec_threat[:2])))

        # Sinir Ağı İçin Tamamen Normalize Gözlem [-1, 1]
        obs = np.array([
            (self.uav_vel - self.v_min) / (self.v_max - self.v_min) * 2.0 - 1.0,
            (self.uav_pos[2] - 100.0) / 100.0,
            np.clip(dist_goal / 1000.0, 0.0, 1.0) * 2.0 - 1.0,
            yaw_goal / np.pi,
            pitch_goal / (np.pi / 2.0),
            np.clip(dist_threat / 1000.0, 0.0, 1.0) * 2.0 - 1.0,
            yaw_threat / np.pi,
            pitch_threat / (np.pi / 2.0)
        ], dtype=np.float32)

        return np.clip(obs, -1.0, 1.0)

    def step(self, action):
        self.current_step += 1
        accel, turn_rate, vz = float(action[0]), float(action[1]), float(action[2])

        # 1. İHA Hareketi
        self.uav_vel = float(np.clip(self.uav_vel + accel * self.dt, self.v_min, self.v_max))
        self.uav_heading += float(np.clip(turn_rate, -self.max_turn_rate, self.max_turn_rate) * self.dt)
        self.uav_heading = (self.uav_heading + np.pi) % (2 * np.pi) - np.pi

        self.uav_pos[0] += self.uav_vel * np.cos(self.uav_heading) * self.dt
        self.uav_pos[1] += self.uav_vel * np.sin(self.uav_heading) * self.dt
        self.uav_pos[2] = float(np.clip(self.uav_pos[2] + vz * self.dt, self.min_alt, self.max_alt))

        # 2. Tehdit Dinamiği
        # Tehdit İHA'nın gerisinde kaldıysa ıskalamıştır (overshoot), takip biter
        if self.threat_pos[0] <= self.uav_pos[0]:
            self.threat_evaded = True

        if not self.threat_evaded:
            vec_to_uav = self.uav_pos - self.threat_pos
            desired_yaw = float(np.arctan2(vec_to_uav[1], vec_to_uav[0]))
            err_yaw = (desired_yaw - self.threat_heading + np.pi) % (2 * np.pi) - np.pi
            turn_t = np.clip(err_yaw, -self.threat_max_turn * self.dt, self.threat_max_turn * self.dt)
            self.threat_heading = (self.threat_heading + turn_t + np.pi) % (2 * np.pi) - np.pi

            dz = np.clip(self.uav_pos[2] - self.threat_pos[2], -self.threat_max_climb * self.dt, self.threat_max_climb * self.dt)
            self.threat_pos[2] += float(dz)

        self.threat_pos[0] += self.threat_vel * np.cos(self.threat_heading) * self.dt
        self.threat_pos[1] += self.threat_vel * np.sin(self.threat_heading) * self.dt

        # 3. Mesafeler ve Ödül
        dist_goal = float(np.linalg.norm(self.goal_pos - self.uav_pos))
        dist_threat = float(np.linalg.norm(self.threat_pos - self.uav_pos))

        progress = self.prev_dist_to_goal - dist_goal
        reward = progress * 4.0
        self.prev_dist_to_goal = dist_goal

        # Uçuş koridoru cezası (Y ekseninde çok açılmayı engelle)
        if abs(self.uav_pos[1]) > self.y_corridor_limit:
            reward -= 5.0

        terminated = False
        truncated = False

        # Tehdit atlatıldıysa ekstra hedefe odaklanma bonusu
        if self.threat_evaded:
            reward += 1.5

        # Tehlike yaklaşma cezası
        if not self.threat_evaded and dist_threat < 80.0:
            reward -= (80.0 - dist_threat) * 0.2

        # Çarpışma durumu
        if not self.threat_evaded and dist_threat < self.danger_radius:
            reward -= 400.0
            terminated = True

        # Hedefe Ulaşma (Tam Başarı)
        if dist_goal < self.goal_radius:
            reward += 1000.0
            terminated = True

        if self.current_step >= self.max_steps:
            truncated = True

        return self._get_obs(), reward, terminated, truncated, {"is_success": (dist_goal < self.goal_radius)}