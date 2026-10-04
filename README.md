# Otonom İHA'larda Derin Pekiştirmeli Öğrenme ile 3B Dinamik Tehdit Kaçınma ve Uçuş Zarfı Koruması
### *Deep Reinforcement Learning-based 3D Dynamic Threat Avoidance & Flight Envelope Protection for Autonomous UAVs*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-v0.29%2B-green.svg)](https://gymnasium.farama.org/)
[![Stable-Baselines3](https://img.shields.io/badge/Stable--Baselines3-PPO-orange.svg)](https://stable-baselines3.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Bu çalışma, **SAYZEK (Savunma Sanayii Yapay Zekâ Yetenek Kümelenmesi)** programı kapsamında; otonom İnsansız Hava Araçlarının (İHA) operasyonel görev sırasında karşılaştığı dinamik tehditlerden (yüksek hızlı güdümlü füzeler / karşı hava araçları) aerodinamik sınırlarını (uçuş zarfını) aşmadan otonom manevra ile sıyrılmasını ve görev noktasına emniyetle ulaşmasını amaçlayan **Derin Pekiştirmeli Öğrenme (Deep Reinforcement Learning - DRL)** tabanlı bir karar/rehberlik katmanıdır.

---

## 📌 İçindekiler
- [Proje Mimarisi ve Karar Katmanı](#-proje-mimarisi-ve-karar-katmanı)
- [Matematiksel Modelleme (Markov Karar Süreci - MDP)](#-matematiksel-modelleme-markov-karar-süreci---mdp)
  - [Durum Uzayı (Observation Space)](#1-durum-uzayı-observation-space)
  - [Eylem Uzayı (Action Space)](#2-eylem-uzayı-action-space)
  - [Ödül Fonksiyonu (Reward Shaping)](#3-ödül-fonksiyonu-reward-shaping)
- [Uçuş Zarfı Koruması (Flight Envelope Protection)](#-uçuş-zarfı-koruması-flight-envelope-protection)
- [Dinamik Tehdit Karakteristiği ve Iskalama Geometrisi](#-dinamik-tehdit-karakteristiği-ve-ıskalama-geometrisi)
- [Dizin Yapısı](#-dizin-yapısı)
- [Kurulum ve Çalıştırma](#-kurulum-ve-çalıştırma)
- [Deneysel Sonuçlar ve Yörünge Analizi](#-deneysel-sonuçlar-ve-yörünge-analizi)
- [Gelecek Çalışmalar ve Yol Haritası (Sim-to-Sim)](#-gelecek-çalışmalar-ve-yol-haritası-sim-to-sim)

---

## 🛰️ Proje Mimarisi ve Karar Katmanı

Geliştirilen mimari, İHA'nın alt seviye aerodinamik/motor kontrolü ile üst seviye taktik kaçınma kararlarını birbirinden ayıran **Hiyerarşik Kontrol (Hierarchical Guidance-Control)** prensibine dayanır:

```
┌────────────────────────────────────────────────────────┐
│             RL Karar Katmanı (Python / PPO)            │
│   Girdi: [V, Hedef Mesafe/Açı/İrtifa, Tehdit Bağıl Vektör]  │
│   Çıktı: [İleri Hız Setpoint, Roll/Yaw Oranı, Tırmanma Oranı] │
└──────────────────────────┬─────────────────────────────┘
                           │ (Kinematik Hız / Açı Setpoint'leri)
                           ▼
┌────────────────────────────────────────────────────────┐
│        PX4 Otopilot / Uçuş Zarfı Filtresi              │
│   - Stall önleme ve g-limiti doyum filtresi             │
│   - PID Tutum ve Hız Döngüleri                          │
└──────────────────────────┬─────────────────────────────┘
                           │ (Kontrol Yüzeyleri / Motor İtki)
                           ▼
┌────────────────────────────────────────────────────────┐
│          Fizik Simülatörü (Gazebo / AirSim)            │
│   - 6-DOF Aerodinamik Model, Rüzgar ve Türbülans       │
└────────────────────────────────────────────────────────┘
```

---

## 📐 Matematiksel Modelleme (Markov Karar Süreci - MDP)

Sistem, bir ayrık zamanlı Markov Karar Süreci $(\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$ olarak modellenmiştir.

### 1. Durum Uzayı (Observation Space)
Derin sinir ağının gradyan kararsızlıklarını önlemek amacıyla gözlem vektörü $s_t \in [-1, 1]^8$ aralığında normalize edilmiştir:

$$s_t = \begin{bmatrix} 
\tilde{V}_{\text{uav}} \\
\tilde{Z}_{\text{uav}} \\
\tilde{d}_{\text{goal}} \\
\psi_{\text{goal}} / \pi \\
\theta_{\text{goal}} / (\pi/2) \\
\tilde{d}_{\text{threat}} \\
\psi_{\text{threat}} / \pi \\
\theta_{\text{threat}} / (\pi/2)
\end{bmatrix}$$

*   $\tilde{V}_{\text{uav}}$: Normalize İHA sürati
*   $\tilde{Z}_{\text{uav}}$: Normalize irtifa
*   $\tilde{d}_{\text{goal}}, \tilde{d}_{\text{threat}}$: Hedefe ve tehdide olan normalize mesafeler
*   $\psi, \theta$: Görüş hattı (Line-of-Sight - LOS) yatay yaw sapması ve düşey pitch eğimi

### 2. Eylem Uzayı (Action Space)
Sürekli (continuous) eylem uzayı $a_t \in \mathcal{A} \subset \mathbb{R}^3$:

$$a_t = \begin{bmatrix} a_x \\ \dot{\psi} \\ v_z \end{bmatrix}, \quad \begin{cases}
a_x \in [-3.0, 3.0] \text{ m/s}^2 & \text{(İleri ivmelenme/yavaşlama)} \\
\dot{\psi} \in [-30^\circ/\text{s}, +30^\circ/\text{s}] & \text{(Dönüş açısı oranı)} \\
v_z \in [-6.0, 6.0] \text{ m/s} & \text{(Tırmanma/Dalma dikey sürati)}
\end{cases}$$

### 3. Ödül Fonksiyonu (Reward Shaping)
Ajanın hedefe kilitlenmesi ve tehditten enerji kaybetmeden kaçınması için potansiyel tabanlı ödül şekillendirme uygulanmıştır:

$$R_t = R_{\text{progress}} + R_{\text{corridor}} + R_{\text{safety}} + R_{\text{terminal}}$$

*   **İlerleme Teşviki (Potential-Based Progress):**
    $$R_{\text{progress}} = 4.0 \times (d_{t-1}^{\text{goal}} - d_t^{\text{goal}})$$
*   **Uçuş Koridoru Kısıtı:**
    $$R_{\text{corridor}} = \begin{cases} -5.0, & \text{eğer } |Y_{\text{uav}}| > Y_{\text{limit}} \\ 0, & \text{aksi halde} \end{cases}$$
*   **Erken Uyarı Emniyet Alanı Cezası ($d < 80\text{ m}$):**
    $$R_{\text{safety}} = -0.2 \times (80.0 - d_t^{\text{threat}})$$
*   **Uç Durum Ödülleri:**
    $$R_{\text{terminal}} = \begin{cases} 
    +1000.0, & \text{Hedefe Başarılı Varış } (d \le R_{\text{goal}}) \\ 
    -400.0, & \text{Çarpışma / İsabet Alma } (d \le R_{\text{danger}}) 
    \end{cases}$$

---

## 🛡️ Uçuş Zarfı Koruması (Flight Envelope Protection)

Havacılık emniyeti gereği RL ajanının ürettiği komutlar aerodinamik sınırları aşamaz. Ortam içerisinde aşağıdaki kinematik ve yapısal kısıtlar her adımda garanti altına alınmıştır:

1.  **Stall ve Maksimum Hız Sınırları:** $V \in [20.0, 30.0] \text{ m/s}$ (Stall hızının altına düşmesi önlenir).
2.  **Maksimum Yapısal Dönüş Oranı:** $|\dot{\psi}| \le 30.0^\circ/\text{s}$ (G-yükü aşımını sınırlar).
3.  **Dikey Sürat Limitleri:** $|v_z| \le 6.0 \text{ m/s}$ (Hücum açısı sınırlarının aşılması engellenir).
4.  **Operasyonel İrtifa Zarfı:** $Z \in [30.0, 250.0] \text{ m}$ (Araziye çarpma ve tavan irtifası kontrolü).

---

## 🎯 Dinamik Tehdit Karakteristiği ve Iskalama Geometrisi

*   **Sınırlı Manevra Kabiliyeti:** Tehdit (füze), İHA'dan daha hızlıdır ($V_{\text{threat}} \in [32, 36]\text{ m/s}$) ancak manevra açısı sınırlandırılmıştır ($|\dot{\psi}_{\text{threat}}| \le 15^\circ/\text{s}$).
*   **Iskalama (Overshoot) Prensibi:** Tehdit $X_{\text{threat}} \le X_{\text{uav}}$ koşulunu sağladığında İHA'yı ıskalamış kabul edilir; takip güdümünü kaybeder ve rota boyunca düz uçuşa geçer.

---

## 📂 Dizin Yapısı

```text
uav_threat_avoidance/
├── envs/
│   ├── __init__.py
│   ├── uav_env.py             # 2B Temel Ortam
│   └── uav_env_3d.py          # 3B Uçuş Zarfı Korumalı Dinamik Tehdit Ortamı
├── models/                    # Eğitilmiş PPO Ağırlıkları (.zip)
├── logs/                      # TensorBoard Logları
├── figures/                   # Başarılı Yörünge Grafikleri
├── train_3d.py                # PPO 3B Eğitim Scripti
├── evaluate_3d.py             # 3B İnteraktif Yörünge Test Scripti
├── requirements.txt           # Bağımlılıklar
└── README.md                  # Proje Dokümantasyonu
```

---

## ⚡ Kurulum ve Çalıştırma

### 1. Depoyu Klonlayın ve Sanal Ortamı Kurun
```bash
git clone https://github.com/KULLANICI_ADINIZ/uav-threat-avoidance-rl.git
cd uav-threat-avoidance-rl

python -m venv venv
# Linux / macOS:
source venv/bin/activate
# Windows:
.\venv\Scripts\Activate.ps1
```

### 2. Bağımlılıkları Yükleyin
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. PPO Modelini Eğitin
```bash
python train_3d.py
```

### 4. Eğitilmiş Modeli 3B Olarak Test Edin
```bash
python evaluate_3d.py
```

---

## 📊 Deneysel Sonuçlar ve Yörünge Analizi

Eğitilen **PPO (Proximal Policy Optimization)** modeli, 3B uzayda yaklaşan dinamik tehdidi algıladığında şu taktiksel davranışları sergilemiştir:

*   **Bileşik Dalış-Açılma Manevrası:** Karşıdan ve aynı irtifadan gelen füzeyi atlatmak için İHA yatayda hafif kavis çizerken dikeyde irtifa kaybederek ($Z \approx 80\text{ m}$) tehdidin görüş açısını bozmuştur.
*   **Enerji Korunumu ve Hedefe Yeniden Kilitlenme:** Tehdit ıskalayıp arkada kaldığı anda ($X \approx 400\text{ m}$) İHA gereksiz manevraları sonlandırarak rotasını doğrudan görev noktasına ($X=1000\text{ m}, Y=0\text{ m}, Z=100\text{ m}$) çevirmiş ve görevi başarıyla tamamlamıştır.

*(Not: `figures/` klasörüne ürettiğiniz 3B başarı grafiklerini ekleyerek bu bölüme görsel referans verebilirsiniz).*

---

## 🚀 Gelecek Çalışmalar ve Yol Haritası (Sim-to-Sim)

Bu çalışma, SAYZEK isterleri doğrultusunda yüksek sadakatli simülasyon ortamına şu adımlarla taşınacaktır:

1.  **ROS 2 & PX4 Offboard Entegrasyonu:** Eğitilen sinir ağının `/fmu/in/trajectory_setpoint` üzerinden Gazebo SITL ortamındaki PX4 otopilotuna hız/yönelim komutu basması.
2.  **6-DOF Aerodinamik Doğrulama:** Rüzgar ve türbülans modelleri altında uçuş zarfı filtrelerinin donanım-çevrimli (HIL) testlerde doğrulanması.
3.  **Kural Tabanlı Yöntemlerle Karşılaştırma:** RL politikasının, Yapay Potansiyel Alanlar (APF) ve Geometrik Kural Tabanlı kaçınma algoritmalarıyla başarı oranı ve enerji tüketimi yönünden kıyaslanması.

---

## 📜 Lisans

Bu proje [MIT Lisansı](LICENSE) kapsamında lisanslanmıştır.