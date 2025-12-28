# HMAPPO: Hierarchical Multi-Agent PPO for Smart Warehouse Robot Optimization

Mã nguồn Python/PyTorch cho đề tài luận văn Thạc sĩ Kỹ thuật về **Tối ưu hóa Robot Nhà kho Thông minh** sử dụng **Hierarchical Multi-Agent Proximal Policy Optimization (HMAPPO)**.

## Tổng quan

Hệ thống này triển khai kiến trúc học tăng cường hai tầng:

### Level 1: Manager Agent (PPO)
- **Vai trò**: Quyết định chiến lược toàn cục
- **Observation**: Vector nén (không dùng grid toàn kho)
  - `avg_travel_time`: Thời gian di chuyển trung bình
  - `robot_density_mean/std`: Mật độ robot
  - `collision_rate`: Tỷ lệ va chạm
  - `deadlock_ratio`: Tỷ lệ deadlock
  - `throughput`: Năng suất
  - `current_Z`: Độ rộng lối đi (chuẩn hóa)
- **Action Space**: Discrete {Giữ Z, Tăng Z, Giảm Z}
- **Reward**: Episode-level (throughput, travel time, collision penalty)

### Level 2: Worker Agents (MAPPO)
- **Vai trò**: Điều khiển từng robot
- **Observation**: Local grid (7x7) + hướng đến mục tiêu + thông tin robot lân cận
- **Action Space**: {UP, DOWN, LEFT, RIGHT, WAIT}
- **Reward**: Step-level (progress, delivery, collision penalty)
- **Training**: Centralized critic, Decentralized execution

## Cấu trúc dự án

```
warehouse_hmappo/
├── __init__.py              # Package initialization
├── config.py                # Configuration classes
├── main.py                  # Main entry point
│
├── envs/                    # Environment modules
│   ├── __init__.py
│   ├── warehouse_env.py     # Main simulation environment
│   └── warehouse_layout.py  # Warehouse layout generator
│
├── agents/                  # Agent implementations
│   ├── __init__.py
│   ├── manager_agent.py     # Manager PPO agent
│   ├── worker_agent.py      # Worker MAPPO agent
│   └── robot_agent.py       # Individual robot wrapper
│
├── networks/                # Neural network architectures
│   ├── __init__.py
│   └── ppo_networks.py      # Actor-Critic networks
│
├── training/                # Training modules
│   ├── __init__.py
│   └── trainer.py           # HMAPPO trainer with 3 phases
│
└── utils/                   # Utility modules
    ├── __init__.py
    ├── buffer.py            # Rollout buffers
    └── logger.py            # Training logger
```

## Cài đặt

### Yêu cầu
- Python >= 3.8
- PyTorch >= 1.9
- NumPy >= 1.20

### Cài đặt dependencies

```bash
pip install torch numpy
```

## Sử dụng

### Training cơ bản

```bash
# Small-scale test
python -m warehouse_hmappo.main --config small --device cpu

# Medium-scale training
python -m warehouse_hmappo.main --config medium --device cuda

# Custom configuration
python -m warehouse_hmappo.main --N 10 --X 4 --Z 5 --pretrain-episodes 500
```

### Sử dụng trong code

```python
from warehouse_hmappo import WarehouseEnv, HMAPPOTrainer
from warehouse_hmappo.training.trainer import TrainingConfig

# Tạo configuration
config = TrainingConfig(
    N=10,          # Số kệ hàng
    X=4,           # Số robot (= số loại hàng)
    Z_init=5,      # Độ rộng lối đi ban đầu
    max_steps=500, # Số bước tối đa mỗi episode
    pretrain_episodes=500,
    manager_episodes=300,
    finetune_episodes=200,
    device="cpu"
)

# Tạo trainer và chạy training
trainer = HMAPPOTrainer(config)
trainer.train()

# Hoặc chỉ chạy evaluation
metrics = trainer.evaluate(n_episodes=10)
print(f"Throughput: {metrics['throughput']}")
```

## Chiến lược huấn luyện

### Phase 1: Pretrain Workers
```python
# Cố định Z ở giá trị lớn để giảm tắc nghẽn
# Train MAPPO cho robots học di chuyển và tránh va chạm
trainer.phase1_pretrain_workers()
```

### Phase 2: Train Manager
```python
# Đóng băng Workers, train Manager PPO
# Manager học tối ưu Z và chính sách luồng
trainer.phase2_train_manager()
```

### Phase 3: Joint Fine-tuning
```python
# Mở lại Workers với learning rate nhỏ
# Tinh chỉnh hành vi chung
trainer.phase3_joint_finetune()
```

## Chi tiết kỹ thuật

### Warehouse Environment

```python
class WarehouseEnv:
    """
    Môi trường mô phỏng nhà kho.
    
    Đầu vào:
        N: Số kệ hàng
        X: Số loại hàng (= số robot = số vị trí phân loại)
        Z: Độ rộng lối đi (biến cần tối ưu, Z >= 2)
    
    Cấu trúc kho:
        - N kệ hàng, mỗi kệ 20 slot, mỗi slot 50 đơn hàng
        - Khu vực phân loại cách kệ 10 dòng
        - X vị trí phân loại tương ứng X loại hàng
    
    Robot:
        - X robots, mỗi robot lấy hàng ở kệ đưa về vị trí phân loại
        - Không va chạm, không deadlock
    """
```

### Manager Observation (Compressed State Vector)

| Feature | Description | Normalization |
|---------|-------------|---------------|
| `avg_travel_time` | Thời gian di chuyển trung bình | / max_steps |
| `robot_density_mean` | Mật độ robot trung bình | / num_robots |
| `robot_density_std` | Độ lệch chuẩn mật độ | / num_robots |
| `collision_rate` | Tỷ lệ va chạm/step | [0, 1] |
| `deadlock_ratio` | Tỷ lệ deadlock/step | [0, 1] |
| `throughput` | Deliveries/step | * 100 |
| `current_Z` | Độ rộng lối đi hiện tại | / max_Z |

### Worker Observation (Local)

| Component | Shape | Description |
|-----------|-------|-------------|
| Local grid | 7x7 | Grid xung quanh robot |
| Target direction | 2 | (dx, dy) normalized |
| Target distance | 1 | Normalized distance |
| Current Z | 1 | Manager signal |
| Robot state | 1 | Encoded state |
| Nearby robots | 8 | 4 nearest neighbors |

### Reward Shaping

**Manager (Episode-level):**
```
R_manager = 100 * throughput 
          - 0.1 * avg_travel_time
          - 10 * collision_rate
          - 50 * deadlock_ratio
```

**Worker (Step-level):**
```
R_worker = 0.1 * progress_to_target
         + 10 * delivery_success
         - 0.01 (step penalty)
         - 1.0 * collision
         - 5.0 * deadlock
```

## Pseudo-code Training Loop

```
Algorithm: HMAPPO Training

Phase 1: PRETRAIN WORKERS
    Z ← Z_large
    for episode = 1 to pretrain_episodes:
        Reset environment with fixed Z
        for t = 1 to max_steps:
            for each robot i:
                o_i ← LocalObservation(robot_i)
            a ← Worker.get_actions(observations)
            Execute(a), observe rewards
            Store experiences
        Update Worker policy (MAPPO)

Phase 2: TRAIN MANAGER  
    Freeze Worker policy
    for episode = 1 to manager_episodes:
        s_M ← CompressedState()
        a_M ← Manager.get_action(s_M)
        Apply Manager action (modify Z)
        Run episode with frozen Workers
        R_M ← EpisodeReward()
        Update Manager policy (PPO)

Phase 3: JOINT FINE-TUNING
    Unfreeze Worker, reduce learning rates
    for episode = 1 to finetune_episodes:
        Train both Manager and Workers
        Manager adjusts Z periodically
        Workers navigate within constraints
```

## Kết quả mong đợi

Sau khi training hoàn tất, hệ thống sẽ:
1. Tự động tìm giá trị Z tối ưu cho cấu hình kho
2. Robots di chuyển hiệu quả với ít va chạm
3. Throughput cao, thời gian di chuyển ngắn
4. Không có deadlock

## Tài liệu tham khảo

1. Schulman, J., et al. "Proximal Policy Optimization Algorithms." (PPO)
2. Yu, C., et al. "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games." (MAPPO)
3. Nachum, O., et al. "Data-Efficient Hierarchical Reinforcement Learning."

## License

MIT License

## Liên hệ

Dành cho mục đích học thuật và nghiên cứu.
