# Huong dan cai moi truong voi uv

File `pyproject.toml` trong repo da khai bao cac thu vien can thiet de train/test PPO cho Super Mario Bros. Nguoi dung chi can cai `uv`, tao virtual environment, sau do chay `uv sync`.

## 1. Cai uv

Neu chua co `uv`, cai bang PowerShell:

```powershell
winget install astral-sh.uv
```

Dong terminal hien tai, mo lai PowerShell moi, kiem tra:

```powershell
uv --version
```

## 2. Tao moi truong Python

Di vao thu muc repo:

```powershell
cd "D:\2025-2026\Reinforcement Learning\Super-mario-bros-PPO-pytorch-master"
```

Dat cache cua `uv` trong repo de tranh loi quyen truy cap cache mac dinh:

```powershell
$env:UV_CACHE_DIR = "$PWD\.uv-cache"
```

Tao virtual environment Python 3.10:

```powershell
uv venv --python 3.10
```

Cai cac thu vien tu `pyproject.toml` / `uv.lock`:

```powershell
uv sync
```

Sau khi cai xong, co the chay lenh bang:

```powershell
uv run python --version
```

## 3. Kiem tra thu vien

```powershell
uv run python -c "import torch, gym_super_mario_bros, cv2; print('env ok')"
```

Neu in ra `env ok` la moi truong da san sang.

## 4. Train thu

Lenh train thu ngan:

```powershell
uv run python train.py --world 1 --stage 1 --max_episodes 10 --save_interval 5
```

Checkpoint mac dinh se luu vao:

```text
trained_models\ppo_super_mario_bros_1_1
```

Neu may yeu, giam so process:

```powershell
uv run python train.py --world 1 --stage 1 --max_episodes 10 --save_interval 5 --num_processes 4
```

## 5. Train World 1

```powershell
uv run python train.py --world 1 --stage 1 --lr 1e-4 --max_episodes 1000 --save_interval 50
uv run python train.py --world 1 --stage 2 --lr 1e-4 --max_episodes 1000 --save_interval 50
uv run python train.py --world 1 --stage 3 --lr 7e-5 --max_episodes 1000 --save_interval 50
uv run python train.py --world 1 --stage 4 --lr 1e-4 --max_episodes 1000 --save_interval 50
```

## 6. Chay demo/test

```powershell
uv run python test.py --world 1 --stage 1 --saved_path trained_models
```

Neu Mario qua man, terminal se in:

```text
World 1 stage 1 completed
```

## 7. Xuat video tren Windows

De ghi file `output\video_1_1.mp4`, may can co `ffmpeg` trong PATH.

Cai `ffmpeg`:

```powershell
winget install Gyan.FFmpeg
```

Dong terminal, mo lai PowerShell moi, kiem tra:

```powershell
ffmpeg -version
```

Chay lai demo:

```powershell
uv run python test.py --world 1 --stage 1 --saved_path trained_models
```

Video se nam trong:

```text
output\video_1_1.mp4
```

## 8. Cac warning co the bo qua

Khi chay, co the thay cac warning ve Gym:

```text
Gym has been unmaintained since 2022...
old step API...
```

Day la do repo dung `gym-super-mario-bros` va Gym API cu. Neu chuong trinh van in `Episode: ... Total loss: ...` hoac tao video thi co the bo qua cac warning nay.
