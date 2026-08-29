<h1 align="center">
ViTaMIn-B:
A Reliable and Efficient Visuo-Tactile BiManual Manipulation Interface
</h1>

<p align="center">
    <a href="https://chuanyune.github.io/">Chuanyu Li</a><sup>1*</sup>,
    Chaoyi Liu<sup>1*</sup>,
    Daotan Wang<sup>1</sup>,
    Shuyu Zhang<sup>4</sup>,
    <br>
    Lusong Li<sup>3</sup>,
    Zecui Zeng<sup>3</sup>,
    <a href="https://fangchenliu.github.io/">Fangchen Liu</a><sup>2</sup>,
    Jing Xu<sup>1†</sup>,
    <a href="https://callmeray.github.io/homepage/">Rui Chen</a><sup>1†</sup>
    <br>
    <sup>1</sup>Tsinghua University &nbsp;&nbsp;
    <sup>2</sup>University of California, Berkeley
    <br>
    <sup>3</sup>JD Explore Academy &nbsp;&nbsp;
    <sup>4</sup>The Hong Kong Polytechnic University
    <br>
    <sup>*</sup>Equal contribution &nbsp;&nbsp;
    <sup>†</sup>Equal advising
</p>

<div align="center">
<a href='https://arxiv.org/abs/2511.05858'><img alt='arXiv' src='https://img.shields.io/badge/arXiv-2503.02881-red.svg'></a>     
<a href='https://chuanyune.github.io/ViTaMIn-B_page/'><img alt='project website' src='https://img.shields.io/website-up-down-green-red/http/cv.lbesson.qc.to.svg'></a>     
<a href='https://huggingface.co/datasets/chuanyune/ViTaMIn-B_data_and_ckpt/tree/main'><img alt='data' src='https://img.shields.io/badge/data-FFD21E?logo=huggingface&logoColor=000'></a>  
<a href='https://huggingface.co/datasets/chuanyune/ViTaMIn-B_data_and_ckpt/tree/main'><img alt='checkpoints' src='https://img.shields.io/badge/checkpoints-FFD21E?logo=huggingface&logoColor=000'></a>    
</div>

---

## 📑 Table of Contents

- [🔧 Hardware Setup](#-hardware-setup)
- [📦 Installation](#-installation)
- [🎥 Data Collection](#-data-collection)
  - [⚙️ Configuration](#️-configuration)
  - [📹 Step 1: Start Data Recorder](#-step-1-start-data-recorder)
  - [🔌 Step 2: Setup ADB Port Forwarding](#-step-2-setup-adb-port-forwarding)
  - [🎯 Step 3: Start Pose Tracking](#-step-3-start-pose-tracking)
  - [🔄 Step 4: Process Collected Data](#-step-4-process-collected-data)
  - [📊 Zarr Data Structure](#-zarr-data-structure)
- [🚀 Training Policy](#-training-policy)
- [🤖 Real-World Deployment](#-real-world-deployment)
- [🙏 Acknowledgement](#-acknowledgement)
- [🔗 Citation](#-citation)
- [📧 Contact](#-contact)

---

cd umi_xMate
./run.sh
docker exec -it ros1_umi bash
cd root/catkin_ws/
source devel/setup.sh 
source /opt/ros/noetic/setup.bash
source /root/catkin_ws/devel/setup.bash

docker rm -f ros1_umi


ls -lh /dev/shm/
rm -rf /dev/shm/*


terminal1: roscore   
terminal2: rosrun impedance_control xMate3_controller -m CP -n A -i 192.168.0.160 -p 1337
if bimanual:
terminal3: rosrun impedance_control xMate3_controller -m CP -n B -i 192.168.2.160 -p 1339

(gripper old version:
sudo chmod 777 /dev/ttyUSB0
sudo chmod 777 /dev/ttyUSB1
)

data:
scp -i ~/.ssh/a800 -P 22 /home/rvsa/codehub/VB-vla/data/_0118/_0118.zarr.zip lcy@166.111.192.71:/hdd/lcy/code/ViTaMIn-B/train_data

ckpt:
scp -i ~/.ssh/a800 -P 22 lcy@166.111.192.71:/hdd/lcy/code/ViTaMIn-B/data/outputs/2026.01.19/19.55.06_vision_tactile_img_umi/checkpoints/epoch=0100-train_loss=0.014.ckpt /home/rvsa/codehub/VB-vla/ckpt/0121

## 🔧 Hardware Setup

We provide multiple hardware configuration options for data collection:

### Visual-Only Data Collection
- **UMI Gripper**: Use the standard gripper from the UMI system (3D-printable gripper models can be downloaded from the [UMI](https://real-stanford.github.io/universal_manipulation_interface/))

### Visual + Tactile Data Collection
We offer three tactile sensor options (grippers require assembly following our provided instructions and molds):
1. **AllTact Gripper**: Vision-based tactile sensor with custom gripper design
2. **DuoTact Gripper**: Dual tactile sensor setup with custom gripper design
3. **GelSight Sensor**: High-resolution tactile sensing solution

For detailed hardware components, assembly instructions, and mold files, please visit our [project page](https://chuanyune.github.io/ViTaMIn-B_page/).

Once you've assembled the data collection device, connect the cameras and foot pedal to your computer.

## 📦 Installation

> **System Requirements:** Ubuntu 20.04 or 22.04

### Clone Repository

```bash
git clone git@github.com:chuanyune/ViTaMIn-B.git
cd ViTaMIn-B
```

### Setup Environment

Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate vitaminb
```

### Install Tracking App

Since we use Meta Quest for pose tracking, you need to install the tracking app [VB-quest](assets/VB-quest.apk) on your Quest headset via [SideQuest](https://sidequestvr.com/).

## 🎥 Data Collection

### ⚙️ Configuration

In the [config file](config/task_config.yaml), modify the `name` parameter to give your task a unique identifier. 

> **⚠️ Important:** Do not change the `name` parameter once you've started collecting data, as all subsequent pipeline steps depend on it.

### 📹 Step 1: Start Data Recorder

Launch the data recorder:

```bash
python ./Data_collection/vitamin_b_data_collection_pipeline/00_get_data.py --cfg ./Data_collection/config/VB_task_config.yaml
```

### 🔌 Step 2: Setup ADB Port Forwarding

1. Put on your Quest headset and launch the installed tracking app
2. In the app, select **Y: TCP Mode**
3. Open a new terminal on your computer and run:

```bash
adb forward tcp:7777 tcp:7777
adb forward tcp:50010 tcp:50010
```

# Quest KEEP OPEN
adb shell
am broadcast -a com.oculus.vrpowermanager.prox_close

> **Troubleshooting:** If ADB forwarding fails, you may need to disable the firewall:
> ```bash
> sudo ufw disable
> ```

If no error is returned, port forwarding is successful.

### 🎯 Step 3: Start Pose Tracking

In the same terminal, launch the pose tracking script:

```bash
python ./Data_collection/vitamin_b_data_collection_pipeline/00_get_data.py --cfg ./Data_collection/config/VB_task_config.yaml
```

You can now start collecting data using the foot pedal.

### 🔄 Step 4: Process Collected Data

After collecting all episodes, run the pipeline to generate training data:

```bash
python run_data_collection_pipeline.py
```

> **Note:** This script sequentially executes all programs in the `vitamin_b_data_collection_pipeline` directory. You can comment out any steps you don't need.


### 📊 Zarr Data Structure

After running the data collection pipeline with `08_generate_replay_buffer.py`, your data will be stored in Zarr format with the following structure:

#### Bimanual Setup Data Format

**Robot State (Left Hand - robot0):**
```
├── robot0_eef_pos (N, 3) float32                  # End-effector position
├── robot0_eef_rot_axis_angle (N, 3) float32      # End-effector rotation
├── robot0_gripper_width (N, 1) float32           # Gripper width
├── robot0_demo_start_pose (N, 7) float32         # Demo start pose
├── robot0_demo_end_pose (N, 7) float32           # Demo end pose
```

**Robot State (Right Hand - robot1):**
```
├── robot1_eef_pos (N, 3) float32                  # End-effector position
├── robot1_eef_rot_axis_angle (N, 3) float32      # End-effector rotation
├── robot1_gripper_width (N, 1) float32           # Gripper width
├── robot1_demo_start_pose (N, 7) float32         # Demo start pose
├── robot1_demo_end_pose (N, 7) float32           # Demo end pose
```

**Vision & Tactile Data (Left Hand - camera0):**
```
├── camera0_rgb (N, H, W, 3) uint8                # Visual camera
├── camera0_left_tactile (N, H_t, W_t, 3) uint8  # Left tactile sensor image
├── camera0_left_tactile_points (N, P, 3) float32 # Left tactile point cloud
├── camera0_right_tactile (N, H_t, W_t, 3) uint8 # Right tactile sensor image
├── camera0_right_tactile_points (N, P, 3) float32 # Right tactile point cloud
```

**Vision & Tactile Data (Right Hand - camera1):**
```
├── camera1_rgb (N, H, W, 3) uint8                # Visual camera
├── camera1_left_tactile (N, H_t, W_t, 3) uint8  # Left tactile sensor image
├── camera1_left_tactile_points (N, P, 3) float32 # Left tactile point cloud
├── camera1_right_tactile (N, H_t, W_t, 3) uint8 # Right tactile sensor image
└── camera1_right_tactile_points (N, P, 3) float32 # Right tactile point cloud
```

#### Variable Definitions

- **N**: Total number of frames across all episodes
- **H × W**: Visual image resolution (set by `visual_out_res` in config)
- **H_t × W_t**: Tactile image resolution (set by `tactile_out_res` in config)
- **P**: Number of points in tactile point cloud (set by `fps_num_points` in config)

#### Important Notes

- **Tactile data** (`*_tactile` and `*_tactile_points`) are only generated when `use_tactile_img` and/or `use_tactile_pc` are enabled in the config. For vision-only policies, set these to `False`.
- **Camera indices**: `camera0` = left hand, `camera1` = right hand
- **Pose format**: `[x, y, z, rx, ry, rz, rw]` where `(x, y, z)` is position and `(rx, ry, rz, rw)` depends on rotation representation
- **Compression**: Images are compressed using JpegXl with configurable compression level


## 🚀 Training Policy

### 🤗 Hugging Face Setup (Optional)

If you experience issues loading models from Hugging Face, configure a mirror:

```bash
pip install -U huggingface_hub
export HF_ENDPOINT=https://hf-mirror.com
```

### Training Commands

#### Single GPU Training

```bash
python train.py --config-name=train_vision_tactile_pc
```

#### Multi-GPU Training

For training with 8 GPUs:

```bash
accelerate launch --num_processes 8 train.py --config-name=train_vision_tactile_pc
```

### 🔧 Troubleshooting

**Issue:** `ImportError: cannot import name 'cached_download' from 'huggingface_hub'`

**Solution:** Locate the `dynamic_modules_utils.py` file and remove the import statement for `cached_download`.


## 🤖 Real-World Deployment

Our reference implementation uses the **Rokae xMate ER3 Pro** robot arm with a **PGI gripper**. 

### Hardware Adaptation

To adapt this system to your own robot hardware, modify the following interfaces:
- [Robot Interface](./real_world/rokae/rokae_interface.py) - Robot arm control
- [Gripper Interface](./real_world/pgi/pgi_interface.py) - Gripper control

### Deploy Trained Policy

Run the deployment script with your trained checkpoint:

```bash
python deploy_scripts/eval_real_bimanual_vb.py -i 'path/to/your/ckpt'
```


## 🙏 Acknowledgement

Our work is built upon [UMI](https://github.com/real-stanford/universal_manipulation_interface)
and [ARCap](https://stanford-tml.github.io/ARCap/).
Thanks for their great work!

## 🔗 Citation

If you find our work useful, please consider citing:

```
@article{li2025vitamin,
  title={ViTaMIn-B: A Reliable and Efficient Visuo-Tactile Bimanual Manipulation Interface},
  author={Li, Chuanyu and Liu, Chaoyi and Wang, Daotan and Zhang, Shuyu and Li, Lusong and Zeng, Zecui and Liu, Fangchen and Xu, Jing and Chen, Rui},
  journal={arXiv preprint arXiv:2511.05858},
  year={2025}
}
```

## 📧 Contact

For questions or collaborations, please contact:
- **Chuanyu Li**: [chuanyu.ne79@gmail.com](mailto:chuanyu.ne79@gmail.com)


# Sync
v4l2-ctl -d /dev/video2 --set-ctrl=backlight_compensation=2
v4l2-ctl -d /dev/video2 --get-ctrl=backlight_compensation
1是正常使用 2是同步开关
arduino:
lsusb 找到对应的板子

IDE 里正确选择板子和串口

插上 Nano R4，用 dmesg 或 IDE 看一下端口，一般会是 /dev/ttyACM0。
在 Arduino IDE 中：
工具 → 开发板：选 Arduino Nano R4（不要选老的 “Arduino Nano”）。
工具 → 端口：选 /dev/ttyACM0 (Arduino Nano R4) 那一项。
如果选成老的 Nano，会出现 avr-g++: no such file or directory 之类的错误。

解决 Linux 权限问题（一次性配置） 2.1 把用户加入 dialout 组（串口权限） sudo usermod -aG dialout $USER

然后 注销当前账户重新登录（或者重启），让组权限生效。
2.2 snap 版 Arduino 允许访问 USB（raw-usb）
（你是 snap 安装的 IDE 才需要这一步）
sudo snap connect arduino:raw-usb
serial-port 接口在你系统上不存在，报错可以忽略。
2.3 给 Nano R4 的两种 DFU 模式加 udev 规则
Nano R4 烧录时会在两种 USB ID 之间切换：
运行态：2341:0074
Bootloader/DFU：2341:0374
我们要给这两个 ID 都放开权限。
编辑规则文件：
sudo nano /etc/udev/rules.d/60-arduino-nano-r4.rules
写入以下内容（全贴进去就行）：
Arduino Nano R4 运行模式（2341:0074）
SUBSYSTEM=="usb", ATTR{idVendor}=="2341", ATTR{idProduct}=="0074", MODE="0666", GROUP="dialout" SUBSYSTEM=="tty", ATTRS{idVendor}=="2341", ATTRS{idProduct}=="0074", MODE="0666", GROUP="dialout"
Arduino Nano R4 Bootloader / DFU 模式（2341:0374）
SUBSYSTEM=="usb", ATTR{idVendor}=="2341", ATTR{idProduct}=="0374", MODE="0666", GROUP="dialout" SUBSYSTEM=="tty", ATTRS{idVendor}=="2341", ATTRS{idProduct}=="0374", MODE="0666", GROUP="dialout"
保存并退出 nano（Ctrl+O 回车，Ctrl+X）。
让规则生效：
sudo udevadm control --reload-rules sudo udevadm trigger
把 Nano R4 拔掉再插上 一次。

上传程序 & 正常的 dfu-util 日志长什么样

回到 Arduino IDE：
再确认一次：
开发板：Arduino Nano R4
端口：/dev/ttyACM0 (Arduino Nano R4)
点左上角「上传」。
成功时你会看到类似这一段 dfu-util 输出（橙色的是正常信息）：
Opening DFU capable USB device... Device ID 2341:0074 ... Device really in Run-Time Mode, send DFU detach request... Device will detach and reattach... Opening DFU USB Device... ... Download [=========================] 100% 39104 bytes Download done. DFU state(7) = dfuMANIFEST, status(0) = No error condition is present DFU state(2) = dfuIDLE, status(0) = No error condition is present Done!
这就表示：
DFU 设备成功打开（运行态 → DFU 态）
程序已经写入板子（Download 100%）
没有权限错误（不再出现 LIBUSB_ERROR_ACCESS）
此时你的 PWM 程序已经在板子上跑了。