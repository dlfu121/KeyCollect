 第四步：重启

  sudo reboot

  第五步：重启后验证

  先进桌面后跑：

  prime-select query

  期望：

  on-demandss  看 Intel/NVIDIA 驱动：

  lsmod | grep -E 'i915|nvidia'

  期望至少有：

  i915
  nvidia
  nvidia_modeset
  nvidia_uvm
  nvidia_drm

  看设备节点：

  ls -l /dev/dri /dev/nvidia*

  期望有：

  /dev/dri/card*
  /dev/dri/renderD*
  /dev/nvidia0
  /dev/nvidiactl
  /dev/nvidia-uvm

  看 NVIDIA：

  nvidia-smi

  如果这一步正常，训练/推理就能用 GPU。

  再看 PyTorch：

  conda activate keycollect

  python - <<'PY'
  import torch
  print("torch:", torch.__version__)
  print("cuda runtime:", torch.version.cuda)
  print("cuda available:", torch.cuda.is_available())
  print("device count:", torch.cuda.device_count())
  if torch.cuda.is_available():
      print("device:", torch.cuda.get_device_name(0))
  PY

  期望：

  cuda available: True
  device count: 1
  device: NVIDIA GeForce RTX 4070 Ti SUPER

  如果 nvidia-smi 仍失败

  这时再判断 Secure Boot/MOK 问题。

  跑：

  sudo modprobe nvidia
  dmesg | grep -iE 'nvidia|module verification|required key|lockdown|secure' | tail -100

  如果看到类似：

  Required key not available
  module verification failed
  unsigned module loading is restricted

  说明 NVIDIA 内核模块签名/MOK 仍然没对上。处理方式是重新配置 DKMS/MOK：

  sudo dpkg-reconfigure nvidia-dkms-570
  sudo update-initramfs -u
  sudo reboot

  重启时会进 MOK Manager，选择：

  Enroll MOK
  Continue
  输入你设置的密码
  Reboot

  但从你现在的输出看，MOK 已经 enrolled，而且 nvidia.ko 上有 signer，所以我会先处理 xorg.conf、prime-select、nomodeset，不要先折腾 Secure Boot。

  如果切 on-demand 后桌面仍黑屏

  还有一个硬件布线点要注意：你的 Xorg 老日志里显示外接显示器接在 NVIDIA 输出上：

  Philips PHL 275S9LRB connected on NVIDIA GPU

  如果显示器插在独显输出口，而你又想让桌面完全跑 Intel，有些台式机/混合显卡配置会黑屏，因为 Intel 核显没有直接控制那个物理输出。

  这种情况下有两个选择：

  1. 显示器插主板视频输出口，让 Intel 核显负责桌面。
  2. 显示器继续插独显，但桌面就不能完全 Intel-only，需要 NVIDIA 参与显示，功耗会高一些。

  你的目标是“默认 Intel 核显，NVIDIA 只用于训练/推理”，那显示器最好接主板 HDMI/DP，也就是 Intel 核显输出。

  训练/推理怎么用 GPU

  只要 nvidia-smi 和 torch.cuda.is_available() 正常，CUDA 计算不需要 prime-run。直接跑训练即可：

  conda activate keycollect

  lerobot-train \
    --policy.type=act \
    --policy.pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1 \
    --policy.push_to_hub=false \
    --dataset.repo_id=local/rm65_dexhand_merged \
    --dataset.root=/home/ee304/dongziyue/KeyCollect/data/rm65_dexhand_merged \
    --dataset.video_backend=pyav \
    --output_dir=outputs/train/act_rm65_dexhand \
    --steps=80000 \
    --save_freq=20000 \
    --env_eval_freq=0

  推理脚本里已经会自动判断：

  if torch.cuda.is_available():
      device = "cuda"
  else:
      device = "cpu"

  所以修好驱动后，它会自动上 CUDA。

  如果是需要 NVIDIA 做 OpenGL/GLX 图形渲染的程序，才需要：

  prime-run your_command

  或者手动：

  __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia your_command

  但纯 PyTorch 训练/推理一般不需要。

  我对你这台机器的判断

  最可能的黑屏链路是：

  MOK 放行后 NVIDIA 驱动终于能加载
  +
  /etc/X11/xorg.conf 强制 Xorg 使用 NVIDIA
  +
  prime-select / nomodeset / 显示器接线状态不匹配
  =
  图形界面黑屏

  当前要做的不是继续“关闭 Secure Boot 或反复 MOK”，而是把模式整理成：

  无 /etc/X11/xorg.conf 强制 NVIDIA
  GRUB 无 nomodeset
  prime-select on-demand
  Intel i915 负责桌面
  NVIDIA 模块可加载，CUDA 可见

  推荐执行的最小修复命令汇总：

  sudo mv /etc/X11/xorg.conf /etc/X11/xorg.conf.nvidia-xconfig.bak
  sudo prime-select on-demand
  sudo sed -i 's/ nomodeset//g' /etc/default/grub
  sudo update-grub
  sudo update-initramfs -u
  sudo reboot

  重启后验证：

  prime-select query
  nvidia-smi

  conda activate keycollect
  python - <<'PY'
  import torch
  print(torch.cuda.is_available())
  print(torch.cuda.device_count())
  print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
  PY

  成功标准很明确：

  prime-select query -> on-demand
  nvidia-smi -> 能显示 RTX 4070 Ti SUPER
  torch.cuda.is_available() -> True

  如果这三项成立，就可以在不把桌面切到 NVIDIA 的情况下使用 GPU 训练和推理。