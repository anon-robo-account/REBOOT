<p align="center">
  <img alt="LeRobot, Hugging Face Robotics Library" src="./media/readme/lerobot-logo-thumbnail.png" width="100%">
</p>

<div align="center">

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/huggingface/lerobot/blob/main/LICENSE)

</div>

# LeRobot (REBOOT fork)

This is a fork of [Trossen Robotics' fork of LeRobot](https://github.com/TrossenRobotics/lerobot_trossen),
extended with depth-stream recording support for use with the **REBOOT** dataset: a bimanual precision-assembly benchmark with paired expert and recovery demonstrations.

The upstream LeRobot library and this fork share the same Python API, dataset format, and training pipelines. The differences are small and additive: this fork is a drop-in replacement for users
who need depth recording on Intel RealSense cameras.

## What's added in this fork

This fork is based on the official LeRobot v0.4.3 release and adds the following:

- **Depth-stream recording integration.** Backports [PR #2604](https://github.com/huggingface/lerobot/pull/2604) from upstream LeRobot, enabling synchronized RGB-D recording from
  Intel RealSense cameras. Depth frames are stored as single-channel arrays in the same parquet files as proprioception, aligned to the corresponding RGB timestamps.

- **Compatibility with the LeRobot v3 dataset format.** Depth data is exposed as a standard dataset feature, so policies that only need RGB ignore it transparently while depth-aware policies
  can consume it directly via the standard data loader.

If you only need RGB recording or are working with non-RealSense cameras, you should use upstream LeRobot directly:
[github.com/huggingface/lerobot](https://github.com/huggingface/lerobot).

## Quick start

```bash
git clone https://github.com/anon-robo-account/REBOOT.git
cd REBOOT
pip install -e .
lerobot-info
```

For the full installation walkthrough, see the upstream
[Installation Documentation](https://huggingface.co/docs/lerobot/installation).
The depth integration adds no extra installation steps; it activates
automatically when an Intel RealSense camera is configured.

## Recording with depth

Depth recording is enabled per-camera in the robot configuration. Each RealSense camera with `depth=True` will produce a synchronized depth stream alongside its RGB stream.

```python
from lerobot.cameras.realsense import RealsenseCamera

cam = RealsenseCamera(
    serial_number="...",
    fps=30,
    width=640,
    height=480,
    depth=True,   # added in this fork
)
```

During recording, depth frames are written to the same parquet files as the RGB and proprioception streams. The dataset visualizer renders depth as a false-color overlay alongside the corresponding
RGB view (see screenshot in the REBOOT paper appendix).

## REBOOT dataset

REBOOT is hosted on the Hugging Face Hub at [huggingface.co/REBOOT26](https://huggingface.co/REBOOT26). It contains 2,160 bimanual demonstrations (1,080 expert + 1,080recovery) across 18 precision-assembly tasks on the NIST Assembly Task Board #1, recorded with this fork.


## Upstream LeRobot

Everything below is unchanged from upstream LeRobot. The library
still supports the same robots, policies, and benchmarks; this fork
only adds depth recording.

**Supported hardware:** SO100, LeKiwi, Koch, HopeJR, OMX,
EarthRover, Reachy2, gamepads, keyboards, phones, OpenARM,
Unitree G1, plus the Trossen WidowX AI bimanual platform tested
with REBOOT.

**Available policies:** ACT, Diffusion Policy, VQ-BeT, Pi0-FAST,
Pi0.5, GR00T N1.5, SmolVLA, XVLA, HIL-SERL, TDMPC.

For training, evaluation, and policy documentation, refer to the
upstream [LeRobot documentation](https://huggingface.co/docs/lerobot/index).



## Acknowledgments

This fork builds on the work of the LeRobot team at Hugging Face and the Trossen Robotics fork. Depth recording integration is based on [PR #2604](https://github.com/huggingface/lerobot/pull/2604)
contributed by the upstream community.

## License

Apache 2.0 
