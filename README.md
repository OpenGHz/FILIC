# FILIC: Dual-Loop Force-Guided Imitation Learning with Impedance Torque Control for Contact-Rich Manipulation Tasks

<div align="center">

*The relevant code will be made public before October 10, 2025*
<!-- [![Paper](https://img.shields.io/badge/Paper-arXiv-red.svg)](https://arxiv.org/) -->
<!-- [![Website](https://img.shields.io/badge/Website-FILIC-blue.svg)](https://filic.github.io/) -->
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
<!-- [![Docker](https://img.shields.io/badge/Docker-Available-blue.svg)](#docker-quick-start) -->

</div>

## 📦 Installation

Clone this repository and install dependencies
```bash
git submodule update --init --recursive
pip install -e .
pip install -r requirements.txt
bash install.sh
```

Our modifications are based on some third-party repositories, and the following patch needs to be applied:

```bash
git -C third_party/DISCOVERSE am $(pwd)/patch/discoverse.patch
```

<!-- Install the third party packages by following the instructions in their respective repositories. -->

## 💡 Usage Examples

### Collecting Demonstrations
To collect demonstrations in simulation, run:
```bash
python3 third_party/DISCOVERSE/discoverse/examples/mocap_ik/mocap_ik_manipulator.py -r airbot_play_force -t peg_in_hole --mouse-3d --record
```
The data will be saved in the `third_party/DISCOVERSE/data` directory by default.

### Converting Data
To convert the collected demonstrations data to the required format for training, run:
```bash
python3 filic/convert_data.py --root third_party/DISCOVERSE/data --task-name airbot_play_force_peg_in_hole --output-dir third_party/DISCOVERSE/policies/act/data/mcap
```
This will convert the data and save it in the specified output directory.

### Training the Policy

To train the policy, run:
```bash
CUDA_VISIBLE_DEVICES=0 python3 train.py -tn airbot_play_force_peg_in_hole
```
The trained models and logs will be saved in the `third_party/DISCOVERSE/policies/act/my_ckpt` directory by default.

### Evaluating the Policy
To evaluate the trained policy, run:
```bash
CUDA_VISIBLE_DEVICES=0 python3 infer.py -tn airbot_play_force_peg_in_hole -ts <timestamp>
```
Please replace `-ts` with the appropriate timestamp of your trained model.

## ⚖️ License

FILIC is released under the [MIT License](LICENSE). See the license file for details.
