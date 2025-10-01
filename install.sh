#!/bin/bash
set -ex

git -C third_party/DISCOVERSE submodule update --init policies/act
git -C third_party/DISCOVERSE/policies/act checkout main
git -C third_party/DISCOVERSE/policies/act pull
cp patch/policy_evaluate_wrapper.py third_party/DISCOVERSE/policies/act/
cp patch/airbot_play_mujoco_env.py third_party/DISCOVERSE/policies/act/envs/
cp configs/airbot_play_force_peg_in_hole.py third_party/DISCOVERSE/policies/act/configurations/task_configs/
pip install -e ./third_party/DISCOVERSE"[act_full]"
pip install -r ./third_party/DISCOVERSE/policies/act/requirements/train_eval.txt
