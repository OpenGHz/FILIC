if __name__ == "__main__":

    import sys
    import subprocess

    args = sys.argv[1:]
    command = "cd third_party/DISCOVERSE/policies/act && python3 policy_evaluate_wrapper.py -cf configurations/basic_configs/example/environment/airbot_ptk_mock.yaml "
    # add args to command
    command += " ".join(args)
    command += " -vm filic.filic.FILIC"
    print(f"Running command: {command}")

    try:
        subprocess.run(command, shell=True, check=True)
    except Exception as e:
        pass
