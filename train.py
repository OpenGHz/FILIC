if __name__ == "__main__":

    import sys
    import subprocess

    print(sys.argv)
    args = sys.argv[1:]
    command = "cd third_party/DISCOVERSE/policies/act && python3 policy_train.py "

    # add args to command
    command += " ".join(args)
    command += " -vm filic.filic.FILIC"
    print(f"Running command: {command}")

    try:
        subprocess.run(command, shell=True, check=True)
    except Exception as e:
        pass
