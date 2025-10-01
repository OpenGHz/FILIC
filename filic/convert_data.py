from mcap_data_loader.utils.mcap_utils import (
    McapFlatBuffersWriter,
    FlatBuffersSchemas,
)
from airbot_data_collection.airbot.samplers.mcap_sampler import (
    AIRBOTMcapDataSampler,
    AIRBOTMcapDataSamplerConfig,
    TaskInfo,
)
from mcap.writer import Writer
from pydantic import BaseModel
from pydantic_settings import CliApp
from typing import Dict
import os
import time
import json


class Config(BaseModel):
    """Configuration for the DISCOVERSE to MCAP conversion.
    Args:
        root (str): Root directory containing the task data.
        task_name (str): Name of the task to process.
        output_dir (str): Directory to save the output MCAP files. If not provided,
            it defaults to `<root>/mcap/<task_name>`.
    """

    root: str
    task_name: str
    output_dir: str = ""


config = CliApp.run(Config)

start = time.perf_counter()
directory = f"{config.root}/{config.task_name}"
output_dir = config.output_dir or f"{config.root}/mcap/{config.task_name}"

os.makedirs(output_dir, exist_ok=True)

# find all folders in the directory
folders = [f.path for f in os.scandir(directory) if f.is_dir()]
# print(folders)

config = AIRBOTMcapDataSamplerConfig(task_info=TaskInfo(task_name=config.task_name))

for folder in folders:
    fd_base = os.path.basename(folder)
    if not fd_base.isdigit():
        print(f"Skipping folder {folder} as it does not match the expected format.")
        continue
    episode = int(fd_base)
    output_file_path = f"{output_dir}/{episode}.mcap"
    print(f"{output_file_path=}")
    mcap_writer = Writer(output_file_path)
    mcap_writer.start()
    flb_writer = McapFlatBuffersWriter()
    flb_writer.set_writer(mcap_writer)
    all_schemas = set(FlatBuffersSchemas)
    all_schemas.remove(FlatBuffersSchemas.COMPRESSED_IMAGE)
    flb_writer.register_schemas(all_schemas)
    AIRBOTMcapDataSampler.add_config_metadata(mcap_writer, config)
    # find all .mp4 files in the folder
    mp4_files = [
        f.path for f in os.scandir(folder) if f.is_file() and f.name.endswith(".mp4")
    ]
    print(f"{mp4_files=}")
    # add video attachments
    for mp4_file in mp4_files:
        with open(mp4_file, "rb") as f:
            name = f"/{os.path.basename(mp4_file).removesuffix('.mp4')}/color/image_raw"
            print(f"Adding video attachment: {name}")
            AIRBOTMcapDataSampler.add_video_attachment(
                mcap_writer,
                name,
                f.read(),
            )

    def to_topic(group: str, component: str) -> str:
        return f"/{group}/arm/pose/position"

    # load json dict
    groups = ["lead", "follow"]
    components = ["arm", "eef"]
    slices = [slice(0, 6), slice(6, 7)]

    path = f"{folder}/obs_action.json"
    if not os.path.exists(path):
        print(f"Skipping file {path} as it does not exist.")
        continue

    topic_mapping: Dict[str, dict] = {
        "jq": {
            "/follow/arm/joint_states/position": slice(0, 6),
            "/follow/eef/joint_states/position": slice(6, 7),
        },
        "jv": {
            "/follow/arm/joint_states/velocity": slice(0, 6),
            "/follow/eef/joint_states/velocity": slice(6, 7),
        },
        "tau": {
            "/follow/arm/joint_states/effort": slice(0, 6),
            "/follow/eef/joint_states/effort": slice(6, 7),
        },
        "eef_pos": "/follow/arm/pose/position",
        "end_force": {
            "/follow/arm/wrench/force": slice(0, 3),
            "/follow/arm/wrench/torque": slice(3, 6),
        },
        "act": "/lead/arm/pose/position",
    }

    topic_names = set()
    for key, value in topic_mapping.items():
        if isinstance(value, str):
            topic_mapping[key] = {value: None}
            topic_names.add(value)
        else:
            assert isinstance(value, dict), f"Value for key {key} must be a dict."
            topic_names.update(value.keys())

    with open(path) as f:
        act_obs: dict = json.load(f)
        print(f"{act_obs.keys()=}")
        obs: dict = act_obs.pop("obs", {})
        print(f"{obs.keys()=}")
        times = act_obs.pop("time", [])
        acts: list = act_obs.pop("act", [])
        not_mapping_keys = (act_obs.keys() | obs.keys()) - topic_mapping.keys()
        print(f"Not mapping keys: {not_mapping_keys}")
        topic_names.update(not_mapping_keys)
        print(f"All topic names: {topic_names}")
        # register joint state channels
        for topic in topic_names:
            flb_writer.register_channel(topic, FlatBuffersSchemas.FLOAT_ARRAY)
        # add joint states messages
        stamps_ns = []

        for stamp in times:
            stamps_ns.append(int(stamp * 1e9))
        AIRBOTMcapDataSampler.add_log_stamps_attachment(
            mcap_writer,
            stamps_ns,
        )

        for i, action in enumerate(acts):
            stamp_ns = stamps_ns[i]
            for topic, slc in topic_mapping["act"].items():
                if slc:
                    flb_writer.add_array(topic, action[slc], stamp_ns, stamp_ns)
                else:
                    flb_writer.add_array(topic, action, stamp_ns, stamp_ns)

        for key, values in obs.items():
            for topic, slc in topic_mapping.get(key, {key: None}).items():
                for i, value in enumerate(values):
                    stamp_ns = stamps_ns[i]
                    if slc:
                        flb_writer.add_array(topic, value[slc], stamp_ns, stamp_ns)
                    else:
                        flb_writer.add_array(topic, value, stamp_ns, stamp_ns)

    mcap_writer.finish()


print(
    f"Time taken: {time.perf_counter() - start:.3f} seconds of {len(folders)} folders"
)
