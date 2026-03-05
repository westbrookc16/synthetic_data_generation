prompt_configs: list[dict[str, str]] = [
    {
        "type": "appliance",
        "prompt": (
            "You are a DIY appliance repair expert (refrigerators, washers, dryers, etc.). "
            "Create realistic homeowner repair question/answer pairs that strictly follow "
            "the required schema."
        ),
    }
]

judge_prompt_configs: list[dict[str, str]] = [
    {
        "type": "appliance_judge",
        "prompt": (
            "You are a strict dataset quality judge for DIY appliance repair records.\n"
            "Evaluate the record given with the required failure modes."
        ),
    }
]

# Backward-compatible defaults used by current scripts.
prompt_config: dict[str, str] = prompt_configs[0]
judge_prompt_config: dict[str, str] = judge_prompt_configs[0]
