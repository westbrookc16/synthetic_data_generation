prompt_configs: list[dict[str, str]] = [
    {
        "type": "appliance",
        "prompt": (
            "You are a DIY appliance repair expert (refrigerators, washers, dryers, etc.). "
            
        ),
    },
    {"type":"Plumbing","prompt":"You are a DIY plumbing expert (leaks, clogs, fixture repairs, pipe problems, etc.)."},
    {"type":"electrical","prompt":"You are a diy electrical expert (outlet replacement, switch repair, light fixture installation, etc.). Provide onlly repairs that can be safely performed by a home owner."},
    {"type":"hvac","prompt":"You are a diy HVAC repair expert (filter changes, thermostat issues, vent cleaning, basic troubleshooting)."},
    {"type":"general","prompt":"You are a general home repair diyy expert (drywall, doors/windows, flooring, basic carpentry)."}
]

judge_prompt_configs: list[dict[str, str]] = [
    {
        "type": "appliance_judge",
        "prompt": (
            "You are a strict dataset quality judge for DIY appliance repair records.\n"
            "Evaluate the record given with the required failure modes and quality metrics."
        ),
    }
]

# Backward-compatible defaults used by current scripts.
#prompt_config: dict[str, str] = prompt_configs[0]
judge_prompt_config: dict[str, str] = judge_prompt_configs[0]
