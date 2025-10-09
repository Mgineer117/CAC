def smooth(
    scalars: list[float], weight: float
) -> list[float]:  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)  # Save it
        last = smoothed_val  # Anchor the last smoothed value

    return smoothed


mAUC = {"CAC": 0.763, "C3M": 0.712, "PPO": 0.634}

COLORS = {
    "CAC": "#F83C32",  # Red (bright red)
    "C3M": "#1A62CF",  # Blue (medium blue)
    "PPO": "#646464",  # Green (distinct)
}
LINESTYLES = {
    "CAC": "-",
    "C3M": "-.",
    "PPO": ":",
}

# LABELS = {
#     "CAC": "CAC (ours)",
#     "C3M": "C3M",
#     "PPO": "PPO",
# }

LABELS = {
    "CAC": 0.763,
    "C3M": 0.712,
    "PPO": 0.634,
}
