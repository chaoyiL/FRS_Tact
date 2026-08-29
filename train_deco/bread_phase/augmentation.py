"""Fixed augmentation contract for the dedicated Bread phase model."""

BREAD_AUGMENTATION_ARGS = (
    "--augmentation-enabled",
    "--augmentation-identity-probability", "0.25",
    "--augmentation-low-light-probability", "0.0",
    "--augmentation-mild-probability", "0.75",
    "--augmentation-exposure-probability", "0.5",
    "--augmentation-exposure-range", "0.8", "1.2",
    "--augmentation-gamma-range", "0.9", "1.1",
    "--augmentation-mild-brightness-range", "0.8", "1.2",
    "--augmentation-contrast-range", "0.85", "1.30",
    "--augmentation-saturation-range", "0.80", "1.15",
    "--augmentation-blur-probability", "0.20",
    "--augmentation-blur-kernel-sizes", "3", "5",
    "--augmentation-blur-sigma-range", "0.1", "1.0",
)

