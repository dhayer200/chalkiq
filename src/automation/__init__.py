"""ChalkIQ automation — scheduled data collection without manual CLI."""

ACTIVE_DIVISIONS = ("mens", "cfb")

# Edge threshold for newsletter bet flags (CFB strategy: lower than CBB 3%)
MIN_EDGE = {
    "mens": 0.025,
    "cfb":  0.020,
}
