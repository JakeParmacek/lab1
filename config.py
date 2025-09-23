# picarx_nav/config.py

# =========================
# Tunables (safe defaults)
# =========================
WIDTH, HEIGHT   = 25, 25      # grid size in cells
RES_CM          = 5           # centimeters per grid cell
INFLATE_CELLS   = 1           # safety buffer (cells)
DRIVE_POWER     = 20          # forward power (0..100)

LEFT_DEG        = -30
RIGHT_DEG       = 30
STRAIGHT_DEG    = 0

STEP_DRIVE_SEC  = 0.3         # drive time per step
PAN_SETTLE_SEC  = 0.08        # servo settle time per pan step
MAX_RANGE_CM    = 100         # ignore hits beyond this
SWEEP_START     = -45         # deg
SWEEP_END       = 45          # deg (exclusive)
SWEEP_STEP      = 2           # deg
