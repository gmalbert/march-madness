# Bracket Layout Adjustment Guide

Quick reference for adjusting the Tournament Bracket visualization in `pages/01_🏀_Tournament_Bracket.py`

## Region Positioning

**Lines 274-281** - Control where each region starts:
```python
region_layout = {
    'South': {'x_start': 50, 'y_start': 550, ...},
    'East': {'x_start': 50, 'y_start': 30, ...},
    'Midwest': {'x_start': 1950, 'y_start': 550, ...},
    'West': {'x_start': 1950, 'y_start': 30, ...}
}
```
- **`x_start`**: Horizontal starting position (left regions: 50, right regions: 1950)
- **`y_start`**: Vertical starting position (higher number = lower on screen)
- Increase right region `x_start` to move right (reduce right margin)
- Adjust `y_start` to prevent team clipping at top/bottom

## Spacing & Layout

**Lines 268-269** - Global spacing parameters:
```python
y_spacing = 32           # Vertical space between teams
x_round_spacing = 140    # Horizontal space between rounds
```
- `y_spacing`: Controls vertical spacing between teams in matchups
- `x_round_spacing`: Controls horizontal distance between rounds

## Final Four Positioning

**Line 516** - Championship center position:
```python
center_x = 900
```
- Adjust to move Final Four circle and championship line left/right
- Should be midpoint between left and right Final Four positions

**Line 534** - Left Final Four position:
```python
final_four_x = 650
```
- Controls where South vs East semifinal is positioned

**Line 627** - Right Final Four position:
```python
final_four_right_x = 1150
```
- Controls where Midwest vs West semifinal is positioned
- Increase to move semifinal right (creates more horizontal spacing)

**Lines 529-530** - Vertical spacing between Final Four teams:
```python
south_y = left_semi_y + (y_spacing * 2)
east_y = left_semi_y - (y_spacing * 2)
```
- Change the `* 2` multiplier to adjust vertical spacing between semifinal teams

## Winner Label Positioning

**Round 2 (Line 409)** - Horizontal offset from marker:
```python
label_x = round2_x + 5 if direction == 1 else round2_x - 5
```
- Change `5` to move text further/closer to winner marker
- Positive = away from bracket, negative = toward bracket

**Round 2 (Line 411)** - Vertical offset:
```python
y=y_mid + 8
```
- Change `8` to move text up (larger) or down (smaller)

**Similar patterns for Round 3 (line ~447) and Round 4 (line ~485)**

## Canvas & Viewport

**Lines 825-827** - Figure dimensions and coordinate system:
```python
xaxis=dict(visible=False, range=[0, 2000])
yaxis=dict(visible=False, range=[0, 1050])
height=1400
width=2200
```
- **`xaxis range`**: Coordinate system width (increase to add right margin)
- **`yaxis range`**: Coordinate system height
- **`width/height`**: Actual figure size in pixels

## Region Labels

**Lines 332-340** - Region name labels (currently commented out):
```python
# Uncomment to show SOUTH, EAST, MIDWEST, WEST labels
# fig.add_annotation(...)
```

## Quick Fixes

**Problem: Right side too cramped**
- Increase `final_four_right_x` (line 627)
- Increase right region `x_start` values (lines 278-279)
- Increase `xaxis range` upper limit (line 826)

**Problem: Teams clipping at top/bottom**
- Adjust region `y_start` values (lines 274-281)

**Problem: Final Four off-center**
- Adjust `center_x` (line 516)
- Should equal `(final_four_x + final_four_right_x) / 2`

**Problem: Winner text overlapping lines**
- Adjust `+ 5` offset in `label_x` calculations (lines 409, ~447, ~485)
- Adjust `+ 8` offset in `y=y_mid + 8` (lines 411, ~449, ~487)

## Recent Changes (what I edited and where)

I made a number of small layout tweaks while iterating on the bracket so you can reproduce or tweak them later. These are the exact areas I changed and why:

- Region label visibility and orientation
    - File: `pages/01_🏀_Tournament_Bracket.py`
    - Lines: region label block around the `for region_name, teams in regions.items()` loop (near where `label_x` is set).
    - What: Re-enabled vertical region labels and positioned them close to the bracket. Left-region labels use `textangle=-90`, right-region labels use `textangle=90` so the start of each word points toward the top of the page. I placed them just inside the bracket edge by setting `label_x = x_start - 25` for left regions and `label_x = x_start - 15` for right regions.
    - How to change: edit `label_x` and `label_angle` (the `textangle` value) in the file. For example, to move labels further away use `x_start - 40` (left) or `x_start + 40` (right).

- Final Four horizontal positions
    - `final_four_x` (left semifinal) and `final_four_right_x` (right semifinal) determine where the two semifinal teams sit horizontally.
    - Current values in the file: `final_four_x = 650`, `final_four_right_x = 1150` (these are set in the Final Four section). Changing these moves each semifinal left/right.
    - Tip: Keep `(final_four_x + final_four_right_x) / 2 == center_x` if you want the championship center to remain visually centered.

- Championship center and canvas
    - `center_x` controls where the Final Four / championship area is drawn (default `center_x = 900`).
    - The coordinate system is controlled by `xaxis.range` (e.g., `[0, 1850]`) and the physical figure `width` (e.g., `width=2200`). If you change `xaxis.range`, update `width` or `center_x` so things remain centered visually.

- Region starting X positions
    - `region_layout['Midwest']['x_start']` and `region_layout['West']['x_start']` control how far to the right the right-side brackets are placed. Moving them right reduces the left-margin effect on the canvas but requires increasing `xaxis.range` accordingly.

- Winner text offsets (Rounds 2-4)
    - The winner-labels for Rounds 2-4 use code like:
        ```python
        text_anchor = 'left' if direction == 1 else 'right'
        label_x = round2_x + 80 if direction == 1 else round2_x - 80
        fig.add_annotation(x=label_x, y=y_mid + 8, ...)
        ```
    - I increased these offsets (used `+80 / -80`) to keep the label text clear of the bracket. Change these numbers to move the labels horizontally.

- Connector behavior
    - `draw_matchup_bracket(x1, y1, y2, x2, y_mid)` draws the horizontal lines from `x1` to `x2` and the vertical connector at `x2`. If you want the same visual indentation on both sides, use consistent `x1/x2` values (we used the region exit `round5_x` values as the `x1` for the Final Four connectors).

## Where to look in the code (quick links)
- Region layout and spacing: `pages/01_🏀_Tournament_Bracket.py` lines ~268-282 (`y_spacing`, `x_round_spacing`, `region_layout`).
- Winner label offsets (Round 2/3/4): lines ~400-420, ~436-456, ~472-492.
- Final Four positions and connectors: lines ~508-760 (Final Four section).
- Figure sizing and axis: `fig.update_layout(... xaxis=dict(range=[0, ...]), width=..., height=...)` near the bottom of the file.

If you'd like, I can also add a small function to expose these layout constants at the top of the file (so you can tweak a single block of constants instead of hunting line numbers). Want me to do that?
