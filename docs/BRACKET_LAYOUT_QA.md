Q&A Log — Recent Edits to Bracket Layout

This supplemental file captures the specific questions and answers we covered while tuning `pages/01_🏀_Tournament_Bracket.py`.

1) Where to adjust Final Four lines?
- Edit the Final Four section inside `create_visual_bracket()` — search for "# Final Four meeting in center".
- Key variables: `final_four_x` (left semifinal), `final_four_right_x` (right semifinal), `center_x` (championship center).
- Connectors originate from `region_exits[<region>]` (set from each region's `round5_x`/`y_final`).

2) What tokens/lines to inspect?
- Search tokens: `# Final Four meeting in center`, `final_four_x`, `final_four_right_x`, `center_x`, `region_exits`.
- The Final Four block contains semifinal Y calculations (`left_semi_y`, `right_semi_y`) and the connector traces.

3) Center off — how fixed?
- `center_x` was changed to be computed as `(final_four_x + final_four_right_x) / 2` so the championship remains centered automatically.

4) Right-side label overlap — what changed?
- Right-side region `x_start` values were shifted left (15px, then tuned to 1825) to reduce overlap.
- Right labels moved to `label_x = x_start + 25` for `direction == -1` so they sit outside the bracket.
- `xaxis.range` was adjusted (temporarily increased then tightened to 1875) so labels were visible but the canvas wasn't oversized.

5) Make right/left outer spacing symmetrical
- With `xaxis.range = 1875` and left `x_start = 50`, right `x_start` was set to `1875 - 50 = 1825` so outer margins match.

6) Shift right brackets to the left by N px
- Change `region_layout['Midwest']['x_start']` and `region_layout['West']['x_start']` (reduced by 15px in our edits).

7) Move right labels further to margin
- Change label computation in the region label block: use `label_x = x_start + <offset>` when `direction == -1`.

8) Team name spacing vs bracket (first round)
- Adjust `box_x` and `text_x` in the Round 1 loop.
  - Left side: `box_x = x_start` and `text_x = box_x + N` (increase N to move name away from bracket).
  - Right side: `box_x = x_start - 100` and `text_x = box_x + M` (modify M to move the name left/right; note `xanchor='right'` behavior).

9) Increase team name sizes
- Round 1 team names: font size changed from 10 -> 12.
- Round winners (Rounds 2–4): label font size changed from 8 -> 10.
- Final Four annotations: font size changed from 10 -> 12.
- Look for `font=dict(size=...)` in the different annotation blocks.

10) Fine horizontal alignment tips
- If you shift `final_four_x` or `final_four_right_x`, keep `(final_four_x + final_four_right_x)/2 == center_x` if you want the championship centered.
- Connector rails use small offsets like `center_x-5` and `center_x+5` — adjust these to nudge the horizontal rail alignment.

11) Canvas clipping
- If labels or markers get clipped after moving anchors, increase `xaxis.range` (in `fig.update_layout(...)`) or adjust `width`.

12) Want easier tuning?
- I can centralize the editable constants into a `LAYOUT_CONFIG` dict at the top of `pages/01_🏀_Tournament_Bracket.py` (single place to tweak `y_spacing`, `x_round_spacing`, `region_layout`, `final_four_x`, `final_four_right_x`, `center_x`, and `xaxis.range`).

If you want, I can merge this Q&A content into `BRACKET_LAYOUT_GUIDE.md` directly (I attempted to append it there but created this supplemental file to avoid modifying the original file's backtick/code-block structure). Want me to insert it into the main guide instead?