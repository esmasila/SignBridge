"""
Extract A-Z alphabet JS animation files and GLB models from SignKit zip.
Outputs:
  - c:/Projects/sign_bridge/web/avatar/signkit_alphabets.json
  - c:/Projects/sign_bridge/web/avatar/models/xbot.glb
  - c:/Projects/sign_bridge/web/avatar/models/ybot.glb
"""

import zipfile
import json
import re
import math
import os

ZIP_PATH = "C:/Projects/sign_bridge/Sign-Kit-An-Avatar-based-ISL-Toolkit-main.zip"
OUT_JSON = "C:/Projects/sign_bridge/web/avatar/signkit_alphabets.json"
OUT_MODELS_DIR = "C:/Projects/sign_bridge/web/avatar/models"

# Ensure output directories exist
os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
os.makedirs(OUT_MODELS_DIR, exist_ok=True)

# --- Math.PI expression evaluator ---
def eval_math_pi(expr):
    """
    Evaluate expressions like:
      Math.PI/9
      -Math.PI/9
      Math.PI/2.5
      Math.PI
      0
      -Math.PI
    Returns a float rounded to 6 decimal places.
    """
    expr = expr.strip()
    # Replace Math.PI with the actual value
    expr = expr.replace("Math.PI", str(math.pi))
    # Now eval safely (only numbers, operators, parens)
    # Restrict to safe characters
    if not re.match(r'^[\d\s\.\+\-\*\/\(\)eE]+$', expr):
        raise ValueError(f"Unsafe expression: {expr!r}")
    result = eval(expr)
    return round(float(result), 6)

# --- Parse one JS file ---
def parse_js_alphabet(js_text):
    """
    Parse a JS alphabet animation file.
    Returns a list of phases. Each phase is a list of [boneName, axis, value] triples.

    Looks for blocks between:
      animations = []
      ...animations.push([...])...
      ref.animations.push(animations)
    """
    phases = []

    # Split into phase blocks by finding "animations = []" resets
    # Each block ends when we see "ref.animations.push(animations)"

    # Find all phase blocks
    # Strategy: find all "animations.push([...])" lines between each pair of
    # "animations = []" and "ref.animations.push(animations)"

    # Regex to match a single push line:
    # animations.push(["boneName", "rotation", "axis", VALUE, "sign"]);
    push_re = re.compile(
        r'animations\.push\(\s*\[\s*'
        r'"([^"]+)"\s*,\s*'      # bone name
        r'"rotation"\s*,\s*'     # "rotation" (literal)
        r'"([xyz])"\s*,\s*'      # axis
        r'([^,]+?)\s*,\s*'       # value expression
        r'"[+\-]"\s*'            # sign
        r'\]\s*\)'
    )

    # Split text into segments by "ref.animations.push(animations)"
    ref_push_re = re.compile(r'ref\.animations\.push\(animations\)')
    reset_re = re.compile(r'animations\s*=\s*\[\s*\]')

    # Walk through the text collecting phases
    current_entries = []
    in_phase = False

    for line in js_text.splitlines():
        stripped = line.strip()

        # Start of a new phase
        if reset_re.search(stripped):
            in_phase = True
            current_entries = []
            continue

        # End of current phase
        if ref_push_re.search(stripped):
            if current_entries:
                phases.append(current_entries)
            current_entries = []
            in_phase = False
            continue

        if in_phase:
            m = push_re.search(stripped)
            if m:
                bone = m.group(1)
                axis = m.group(2)
                val_expr = m.group(3).strip()
                value = eval_math_pi(val_expr)
                current_entries.append([bone, axis, value])

    return phases

# --- Main ---
alphabet_data = {}

with zipfile.ZipFile(ZIP_PATH) as zf:
    namelist = zf.namelist()

    # Extract A-Z JS files
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        path = f"Sign-Kit-An-Avatar-based-ISL-Toolkit-main/client/src/Animations/Alphabets/{letter}.js"
        if path not in namelist:
            print(f"WARNING: {letter}.js not found in zip")
            continue
        js_bytes = zf.read(path)
        js_text = js_bytes.decode("utf-8")
        phases = parse_js_alphabet(js_text)
        alphabet_data[letter] = phases
        print(f"  {letter}: {len(phases)} phases, {sum(len(p) for p in phases)} total entries")

    # Extract GLB models
    for glb_name, dest_subpath in [
        ("Sign-Kit-An-Avatar-based-ISL-Toolkit-main/client/src/Models/xbot/xbot.glb", "xbot.glb"),
        ("Sign-Kit-An-Avatar-based-ISL-Toolkit-main/client/src/Models/ybot/ybot.glb", "ybot.glb"),
    ]:
        if glb_name not in namelist:
            print(f"WARNING: {glb_name} not found in zip")
            continue
        data = zf.read(glb_name)
        dest = os.path.join(OUT_MODELS_DIR, dest_subpath)
        with open(dest, "wb") as f:
            f.write(data)
        print(f"  Extracted {dest_subpath} -> {dest} ({len(data):,} bytes)")

# Write JSON
with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(alphabet_data, f, indent=2, ensure_ascii=False)

size = os.path.getsize(OUT_JSON)
print(f"\nWrote {OUT_JSON}")
print(f"  Letters: {len(alphabet_data)}")
print(f"  File size: {size:,} bytes ({size/1024:.1f} KB)")
