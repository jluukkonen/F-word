import pandas as pd
import re
import os

EXISTING = "/Volumes/United/Work/F-word/swearing-nlp/data/labeled/chatgpt_labels.tsv"
OUTPUT = "/Volumes/United/Work/F-word/swearing-nlp/data/labeled/chatgpt_labels.tsv"

# ── Raw paste from user ──
RAW = """
mswnr67 | aggression
mpribsw | bonding
mq23sdw | aggression
msxe4fz | bonding
mszwrte | frustration
msknwv6 | aggression
mr5y50b | frustration
msofwpy | ambiguous
mrl93z4 | aggression
mqh81qv | emphasis
mt1cqbc | frustration
mre5onn | emphasis
msl9vza | bonding
mphyqov | emphasis
mquaiy2 | bonding
mrdxf0p | emphasis
mql8jsg | aggression
mr2d43d | aggression
mt3714u | aggression
mpp07ck | emphasis
mt2fah7 | frustration
mt1fine | bonding
mrdc67m | frustration
mrzm7u2 | aggression
msvx2hx | aggression
mssdkch | aggression
mt1l702 | emphasis
mqyluqm | bonding
mszaaef | aggression
mrdz1ho | ambiguous
mre3028 | emphasis
mqh8pvg | emphasis
msxg9nl | aggression
mqpccfg | emphasis
mqwoe90 | bonding
mre2n9l | aggression
msueh17 | aggression
mr8bqma | bonding
mrca1h5 | aggression
mt0dy2d | aggression
mpxlm9j | bonding
msgcicl | bonding
mt43lxu | aggression
mszbmfx | aggression
mt05dvn | bonding
mrdy50n | aggression
mntj7au | bonding
mrduhcz | bonding
mr458lu | aggression
mre3zyo | bonding
mqaggl5 | bonding
msxypas | bonding
mrdyvmp | bonding
mnt7t7u | bonding
mrilw38 | bonding
mqmsjlf | bonding
mrmkg19 | aggression
mt2ivxb | bonding
msed274 | aggression
mspy7vj | bonding
mre4c8q | bonding
mt37zri | aggression
mre0g7z | aggression
msywgcq | aggression
mr2mei7 | bonding
mt0gf7m | bonding
mr1ubnk | bonding
mskon5h | aggression
mt0lhw0 | aggression
mriuymi | bonding
mr0hxsq | bonding
mqhi4z4 | aggression
mpr61d3 | aggression
msxgwvx | aggression
mspc5o7 | aggression
msttv1r | aggression
ms3gms2 | aggression
mse0ncp | bonding
mt25w9p | aggression
msh09pk | bonding
mnwds7y | bonding
mpcynks | bonding
mrdvmpf | aggression
mrzcn07 | bonding
mrq2yw6 | bonding
mswf69a | aggression
mqu7d8f | bonding
mswezv0 | bonding
mpnvx29 | aggression
mrdth2a | bonding
mstfw5m | aggression
msubwrj | bonding
msswkq4 | aggression
mrdwg6g | bonding
mre0k5t | bonding
msn6jab | aggression
mqoy07k | aggression
mrtxr82 | bonding
msup56m | bonding
msvekf5 | aggression
msw99ux | bonding
mrdx736 | aggression
msgrtfr | aggression
mszk399 | aggression
mspapov | bonding
mr3415m | aggression
msxfeaw | aggression
mqy6o73 | bonding
mrcs4pn | bonding
mpo2ok0 | aggression
msw1c9f | aggression
msx80g5 | bonding
msysqnx | bonding
mrdysk0 | aggression
mqidtwz | bonding
mqw6uut | bonding
mqk0oge | bonding
mqefbgv | aggression
msueunx | ambiguous
mr3dujk | bonding
mqfs7h4 | aggression
msw0v6a | bonding
mqdi8e1 | ambiguous
ms5z993 | aggression
mswfssm | bonding
msxh1ze | aggression
mqtn356 | bonding
moisfxy | bonding
mt0em5c | aggression
msu4ncl | aggression
mswhyfg | bonding
mqsutmo | bonding
msvklrp | bonding
mqk7295 | aggression
mr89uvw | bonding
mt3jftj | aggression
msxoyr2 | aggression
mrl93b0 | ambiguous
mrdxyjg | bonding
mt0ce0p | aggression
msp4dqe | aggression
mrdwpfs | bonding
mqt23at | aggression
mqso1fu | bonding
mq47hnv | aggression
msuqvtm | bonding
mswqi4e | aggression
mshyxmv | aggression
mszq43v | bonding
mt1oa3v | bonding
mq7bm21 | bonding
mt0860f | aggression
mq6cuwd | bonding
"""

# Parse all lines
results = []
for line in RAW.strip().split("\n"):
    line = line.strip()
    if not line or line.startswith("---") or line.startswith("|") and "id" in line.lower():
        continue
    
    # Try multiple formats:
    # "| id | label |" table format
    # "id — **label**" or "id — label"  
    # "id → **label**"
    # "id | label"
    
    # Clean markdown bold
    line = line.replace("**", "")
    
    # Try pipe-separated (table)
    if "|" in line:
        parts = [p.strip() for p in line.split("|") if p.strip()]
        if len(parts) >= 2:
            item_id = parts[0].strip()
            label = parts[1].strip()
        else:
            continue
    # Try arrow formats
    elif "→" in line or "—" in line:
        sep = "→" if "→" in line else "—"
        parts = line.split(sep)
        if len(parts) >= 2:
            item_id = parts[0].strip().lstrip("* ")
            label = parts[1].strip()
        else:
            continue
    else:
        continue
    
    # Clean the label: remove parenthetical explanations
    label = label.split("(")[0].strip().lower()
    # Map neutral/other to ambiguous
    if label in ["neutral", "neutral / other", "other"]:
        label = "ambiguous"
    
    # Validate
    if label in ["aggression", "bonding", "emphasis", "frustration", "ambiguous"] and len(item_id) >= 5:
        results.append({"id": item_id, "label": label})

new_df = pd.DataFrame(results)
print(f"Parsed {len(new_df)} new labels from paste")
print(f"Unique IDs: {new_df['id'].nunique()}")

# Load existing
existing_df = pd.read_csv(EXISTING, sep="\t")
print(f"Existing labels: {len(existing_df)}")

# Merge: existing takes priority, then new
combined = pd.concat([existing_df, new_df]).drop_duplicates(subset="id", keep="first")
print(f"Combined (deduplicated): {len(combined)}")

# Label distribution
print("\nLabel distribution:")
print(combined["label"].value_counts())

# Save
combined.to_csv(OUTPUT, sep="\t", index=False)
print(f"\nSaved to {OUTPUT}")
