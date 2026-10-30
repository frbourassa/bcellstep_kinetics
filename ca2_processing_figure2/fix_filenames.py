#!/usr/bin/env python3

import sys
from pathlib import Path

def main():
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: rename_files.py <directory>\n"
            "Example: rename_files.py JR064_tracks/"
        )

    data_dir = Path(sys.argv[1]).expanduser().resolve()

    if not data_dir.is_dir():
        raise SystemExit(f"Directory not found: {data_dir}")

    rename_map = {
        "IgM_10ug:mL_z1.csv":
            "IgM_10ug:mL.csv",

        "IgM_10ug:mL_z2_.csv":
            "IgM_10ug:mL_2.csv",

        "pHELD-AF594_10pM_2_z2.csv":
            "pHELD-AF594_10pM_2_2.csv",

        "pHELD-AF594_10pM_z2.csv":
            "pHELD-AF594_10pM_2.csv",
    }

    renamed = 0

    for old_name, new_name in rename_map.items():
        old_path = data_dir / old_name
        new_path = data_dir / new_name

        if not old_path.exists():
            print(f"NOT FOUND: {old_name}")
            continue

        if new_path.exists():
            print(f"SKIPPED: {new_name} already exists")
            continue

        old_path.rename(new_path)
        renamed += 1

        print(f"RENAMED: {old_name} -> {new_name}")

    print(f"\nDone. Renamed {renamed} file(s).")

if __name__ == "__main__":
    main()