"""生成 problem1_v2/对齐结果 交付包。"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from problem1_build_alignment_packages import write_package

if __name__ == "__main__":
    print(write_package("v2"))
