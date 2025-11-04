import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.prepare_data import generate_fly_splits

if __name__ == "__main__":
    generate_fly_splits(
        '../../../../data/fly_data/fly_group_train.npy',
        val_fraction=0.1,
        seed=42,
        save_path='fly_data/fly_split.json'
    )