"""
SE Detection DataLoader
-----------------------
build_dataset.py로 생성한 dataset.json을 읽어
turn마다 text와 label을 뱉습니다.

출력 샘플:
    {
        "text"     : str,  # "Kelly: Hello Monica, I hope..."
        "label"    : int,  # 0 or 1
        "conv_id"  : str,  # "seconvo_0"
        "turn_idx" : int,  # 1, 2, 3, ...
    }
"""

import json
from torch.utils.data import Dataset, DataLoader


class SEDataset(Dataset):

    def __init__(self, dataset_path: str, split: str):
        assert split in ("train", "test"), "split은 'train' 또는 'test'"

        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.samples = []
        for conv in data[split]:
            for turn in conv["turns"]:
                self.samples.append({
                    "text"     : f"{turn['name']}: {turn['message']}",
                    "label"    : turn["label"],
                    "conv_id"  : conv["conv_id"],
                    "turn_idx" : turn["turn_idx"],
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


if __name__ == "__main__":
    DATASET_PATH = "dataset.json"

    train_ds = SEDataset(DATASET_PATH, split="train")
    test_ds  = SEDataset(DATASET_PATH, split="test")

    print(f"train samples: {len(train_ds)}")
    print(f"test  samples: {len(test_ds)}")

    print("\n[ 샘플 예시 ]")
    for i in [0, 1, 2]:
        s = train_ds[i]
        print(f"  conv_id={s['conv_id']}, turn_idx={s['turn_idx']}, label={s['label']}")
        print(f"  text: {s['text'][:80]}...")