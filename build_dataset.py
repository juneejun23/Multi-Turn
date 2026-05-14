"""
SE Detection Dataset Builder
-----------------------------
SEConvo + SE-VSim 전체 데이터를 합쳐서
conversation 단위 8:2 train/test split 후
turn마다 IsMalicious 기반 label(0/1)을 붙여 JSON으로 저장합니다.

출력 형식:
{
    "stats": {
        "total_conversations": 2750,
        "train_conversations": 2200,
        "test_conversations": 550,
        "train_turns": ...,
        "test_turns": ...
    },
    "train": [
        {
            "conv_id": "seconvo_0",
            "source": "seconvo",
            "is_malicious": true,
            "turns": [
                {"turn_idx": 1, "name": "Kelly", "message": "...", "label": 1},
                {"turn_idx": 2, "name": "Monica", "message": "...", "label": 1},
                ...
            ]
        },
        ...
    ],
    "test": [ ... ]
}

Note:
    현재는 IsMalicious를 conversation 전체 turn에 동일하게 propagation합니다.
    (Late Labeling: malicious 대화의 앞 절반을 label=0으로 처리하는 방식도
     고려할 수 있으나, 현재는 적용하지 않습니다.)
"""

import json
import random
from pathlib import Path


# ─────────────────────────────────────────
# 설정
# ─────────────────────────────────────────

DATA_DIR    = "data"        # 이 아래 모든 .json 파일을 재귀적으로 읽음
SPLIT_RATIO = 0.9           # train 비율
RANDOM_SEED = 42
OUTPUT_PATH = "dataset.json"


# ─────────────────────────────────────────
# 로드
# ─────────────────────────────────────────

def load_all_conversations(data_dir: str) -> list[dict]:
    """
    data_dir 아래 모든 .json 파일을 재귀적으로 읽어
    conversation 리스트로 반환합니다.

    source는 파일이 속한 첫 번째 하위 폴더명을 소문자로 사용합니다.
    예) data/SEConvo/annotated_train.json -> source="seconvo"
        data/Sevsim/train_data.json       -> source="sevsim"

    각 conversation:
        {
            "conv_id"     : str,
            "source"      : str,
            "is_malicious": bool,
            "turns"       : [{"name": str, "message": str}, ...]
        }
    """
    conversations = []
    conv_idx = 0

    json_files = sorted(Path(data_dir).rglob("*.json"))
    print(f"  발견된 파일 {len(json_files)}개:")

    for path in json_files:
        # data/ 바로 아래 첫 번째 폴더명을 source로 사용
        source = path.relative_to(data_dir).parts[0].lower()
        print(f"    [{source}] {path.name}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for c in data["Conversations"]:
            is_malicious = bool(c["GroundTruth"]["IsMalicious"])
            turns = [
                {"name": t["Name"], "message": t["Message"]}
                for t in c["Conversation"]
            ]
            conversations.append({
                "conv_id"     : f"{source}_{conv_idx}",
                "source"      : source,
                "is_malicious": is_malicious,
                "turns"       : turns,
            })
            conv_idx += 1

    return conversations


# ─────────────────────────────────────────
# 샘플 생성 (turn마다 label 부여)
# ─────────────────────────────────────────

def build_labeled_conversation(conv: dict) -> dict:
    """
    conversation의 각 turn에 label(0/1)을 붙입니다.
    label = IsMalicious ? 1 : 0  (전체 turn 동일 적용)
    """
    label = 1 if conv["is_malicious"] else 0
    labeled_turns = [
        {
            "turn_idx": i + 1,
            "name"    : t["name"],
            "message" : t["message"],
            "label"   : label,
        }
        for i, t in enumerate(conv["turns"])
    ]
    return {
        "conv_id"     : conv["conv_id"],
        "source"      : conv["source"],
        "is_malicious": conv["is_malicious"],
        "turns"       : labeled_turns,
    }


# ─────────────────────────────────────────
# Split (conversation 단위)
# ─────────────────────────────────────────

def split_conversations(
    conversations: list[dict],
    ratio: float = 0.8,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """
    conversation 단위로 train/test split합니다.
    같은 conversation의 turn이 train/test에 나뉘지 않도록
    반드시 conversation 단위로 먼저 나눕니다.
    """
    random.seed(seed)
    shuffled = conversations[:]
    random.shuffle(shuffled)

    n_train = int(len(shuffled) * ratio)
    return shuffled[:n_train], shuffled[n_train:]


# ─────────────────────────────────────────
# 메인
# ─────────────────────────────────────────

def main():
    # 1. 전체 로드
    print("Loading conversations...")
    conversations = load_all_conversations(DATA_DIR)
    print(f"  총 conversation: {len(conversations)}개")

    # 2. conversation 단위 split
    train_convs, test_convs = split_conversations(
        conversations, ratio=SPLIT_RATIO, seed=RANDOM_SEED
    )
    print(f"  train: {len(train_convs)}개 / test: {len(test_convs)}개")

    # 3. turn마다 label 부여
    train_data = [build_labeled_conversation(c) for c in train_convs]
    test_data  = [build_labeled_conversation(c) for c in test_convs]

    # 4. 통계
    train_turns = sum(len(c["turns"]) for c in train_data)
    test_turns  = sum(len(c["turns"]) for c in test_data)
    train_mal   = sum(1 for c in train_data if c["is_malicious"])
    test_mal    = sum(1 for c in test_data  if c["is_malicious"])

    stats = {
        "total_conversations" : len(conversations),
        "train_conversations" : len(train_data),
        "test_conversations"  : len(test_data),
        "train_turns"         : train_turns,
        "test_turns"          : test_turns,
        "train_malicious"     : train_mal,
        "train_benign"        : len(train_data) - train_mal,
        "test_malicious"      : test_mal,
        "test_benign"         : len(test_data) - test_mal,
    }

    print("\n[ 통계 ]")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # 5. JSON 저장 (stats 제외)
    output = {
        "train": train_data,
        "test" : test_data,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n저장 완료: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()