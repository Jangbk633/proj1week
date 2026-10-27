# === [파일 최상단 import 목록] ===
import gc
import inspect
import json
import os
from pathlib import Path
import sqlite3

import torch
from datasets import Dataset
from dotenv import load_dotenv
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from transformers.models.llama.configuration_llama import LlamaConfig

from preprocessing import clean_article_content

# === [단독 실행용 기본 경로 설정] ===
ROOT_DIR = Path(__file__).resolve().parent
sft_jsonl_path = ROOT_DIR / "summarizer_sft_train.jsonl"

# 만약 summarizer_sft_train.jsonl이 아직 없으면 news.db에서 150건 추출해 기본 틀 생성
if not sft_jsonl_path.exists():
    con = sqlite3.connect(str(ROOT_DIR / "news.db"))
    rows = con.execute("SELECT PK, TITLE, CONTENT FROM NEWS WHERE LENGTH(CONTENT) >= 150 LIMIT 150").fetchall()
    con.close()
    with open(sft_jsonl_path, "w", encoding="utf-8") as f:
        for pk, title, content in rows:
            body = clean_article_content(content, max_chars=1200)
            item = {
                "news_pk": pk,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "너는 한국어 뉴스 기사를 정확하고 간결하게 요약하는 AI 요약기야. "
                            "기사에 없는 내용은 절대 지어내지 말고, 핵심 사실과 요지를 한국어 2~3문장으로만 요약해."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"[제목] {title}\n[본문]\n{body}\n\n위 기사의 핵심 내용을 2~3문장으로 요약해 줘.",
                    },
                    {"role": "assistant", "content": ""},
                ],
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def train_kanana_summarizer_qlora(
    train_jsonl: Path = sft_jsonl_path,
    base_model_id: str = "kakaocorp/kanana-nano-2.1b-instruct",
    output_dir: Path = ROOT_DIR / "kanana_summarizer_full_ft_model",
    openai_teacher_jsonl: Path = ROOT_DIR / "summarizer_sft_openai_teacher.jsonl",
    openai_model: str | None = None,
    force_regenerate_teacher: bool = False,
    num_train_epochs: int = 1,
    learning_rate: float = 2e-5,
    run_training: bool = True,
) -> dict:
    """
    [OpenAI API Teacher 정답(True Value) 기반 요약기 Full Fine-tuning 파이프라인 - 16GB VRAM 최적화]
    1. OpenAI API를 호출하여 각 기사의 정답 요약문(True Value)을 생성하고 `openai_teacher_jsonl`에 캐싱합니다.
    2. 프롬프트(system + user) 토큰은 label=-100으로 마스킹하여, 오직 OpenAI API가 생성한 정답 요약(assistant)
       토큰에 대해서만 Cross-Entropy Loss를 계산합니다.
    """
    import gc
    import inspect
    import json
    import os
    from pathlib import Path

    # 1. .env에서 OpenAI API 설정 로드
    env_path = ROOT_DIR / ".env"
    if env_path.exists():
        for _line in env_path.read_text(encoding="utf-8").splitlines():
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())
    try:
        from dotenv import load_dotenv

        load_dotenv(env_path, override=False)
    except ImportError:
        pass

    active_oai_key = os.environ.get("OPENAI_API_KEY", "").strip()
    active_oai_url = (os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").strip()
    active_oai_model = (openai_model or os.environ.get("OPENAI_MODEL") or "gpt-4o-mini").strip()
    if "api.openai.com" in active_oai_url and ("/" in active_oai_model or not active_oai_model):
        active_oai_model = "gpt-4o-mini"

    # 2. 원본 SFT 데이터셋 로드 후 OpenAI API 결과로 정답(True Value) 구축/캐싱
    raw_items = [
        json.loads(line)
        for line in Path(train_jsonl).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    teacher_items = []
    if Path(openai_teacher_jsonl).exists() and not force_regenerate_teacher:
        teacher_items = [
            json.loads(line)
            for line in Path(openai_teacher_jsonl).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        print(f"[OpenAI Teacher] 캐시된 OpenAI 정답 요약 데이터셋 로드 완료: {len(teacher_items)}건 ({openai_teacher_jsonl.name})")

    if len(teacher_items) < len(raw_items):
        if not active_oai_key:
            raise ValueError("OPENAI_API_KEY가 설정되어 있지 않습니다. .env 파일에 OPENAI_API_KEY를 설정해 주세요.")

        from openai import OpenAI

        oai_client = OpenAI(base_url=active_oai_url, api_key=active_oai_key)
        print(f"[OpenAI Teacher] OpenAI API({active_oai_model})로 정답 요약(True Value) 생성 중 ({len(raw_items)}건)...")

        cached_by_pk = {item.get("news_pk"): item for item in teacher_items if item.get("news_pk") is not None}
        updated_items = []

        for idx, item in enumerate(raw_items, 1):
            pk = item.get("news_pk")
            if pk in cached_by_pk and not force_regenerate_teacher:
                updated_items.append(cached_by_pk[pk])
                continue

            msgs = item["messages"]
            sys_msg = next((m["content"] for m in msgs if m["role"] == "system"), "")
            usr_msg = next((m["content"] for m in msgs if m["role"] == "user"), "")

            try:
                resp = oai_client.chat.completions.create(
                    model=active_oai_model,
                    messages=[
                        {"role": "system", "content": sys_msg},
                        {"role": "user", "content": usr_msg},
                    ],
                    temperature=0.1,
                    max_tokens=220,
                    timeout=30.0,
                )
                openai_true_summary = (resp.choices[0].message.content or "").strip()
            except Exception as e:
                print(f"  [경고] {idx}번째 샘플 OpenAI 호출 실패({e}) -> 기존 요약문 유지")
                openai_true_summary = next((m["content"] for m in msgs if m["role"] == "assistant"), "")

            new_item = dict(item)
            new_item["target_summary"] = openai_true_summary
            new_item["teacher_model"] = active_oai_model
            new_item["messages"] = [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": usr_msg},
                {"role": "assistant", "content": openai_true_summary},  # True Value (OpenAI 정답 라벨)
            ]
            updated_items.append(new_item)

            if idx % 20 == 0 or idx == len(raw_items):
                print(f"  - 진행률: {idx} / {len(raw_items)}건 완료")

        teacher_items = updated_items
        with open(openai_teacher_jsonl, "w", encoding="utf-8") as f:
            for it in teacher_items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        print(f"[OpenAI Teacher] OpenAI 정답 데이터셋 저장 완료: {openai_teacher_jsonl}")

    ft_config_spec = {
        "base_model": base_model_id,
        "teacher_model": active_oai_model,
        "ground_truth_source": f"OpenAI API ({openai_teacher_jsonl.name})",
        "loss_masking": "Prompt masked (-100), Loss computed ONLY on OpenAI target summary tokens",
        "training_type": "Full Fine-Tuning (All Parameters Updated)",
        "precision": "Pure bfloat16 (torch.bfloat16)",
        "optimizer": "adafactor",
        "gradient_checkpointing": True,
        "learning_rate": learning_rate,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "max_seq_length": 1024,
        "train_samples": len(teacher_items),
        "output_dir": str(output_dir),
    }

    if not run_training:
        print("[안내] Full Fine-tuning 하이퍼파라미터 구성 완료 (실제 학습을 시작하려면 run_training=True로 호출하세요).")
        for k, v in ft_config_spec.items():
            print(f"  - {k}: {v}")
        return ft_config_spec

    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    from transformers.models.llama.configuration_llama import LlamaConfig

    # [핵심 패치] Kanana-Nano-2.1B 비표준 차원 검증 우회
    LlamaConfig.validate_architecture = lambda self: None
    if hasattr(LlamaConfig, "validate"):
        LlamaConfig.validate = lambda *args, **kwargs: None

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"[Full FT] 토크나이저 및 베이스 모델 로드 중: {base_model_id} (dtype=bfloat16)...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_load_kwargs = dict(
        pretrained_model_name_or_path=base_model_id,
        device_map={"": 0} if torch.cuda.is_available() else "auto",
        trust_remote_code=True,
    )
    if "dtype" in inspect.signature(AutoModelForCausalLM.from_pretrained).parameters:
        model_load_kwargs["dtype"] = torch.bfloat16
    else:
        model_load_kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(**model_load_kwargs)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    for param in model.parameters():
        param.requires_grad = True

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Full FT] 학습 대상 파라미터: {trainable_params:,} / {total_params:,} (100.0%)")

    # 3. OpenAI 정답 요약(assistant) 토큰에 대해서만 Loss를 구하도록 토큰화 & 마스킹(-100)
    max_seq_len = 1024

    def tokenize_with_openai_target_mask(example):
        msgs = example["messages"]
        prompt_msgs = msgs[:-1]  # system + user (입력 프롬프트)
        full_msgs = msgs         # system + user + assistant (OpenAI 정답 요약 포함)

        prompt_text = tokenizer.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True
        )
        full_text = tokenizer.apply_chat_template(
            full_msgs, tokenize=False, add_generation_prompt=False
        )

        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_encoded = tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_seq_len,
        )
        input_ids = full_encoded["input_ids"]
        attention_mask = full_encoded["attention_mask"]

        # 프롬프트 구간은 -100으로 마스킹하고, OpenAI API가 생성한 요약(True Value) 토큰만 정답 라벨로 남김
        prompt_len = min(len(prompt_ids), len(input_ids))
        labels = [-100] * prompt_len + input_ids[prompt_len:]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    raw_hf_ds = Dataset.from_list(teacher_items)
    tokenized_ds = raw_hf_ds.map(
        tokenize_with_openai_target_mask,
        remove_columns=raw_hf_ds.column_names,
    )

    def collate_fn(batch):
        max_len = max(len(x["input_ids"]) for x in batch)
        input_ids, attention_mask, labels = [], [], []
        for x in batch:
            pad_len = max_len - len(x["input_ids"])
            input_ids.append(x["input_ids"] + [tokenizer.pad_token_id] * pad_len)
            attention_mask.append(x["attention_mask"] + [0] * pad_len)
            labels.append(x["labels"] + [-100] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        fp16=False,
        bf16=False,
        gradient_checkpointing=True,
        optim="adafactor",
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds,
        data_collator=collate_fn,
    )

    print(f"[Full FT] OpenAI({active_oai_model}) 정답 요약(True Value) 기준 전체 가중치 파인튜닝 시작...")
    train_result = trainer.train()

    print(f"[Full FT] 학습 완료. 모델 및 토크나이저 저장 중: {output_dir}")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    ft_config_spec["train_loss"] = round(float(train_result.training_loss), 4)
    return ft_config_spec


# 즉시 학습 실행 (run_training=True 기본값 적용)
qlora_spec = train_kanana_summarizer_qlora()