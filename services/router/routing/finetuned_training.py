from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple
import json
import random
import re
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def collect_training_samples(
    *,
    allowed_intents: Dict[str, Dict[str, Any]],
    feedback_path: str,
    max_text_chars: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    seen = set()
    rows: List[Dict[str, Any]] = []
    source_counts: Dict[str, int] = defaultdict(int)
    class_counts: Dict[str, int] = defaultdict(int)

    for intent_id, meta in allowed_intents.items():
        base_examples = list(meta.get("examples") or [])
        if meta.get("title"):
            base_examples.append(str(meta["title"]))
        for example in base_examples:
            text = _prepare_training_text(str(example), max_text_chars=max_text_chars)
            if not text:
                continue
            key = (intent_id, text.lower())
            if key in seen:
                continue
            seen.add(key)
            rows.append({"text": text, "intent_id": intent_id, "source": "intent_examples"})
            source_counts["intent_examples"] += 1
            class_counts[intent_id] += 1

    feedback_file = Path(feedback_path)
    if feedback_file.exists() and feedback_file.is_file():
        for raw_line in feedback_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue

            final = item.get("final") or {}
            intent_id = str(final.get("intent_id") or "").strip()
            if intent_id not in allowed_intents:
                continue

            text = str(item.get("training_sample") or "").strip()
            if not text:
                text = str(item.get("transcript_text") or "").strip()
            text = _prepare_training_text(text, max_text_chars=max_text_chars)
            if not text:
                continue

            key = (intent_id, text.lower())
            if key in seen:
                continue
            seen.add(key)
            rows.append({"text": text, "intent_id": intent_id, "source": "operator_feedback"})
            source_counts["operator_feedback"] += 1
            class_counts[intent_id] += 1

    dataset_meta = {
        "source_counts": dict(source_counts),
        "class_counts": dict(class_counts),
        "feedback_path": str(feedback_file),
    }
    return rows, dataset_meta


def stratified_split(labels: List[int], val_ratio: float, random_seed: int) -> Tuple[List[int], List[int]]:
    by_class: Dict[int, List[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        by_class[int(label)].append(idx)

    rnd = random.Random(int(random_seed))
    train_idx: List[int] = []
    val_idx: List[int] = []
    val_ratio = max(0.0, min(0.5, float(val_ratio)))

    for indices in by_class.values():
        rnd.shuffle(indices)
        if len(indices) <= 1 or val_ratio <= 0.0:
            train_idx.extend(indices)
            continue
        take_val = max(1, int(len(indices) * val_ratio))
        take_val = min(take_val, len(indices) - 1)
        val_idx.extend(indices[:take_val])
        train_idx.extend(indices[take_val:])

    rnd.shuffle(train_idx)
    rnd.shuffle(val_idx)
    return train_idx, val_idx


def train_finetuned_model(
    *,
    model_name: str,
    device: str,
    finetuned_model_path: str,
    finetuned_max_length: int,
    finetuned_weight_decay: float,
    texts: List[str],
    labels: List[int],
    intent_ids: List[str],
    train_idx: List[int],
    val_idx: List[int],
    random_seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    model_path = str(finetuned_model_path or "").strip()
    if not model_path:
        raise RuntimeError("ROUTER_FINETUNED_MODEL_PATH is empty")
    if not train_idx:
        raise RuntimeError("empty train set for fine-tuned model")
    if not val_idx:
        val_idx = list(train_idx)

    train_texts = [texts[i] for i in train_idx]
    val_texts = [texts[i] for i in val_idx]
    train_labels = torch.tensor([labels[i] for i in train_idx], dtype=torch.long)
    val_labels = torch.tensor([labels[i] for i in val_idx], dtype=torch.long)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_enc = tokenizer(
        train_texts,
        truncation=True,
        padding=True,
        max_length=finetuned_max_length,
        return_tensors="pt",
    )
    val_enc = tokenizer(
        val_texts,
        truncation=True,
        padding=True,
        max_length=finetuned_max_length,
        return_tensors="pt",
    )

    train_ds = TensorDataset(train_enc["input_ids"], train_enc["attention_mask"], train_labels)
    val_ds = TensorDataset(val_enc["input_ids"], val_enc["attention_mask"], val_labels)

    batch_size = int(max(4, min(64, batch_size)))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(intent_ids),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(max(1e-6, min(1e-3, learning_rate))),
        weight_decay=float(max(0.0, min(0.2, finetuned_weight_decay))),
    )
    class_weights = _build_class_weights(train_labels, len(intent_ids)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    torch.manual_seed(int(random_seed))
    random.seed(int(random_seed))

    best_state = None
    best_val_f1 = -1.0
    best_epoch = 0
    patience = 2
    no_improve = 0
    epochs = int(max(1, min(12, epochs)))

    for epoch in range(1, epochs + 1):
        model.train()
        for input_ids, attention_mask, yb in train_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        val_metrics = _evaluate_model(model, val_loader, criterion, device)
        val_f1 = float(val_metrics.get("macro_f1", 0.0))
        if val_f1 > best_val_f1 + 1e-6:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            break

    if best_state is None:
        raise RuntimeError("fine-tuning failed: no best checkpoint")

    model.load_state_dict(best_state)
    train_eval_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    train_metrics = _evaluate_model(model, train_eval_loader, criterion, device)
    val_metrics = _evaluate_model(model, val_loader, criterion, device)

    save_dir = Path(model_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))

    trained_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    meta_payload = {
        "model_name": model_name,
        "intent_ids": intent_ids,
        "trained_at": trained_at,
        "max_length": finetuned_max_length,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
    }
    (save_dir / "intent_ids.json").write_text(
        json.dumps(meta_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report = {
        "enabled": True,
        "best_epoch": best_epoch,
        "model_path": str(save_dir),
        "metrics": {
            "train": train_metrics,
            "val": val_metrics,
            "epochs_requested": epochs,
        },
        "dataset": {
            "samples_total": len(texts),
            "samples_train": len(train_idx),
            "samples_val": len(val_idx),
        },
    }
    artifact = {
        "enabled": True,
        "model_path": str(save_dir),
        "intent_ids": intent_ids,
        "trained_at": trained_at,
        "max_length": finetuned_max_length,
        "metrics": report["metrics"],
        "dataset": report["dataset"],
    }
    return report, artifact


def _prepare_training_text(text: str, *, max_text_chars: int) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) < 6:
        return ""
    if len(cleaned) > max_text_chars:
        cleaned = _extract_text_with_context(cleaned, max_text_chars)
    return cleaned


def _evaluate_model(
    model: AutoModelForSequenceClassification,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> Dict[str, float]:
    model.eval()
    losses: List[float] = []
    preds: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []
    with torch.inference_mode():
        for input_ids, attention_mask, yb in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            yb = yb.to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            loss = criterion(logits, yb)
            losses.append(float(loss.item()))
            preds.append(torch.argmax(logits, dim=1).detach().cpu())
            targets.append(yb.detach().cpu())

    pred = torch.cat(preds, dim=0) if preds else torch.empty(0, dtype=torch.long)
    target = torch.cat(targets, dim=0) if targets else torch.empty(0, dtype=torch.long)
    acc = float((pred == target).float().mean().item()) if target.numel() > 0 else 0.0
    macro_precision, macro_recall, macro_f1 = _macro_precision_recall_f1(pred, target)
    return {
        "loss": round(sum(losses) / max(1, len(losses)), 6),
        "accuracy": round(acc, 6),
        "macro_precision": round(macro_precision, 6),
        "macro_recall": round(macro_recall, 6),
        "macro_f1": round(macro_f1, 6),
    }


def _build_class_weights(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    counts = torch.bincount(labels, minlength=num_classes).float().clamp(min=1.0)
    inv = 1.0 / counts
    return inv / inv.mean()


def _macro_precision_recall_f1(pred: torch.Tensor, target: torch.Tensor) -> Tuple[float, float, float]:
    if target.numel() == 0:
        return 0.0, 0.0, 0.0
    labels = sorted({int(x.item()) for x in target})
    precision_scores: List[float] = []
    recall_scores: List[float] = []
    f1_scores: List[float] = []
    for label in labels:
        p = pred == label
        t = target == label
        tp = float((p & t).sum().item())
        fp = float((p & ~t).sum().item())
        fn = float((~p & t).sum().item())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision_scores.append(precision)
        recall_scores.append(recall)
        if precision + recall == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(2 * precision * recall / (precision + recall))

    macro_precision = float(sum(precision_scores) / max(1, len(precision_scores)))
    macro_recall = float(sum(recall_scores) / max(1, len(recall_scores)))
    macro_f1 = float(sum(f1_scores) / max(1, len(f1_scores)))
    return macro_precision, macro_recall, macro_f1


def _extract_text_with_context(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    start_chars = int(max_chars * 0.6)
    end_chars = int(max_chars * 0.4)
    return text[:start_chars] + "\n[...]\n" + text[-end_chars:]
