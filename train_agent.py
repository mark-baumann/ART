#!/usr/bin/env python3
"""
ART – Agent Reinforcement Trainer
GRPO-Training für Qwen/Llama mit LoRA.

Führt Group Relative Policy Optimization (GRPO) Training mit
Low-Rank Adaptation (LoRA) durch. Unterstützt Qwen2.5 und Llama 3.1
Modellfamilien.

Verwendung:
    python train_agent.py                          # Standard-Training (Qwen 7B)
    python train_agent.py --model llama            # Llama 3.1 8B
    python train_agent.py --model qwen --test-mode # Schneller Test-Modus
    python train_agent.py --config my_config.py    # Benutzerdefinierte Config
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    TrainingArguments,
)
from trl import GRPOConfig, GRPOTrainer

# Lokale Imports
from config import (
    GRPOConfig as LocalGRPOConfig,
    LoRAConfig as LocalLoRAConfig,
    ModelConfig,
    TrainingConfig,
    get_llama_config,
    get_qwen_config,
    get_small_test_config,
)
from reward_model import AgentRewardModel, create_reward_function

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset Utilities
# ---------------------------------------------------------------------------


def load_training_data(
    train_file: str,
    eval_file: Optional[str] = None,
    prompt_template: str = "{prompt}",
    max_samples: Optional[int] = None,
) -> tuple[Dataset, Optional[Dataset]]:
    """
    Lädt Trainings- und Evaluierungsdaten.

    Erwartet JSONL-Dateien mit einem "prompt"-Feld pro Zeile.

    Args:
        train_file: Pfad zur Trainings-JSONL-Datei.
        eval_file: Pfad zur Evaluierungs-JSONL-Datei.
        prompt_template: Template für die Prompt-Formatierung.
        max_samples: Maximale Anzahl Samples (für Tests).

    Returns:
        Tuple aus (train_dataset, eval_dataset).
    """
    logger.info(f"Lade Trainingsdaten aus {train_file}")

    if not os.path.exists(train_file):
        logger.warning(
            f"Trainingsdatei {train_file} nicht gefunden. "
            f"Erstelle synthetischen Demo-Datensatz."
        )
        return _create_demo_dataset(prompt_template, max_samples or 100)

    train_data = _load_jsonl(train_file, prompt_template, max_samples)
    train_dataset = Dataset.from_list(train_data)

    eval_dataset = None
    if eval_file and os.path.exists(eval_file):
        eval_data = _load_jsonl(eval_file, prompt_template, max_samples)
        eval_dataset = Dataset.from_list(eval_data)

    logger.info(
        f"Geladen: {len(train_dataset)} Trainings-Samples"
        + (f", {len(eval_dataset)} Eval-Samples" if eval_dataset else "")
    )
    return train_dataset, eval_dataset


def _load_jsonl(
    filepath: str,
    prompt_template: str,
    max_samples: Optional[int] = None,
) -> list[dict[str, str]]:
    """Lädt und formatiert JSONL-Daten."""
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            try:
                item = json.loads(line.strip())
                prompt = item.get("prompt", "")
                formatted = prompt_template.format(prompt=prompt)
                data.append({"prompt": formatted})
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Überspringe Zeile {i}: {e}")
    return data


def _create_demo_dataset(
    prompt_template: str,
    num_samples: int = 100,
) -> tuple[Dataset, Optional[Dataset]]:
    """Erstellt einen synthetischen Demo-Datensatz für Tests."""
    demo_prompts = [
        "Erkläre den Unterschied zwischen supervised und reinforcement learning.",
        "Schreibe eine Python-Funktion, die Fibonacci-Zahlen berechnet.",
        "Was ist der Unterschied zwischen GRPO und PPO?",
        "Erstelle eine SQL-Abfrage, die alle Benutzer mit Admin-Rechten findet.",
        "Beschreibe den Ablauf einer HTTP-Anfrage vom Browser zum Server.",
        "Wie funktioniert die LoRA (Low-Rank Adaptation) Methode?",
        "Erkläre das Konzept der Attention in Transformer-Modellen.",
        "Schreibe einen Bash-Befehl, der alle .log-Dateien der letzten 7 Tage findet.",
        "Was sind die Vorteile von Type Hints in Python?",
        "Beschreibe den Unterschied zwischen Git Merge und Git Rebase.",
        "Wie implementiert man einen LRU-Cache in Python?",
        "Erkläre das CAP-Theorem in verteilten Systemen.",
        "Schreibe eine Regex, die alle E-Mail-Adressen in einem Text findet.",
        "Was ist der Unterschied zwischen Docker und einer VM?",
        "Erkläre den Gradient Descent Algorithmus.",
        "Wie funktioniert JWT (JSON Web Token) Authentifizierung?",
        "Schreibe einen Kubernetes Deployment YAML für eine Web-App.",
        "Was ist der Unterschied zwischen TCP und UDP?",
        "Erkläre das Konzept von Dependency Injection.",
        "Wie optimiert man eine langsame PostgreSQL-Abfrage?",
    ]

    # Wiederhole Prompts um genügend Samples zu haben
    prompts = (demo_prompts * ((num_samples // len(demo_prompts)) + 1))[:num_samples]

    train_data = [
        {"prompt": prompt_template.format(prompt=p)}
        for p in prompts[: int(num_samples * 0.8)]
    ]
    eval_data = [
        {"prompt": prompt_template.format(prompt=p)}
        for p in prompts[int(num_samples * 0.8) :]
    ]

    train_dataset = Dataset.from_list(train_data)
    eval_dataset = Dataset.from_list(eval_data)

    logger.info(f"Demo-Datensatz erstellt: {len(train_dataset)} train, {len(eval_dataset)} eval")
    return train_dataset, eval_dataset


# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------


def load_model_and_tokenizer(
    config: ModelConfig,
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Lädt das Basis-Modell und den Tokenizer.

    Args:
        config: ModelConfig mit Modell-Parametern.

    Returns:
        Tuple aus (model, tokenizer).
    """
    logger.info(f"Lade Modell: {config.model_name_or_path}")

    # Quantisierungskonfiguration
    bnb_config = None
    if config.load_in_4bit:
        compute_dtype = getattr(torch, config.bnb_4bit_compute_dtype)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=config.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,
        )
        logger.info("4-Bit Quantisierung aktiviert")

    # Tokenizer
    tokenizer_path = config.tokenizer_name_or_path or config.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=config.trust_remote_code,
    )

    # Padding-Token setzen falls nicht vorhanden
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("pad_token auf eos_token gesetzt")

    # Modell
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": config.trust_remote_code,
    }

    if config.attn_implementation:
        model_kwargs["attn_implementation"] = config.attn_implementation

    if bnb_config:
        model_kwargs["quantization_config"] = bnb_config
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        **model_kwargs,
    )

    logger.info(f"Modell geladen: {type(model).__name__}")
    return model, tokenizer


def apply_lora(
    model: PreTrainedModel,
    lora_config: LocalLoRAConfig,
) -> PreTrainedModel:
    """
    Wendet LoRA-Adapter auf das Modell an.

    Args:
        model: Das Basis-Modell.
        lora_config: LoRA-Konfiguration.

    Returns:
        Modell mit LoRA-Adaptern.
    """
    logger.info(
        f"Wende LoRA an: r={lora_config.r}, alpha={lora_config.lora_alpha}"
    )

    # Modell für k-bit Training vorbereiten
    model = prepare_model_for_kbit_training(model)

    # LoRA-Konfiguration
    peft_config = LoraConfig(
        r=lora_config.r,
        lora_alpha=lora_config.lora_alpha,
        target_modules=lora_config.target_modules,
        lora_dropout=lora_config.lora_dropout,
        bias=lora_config.bias,
        task_type=lora_config.task_type,
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    return model


# ---------------------------------------------------------------------------
# GRPO Training
# ---------------------------------------------------------------------------


def create_grpo_config(config: TrainingConfig) -> GRPOConfig:
    """
    Erstellt eine TRL GRPOConfig aus der lokalen Konfiguration.

    Args:
        config: TrainingConfig mit allen Parametern.

    Returns:
        TRL GRPOConfig.
    """
    grpo = config.grpo

    return GRPOConfig(
        # GRPO-spezifisch
        num_generations=grpo.num_generations,
        max_prompt_length=grpo.max_prompt_length,
        max_completion_length=grpo.max_completion_length,
        temperature=grpo.temperature,
        # Training
        learning_rate=grpo.learning_rate,
        num_train_epochs=grpo.num_epochs,
        per_device_train_batch_size=grpo.per_device_train_batch_size,
        gradient_accumulation_steps=grpo.gradient_accumulation_steps,
        # Optimizer
        optim=grpo.optim,
        lr_scheduler_type=grpo.lr_scheduler_type,
        warmup_ratio=grpo.warmup_ratio,
        weight_decay=grpo.weight_decay,
        # Logging & Saving
        logging_steps=grpo.logging_steps,
        save_steps=grpo.save_steps,
        eval_strategy=grpo.eval_strategy,
        eval_steps=grpo.eval_steps,
        # Precision
        bf16=grpo.bf16,
        fp16=grpo.fp16,
        gradient_checkpointing=grpo.gradient_checkpointing,
        # Output
        output_dir=grpo.output_dir,
        report_to=grpo.report_to,
        run_name=config.experiment_name,
        seed=grpo.seed,
        # GRPO Beta
        beta=grpo.beta,
    )


def train(
    config: TrainingConfig,
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset] = None,
) -> str:
    """
    Führt das GRPO-Training durch.

    Args:
        config: Vollständige Trainingskonfiguration.
        train_dataset: Trainingsdatensatz.
        eval_dataset: Optionaler Evaluierungsdatensatz.

    Returns:
        Pfad zum gespeicherten Modell.
    """
    logger.info("=" * 60)
    logger.info(f"Starte GRPO-Training: {config.experiment_name}")
    logger.info("=" * 60)

    # 1. Modell & Tokenizer laden
    model, tokenizer = load_model_and_tokenizer(config.model)

    # 2. LoRA anwenden
    model = apply_lora(model, config.lora)

    # 3. Reward-Modell erstellen
    reward_model = AgentRewardModel(
        model_name_or_path=config.reward.reward_model_name_or_path,
        reward_weights=config.reward.reward_weights,
        use_model=False,  # Regelbasiert für Geschwindigkeit
    )
    reward_func = create_reward_function(reward_model)

    # 4. GRPO-Konfiguration
    grpo_config = create_grpo_config(config)

    # 5. GRPO-Trainer
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=grpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        reward_funcs=reward_func,
    )

    # 6. Training
    logger.info("Beginne Training...")
    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.info("Training durch Benutzer unterbrochen.")
    except Exception as e:
        logger.error(f"Fehler während des Trainings: {e}")
        raise

    # 7. Modell speichern
    output_dir = config.grpo.output_dir
    logger.info(f"Speichere Modell nach {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    logger.info("Training abgeschlossen!")
    return output_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """CLI-Argumente parsen."""
    parser = argparse.ArgumentParser(
        description="ART – GRPO-Training für Agenten mit LoRA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  python train_agent.py                          # Qwen 2.5 7B (Standard)
  python train_agent.py --model llama            # Llama 3.1 8B
  python train_agent.py --test-mode              # Schneller Test (Qwen 1.5B)
  python train_agent.py --train-file data.jsonl  # Eigene Daten
  python train_agent.py --output-dir ./my_model  # Ausgabepfad
        """,
    )

    parser.add_argument(
        "--model",
        choices=["qwen", "llama"],
        default="qwen",
        help="Modellfamilie (default: qwen)",
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Schneller Test-Modus mit kleinem Modell",
    )
    parser.add_argument(
        "--train-file",
        type=str,
        default="data/train.jsonl",
        help="Pfad zur Trainings-JSONL-Datei",
    )
    parser.add_argument(
        "--eval-file",
        type=str,
        default=None,
        help="Pfad zur Evaluierungs-JSONL-Datei",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Ausgabeverzeichnis für Checkpoints",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximale Anzahl Trainings-Samples",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="W&B-Logging deaktivieren",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning Rate (überschreibt Config)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximale Trainingsschritte (überschreibt Config)",
    )

    return parser.parse_args()


def main() -> None:
    """Hauptfunktion."""
    args = parse_args()

    # Konfiguration auswählen
    if args.test_mode:
        config = get_small_test_config()
        logger.info("Test-Modus: Verwende kleine Konfiguration")
    elif args.model == "llama":
        config = get_llama_config()
        logger.info("Llama-Modus: Verwende Llama 3.1 8B Konfiguration")
    else:
        config = get_qwen_config()
        logger.info("Qwen-Modus: Verwende Qwen 2.5 7B Konfiguration")

    # CLI-Überschreibungen
    if args.output_dir:
        config.grpo.output_dir = args.output_dir
    if args.learning_rate is not None:
        config.grpo.learning_rate = args.learning_rate
    if args.max_steps is not None:
        config.grpo.max_steps = args.max_steps
    if args.no_wandb:
        config.grpo.report_to = "none"

    # Daten laden
    train_dataset, eval_dataset = load_training_data(
        train_file=args.train_file,
        eval_file=args.eval_file,
        prompt_template=config.data.prompt_template,
        max_samples=args.max_samples,
    )

    # Training starten
    output_dir = train(
        config=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    logger.info(f"Training abgeschlossen. Modell gespeichert in: {output_dir}")


if __name__ == "__main__":
    main()
