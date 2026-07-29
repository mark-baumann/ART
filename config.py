"""
ART – Agent Reinforcement Trainer
Trainingskonfigurationen für GRPO-Training mit LoRA.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class ModelConfig:
    """Konfiguration für das zu trainierende Modell."""

    # Modell-Identifier (HuggingFace Hub oder lokaler Pfad)
    model_name_or_path: str = "Qwen/Qwen2.5-7B-Instruct"

    # Alternativ: Llama-basierte Modelle
    # model_name_or_path: str = "meta-llama/Llama-3.1-8B-Instruct"

    # Tokenizer (default = model_name_or_path)
    tokenizer_name_or_path: Optional[str] = None

    # Quantisierung (4-bit für Speichereffizienz)
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True

    # Attention-Implementierung
    attn_implementation: str = "flash_attention_2"

    # Trust remote code (für benutzerdefinierte Modelle)
    trust_remote_code: bool = False


@dataclass
class LoRAConfig:
    """LoRA (Low-Rank Adaptation) Konfiguration."""

    # LoRA Rank
    r: int = 16

    # LoRA Alpha (Skalierungsfaktor)
    lora_alpha: int = 32

    # Zielmodule für LoRA-Adapter
    target_modules: list[str] = field(default_factory=lambda: [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ])

    # LoRA Dropout
    lora_dropout: float = 0.05

    # Bias-Typ
    bias: str = "none"

    # Task-Typ
    task_type: str = "CAUSAL_LM"


@dataclass
class GRPOConfig:
    """GRPO (Group Relative Policy Optimization) Konfiguration."""

    # Anzahl der generierten Samples pro Prompt (Gruppengröße)
    num_generations: int = 4

    # Maximale Prompt-Länge in Tokens
    max_prompt_length: int = 2048

    # Maximale Completion-Länge in Tokens
    max_completion_length: int = 1024

    # Temperatur für Sampling
    temperature: float = 0.9

    # Top-p Sampling
    top_p: float = 1.0

    # Anzahl der Epochen pro GRPO-Schritt
    num_epochs: int = 1

    # Learning Rate
    learning_rate: float = 5e-6

    # Beta (KL-Divergence-Koeffizient)
    beta: float = 0.04

    # Gradient Accumulation Steps
    gradient_accumulation_steps: int = 4

    # Per-Device Train Batch Size
    per_device_train_batch_size: int = 2

    # Optimizer
    optim: str = "adamw_8bit"

    # LR Scheduler
    lr_scheduler_type: str = "cosine"

    # Warmup Ratio
    warmup_ratio: float = 0.1

    # Weight Decay
    weight_decay: float = 0.01

    # Max Steps (-1 = voller Datensatz)
    max_steps: int = -1

    # Logging Steps
    logging_steps: int = 10

    # Save Steps
    save_steps: int = 100

    # Evaluation Strategy
    eval_strategy: str = "steps"
    eval_steps: int = 100

    # Mixed Precision
    bf16: bool = True
    fp16: bool = False

    # Gradient Checkpointing
    gradient_checkpointing: bool = True

    # Seed
    seed: int = 42

    # Report To (wandb, tensorboard, etc.)
    report_to: str = "wandb"

    # Output Directory
    output_dir: str = "./output/grpo-lora"


@dataclass
class RewardConfig:
    """Konfiguration für das Reward-Modell."""

    # Reward-Modell Identifier
    reward_model_name_or_path: str = "Qwen/Qwen2.5-7B-Instruct"

    # Maximale Sequenzlänge für Reward-Berechnung
    max_length: int = 4096

    # Reward-Typen und ihre Gewichte
    reward_weights: dict[str, float] = field(default_factory=lambda: {
        "correctness": 1.0,       # Korrektheit der Antwort
        "format": 0.3,            # Einhaltung des Ausgabeformats
        "helpfulness": 0.5,       # Hilfreichkeit
        "safety": 0.8,            # Sicherheit
        "tool_usage": 0.4,        # Korrekte Tool-Nutzung
    })

    # Schwellwerte für Reward-Komponenten
    correctness_threshold: float = 0.5
    safety_threshold: float = 0.3


@dataclass
class DataConfig:
    """Konfiguration für Trainingsdaten."""

    # Pfad zum Trainingsdatensatz (JSONL mit "prompt"-Feld)
    train_file: str = "data/train.jsonl"

    # Pfad zum Evaluierungsdatensatz
    eval_file: str = "data/eval.jsonl"

    # Dataset-Format
    dataset_format: Literal["standard", "sharegpt", "custom"] = "standard"

    # Prompt-Template
    prompt_template: str = (
        "<|im_start|>system\n"
        "Du bist ein hilfreicher KI-Agent. Nutze Tools wenn nötig und "
        "antworte präzise und korrekt.<|im_end|>\n"
        "<|im_start|>user\n"
        "{prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


@dataclass
class TrainingConfig:
    """Gesamtkonfiguration für das GRPO-Training."""

    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    data: DataConfig = field(default_factory=DataConfig)

    # Experiment-Name (für Logging)
    experiment_name: str = "art-grpo-lora"

    # Resume from checkpoint
    resume_from_checkpoint: Optional[str] = None


# Vordefinierte Konfigurationen für verschiedene Setups

def get_qwen_config() -> TrainingConfig:
    """Standard-Konfiguration für Qwen2.5-7B."""
    return TrainingConfig(
        model=ModelConfig(model_name_or_path="Qwen/Qwen2.5-7B-Instruct"),
        experiment_name="art-qwen-7b-grpo",
    )


def get_llama_config() -> TrainingConfig:
    """Standard-Konfiguration für Llama-3.1-8B."""
    return TrainingConfig(
        model=ModelConfig(
            model_name_or_path="meta-llama/Llama-3.1-8B-Instruct",
        ),
        lora=LoRAConfig(
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        ),
        data=DataConfig(
            prompt_template=(
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                "Du bist ein hilfreicher KI-Agent. Nutze Tools wenn nötig und "
                "antworte präzise und korrekt.<|eot_id|>"
                "<|start_header_id|>user<|end_header_id|>\n\n"
                "{prompt}<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>\n\n"
            ),
        ),
        experiment_name="art-llama-8b-grpo",
    )


def get_small_test_config() -> TrainingConfig:
    """Kleine Test-Konfiguration für schnelle Experimente."""
    return TrainingConfig(
        model=ModelConfig(
            model_name_or_path="Qwen/Qwen2.5-1.5B-Instruct",
            load_in_4bit=False,  # Kleines Modell, kein 4-bit nötig
        ),
        lora=LoRAConfig(r=8, lora_alpha=16),
        grpo=GRPOConfig(
            num_generations=2,
            max_prompt_length=512,
            max_completion_length=256,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=2,
            max_steps=50,
            save_steps=25,
            eval_steps=25,
            logging_steps=5,
        ),
        experiment_name="art-test-grpo",
    )
