# ART – GRPO-Training mit LoRA

Ergänzende Trainingsskripte für **Group Relative Policy Optimization (GRPO)** mit
**Low-Rank Adaptation (LoRA)** im OpenPipe ART Framework.

## 📁 Dateien

| Datei | Beschreibung |
|-------|-------------|
| `config.py` | Trainingskonfigurationen (Model, LoRA, GRPO, Reward, Data) |
| `reward_model.py` | Reward-Modell für Agent-Bewertung (regelbasiert + modellbasiert) |
| `train_agent.py` | Haupt-Trainingsskript für GRPO mit LoRA |

## 🚀 Schnellstart

### 1. Installation

```bash
# Im ART-Repository (Dependencies sind bereits in pyproject.toml)
uv sync --extra backend
```

### 2. Training starten

```bash
# Standard-Training mit Qwen 2.5 7B
python train_agent.py

# Training mit Llama 3.1 8B
python train_agent.py --model llama

# Schneller Test-Modus (Qwen 1.5B, wenige Steps)
python train_agent.py --test-mode

# Mit eigenen Daten
python train_agent.py --train-file data/my_tasks.jsonl --eval-file data/my_eval.jsonl

# Ohne W&B-Logging
python train_agent.py --no-wandb

# Mit angepasster Learning Rate
python train_agent.py --learning-rate 1e-5 --max-steps 500
```

## ⚙️ Konfiguration

### Vordefinierte Konfigurationen

```python
from config import get_qwen_config, get_llama_config, get_small_test_config

# Qwen 2.5 7B (Standard)
config = get_qwen_config()

# Llama 3.1 8B
config = get_llama_config()

# Test-Modus (Qwen 1.5B)
config = get_small_test_config()
```

### Benutzerdefinierte Konfiguration

```python
from config import TrainingConfig, ModelConfig, LoRAConfig, GRPOConfig

config = TrainingConfig(
    model=ModelConfig(
        model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
        load_in_4bit=True,
    ),
    lora=LoRAConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
    ),
    grpo=GRPOConfig(
        learning_rate=5e-6,
        num_generations=4,
        beta=0.04,
        max_steps=1000,
    ),
    experiment_name="my-experiment",
)
```

## 🎯 Reward-Modell

Das Reward-Modell bewertet Agent-Antworten anhand von 5 Kriterien:

| Kriterium | Gewicht | Beschreibung |
|-----------|---------|-------------|
| `correctness` | 1.0 | Fachliche Korrektheit der Antwort |
| `format` | 0.3 | Struktur und Formatierung |
| `helpfulness` | 0.5 | Nützlichkeit und Verständlichkeit |
| `safety` | 0.8 | Sicherheit (keine schädlichen Inhalte) |
| `tool_usage` | 0.4 | Korrekte Nutzung von Tools |

### Verwendung

```python
from reward_model import AgentRewardModel, create_reward_function

# Regelbasiert (schnell, keine GPU nötig)
reward_model = AgentRewardModel(
    reward_weights={"correctness": 1.0, "safety": 0.8},
    use_model=False,
)

# Reward berechnen
result = reward_model.compute_reward(
    prompt="Erkläre Quantencomputing",
    completion="Quantencomputing nutzt Qubits...",
    ground_truth="Quantencomputing verwendet Quantenbits...",
)
print(f"Total Reward: {result['total']:.3f}")

# Als TRL-kompatible Reward-Funktion
reward_func = create_reward_function(reward_model)
```

## 📊 Trainingsdaten-Format

JSONL-Datei mit einem `prompt`-Feld pro Zeile:

```jsonl
{"prompt": "Erkläre den Unterschied zwischen GRPO und PPO."}
{"prompt": "Schreibe eine Python-Funktion für Binary Search."}
{"prompt": "Was ist der Unterschied zwischen TCP und UDP?"}
```

## 🔧 Abhängigkeiten

Alle Abhängigkeiten sind bereits im ART-`pyproject.toml` unter dem `backend`-Extra definiert:

- `transformers>=5.2.0`
- `peft>=0.14.0`
- `trl==0.20.0`
- `torch==2.11.0`
- `bitsandbytes>=0.45.2`
- `datasets` (via HuggingFace)
- `accelerate==1.7.0`

## 📝 Hinweise

- **GPU**: Für 7B/8B-Modelle wird eine GPU mit ≥24GB VRAM empfohlen (mit 4-bit Quantisierung).
- **Test-Modus**: `--test-mode` nutzt Qwen 1.5B ohne Quantisierung – läuft auch auf kleineren GPUs.
- **Daten**: Ohne `--train-file` wird ein synthetischer Demo-Datensatz verwendet.
- **W&B**: Standardmäßig wird zu Weights & Biases geloggt. Mit `--no-wandb` deaktivierbar.
