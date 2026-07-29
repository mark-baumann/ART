"""Unit-Tests für die Trainingskonfiguration (config.py)."""

import pytest

from config import (
    DataConfig,
    GRPOConfig,
    LoRAConfig,
    ModelConfig,
    RewardConfig,
    TrainingConfig,
    get_llama_config,
    get_qwen_config,
    get_small_test_config,
)


class TestModelConfig:
    """Tests für ModelConfig."""

    def test_default_values(self):
        """Test: Standardwerte sind korrekt."""
        cfg = ModelConfig()
        assert cfg.model_name_or_path == "Qwen/Qwen2.5-7B-Instruct"
        assert cfg.load_in_4bit is True
        assert cfg.bnb_4bit_compute_dtype == "bfloat16"
        assert cfg.bnb_4bit_quant_type == "nf4"
        assert cfg.bnb_4bit_use_double_quant is True
        assert cfg.attn_implementation == "flash_attention_2"
        assert cfg.trust_remote_code is False
        assert cfg.tokenizer_name_or_path is None

    def test_custom_model(self):
        """Test: Benutzerdefinierte Werte werden übernommen."""
        cfg = ModelConfig(
            model_name_or_path="meta-llama/Llama-3.1-8B-Instruct",
            load_in_4bit=False,
            trust_remote_code=True,
        )
        assert cfg.model_name_or_path == "meta-llama/Llama-3.1-8B-Instruct"
        assert cfg.load_in_4bit is False
        assert cfg.trust_remote_code is True


class TestLoRAConfig:
    """Tests für LoRAConfig."""

    def test_default_values(self):
        """Test: Standardwerte sind korrekt."""
        cfg = LoRAConfig()
        assert cfg.r == 16
        assert cfg.lora_alpha == 32
        assert cfg.lora_dropout == 0.05
        assert cfg.bias == "none"
        assert cfg.task_type == "CAUSAL_LM"
        assert "q_proj" in cfg.target_modules
        assert "v_proj" in cfg.target_modules

    def test_custom_rank(self):
        """Test: Benutzerdefinierter Rank."""
        cfg = LoRAConfig(r=8, lora_alpha=16)
        assert cfg.r == 8
        assert cfg.lora_alpha == 16

    def test_target_modules_is_list(self):
        """Test: target_modules ist eine Liste von Strings."""
        cfg = LoRAConfig()
        assert isinstance(cfg.target_modules, list)
        assert all(isinstance(m, str) for m in cfg.target_modules)


class TestGRPOConfig:
    """Tests für GRPOConfig."""

    def test_default_values(self):
        """Test: Standardwerte sind korrekt."""
        cfg = GRPOConfig()
        assert cfg.num_generations == 4
        assert cfg.max_prompt_length == 2048
        assert cfg.max_completion_length == 1024
        assert cfg.temperature == 0.9
        assert cfg.learning_rate == 5e-6
        assert cfg.beta == 0.04
        assert cfg.per_device_train_batch_size == 2
        assert cfg.gradient_accumulation_steps == 4
        assert cfg.bf16 is True
        assert cfg.fp16 is False
        assert cfg.seed == 42
        assert cfg.output_dir == "./output/grpo-lora"

    def test_custom_learning_rate(self):
        """Test: Benutzerdefinierte Learning Rate."""
        cfg = GRPOConfig(learning_rate=1e-4)
        assert cfg.learning_rate == 1e-4

    def test_max_steps_default(self):
        """Test: max_steps default ist -1 (voller Datensatz)."""
        cfg = GRPOConfig()
        assert cfg.max_steps == -1


class TestRewardConfig:
    """Tests für RewardConfig."""

    def test_default_values(self):
        """Test: Standardwerte sind korrekt."""
        cfg = RewardConfig()
        assert cfg.reward_model_name_or_path == "Qwen/Qwen2.5-7B-Instruct"
        assert cfg.max_length == 4096
        assert cfg.reward_weights["correctness"] == 1.0
        assert cfg.reward_weights["safety"] == 0.8
        assert cfg.correctness_threshold == 0.5
        assert cfg.safety_threshold == 0.3

    def test_custom_weights(self):
        """Test: Benutzerdefinierte Reward-Gewichte."""
        custom = {"correctness": 2.0, "format": 0.5}
        cfg = RewardConfig(reward_weights=custom)
        assert cfg.reward_weights == custom


class TestDataConfig:
    """Tests für DataConfig."""

    def test_default_values(self):
        """Test: Standardwerte sind korrekt."""
        cfg = DataConfig()
        assert cfg.train_file == "data/train.jsonl"
        assert cfg.eval_file == "data/eval.jsonl"
        assert cfg.dataset_format == "standard"
        assert "{prompt}" in cfg.prompt_template
        assert "<|im_start|>" in cfg.prompt_template

    def test_custom_files(self):
        """Test: Benutzerdefinierte Dateipfade."""
        cfg = DataConfig(
            train_file="my_train.jsonl",
            eval_file="my_eval.jsonl",
        )
        assert cfg.train_file == "my_train.jsonl"
        assert cfg.eval_file == "my_eval.jsonl"


class TestTrainingConfig:
    """Tests für TrainingConfig (Gesamtkonfiguration)."""

    def test_default_values(self):
        """Test: Standardwerte sind korrekt."""
        cfg = TrainingConfig()
        assert isinstance(cfg.model, ModelConfig)
        assert isinstance(cfg.lora, LoRAConfig)
        assert isinstance(cfg.grpo, GRPOConfig)
        assert isinstance(cfg.reward, RewardConfig)
        assert isinstance(cfg.data, DataConfig)
        assert cfg.experiment_name == "art-grpo-lora"
        assert cfg.resume_from_checkpoint is None

    def test_custom_experiment_name(self):
        """Test: Benutzerdefinierter Experiment-Name."""
        cfg = TrainingConfig(experiment_name="my-experiment")
        assert cfg.experiment_name == "my-experiment"

    def test_resume_from_checkpoint(self):
        """Test: Resume-Checkpoint wird gespeichert."""
        cfg = TrainingConfig(resume_from_checkpoint="./checkpoints/step-100")
        assert cfg.resume_from_checkpoint == "./checkpoints/step-100"


class TestPresetConfigs:
    """Tests für die vordefinierten Konfigurationen."""

    def test_get_qwen_config(self):
        """Test: Qwen-Konfiguration hat korrekte Werte."""
        cfg = get_qwen_config()
        assert cfg.model.model_name_or_path == "Qwen/Qwen2.5-7B-Instruct"
        assert cfg.experiment_name == "art-qwen-7b-grpo"

    def test_get_llama_config(self):
        """Test: Llama-Konfiguration hat korrekte Werte."""
        cfg = get_llama_config()
        assert cfg.model.model_name_or_path == "meta-llama/Llama-3.1-8B-Instruct"
        assert cfg.experiment_name == "art-llama-8b-grpo"
        # Llama hat ein anderes Prompt-Template
        assert "<|begin_of_text|>" in cfg.data.prompt_template

    def test_get_small_test_config(self):
        """Test: Test-Konfiguration hat reduzierte Werte."""
        cfg = get_small_test_config()
        assert cfg.model.model_name_or_path == "Qwen/Qwen2.5-1.5B-Instruct"
        assert cfg.model.load_in_4bit is False
        assert cfg.lora.r == 8
        assert cfg.lora.lora_alpha == 16
        assert cfg.grpo.num_generations == 2
        assert cfg.grpo.max_prompt_length == 512
        assert cfg.grpo.max_completion_length == 256
        assert cfg.grpo.max_steps == 50
        assert cfg.experiment_name == "art-test-grpo"

    def test_all_configs_are_training_config(self):
        """Test: Alle vordefinierten Konfigurationen sind TrainingConfig-Instanzen."""
        for cfg in [get_qwen_config(), get_llama_config(), get_small_test_config()]:
            assert isinstance(cfg, TrainingConfig)
