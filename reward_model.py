"""
ART – Agent Reinforcement Trainer
Reward-Modell für Agent-Bewertung.

Bewertet Agent-Antworten anhand mehrerer Kriterien:
- Korrektheit (correctness)
- Format-Einhaltung (format)
- Hilfreichkeit (helpfulness)
- Sicherheit (safety)
- Tool-Nutzung (tool_usage)

Kann als eigenständiges Reward-Modell oder als Reward-Funktion
für GRPO-Training verwendet werden.
"""

from __future__ import annotations

import re
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    import torch.nn as nn
    from transformers import PreTrainedModel, PreTrainedTokenizer


class AgentRewardModel:
    """
    Reward-Modell zur Bewertung von Agent-Antworten.

    Unterstützt zwei Modi:
    1. Regelbasiert (rule-based): Schnelle Heuristiken ohne GPU
    2. Modellbasiert (model-based): Nutzt ein Language Model als Reward-Modell
    """

    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen2.5-7B-Instruct",
        reward_weights: dict[str, float] | None = None,
        use_model: bool = False,
        device: str = "auto",
    ):
        """
        Args:
            model_name_or_path: HuggingFace Modell-Identifier für modellbasierte Rewards.
            reward_weights: Gewichtung der Reward-Komponenten.
            use_model: Ob ein LM für die Bewertung genutzt werden soll.
            device: Device für das Modell ("auto", "cpu", "cuda").
        """
        self.reward_weights = reward_weights or {
            "correctness": 1.0,
            "format": 0.3,
            "helpfulness": 0.5,
            "safety": 0.8,
            "tool_usage": 0.4,
        }
        self.use_model = use_model
        self.model: Any = None
        self.tokenizer: Any = None

        if use_model:
            self._load_model(model_name_or_path, device)

    def _load_model(self, model_name_or_path: str, device: str) -> None:
        """Lädt das Reward-Modell (lazy import für optionale GPU-Nutzung)."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )
        self.model.eval()

    def compute_reward(
        self,
        prompt: str,
        completion: str,
        ground_truth: Optional[str] = None,
        tools_expected: Optional[list[str]] = None,
    ) -> dict[str, float]:
        """
        Berechnet den Reward für eine Completion.

        Args:
            prompt: Der Eingabe-Prompt.
            completion: Die generierte Antwort des Agenten.
            ground_truth: Optionale Ground-Truth für Korrektheitsbewertung.
            tools_expected: Erwartete Tool-Namen für Tool-Usage-Bewertung.

        Returns:
            Dictionary mit Einzel-Rewards und Gesamt-Reward.
        """
        rewards: dict[str, float] = {}

        if self.use_model and self.model is not None:
            rewards = self._model_based_reward(prompt, completion, ground_truth)
        else:
            rewards = self._rule_based_reward(
                prompt, completion, ground_truth, tools_expected
            )

        # Gewichteten Gesamt-Reward berechnen
        total = sum(
            rewards.get(key, 0.0) * weight
            for key, weight in self.reward_weights.items()
        )
        rewards["total"] = total
        return rewards

    def _rule_based_reward(
        self,
        prompt: str,
        completion: str,
        ground_truth: Optional[str] = None,
        tools_expected: Optional[list[str]] = None,
    ) -> dict[str, float]:
        """Regelbasierte Reward-Berechnung."""
        rewards: dict[str, float] = {}

        # 1. Korrektheit
        rewards["correctness"] = self._score_correctness(completion, ground_truth)

        # 2. Format
        rewards["format"] = self._score_format(completion)

        # 3. Hilfreichkeit
        rewards["helpfulness"] = self._score_helpfulness(completion)

        # 4. Sicherheit
        rewards["safety"] = self._score_safety(completion)

        # 5. Tool-Nutzung
        rewards["tool_usage"] = self._score_tool_usage(completion, tools_expected)

        return rewards

    def _score_correctness(
        self, completion: str, ground_truth: Optional[str]
    ) -> float:
        """Bewertet die Korrektheit der Antwort."""
        if ground_truth is None:
            return 0.5  # Neutral wenn keine Ground-Truth

        # Exakte Übereinstimmung
        if completion.strip().lower() == ground_truth.strip().lower():
            return 1.0

        # Teilweise Übereinstimmung (Ground-Truth in Completion enthalten)
        if ground_truth.strip().lower() in completion.strip().lower():
            return 0.7

        # Keyword-Überlappung
        gt_words = set(ground_truth.lower().split())
        comp_words = set(completion.lower().split())
        if gt_words:
            overlap = len(gt_words & comp_words) / len(gt_words)
            return min(overlap, 0.5)

        return 0.0

    def _score_format(self, completion: str) -> float:
        """Bewertet die Einhaltung des Ausgabeformats."""
        score = 0.0

        # Prüfe auf strukturierte Ausgabe (JSON, Markdown, etc.)
        if re.search(r"```(?:json|python|yaml)?\s*\n", completion):
            score += 0.3

        # Prüfe auf klare Abschnitte
        if re.search(r"^#{1,3}\s", completion, re.MULTILINE):
            score += 0.2

        # Prüfe auf Listen
        if re.search(r"^\s*[-*]\s", completion, re.MULTILINE):
            score += 0.2

        # Prüfe auf angemessene Länge (nicht zu kurz, nicht zu lang)
        length = len(completion.split())
        if 20 <= length <= 500:
            score += 0.3
        elif 10 <= length <= 1000:
            score += 0.15

        return min(score, 1.0)

    def _score_helpfulness(self, completion: str) -> float:
        """Bewertet die Hilfreichkeit der Antwort."""
        score = 0.0

        # Enthält die Antwort erklärende Elemente?
        explanation_patterns = [
            r"(?:weil|da|denn|deshalb|daher|somit)\b",
            r"\b(?:first|second|finally|therefore|because|thus)\b",
            r"^(?:Schritt|Step)\s+\d",
        ]
        for pattern in explanation_patterns:
            if re.search(pattern, completion, re.IGNORECASE):
                score += 0.2
                break

        # Enthält die Antwort Beispiele?
        if re.search(r"(?:z\.B\.|e\.g\.|for example|Beispiel)", completion, re.IGNORECASE):
            score += 0.2

        # Ausreichende Länge für hilfreiche Antwort
        word_count = len(completion.split())
        if word_count >= 30:
            score += 0.3
        elif word_count >= 15:
            score += 0.15

        # Keine leere Antwort
        if completion.strip():
            score += 0.3

        return min(score, 1.0)

    def _score_safety(self, completion: str) -> float:
        """Bewertet die Sicherheit der Antwort (1.0 = sicher)."""
        score = 1.0

        # Liste unsicherer Muster
        unsafe_patterns = [
            r"\b(?:hack|exploit|bypass|inject)\b",
            r"\b(?:password|token|secret|api[_\s]?key)\s*[:=]\s*\S+",
            r"\b(?:rm\s+-rf|DROP\s+TABLE|DELETE\s+FROM)\b",
            r"\b(?:illegal|malware|ransomware|phishing)\b",
        ]

        for pattern in unsafe_patterns:
            if re.search(pattern, completion, re.IGNORECASE):
                score -= 0.3

        # Prüfe auf Refusal (Ablehnung unsicherer Anfragen)
        refusal_patterns = [
            r"\b(?:cannot|can't|unable to|not able to|won't)\b",
            r"\b(?:entschuldigung|tut mir leid|kann (?:ich )?nicht)\b",
        ]
        for pattern in refusal_patterns:
            if re.search(pattern, completion, re.IGNORECASE):
                score = max(score, 0.8)  # Refusal ist sicher
                break

        return max(score, 0.0)

    def _score_tool_usage(
        self, completion: str, tools_expected: Optional[list[str]]
    ) -> float:
        """Bewertet die korrekte Tool-Nutzung."""
        if tools_expected is None:
            return 0.5  # Neutral wenn keine Erwartung

        score = 0.0
        completion_lower = completion.lower()

        for tool in tools_expected:
            if tool.lower() in completion_lower:
                score += 1.0 / len(tools_expected)

        # Bonus für korrektes Tool-Call-Format
        if re.search(r"<tool_call>|function_call|tool_calls", completion_lower):
            score = min(score + 0.2, 1.0)

        return score

    def _model_based_reward(
        self,
        prompt: str,
        completion: str,
        ground_truth: Optional[str] = None,
    ) -> dict[str, float]:
        """
        Modellbasierte Reward-Berechnung mittels eines Language Models.

        Nutzt das geladene Modell um die Qualität der Completion zu bewerten.
        """
        import torch

        if self.model is None or self.tokenizer is None:
            return self._rule_based_reward(prompt, completion, ground_truth)

        # Reward-Prompt für das Bewertungsmodell
        reward_prompt = self._build_reward_prompt(prompt, completion, ground_truth)

        inputs = self.tokenizer(
            reward_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Nutze die Logits des letzten Tokens als Reward-Signal
            logits = outputs.logits[:, -1, :]
            # Konvertiere zu einem skalaren Reward (gemittelt über Vocab)
            reward_signal = logits.mean(dim=-1).sigmoid().item()

        return {
            "correctness": reward_signal,
            "format": reward_signal * 0.8,
            "helpfulness": reward_signal * 0.9,
            "safety": 0.9,
            "tool_usage": reward_signal * 0.7,
        }

    def _build_reward_prompt(
        self,
        prompt: str,
        completion: str,
        ground_truth: Optional[str] = None,
    ) -> str:
        """Erstellt den Prompt für die modellbasierte Bewertung."""
        parts = [
            "Bewerte die folgende Agent-Antwort auf einer Skala von 0.0 bis 1.0.",
            "",
            f"### Prompt:\n{prompt}",
            "",
            f"### Antwort:\n{completion}",
        ]
        if ground_truth:
            parts.append(f"\n### Erwartete Antwort:\n{ground_truth}")

        parts.extend([
            "",
            "### Bewertungskriterien:",
            "- Korrektheit: Ist die Antwort fachlich richtig?",
            "- Format: Ist die Antwort gut strukturiert?",
            "- Hilfreichkeit: Ist die Antwort nützlich und verständlich?",
            "- Sicherheit: Enthält die Antwort keine schädlichen Inhalte?",
            "",
            "Gib NUR eine Zahl zwischen 0.0 und 1.0 zurück:",
        ])
        return "\n".join(parts)


class RewardModelWrapper:
    """
    PyTorch-Modul-Wrapper für das Reward-Modell.

    Kann als eigenständiges Reward-Modell für TRL/GRPO-Training verwendet werden.
    Erfordert torch + transformers (lazy import).
    """

    def __init__(
        self,
        base_model: Any,
        reward_dim: int = 1,
        dropout: float = 0.1,
    ):
        """
        Args:
            base_model: Das Basis-Sprachmodell (transformers.PreTrainedModel).
            reward_dim: Dimension des Reward-Heads.
            dropout: Dropout-Rate.
        """
        import torch.nn as nn

        self.base_model = base_model
        hidden_size = base_model.config.hidden_size

        self.reward_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, reward_dim),
        )

    def forward(
        self,
        input_ids: Any,
        attention_mask: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Forward-Pass des Reward-Modells.

        Args:
            input_ids: Token-IDs (torch.Tensor).
            attention_mask: Attention-Maske (torch.Tensor).

        Returns:
            Reward-Scores (batch_size, reward_dim).
        """
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs,
        )

        # Nutze das letzte Hidden-State des letzten Tokens
        last_hidden = outputs.hidden_states[-1][:, -1, :]
        reward = self.reward_head(last_hidden)
        return reward


def create_reward_function(
    reward_model: AgentRewardModel,
):
    """
    Erstellt eine Reward-Funktion kompatibel mit TRL's GRPOTrainer.

    Args:
        reward_model: Eine AgentRewardModel-Instanz.

    Returns:
        Funktion die (prompts, completions, **kwargs) -> rewards liefert.
    """

    def reward_func(
        prompts: list[str],
        completions: list[str],
        **kwargs: Any,
    ) -> list[float]:
        rewards = []
        for prompt, completion in zip(prompts, completions):
            result = reward_model.compute_reward(
                prompt=prompt,
                completion=completion,
                ground_truth=kwargs.get("ground_truth"),
            )
            rewards.append(result["total"])
        return rewards

    return reward_func
