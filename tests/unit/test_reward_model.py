"""Unit-Tests für das Reward-Modell (reward_model.py)."""

import pytest

from reward_model import AgentRewardModel, RewardModelWrapper, create_reward_function


class TestAgentRewardModel:
    """Tests für die regelbasierte Reward-Berechnung."""

    @pytest.fixture
    def reward_model(self):
        """Erstellt ein regelbasiertes Reward-Modell mit Standard-Gewichten."""
        return AgentRewardModel(use_model=False)

    def test_init_default_weights(self):
        """Test: Standard-Gewichte werden korrekt gesetzt."""
        model = AgentRewardModel(use_model=False)
        assert model.reward_weights["correctness"] == 1.0
        assert model.reward_weights["format"] == 0.3
        assert model.reward_weights["helpfulness"] == 0.5
        assert model.reward_weights["safety"] == 0.8
        assert model.reward_weights["tool_usage"] == 0.4
        assert model.use_model is False
        assert model.model is None

    def test_init_custom_weights(self):
        """Test: Benutzerdefinierte Gewichte werden übernommen."""
        custom = {"correctness": 2.0, "safety": 1.0}
        model = AgentRewardModel(reward_weights=custom, use_model=False)
        assert model.reward_weights == custom

    def test_compute_reward_returns_total(self, reward_model):
        """Test: compute_reward gibt ein Dictionary mit 'total'-Key zurück."""
        result = reward_model.compute_reward(
            prompt="Was ist Python?",
            completion="Python ist eine Programmiersprache.",
        )
        assert "total" in result
        assert isinstance(result["total"], float)
        assert 0.0 <= result["total"] <= 5.0  # Max sum of weights

    def test_compute_reward_all_keys_present(self, reward_model):
        """Test: Alle Reward-Komponenten sind im Ergebnis enthalten."""
        result = reward_model.compute_reward(
            prompt="Erkläre Docker.",
            completion="Docker ist eine Container-Plattform.",
        )
        for key in ["correctness", "format", "helpfulness", "safety", "tool_usage", "total"]:
            assert key in result, f"Key '{key}' fehlt im Ergebnis"

    def test_correctness_exact_match(self, reward_model):
        """Test: Exakte Übereinstimmung mit Ground-Truth ergibt 1.0."""
        result = reward_model.compute_reward(
            prompt="Was ist 2+2?",
            completion="4",
            ground_truth="4",
        )
        assert result["correctness"] == 1.0

    def test_correctness_partial_match(self, reward_model):
        """Test: Teilweise Übereinstimmung (Ground-Truth in Completion) ergibt 0.7."""
        result = reward_model.compute_reward(
            prompt="Erkläre Python.",
            completion="Python ist eine Programmiersprache. Sie ist weit verbreitet.",
            ground_truth="Python ist eine Programmiersprache.",
        )
        assert result["correctness"] == 0.7

    def test_correctness_keyword_overlap(self, reward_model):
        """Test: Keyword-Überlappung ohne exakte Teilstring-Übereinstimmung."""
        result = reward_model.compute_reward(
            prompt="Erkläre Python.",
            completion="Python ist eine großartige Programmiersprache.",
            ground_truth="Python ist eine Programmiersprache.",
        )
        # Keyword-Überlappung: 4 von 4 Wörtern → min(1.0, 0.5) = 0.5
        assert result["correctness"] == 0.5

    def test_correctness_no_ground_truth(self, reward_model):
        """Test: Ohne Ground-Truth wird 0.5 zurückgegeben."""
        result = reward_model.compute_reward(
            prompt="Was ist Rust?",
            completion="Rust ist eine Systems-Programmiersprache.",
        )
        assert result["correctness"] == 0.5

    def test_correctness_no_match(self, reward_model):
        """Test: Keine Übereinstimmung ergibt 0.0."""
        result = reward_model.compute_reward(
            prompt="Erkläre Python.",
            completion="Ein völlig anderes Thema ohne Bezug.",
            ground_truth="Python ist eine Programmiersprache.",
        )
        assert result["correctness"] == 0.0

    def test_format_code_block(self, reward_model):
        """Test: Code-Blöcke erhöhen den Format-Score."""
        result = reward_model.compute_reward(
            prompt="Schreibe Code.",
            completion="Hier ist der Code:\n```python\nprint('hello')\n```",
        )
        assert result["format"] > 0.0

    def test_format_empty_completion(self, reward_model):
        """Test: Leere Completion hat niedrigen Format-Score."""
        result = reward_model.compute_reward(
            prompt="Was ist das?",
            completion="",
        )
        assert result["format"] < 0.5

    def test_helpfulness_non_empty(self, reward_model):
        """Test: Nicht-leere Antwort erhält Basis-Score."""
        result = reward_model.compute_reward(
            prompt="Hilfe!",
            completion="Hier ist eine ausführliche Erklärung mit vielen Details und Beispielen.",
        )
        assert result["helpfulness"] > 0.0

    def test_helpfulness_empty(self, reward_model):
        """Test: Leere Antwort hat niedrigen Helpfulness-Score."""
        result = reward_model.compute_reward(
            prompt="Hilfe!",
            completion="",
        )
        assert result["helpfulness"] < 0.5

    def test_safety_clean_content(self, reward_model):
        """Test: Saubere Inhalte haben hohen Safety-Score."""
        result = reward_model.compute_reward(
            prompt="Wie geht's?",
            completion="Mir geht es gut, danke der Nachfrage!",
        )
        assert result["safety"] >= 0.7

    def test_safety_unsafe_pattern(self, reward_model):
        """Test: Unsichere Muster reduzieren den Safety-Score."""
        result = reward_model.compute_reward(
            prompt="Wie hacke ich?",
            completion="Du kannst das System hacken mit rm -rf /",
        )
        assert result["safety"] < 1.0

    def test_safety_refusal(self, reward_model):
        """Test: Refusal-Antworten haben hohen Safety-Score."""
        result = reward_model.compute_reward(
            prompt="Wie hacke ich?",
            completion="Entschuldigung, ich kann nicht bei illegalen Aktivitäten helfen.",
        )
        assert result["safety"] >= 0.8

    def test_tool_usage_expected_tools(self, reward_model):
        """Test: Erwartete Tools werden erkannt."""
        result = reward_model.compute_reward(
            prompt="Suche etwas.",
            completion="Ich nutze die search-Funktion und den calculator.",
            tools_expected=["search", "calculator"],
        )
        assert result["tool_usage"] > 0.5

    def test_tool_usage_no_expected(self, reward_model):
        """Test: Ohne erwartete Tools wird 0.5 zurückgegeben."""
        result = reward_model.compute_reward(
            prompt="Suche etwas.",
            completion="Ich suche mit der search-Funktion.",
        )
        assert result["tool_usage"] == 0.5

    def test_tool_usage_none_found(self, reward_model):
        """Test: Keine der erwarteten Tools gefunden."""
        result = reward_model.compute_reward(
            prompt="Rechne etwas.",
            completion="Das Ergebnis ist 42.",
            tools_expected=["calculator", "math"],
        )
        assert result["tool_usage"] == 0.0

    def test_total_reward_weighted_sum(self, reward_model):
        """Test: Der Gesamt-Reward ist die gewichtete Summe der Einzel-Rewards."""
        result = reward_model.compute_reward(
            prompt="Test",
            completion="Test-Antwort",
        )
        expected_total = (
            result["correctness"] * 1.0
            + result["format"] * 0.3
            + result["helpfulness"] * 0.5
            + result["safety"] * 0.8
            + result["tool_usage"] * 0.4
        )
        assert result["total"] == pytest.approx(expected_total)


class TestCreateRewardFunction:
    """Tests für die create_reward_function-Hilfsfunktion."""

    def test_returns_callable(self):
        """Test: Gibt eine aufrufbare Funktion zurück."""
        model = AgentRewardModel(use_model=False)
        func = create_reward_function(model)
        assert callable(func)

    def test_reward_func_returns_list_of_floats(self):
        """Test: Die Reward-Funktion gibt eine Liste von Floats zurück."""
        model = AgentRewardModel(use_model=False)
        func = create_reward_function(model)
        prompts = ["Frage 1", "Frage 2"]
        completions = ["Antwort 1", "Antwort 2"]
        rewards = func(prompts, completions)
        assert len(rewards) == 2
        assert all(isinstance(r, float) for r in rewards)

    def test_reward_func_empty_lists(self):
        """Test: Leere Listen ergeben leere Reward-Liste."""
        model = AgentRewardModel(use_model=False)
        func = create_reward_function(model)
        rewards = func([], [])
        assert rewards == []

    def test_reward_func_passes_ground_truth(self):
        """Test: Ground-Truth wird als kwargs durchgereicht."""
        model = AgentRewardModel(use_model=False)
        func = create_reward_function(model)
        rewards = func(
            ["Was ist 2+2?"],
            ["4"],
            ground_truth="4",
        )
        assert len(rewards) == 1
        assert rewards[0] > 0.0


class TestRewardModelWrapper:
    """Tests für den RewardModelWrapper (ohne echtes Modell)."""

    def test_init_creates_reward_head(self):
        """Test: Der Wrapper erstellt einen Reward-Head."""
        pytest.importorskip("torch", reason="torch nicht installiert")
        # Mock ein base_model mit config.hidden_size
        class MockConfig:
            hidden_size = 768

        class MockModel:
            config = MockConfig()

        wrapper = RewardModelWrapper(base_model=MockModel())
        assert hasattr(wrapper, "reward_head")
        assert hasattr(wrapper, "base_model")
