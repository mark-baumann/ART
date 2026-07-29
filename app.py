"""
Streamlit-App: ART — Agent Reinforcement Trainer
=================================================
GRPO-Training konfigurieren, Reward-Modell testen, LoRA-Parameter einstellen.
"""

import streamlit as st
import numpy as np
import re
import os

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="ART — Agent Reinforcement Trainer",
    page_icon="🎯",
    layout="wide",
)

st.title("🎯 ART — Agent Reinforcement Trainer")
st.markdown("GRPO-Training · Reward-Modell · LoRA-Parameter")

# ── Sidebar: Modus ───────────────────────────────────────────
mode = st.sidebar.selectbox(
    "Modus wählen",
    ["GRPO-Training konfigurieren", "Reward-Modell testen", "LoRA-Parameter einstellen"],
)

# ═══════════════════════════════════════════════════════════════
# 1. GRPO-Training konfigurieren
# ═══════════════════════════════════════════════════════════════

if mode == "GRPO-Training konfigurieren":
    st.header("⚙️ GRPO-Training konfigurieren")

    st.markdown("""
    **Group Relative Policy Optimization (GRPO)** — Konfiguriere das Training
    für Qwen2.5 oder Llama 3.1 mit LoRA-Adaptern.
    """)

    # Modell-Auswahl
    st.subheader("🤖 Modell")
    col1, col2 = st.columns(2)
    with col1:
        model_family = st.selectbox("Modellfamilie", ["Qwen 2.5 7B", "Llama 3.1 8B", "Qwen 2.5 1.5B (Test)"], index=0)
    with col2:
        load_4bit = st.checkbox("4-Bit Quantisierung", value=True)
        attn_impl = st.selectbox("Attention", ["flash_attention_2", "sdpa", "eager"], index=0)

    # GRPO-Parameter
    st.subheader("🎯 GRPO-Parameter")
    col1, col2, col3 = st.columns(3)
    with col1:
        num_generations = st.slider("Generations (G)", 1, 8, 4, help="Anzahl Samples pro Prompt")
        max_prompt_len = st.number_input("Max Prompt-Länge", 256, 4096, 2048, 256)
    with col2:
        max_completion_len = st.number_input("Max Completion-Länge", 128, 2048, 1024, 128)
        temperature = st.slider("Temperatur", 0.1, 2.0, 0.9, 0.1)
    with col3:
        beta = st.slider("Beta (KL-Koeffizient)", 0.0, 0.2, 0.04, 0.01, help="KL-Divergence Gewicht")
        top_p = st.slider("Top-P", 0.5, 1.0, 1.0, 0.05)

    # Training-Parameter
    st.subheader("🏋️ Training")
    col1, col2, col3 = st.columns(3)
    with col1:
        learning_rate = st.selectbox("Learning Rate", [1e-6, 3e-6, 5e-6, 1e-5, 5e-5], index=2, format_func=lambda x: f"{x:.0e}")
        num_epochs = st.slider("Epochen", 1, 10, 1)
    with col2:
        batch_size = st.slider("Batch Size (pro Device)", 1, 8, 2)
        grad_accum = st.slider("Gradient Accumulation", 1, 16, 4)
    with col3:
        optim = st.selectbox("Optimizer", ["adamw_8bit", "adamw_torch", "sgd"], index=0)
        lr_scheduler = st.selectbox("LR Scheduler", ["cosine", "linear", "constant"], index=0)

    col1, col2 = st.columns(2)
    with col1:
        warmup_ratio = st.slider("Warmup Ratio", 0.0, 0.3, 0.1, 0.05)
        weight_decay = st.slider("Weight Decay", 0.0, 0.2, 0.01, 0.01)
    with col2:
        max_steps = st.number_input("Max Steps (-1 = voller Datensatz)", -1, 10000, -1, 100)
        seed = st.number_input("Seed", 0, 9999, 42)

    # Logging
    st.subheader("📊 Logging & Output")
    col1, col2, col3 = st.columns(3)
    with col1:
        logging_steps = st.number_input("Logging Steps", 1, 500, 10)
        save_steps = st.number_input("Save Steps", 10, 1000, 100)
    with col2:
        eval_steps = st.number_input("Eval Steps", 10, 1000, 100)
        report_to = st.selectbox("Report To", ["wandb", "tensorboard", "none"], index=0)
    with col3:
        output_dir = st.text_input("Output Dir", "./output/grpo-lora")
        experiment_name = st.text_input("Experiment", "art-grpo-lora")

    # Zusammenfassung
    st.divider()
    st.subheader("📋 Vollständige Konfiguration")

    config_summary = {
        "Modell": {
            "Familie": model_family,
            "4-Bit": load_4bit,
            "Attention": attn_impl,
        },
        "GRPO": {
            "num_generations": num_generations,
            "max_prompt_length": max_prompt_len,
            "max_completion_length": max_completion_len,
            "temperature": temperature,
            "beta": beta,
            "top_p": top_p,
        },
        "Training": {
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "gradient_accumulation_steps": grad_accum,
            "optim": optim,
            "lr_scheduler": lr_scheduler,
            "warmup_ratio": warmup_ratio,
            "weight_decay": weight_decay,
            "max_steps": max_steps,
            "seed": seed,
        },
        "Logging": {
            "logging_steps": logging_steps,
            "save_steps": save_steps,
            "eval_steps": eval_steps,
            "report_to": report_to,
            "output_dir": output_dir,
            "experiment_name": experiment_name,
        },
    }

    st.json(config_summary)

    # CLI-Befehl generieren
    st.subheader("💻 CLI-Befehl")
    cmd_parts = ["python train_agent.py"]
    if "Qwen" in model_family:
        cmd_parts.append("--model qwen")
    else:
        cmd_parts.append("--model llama")
    if "1.5B" in model_family:
        cmd_parts.append("--test-mode")
    if output_dir != "./output/grpo-lora":
        cmd_parts.append(f"--output-dir {output_dir}")
    if learning_rate != 5e-6:
        cmd_parts.append(f"--learning-rate {learning_rate}")
    if max_steps > 0:
        cmd_parts.append(f"--max-steps {max_steps}")
    if report_to == "none":
        cmd_parts.append("--no-wandb")

    st.code(" \\\n  ".join(cmd_parts), language="bash")

    if st.button("💾 Konfiguration speichern", type="primary"):
        st.success(f"✅ Konfiguration '{experiment_name}' bereit zum Training!")

# ═══════════════════════════════════════════════════════════════
# 2. Reward-Modell testen
# ═══════════════════════════════════════════════════════════════

elif mode == "Reward-Modell testen":
    st.header("🏆 Reward-Modell testen")

    st.markdown("""
    Teste das regelbasierte Reward-Modell mit eigenen Prompts und Completions.
    Bewertet werden: **Korrektheit, Format, Hilfreichkeit, Sicherheit, Tool-Nutzung**.
    """)

    # Reward-Gewichte
    st.subheader("⚖️ Reward-Gewichte")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        w_correctness = st.slider("Korrektheit", 0.0, 2.0, 1.0, 0.1)
    with col2:
        w_format = st.slider("Format", 0.0, 2.0, 0.3, 0.1)
    with col3:
        w_helpfulness = st.slider("Hilfreichkeit", 0.0, 2.0, 0.5, 0.1)
    with col4:
        w_safety = st.slider("Sicherheit", 0.0, 2.0, 0.8, 0.1)
    with col5:
        w_tool_usage = st.slider("Tool-Nutzung", 0.0, 2.0, 0.4, 0.1)

    # Eingabe
    st.subheader("📝 Test-Eingabe")
    prompt_input = st.text_area("Prompt", value="Erkläre den Unterschied zwischen GRPO und PPO.", height=80)
    completion_input = st.text_area("Completion (Agent-Antwort)", value="GRPO (Group Relative Policy Optimization) ist eine Weiterentwicklung von PPO (Proximal Policy Optimization). Der Hauptunterschied: GRPO vergleicht mehrere generierte Antworten innerhalb einer Gruppe relativ zueinander, während PPO mit einem einzelnen Value-Netzwerk arbeitet. GRPO benötigt daher kein separates Critic-Modell und ist speichereffizienter.", height=120)
    ground_truth = st.text_input("Ground Truth (optional)", value="", placeholder="Erwartete Antwort für Korrektheits-Check")

    if st.button("🔍 Reward berechnen", type="primary"):
        # ── Regelbasierte Reward-Berechnung ──────────────────
        rewards = {}

        # 1. Korrektheit
        if ground_truth.strip():
            if completion_input.strip().lower() == ground_truth.strip().lower():
                rewards["correctness"] = 1.0
            elif ground_truth.strip().lower() in completion_input.strip().lower():
                rewards["correctness"] = 0.7
            else:
                gt_words = set(ground_truth.lower().split())
                comp_words = set(completion_input.lower().split())
                if gt_words:
                    overlap = len(gt_words & comp_words) / len(gt_words)
                    rewards["correctness"] = min(overlap, 0.5)
                else:
                    rewards["correctness"] = 0.0
        else:
            rewards["correctness"] = 0.5

        # 2. Format
        format_score = 0.0
        if re.search(r"```(?:json|python|yaml)?\s*\n", completion_input):
            format_score += 0.3
        if re.search(r"^#{1,3}\s", completion_input, re.MULTILINE):
            format_score += 0.2
        if re.search(r"^\s*[-*]\s", completion_input, re.MULTILINE):
            format_score += 0.2
        length = len(completion_input.split())
        if 20 <= length <= 500:
            format_score += 0.3
        elif 10 <= length <= 1000:
            format_score += 0.15
        rewards["format"] = min(format_score, 1.0)

        # 3. Hilfreichkeit
        helpful_score = 0.0
        explanation_patterns = [
            r"(?:weil|da|denn|deshalb|daher|somit)\b",
            r"\b(?:first|second|finally|therefore|because|thus)\b",
            r"^(?:Schritt|Step)\s+\d",
        ]
        for pat in explanation_patterns:
            if re.search(pat, completion_input, re.IGNORECASE):
                helpful_score += 0.2
                break
        if re.search(r"(?:z\.B\.|e\.g\.|for example|Beispiel)", completion_input, re.IGNORECASE):
            helpful_score += 0.2
        word_count = len(completion_input.split())
        if word_count >= 30:
            helpful_score += 0.3
        elif word_count >= 15:
            helpful_score += 0.15
        if completion_input.strip():
            helpful_score += 0.3
        rewards["helpfulness"] = min(helpful_score, 1.0)

        # 4. Sicherheit
        safety_score = 1.0
        unsafe_patterns = [
            r"\b(?:hack|exploit|bypass|inject)\b",
            r"\b(?:password|token|secret|api[_\s]?key)\s*[:=]\s*\S+",
            r"\b(?:rm\s+-rf|DROP\s+TABLE|DELETE\s+FROM)\b",
            r"\b(?:illegal|malware|ransomware|phishing)\b",
        ]
        for pat in unsafe_patterns:
            if re.search(pat, completion_input, re.IGNORECASE):
                safety_score -= 0.3
        refusal_patterns = [
            r"\b(?:cannot|can't|unable to|not able to|won't)\b",
            r"\b(?:entschuldigung|tut mir leid|kann (?:ich )?nicht)\b",
        ]
        for pat in refusal_patterns:
            if re.search(pat, completion_input, re.IGNORECASE):
                safety_score = max(safety_score, 0.8)
                break
        rewards["safety"] = max(safety_score, 0.0)

        # 5. Tool-Nutzung
        rewards["tool_usage"] = 0.5  # Neutral

        # Gewichteter Total-Reward
        total = (
            rewards["correctness"] * w_correctness +
            rewards["format"] * w_format +
            rewards["helpfulness"] * w_helpfulness +
            rewards["safety"] * w_safety +
            rewards["tool_usage"] * w_tool_usage
        )
        rewards["total"] = total

        # ── Ergebnisse anzeigen ──────────────────────────────
        st.divider()
        st.subheader("📊 Reward-Ergebnisse")

        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col1:
            st.metric("Korrektheit", f"{rewards['correctness']:.2f}", delta=None)
        with col2:
            st.metric("Format", f"{rewards['format']:.2f}", delta=None)
        with col3:
            st.metric("Hilfreichkeit", f"{rewards['helpfulness']:.2f}", delta=None)
        with col4:
            st.metric("Sicherheit", f"{rewards['safety']:.2f}", delta=None)
        with col5:
            st.metric("Tool-Nutzung", f"{rewards['tool_usage']:.2f}", delta=None)
        with col6:
            st.metric("**Total**", f"{rewards['total']:.2f}", delta=None)

        # Balkendiagramm
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 3))
        categories = ["Korrektheit", "Format", "Hilfreichkeit", "Sicherheit", "Tool-Nutzung"]
        values = [rewards["correctness"], rewards["format"], rewards["helpfulness"], rewards["safety"], rewards["tool_usage"]]
        colors = ["#4CAF50", "#2196F3", "#FF9800", "#f44336", "#9C27B0"]
        bars = ax.bar(categories, values, color=colors)
        ax.axhline(y=rewards["total"], color="black", linestyle="--", label=f"Total: {rewards['total']:.2f}")
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Score")
        ax.legend()
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f"{val:.2f}", ha="center", fontsize=10)
        st.pyplot(fig)

        # Bewertung
        if total >= 0.7:
            st.success(f"🌟 Gute Antwort! Total Reward: {total:.2f}")
        elif total >= 0.4:
            st.warning(f"⚡ Durchschnittliche Antwort. Total Reward: {total:.2f}")
        else:
            st.error(f"❌ Schwache Antwort. Total Reward: {total:.2f}")

# ═══════════════════════════════════════════════════════════════
# 3. LoRA-Parameter einstellen
# ═══════════════════════════════════════════════════════════════

elif mode == "LoRA-Parameter einstellen":
    st.header("🔧 LoRA-Parameter einstellen")

    st.markdown("""
    **Low-Rank Adaptation (LoRA)** — Konfiguriere die LoRA-Adapter für effizientes Fine-Tuning.
    Nur ein Bruchteil der Parameter wird trainiert.
    """)

    # LoRA-Kernparameter
    st.subheader("📐 LoRA-Kernparameter")
    col1, col2, col3 = st.columns(3)
    with col1:
        lora_r = st.slider("Rank (r)", 1, 128, 16, help="LoRA-Rank — niedriger = weniger Parameter")
    with col2:
        lora_alpha = st.slider("Alpha", 1, 128, 32, help="Skalierungsfaktor")
    with col3:
        lora_dropout = st.slider("Dropout", 0.0, 0.5, 0.05, 0.05)

    # Effektive Skalierung
    effective_scale = lora_alpha / lora_r
    st.info(f"📏 Effektive Skalierung: α/r = {lora_alpha}/{lora_r} = **{effective_scale:.2f}**")

    # Target-Module
    st.subheader("🎯 Target-Module")
    st.markdown("Wähle aus, welche Layer mit LoRA-Adaptern versehen werden:")

    col1, col2 = st.columns(2)
    with col1:
        target_q = st.checkbox("q_proj (Query)", value=True)
        target_k = st.checkbox("k_proj (Key)", value=True)
        target_v = st.checkbox("v_proj (Value)", value=True)
        target_o = st.checkbox("o_proj (Output)", value=True)
    with col2:
        target_gate = st.checkbox("gate_proj", value=True)
        target_up = st.checkbox("up_proj", value=True)
        target_down = st.checkbox("down_proj", value=True)

    selected_targets = []
    if target_q: selected_targets.append("q_proj")
    if target_k: selected_targets.append("k_proj")
    if target_v: selected_targets.append("v_proj")
    if target_o: selected_targets.append("o_proj")
    if target_gate: selected_targets.append("gate_proj")
    if target_up: selected_targets.append("up_proj")
    if target_down: selected_targets.append("down_proj")

    st.write(f"**{len(selected_targets)} Module** ausgewählt: `{', '.join(selected_targets) if selected_targets else 'keine'}`")

    # Parameter-Schätzung
    st.subheader("📊 Parameter-Schätzung")

    model_size = st.selectbox("Basis-Modell", ["Qwen 2.5 7B", "Llama 3.1 8B", "Qwen 2.5 1.5B"], index=0)

    # Grobe Schätzung
    if "7B" in model_size:
        base_params = 7_000_000_000
        hidden_size = 4096
        num_layers = 32
    elif "8B" in model_size:
        base_params = 8_000_000_000
        hidden_size = 4096
        num_layers = 32
    else:
        base_params = 1_500_000_000
        hidden_size = 1536
        num_layers = 28

    # LoRA-Parameter: 2 * r * hidden_size * num_target_modules * num_layers
    num_targets = len(selected_targets)
    lora_params = 2 * lora_r * hidden_size * num_targets * num_layers

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Basis-Parameter", f"{base_params/1e9:.1f}B")
    with col2:
        st.metric("LoRA-Parameter", f"{lora_params/1e6:.1f}M")
    with col3:
        ratio = lora_params / base_params * 100
        st.metric("Anteil", f"{ratio:.2f}%")

    # Visualisierung
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    sizes = [base_params - lora_params, lora_params]
    labels = ["Eingefroren", "LoRA (trainierbar)"]
    colors = ["#BBDEFB", "#1565C0"]
    ax.pie(sizes, labels=labels, autopct="%1.2f%%", colors=colors, startangle=90)
    ax.set_title(f"Parameter-Verteilung: {model_size}")
    st.pyplot(fig)

    # Vordefinierte Presets
    st.subheader("🎛️ Vordefinierte Presets")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔬 Konservativ (r=8, α=16)", use_container_width=True):
            st.session_state["lora_preset"] = "conservative"
    with col2:
        if st.button("⚖️ Standard (r=16, α=32)", use_container_width=True):
            st.session_state["lora_preset"] = "standard"
    with col3:
        if st.button("🚀 Aggressiv (r=64, α=128)", use_container_width=True):
            st.session_state["lora_preset"] = "aggressive"

    # Konfigurations-Code
    st.subheader("💻 LoRA-Konfiguration (Python)")
    lora_code = f"""from peft import LoraConfig

lora_config = LoraConfig(
    r={lora_r},
    lora_alpha={lora_alpha},
    target_modules={selected_targets},
    lora_dropout={lora_dropout},
    bias="none",
    task_type="CAUSAL_LM",
)"""
    st.code(lora_code, language="python")

    # Empfehlungen
    st.subheader("💡 Empfehlungen")
    st.markdown(f"""
    - **Rank (r={lora_r})**: {'Niedrig' if lora_r <= 8 else 'Mittel' if lora_r <= 32 else 'Hoch'} — {
        'Gut für einfache Tasks, sehr speichereffizient' if lora_r <= 8 else
        'Gute Balance für die meisten Anwendungen' if lora_r <= 32 else
        'Für komplexe Tasks, höherer Speicherverbrauch'
    }
    - **Alpha (α={lora_alpha})**: Skalierung = {effective_scale:.1f}× — {
        'Starke Regularisierung' if effective_scale < 1 else
        'Standard-Skalierung' if effective_scale <= 2 else
        'Schwache Regularisierung, stärkere Anpassung'
    }
    - **Target-Module**: {num_targets} von 7 möglichen — {
        'Minimale Anpassung' if num_targets <= 3 else
        'Ausgewogen' if num_targets <= 5 else
        'Volle Abdeckung (empfohlen für GRPO)'
    }
    - **Dropout ({lora_dropout})**: {
        'Keine Regularisierung' if lora_dropout == 0 else
        'Leichte Regularisierung' if lora_dropout <= 0.1 else
        'Starke Regularisierung (gegen Overfitting)'
    }
    """)

st.sidebar.markdown("---")
st.sidebar.caption("ART · Streamlit Dashboard")
