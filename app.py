"""
Streamlit-App: ART — Agent Reinforcement Trainer
================================================
GRPO-Training konfigurieren, Reward-Modell testen, LoRA-Parameter einstellen.
"""

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    TrainingConfig, ModelConfig, LoRAConfig, GRPOConfig, RewardConfig, DataConfig,
    get_qwen_config, get_llama_config, get_small_test_config,
)
from reward_model import AgentRewardModel

st.set_page_config(
    page_title="ART — Agent Reinforcement Trainer",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 ART — Agent Reinforcement Trainer")
st.markdown("### GRPO-Training konfigurieren · Reward-Modell testen · LoRA-Parameter einstellen")

# ── Sidebar ──
seite = st.sidebar.radio(
    "📂 Bereich wählen",
    ["🎯 GRPO-Training", "🏆 Reward-Modell", "🔧 LoRA-Konfiguration", "📋 Gesamtkonfiguration"],
)

# ═══════════════════════════════════════════════════════════════
# GRPO-TRAINING
# ═══════════════════════════════════════════════════════════════
if seite == "🎯 GRPO-Training":
    st.header("🎯 GRPO-Training konfigurieren")

    st.markdown("""
    **GRPO** (Group Relative Policy Optimization) ist ein RL-Verfahren für
    Sprachmodelle. Es generiert mehrere Antworten pro Prompt, bewertet sie
    mit einem Reward-Modell und optimiert die Policy relativ zur Gruppe.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 GRPO-Parameter")

        num_generations = st.slider("Anzahl Generierungen (G)", 1, 8, 4,
                                     help="Gruppengröße: Wie viele Antworten pro Prompt generiert werden.")
        max_prompt_length = st.slider("Max. Prompt-Länge", 256, 4096, 2048, 128)
        max_completion_length = st.slider("Max. Completion-Länge", 128, 2048, 1024, 128)
        temperature = st.slider("Temperatur", 0.1, 2.0, 0.9, 0.1)
        top_p = st.slider("Top-p", 0.5, 1.0, 1.0, 0.05)

    with col2:
        st.subheader("⚡ Training")

        learning_rate = st.number_input("Learning Rate", 1e-7, 1e-3, 5e-6, format="%.1e")
        beta = st.slider("Beta (KL-Koeffizient)", 0.0, 0.2, 0.04, 0.01,
                         help="Steuert, wie stark die Policy vom Reference-Modell abweichen darf.")
        num_epochs = st.slider("Epochen pro GRPO-Schritt", 1, 5, 1)
        grad_accum = st.slider("Gradient Accumulation Steps", 1, 16, 4)
        batch_size = st.slider("Batch Size pro Device", 1, 8, 2)

    # GRPO-Workflow visualisieren
    st.markdown("---")
    st.subheader("🔄 GRPO-Workflow")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')

    steps = [
        (1, 4.5, "1. Prompt", '#4ECDC4'),
        (3, 4.5, "2. Generiere\nG Antworten", '#45B7D1'),
        (5, 4.5, "3. Reward\nberechnen", '#FFE66D'),
        (7, 4.5, "4. Advantage\n(Gruppen-Norm.)", '#F38181'),
        (9, 4.5, "5. Policy\nUpdate (PPO)", '#FF6B6B'),
        (11, 4.5, "6. KL-Div\nprüfen", '#AA96DA'),
    ]

    for x, y, text, color in steps:
        rect = plt.Rectangle((x - 0.8, y - 0.6), 1.6, 1.2,
                             facecolor=color, edgecolor='white',
                             linewidth=2, alpha=0.9, zorder=2)
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=9,
                fontweight='bold', color='white', zorder=3)

    # Pfeile
    for i in range(len(steps) - 1):
        ax.annotate('', xy=(steps[i + 1][0] - 0.8, steps[i + 1][1]),
                    xytext=(steps[i][0] + 0.8, steps[i][1]),
                    arrowprops=dict(arrowstyle='->', color='#888888', lw=2))

    # Rückkopplung
    ax.annotate('', xy=(1, 2.5), xytext=(11, 2.5),
                arrowprops=dict(arrowstyle='->', color='#FF6B6B', lw=1.5,
                                connectionstyle='arc3,rad=-0.3'))
    ax.text(6, 2.0, "↻ Iteriere bis Konvergenz", ha='center', fontsize=10,
            color='#FF6B6B', style='italic')

    ax.set_title("GRPO Training Loop", fontsize=14, fontweight='bold')
    st.pyplot(fig)
    plt.close(fig)

    # Parameter-Impact
    st.markdown("---")
    st.subheader("📈 Parameter-Einfluss")

    # Beta vs KL-Divergence
    betas = np.linspace(0.01, 0.2, 20)
    kl_penalty = betas * 10  # Simulierter Effekt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(betas, kl_penalty, color='#FF6B6B', linewidth=2)
    ax1.fill_between(betas, 0, kl_penalty, alpha=0.2, color='#FF6B6B')
    ax1.set_xlabel("Beta (KL-Koeffizient)")
    ax1.set_ylabel("KL-Penalty")
    ax1.set_title("Beta → KL-Divergence-Kontrolle")
    ax1.grid(True, alpha=0.3)

    # Gruppengröße vs Variance
    groups = np.arange(1, 9)
    advantage_variance = 1.0 / np.sqrt(groups)
    ax2.bar(groups, advantage_variance, color='#4ECDC4', edgecolor='white')
    ax2.set_xlabel("Gruppengröße (G)")
    ax2.set_ylabel("Advantage-Varianz (relativ)")
    ax2.set_title("Gruppengröße → Schätzgenauigkeit")
    ax2.grid(True, alpha=0.3, axis='y')

    st.pyplot(fig)
    plt.close(fig)

# ═══════════════════════════════════════════════════════════════
# REWARD-MODELL
# ═══════════════════════════════════════════════════════════════
elif seite == "🏆 Reward-Modell":
    st.header("🏆 Reward-Modell testen")

    st.markdown("""
    Das Reward-Modell bewertet Agent-Antworten anhand mehrerer Kriterien.
    Hier kannst du es mit eigenen Beispielen testen.
    """)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📝 Eingabe")

        prompt = st.text_area(
            "Prompt",
            value="Erkläre, wie ein Transformer-Modell funktioniert.",
            height=80,
        )

        completion = st.text_area(
            "Agent-Antwort",
            value=(
                "Ein Transformer-Modell basiert auf dem Attention-Mechanismus. "
                "Es verarbeitet Eingaben parallel statt sequentiell. "
                "Die Self-Attention berechnet für jedes Token die Relevanz "
                "aller anderen Tokens im Kontext. Das ermöglicht es dem Modell, "
                "langreichweitige Abhängigkeiten zu erfassen."
            ),
            height=150,
        )

        ground_truth = st.text_input(
            "Ground Truth (optional)",
            value="Transformer nutzen Self-Attention zur parallelen Verarbeitung von Sequenzen.",
        )

        tools_expected = st.multiselect(
            "Erwartete Tools",
            ["search", "calculator", "code_interpreter", "web_browser"],
            [],
        )

        if st.button("🏆 Reward berechnen", type="primary", use_container_width=True):
            st.session_state.reward_clicked = True
        else:
            if "reward_clicked" not in st.session_state:
                st.session_state.reward_clicked = False

    with col2:
        st.subheader("📊 Reward-Ergebnis")

        if st.session_state.reward_clicked:
            with st.spinner("Berechne Reward..."):
                reward_model = AgentRewardModel(use_model=False)
                result = reward_model.compute_reward(
                    prompt=prompt,
                    completion=completion,
                    ground_truth=ground_truth if ground_truth else None,
                    tools_expected=tools_expected if tools_expected else None,
                )

            # Gesamt-Reward
            total = result["total"]
            color = "green" if total > 1.5 else "orange" if total > 0.8 else "red"
            st.markdown(f"### Gesamt-Reward: <span style='color:{color}'>{total:.3f}</span>",
                        unsafe_allow_html=True)

            # Einzel-Rewards
            reward_items = {k: v for k, v in result.items() if k != "total"}
            fig, ax = plt.subplots(figsize=(6, 4))
            names = list(reward_items.keys())
            values = list(reward_items.values())
            colors = ['#4ECDC4', '#45B7D1', '#FFE66D', '#FF6B6B', '#F38181'][:len(names)]
            bars = ax.barh(names, values, color=colors, edgecolor='white')
            ax.set_xlim(0, 1.0)
            ax.set_xlabel("Score")
            ax.set_title("Reward-Komponenten")
            for bar, val in zip(bars, values):
                ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                        f"{val:.2f}", va='center', fontsize=10)
            st.pyplot(fig)
            plt.close(fig)

            # Details
            st.markdown("**Details:**")
            for name, val in reward_items.items():
                emoji = "✅" if val > 0.7 else "⚠️" if val > 0.3 else "❌"
                st.markdown(f"{emoji} **{name}**: {val:.3f}")
        else:
            st.info("👈 Gib links Prompt und Antwort ein und klicke auf **Reward berechnen**.")

    # Reward-Gewichte
    st.markdown("---")
    st.subheader("⚖️ Reward-Gewichte konfigurieren")

    col_a, col_b = st.columns(2)

    with col_a:
        w_correctness = st.slider("Correctness", 0.0, 2.0, 1.0, 0.1)
        w_format = st.slider("Format", 0.0, 2.0, 0.3, 0.1)
        w_helpfulness = st.slider("Helpfulness", 0.0, 2.0, 0.5, 0.1)

    with col_b:
        w_safety = st.slider("Safety", 0.0, 2.0, 0.8, 0.1)
        w_tool_usage = st.slider("Tool Usage", 0.0, 2.0, 0.4, 0.1)

    # Gewichte visualisieren
    weights = {
        "Correctness": w_correctness,
        "Format": w_format,
        "Helpfulness": w_helpfulness,
        "Safety": w_safety,
        "Tool Usage": w_tool_usage,
    }

    fig, ax = plt.subplots(figsize=(6, 3))
    names = list(weights.keys())
    values = list(weights.values())
    colors = ['#4ECDC4', '#45B7D1', '#FFE66D', '#FF6B6B', '#F38181']
    ax.bar(names, values, color=colors, edgecolor='white')
    ax.set_ylabel("Gewicht")
    ax.set_title("Reward-Gewichte")
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    st.pyplot(fig)
    plt.close(fig)

# ═══════════════════════════════════════════════════════════════
# LORA-KONFIGURATION
# ═══════════════════════════════════════════════════════════════
elif seite == "🔧 LoRA-Konfiguration":
    st.header("🔧 LoRA-Parameter einstellen")

    st.markdown("""
    **LoRA** (Low-Rank Adaptation) fügt trainierbare Low-Rank-Matrizen zu
    eingefrorenen Gewichten hinzu. Das reduziert die trainierbaren Parameter
    drastisch (typisch <1% des Originalmodells).
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📐 LoRA-Dimensionen")

        lora_r = st.slider("Rank (r)", 1, 128, 16,
                           help="Rank der Low-Rank-Approximation. Höher = mehr Kapazität, mehr Parameter.")
        lora_alpha = st.slider("Alpha", 1, 128, 32,
                               help="Skalierungsfaktor. Effektive LR skaliert mit alpha/r.")
        lora_dropout = st.slider("Dropout", 0.0, 0.5, 0.05, 0.01)

        st.markdown(f"**Effektiver Skalierungsfaktor:** α/r = {lora_alpha / lora_r:.2f}")

    with col2:
        st.subheader("🎯 Zielmodule")

        target_modules = st.multiselect(
            "Module für LoRA-Adapter",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )

        bias_type = st.selectbox("Bias-Typ", ["none", "all", "lora_only"])

    # LoRA-Parameter berechnen
    st.markdown("---")
    st.subheader("📊 LoRA-Parameter-Analyse")

    # Beispiel: Qwen2.5-7B
    base_params = 7_000_000_000  # 7B
    hidden_size = 4096  # typisch für 7B-Modelle

    lora_params_per_module = 2 * hidden_size * lora_r  # A und B Matrix
    total_lora_params = lora_params_per_module * len(target_modules)
    lora_ratio = total_lora_params / base_params * 100

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Basis-Parameter", f"{base_params / 1e9:.1f}B")
    with col_b:
        st.metric("LoRA-Parameter", f"{total_lora_params / 1e6:.2f}M")
    with col_c:
        st.metric("LoRA-Anteil", f"{lora_ratio:.4f}%")

    # Rank vs Parameter
    ranks = [1, 2, 4, 8, 16, 32, 64, 128]
    params_for_rank = [2 * hidden_size * r * len(target_modules) / 1e6 for r in ranks]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(ranks, params_for_rank, 'o-', color='#4ECDC4', linewidth=2, markersize=8)
    ax1.set_xlabel("LoRA Rank (r)")
    ax1.set_ylabel("Trainierbare Parameter (M)")
    ax1.set_title("Rank → Parameter")
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=lora_r, color='#FF6B6B', linestyle='--', alpha=0.5,
                label=f'Aktuell: r={lora_r}')
    ax1.legend()

    # Modul-Beitrag
    modules = target_modules if target_modules else ["q_proj", "k_proj", "v_proj", "o_proj"]
    mod_params = [2 * hidden_size * lora_r / 1e6] * len(modules)
    ax2.pie(mod_params, labels=modules, autopct='%1.1f%%',
            colors=plt.cm.Set3(np.linspace(0, 1, len(modules))))
    ax2.set_title(f"Parameter-Verteilung (r={lora_r})")

    st.pyplot(fig)
    plt.close(fig)

    # LoRA-Architektur-Diagramm
    st.markdown("---")
    st.subheader("🎨 LoRA-Architektur")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Original Weight
    rect = plt.Rectangle((1, 2.5), 3, 2, facecolor='#E8E8E8', edgecolor='#888888',
                         linewidth=2, alpha=0.8, zorder=1)
    ax.add_patch(rect)
    ax.text(2.5, 3.5, "Eingefrorene\nGewichte W\n(d × k)", ha='center', va='center',
            fontsize=11, fontweight='bold', color='#555')

    # LoRA A
    rect_a = plt.Rectangle((5, 3.5), 1.5, 1, facecolor='#4ECDC4', edgecolor='white',
                           linewidth=2, alpha=0.9, zorder=2)
    ax.add_patch(rect_a)
    ax.text(5.75, 4.0, "A\n(d × r)", ha='center', va='center',
            fontsize=10, fontweight='bold', color='white')

    # LoRA B
    rect_b = plt.Rectangle((7, 3.5), 1.5, 1, facecolor='#FF6B6B', edgecolor='white',
                           linewidth=2, alpha=0.9, zorder=2)
    ax.add_patch(rect_b)
    ax.text(7.75, 4.0, "B\n(r × k)", ha='center', va='center',
            fontsize=10, fontweight='bold', color='white')

    # Output
    ax.text(9.5, 4.0, "ΔW = α/r · BA", ha='center', va='center',
            fontsize=10, fontweight='bold', color='#333')

    # Pfeile
    ax.annotate('', xy=(5, 4.0), xytext=(4, 4.0),
                arrowprops=dict(arrowstyle='->', color='#888888', lw=2))
    ax.annotate('', xy=(7, 4.0), xytext=(6.5, 4.0),
                arrowprops=dict(arrowstyle='->', color='#888888', lw=2))
    ax.annotate('', xy=(9, 4.0), xytext=(8.5, 4.0),
                arrowprops=dict(arrowstyle='->', color='#888888', lw=2))

    # Input
    ax.text(0.5, 4.0, "Input x", ha='center', fontsize=10, fontweight='bold')
    ax.annotate('', xy=(1, 4.0), xytext=(0.8, 4.0),
                arrowprops=dict(arrowstyle='->', color='#888888', lw=2))

    # Formel
    ax.text(5, 1.5, "h = Wx + (α/r) · BAx", ha='center', fontsize=14,
            fontweight='bold', color='#333',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#F0F0F0', alpha=0.8))

    ax.set_title("LoRA: Low-Rank Adaptation", fontsize=14, fontweight='bold')
    st.pyplot(fig)
    plt.close(fig)

# ═══════════════════════════════════════════════════════════════
# GESAMTKONFIGURATION
# ═══════════════════════════════════════════════════════════════
elif seite == "📋 Gesamtkonfiguration":
    st.header("📋 Gesamtkonfiguration")

    preset = st.selectbox(
        "Vordefinierte Konfiguration",
        ["Qwen2.5-7B", "Llama-3.1-8B", "Small Test (1.5B)"],
    )

    if preset == "Qwen2.5-7B":
        config = get_qwen_config()
    elif preset == "Llama-3.1-8B":
        config = get_llama_config()
    else:
        config = get_small_test_config()

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Modell", "LoRA", "GRPO", "Reward", "Daten"]
    )

    with tab1:
        st.json({
            "model_name": config.model.model_name_or_path,
            "load_in_4bit": config.model.load_in_4bit,
            "attn_implementation": config.model.attn_implementation,
        })

    with tab2:
        st.json({
            "r": config.lora.r,
            "lora_alpha": config.lora.lora_alpha,
            "target_modules": config.lora.target_modules,
            "lora_dropout": config.lora.lora_dropout,
        })

    with tab3:
        st.json({
            "num_generations": config.grpo.num_generations,
            "learning_rate": config.grpo.learning_rate,
            "beta": config.grpo.beta,
            "max_prompt_length": config.grpo.max_prompt_length,
            "max_completion_length": config.grpo.max_completion_length,
            "temperature": config.grpo.temperature,
            "per_device_train_batch_size": config.grpo.per_device_train_batch_size,
            "gradient_accumulation_steps": config.grpo.gradient_accumulation_steps,
        })

    with tab4:
        st.json({
            "reward_model": config.reward.reward_model_name_or_path,
            "reward_weights": config.reward.reward_weights,
            "correctness_threshold": config.reward.correctness_threshold,
            "safety_threshold": config.reward.safety_threshold,
        })

    with tab5:
        st.json({
            "train_file": config.data.train_file,
            "eval_file": config.data.eval_file,
            "dataset_format": config.data.dataset_format,
        })

    st.markdown("---")
    st.markdown(f"**Experiment:** `{config.experiment_name}`")
    st.markdown(f"**Output:** `{config.grpo.output_dir}`")

st.sidebar.markdown("---")
st.sidebar.markdown("📁 **Repo:** [ART](https://github.com/mark-baumann/ART)")
st.sidebar.markdown("🐍 **Python 3.13** · **Streamlit** · **GRPO + LoRA**")
