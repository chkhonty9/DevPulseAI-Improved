"""Streamlit interface for the DevPulseAI improvement project."""

from __future__ import annotations

import os
from time import perf_counter
from typing import Any, Dict, List, Tuple

import streamlit as st

from adapters.arxiv import fetch_arxiv_papers
from adapters.github import fetch_github_trending
from adapters.hackernews import fetch_hackernews_stories
from adapters.huggingface import fetch_huggingface_models
from adapters.medium import fetch_medium_blogs
from agents import (
    AnalysisFailedError,
    CriticAgent,
    RelevanceAgent,
    RiskAgent,
    SignalCollector,
    SynthesisAgent,
)
from config import (
    DEFAULT_GETQHSE_MODEL,
    MissingGetQHSEKeyError,
    resolve_getqhse_settings,
)
from main import DEFAULT_SIGNAL_LIMIT
from models import (
    ContractValidationError,
    DeveloperProfile,
    validate_contract,
)
from services import (
    LLMClient,
    LLMRequestError,
    ObservabilityLLMClient,
    PROVIDERS,
    GetQHSEClient,
    ProviderConfigurationError,
    create_provider_client,
    validate_provider_key,
)
from workflows import ReviewedSynthesisWorkflow


st.set_page_config(
    page_title="DevPulseAI — Developer Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
        :root {
            --dp-bg: #080d19;
            --dp-panel: #11192b;
            --dp-panel-soft: #151f35;
            --dp-border: rgba(148, 163, 184, 0.17);
            --dp-text: #edf3ff;
            --dp-muted: #9cabca;
            --dp-blue: #6ea8fe;
            --dp-violet: #a78bfa;
            --dp-green: #41d6a3;
        }

        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(circle at 82% 2%, rgba(78, 116, 255, 0.13), transparent 34rem),
                radial-gradient(circle at 12% 86%, rgba(124, 58, 237, 0.09), transparent 30rem),
                var(--dp-bg);
            color: var(--dp-text);
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        [data-testid="stSidebar"] {
            background: #0d1424;
            border-right: 1px solid var(--dp-border);
        }

        [data-testid="stSidebar"] > div:first-child {
            padding-top: 1.2rem;
        }

        .block-container {
            max-width: 1220px;
            padding-top: 2.4rem;
            padding-bottom: 4rem;
        }

        h1, h2, h3, p, label, [data-testid="stMarkdownContainer"] {
            color: var(--dp-text);
        }

        .dp-sidebar-brand {
            display: flex;
            align-items: center;
            gap: 0.7rem;
            padding: 0.35rem 0 1.1rem;
        }

        .dp-sidebar-mark {
            width: 2.25rem;
            height: 2.25rem;
            display: grid;
            place-items: center;
            border-radius: 0.75rem;
            background: linear-gradient(135deg, #6196ff, #8b5cf6);
            color: white;
            font-size: 1.2rem;
            box-shadow: 0 10px 30px rgba(78, 116, 255, 0.25);
        }

        .dp-sidebar-name {
            font-size: 1.16rem;
            font-weight: 750;
            letter-spacing: -0.02em;
        }

        .dp-sidebar-version {
            color: var(--dp-muted);
            font-size: 0.76rem;
            margin-top: -0.12rem;
        }

        .dp-section-label {
            color: #7887a8;
            font-size: 0.72rem;
            font-weight: 750;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin: 0.3rem 0 0.55rem;
        }

        .dp-connection {
            border: 1px solid rgba(65, 214, 163, 0.22);
            background: rgba(65, 214, 163, 0.07);
            border-radius: 0.75rem;
            padding: 0.75rem 0.85rem;
            color: #c9ffeb;
            font-size: 0.84rem;
            margin-bottom: 0.75rem;
        }

        .dp-dot {
            display: inline-block;
            width: 0.5rem;
            height: 0.5rem;
            border-radius: 50%;
            background: var(--dp-green);
            box-shadow: 0 0 0 4px rgba(65, 214, 163, 0.10);
            margin-right: 0.55rem;
        }

        .dp-hero {
            position: relative;
            overflow: hidden;
            border: 1px solid var(--dp-border);
            border-radius: 1.5rem;
            padding: 3.1rem 3.2rem;
            background: linear-gradient(135deg, rgba(18, 27, 47, 0.97), rgba(11, 17, 31, 0.96));
            box-shadow: 0 24px 70px rgba(0, 0, 0, 0.24);
            margin-bottom: 1.25rem;
        }

        .dp-hero::after {
            content: "";
            position: absolute;
            width: 19rem;
            height: 19rem;
            right: -4rem;
            top: -8rem;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(110, 168, 254, 0.24), transparent 67%);
        }

        .dp-eyebrow {
            color: var(--dp-blue);
            font-size: 0.76rem;
            font-weight: 800;
            letter-spacing: 0.15em;
            text-transform: uppercase;
            margin-bottom: 1rem;
        }

        .dp-title {
            color: white;
            font-size: clamp(2.4rem, 5vw, 4.45rem);
            line-height: 0.98;
            letter-spacing: -0.055em;
            font-weight: 820;
            max-width: 780px;
            margin: 0;
        }

        .dp-title span {
            background: linear-gradient(90deg, #77aaff, #b39aff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .dp-subtitle {
            color: var(--dp-muted);
            font-size: 1.05rem;
            line-height: 1.7;
            max-width: 720px;
            margin: 1.35rem 0 0;
        }

        .dp-chips {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            margin-top: 1.6rem;
        }

        .dp-chip {
            color: #cbd8f3;
            background: rgba(255, 255, 255, 0.045);
            border: 1px solid var(--dp-border);
            border-radius: 999px;
            padding: 0.42rem 0.75rem;
            font-size: 0.79rem;
        }

        .dp-card {
            min-height: 148px;
            border: 1px solid var(--dp-border);
            background: rgba(17, 25, 43, 0.88);
            border-radius: 1rem;
            padding: 1.3rem 1.35rem;
        }

        .dp-card-number {
            color: var(--dp-blue);
            font-size: 0.74rem;
            font-weight: 800;
            letter-spacing: 0.12em;
        }

        .dp-card-title {
            color: white;
            font-size: 1.02rem;
            font-weight: 720;
            margin: 0.62rem 0 0.42rem;
        }

        .dp-card-copy {
            color: var(--dp-muted);
            font-size: 0.86rem;
            line-height: 1.55;
        }

        .dp-result-title {
            color: white;
            font-size: 1.8rem;
            font-weight: 760;
            letter-spacing: -0.03em;
            margin: 1rem 0 0.3rem;
        }

        div[data-testid="stMetric"] {
            background: rgba(17, 25, 43, 0.9);
            border: 1px solid var(--dp-border);
            border-radius: 1rem;
            padding: 1rem 1.1rem;
        }

        div[data-testid="stExpander"] {
            border: 1px solid var(--dp-border);
            background: rgba(17, 25, 43, 0.78);
            border-radius: 0.9rem;
        }

        .stButton > button {
            border-radius: 0.75rem;
            min-height: 2.8rem;
            font-weight: 700;
        }

        .stButton > button[kind="primary"] {
            border: 0;
            background: linear-gradient(90deg, #4f7df3, #7657e8);
            box-shadow: 0 10px 24px rgba(79, 125, 243, 0.22);
        }

        [data-testid="stAlert"] {
            border-radius: 0.85rem;
        }

        @media (max-width: 760px) {
            .dp-hero { padding: 2rem 1.4rem; }
            .block-container { padding-top: 1.1rem; }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def _load_streamlit_secrets() -> Dict[str, Any]:
    try:
        return dict(st.secrets)
    except (FileNotFoundError, RuntimeError):
        return {}


def _build_runtime_client(
    *,
    key_mode: str,
    provider: str,
    user_api_key: str | None,
    model: str,
    secrets: Dict[str, Any],
) -> Tuple[LLMClient, str]:
    """Construct the selected provider client without persisting a user key."""
    if key_mode == "local":
        settings = resolve_getqhse_settings(
            model=model,
            secrets=secrets,
        )
        return GetQHSEClient(settings), settings.key_source

    client = create_provider_client(
        provider,
        api_key=user_api_key or "",
        model=model,
    )
    return client, "current browser session"


def _collect_selected_signals(
    sources: List[str], signal_count: int
) -> List[Dict[str, Any]]:
    signals: List[Dict[str, Any]] = []

    if "GitHub" in sources:
        signals.extend(fetch_github_trending(limit=signal_count))
    if "ArXiv" in sources:
        signals.extend(fetch_arxiv_papers(limit=signal_count))
    if "Hacker News" in sources:
        signals.extend(fetch_hackernews_stories(limit=signal_count))
    if "Medium" in sources:
        signals.extend(fetch_medium_blogs(limit=min(signal_count, 3)))
    if "Hugging Face" in sources:
        signals.extend(fetch_huggingface_models(limit=signal_count))

    return signals


def _parse_profile_items(value: str) -> List[str]:
    """Parse a comma-separated profile field while preserving order."""
    items: List[str] = []
    seen = set()
    for raw_item in value.split(","):
        item = raw_item.strip()
        normalized = item.casefold()
        if item and normalized not in seen:
            items.append(item)
            seen.add(normalized)
    return items


def _render_signal(signal: Dict[str, Any]) -> None:
    relevance = signal.get("relevance", {})
    risk = signal.get("risk", {})
    risk_level = risk.get("risk_level", "UNKNOWN")
    score = relevance.get("score", 0)

    with st.expander(
        f"{score:>3} · {risk_level:<8} · "
        f"[{signal.get('source', 'unknown').upper()}] {signal.get('title', 'Untitled')}"
    ):
        details, assessment = st.columns([3, 1])
        with details:
            st.markdown(signal.get("description") or "No description available.")
            matched_technologies = relevance.get("matched_technologies") or []
            matched_interests = relevance.get("matched_interests") or []
            if matched_technologies:
                st.caption(
                    "Technology matches: " + ", ".join(matched_technologies)
                )
            if matched_interests:
                st.caption("Interest matches: " + ", ".join(matched_interests))
            if signal.get("url"):
                st.link_button("Open source", signal["url"])
            concerns = risk.get("concerns") or []
            evidence_quotes = risk.get("evidence_quotes") or []
            recommended_actions = risk.get("recommended_actions") or []
            if concerns:
                st.markdown("**Risk concerns**")
                for concern, evidence_quote, recommended_action in zip(
                    concerns,
                    evidence_quotes,
                    recommended_actions,
                ):
                    st.write(f"• {concern}")
                    st.caption(f'Evidence: “{evidence_quote}”')
                    st.caption(f"Recommended action: {recommended_action}")

        with assessment:
            st.metric("Relevance", f"{score}/100")
            st.metric("Risk", risk_level)
            st.metric("Actionability", relevance.get("actionability", "UNKNOWN"))
            if relevance.get("reasoning"):
                st.caption(relevance["reasoning"])


def _render_pipeline_observability(
    *,
    observed_client: ObservabilityLLMClient,
    timeline: List[Dict[str, Any]],
    raw_signals: List[Dict[str, Any]],
    normalized_signals: List[Dict[str, Any]],
    analyzed_signals: List[Dict[str, Any]],
    provider: str,
    model: str,
    key_source: str,
    total_duration_ms: int,
) -> None:
    """Render secret-safe call, timing, token, and provenance metadata."""
    snapshot = observed_client.snapshot()
    with st.expander("Pipeline observability", expanded=True):
        st.caption(
            f"Provider: {provider} · Model: {model} · Key source: {key_source}. "
            "Prompts, responses, and API keys are not retained here."
        )
        obs_one, obs_two, obs_three, obs_four = st.columns(4)
        obs_one.metric("End-to-end time", f"{total_duration_ms / 1000:.1f}s")
        obs_two.metric("Model calls", snapshot["call_count"])
        obs_three.metric("Total tokens", snapshot["total_tokens"])
        obs_four.metric(
            "Provider latency",
            f"{snapshot['provider_latency_ms'] / 1000:.1f}s",
        )

        st.markdown("**Execution timeline**")
        st.dataframe(timeline, hide_index=True, use_container_width=True)

        st.markdown("**AI usage by stage**")
        if snapshot["by_stage"]:
            st.dataframe(
                snapshot["by_stage"],
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.caption("No model calls were recorded.")

        raw_counts: Dict[str, int] = {}
        normalized_counts: Dict[str, int] = {}
        analyzed_counts: Dict[str, int] = {}
        for item in raw_signals:
            source = item.get("source", "unknown")
            raw_counts[source] = raw_counts.get(source, 0) + 1
        for item in normalized_signals:
            source = item.get("source", "unknown")
            normalized_counts[source] = normalized_counts.get(source, 0) + 1
        for item in analyzed_signals:
            source = item.get("source", "unknown")
            analyzed_counts[source] = analyzed_counts.get(source, 0) + 1
        all_sources = sorted(
            set(raw_counts) | set(normalized_counts) | set(analyzed_counts)
        )
        provenance_rows = [
            {
                "source": source,
                "raw": raw_counts.get(source, 0),
                "normalized": normalized_counts.get(source, 0),
                "analyzed": analyzed_counts.get(source, 0),
            }
            for source in all_sources
        ]
        st.markdown("**Signal provenance**")
        st.dataframe(
            provenance_rows,
            hide_index=True,
            use_container_width=True,
        )

        with st.expander("Individual model calls"):
            st.dataframe(
                snapshot["calls"],
                hide_index=True,
                use_container_width=True,
            )


secret_values = _load_streamlit_secrets()

with st.sidebar:
    st.markdown(
        """
        <div class="dp-sidebar-brand">
            <div class="dp-sidebar-mark">◈</div>
            <div>
                <div class="dp-sidebar-name">DevPulseAI</div>
                <div class="dp-sidebar-version">Improvement workspace</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="dp-section-label">AI connection</div>', unsafe_allow_html=True)
    key_source_label = st.selectbox(
        "Key source",
        ["Local GetQHSE configuration", "Enter my own API key"],
        label_visibility="collapsed",
        help=(
            "Local configuration reads a protected GetQHSE key from the backend. "
            "A key entered here is kept only in the current Streamlit session."
        ),
    )
    key_mode = (
        "local"
        if key_source_label == "Local GetQHSE configuration"
        else "user"
    )

    active_provider = "GetQHSE Hosted LLM"
    provider_definition = None
    if key_mode == "user":
        active_provider = st.selectbox(
            "AI provider",
            list(PROVIDERS),
            help="Choose the company that issued the API key.",
        )
        provider_definition = PROVIDERS[active_provider]

    user_api_key = None
    if key_mode == "user":
        assert provider_definition is not None
        user_api_key = st.text_input(
            provider_definition.key_label,
            type="password",
            placeholder=provider_definition.key_placeholder,
            help=(
                f"Used directly with {active_provider} and kept only for the "
                "current Streamlit session."
            ),
            key=f"api_key_{active_provider}",
        )

    local_key_available = bool(
        os.environ.get("GETQHSE_API_KEY")
        or os.environ.get("OLLAMA_API_KEY")
        or secret_values.get("GETQHSE_API_KEY")
        or secret_values.get("OLLAMA_API_KEY")
    )
    key_available = local_key_available if key_mode == "local" else bool(user_api_key)
    key_configuration_error = None
    if key_mode == "user" and user_api_key:
        try:
            validate_provider_key(active_provider, user_api_key)
        except ProviderConfigurationError as exc:
            key_configuration_error = str(exc)
    configuration_ready = key_available and not key_configuration_error

    if key_configuration_error:
        st.error(key_configuration_error)
    elif key_available:
        connection_copy = (
            "Protected GetQHSE key ready · test before running"
            if key_mode == "local"
            else f"{active_provider} key entered · test before running"
        )
        st.markdown(
            '<div class="dp-connection"><span class="dp-dot"></span>'
            f"{connection_copy}</div>",
            unsafe_allow_html=True,
        )
    else:
        if key_mode == "local":
            st.warning(
                "Add GETQHSE_API_KEY to the environment or "
                ".streamlit/secrets.toml."
            )
        else:
            assert provider_definition is not None
            st.warning(f"Add a {provider_definition.key_label} to continue.")

    if key_mode == "local":
        configured_model = str(
            os.environ.get("GETQHSE_MODEL")
            or secret_values.get("GETQHSE_MODEL")
            or DEFAULT_GETQHSE_MODEL
        )
    else:
        assert provider_definition is not None
        configured_model = provider_definition.default_model

    with st.expander("Model settings", expanded=False):
        discovered_models = st.session_state.get("getqhse_models", [])
        if key_mode == "local" and st.button(
            "Refresh available models",
            use_container_width=True,
            disabled=not key_available,
        ):
            try:
                discovery_client, _ = _build_runtime_client(
                    key_mode="local",
                    provider=active_provider,
                    user_api_key=None,
                    model=configured_model,
                    secrets=secret_values,
                )
                with st.spinner("Loading models from GetQHSE…"):
                    discovered_models = discovery_client.list_models()
                st.session_state["getqhse_models"] = discovered_models
                st.success(f"Found {len(discovered_models)} available models.")
            except (MissingGetQHSEKeyError, LLMRequestError) as exc:
                st.error(str(exc))

        if key_mode == "local" and discovered_models:
            default_index = (
                discovered_models.index(configured_model)
                if configured_model in discovered_models
                else 0
            )
            model_id = st.selectbox(
                "GetQHSE model",
                discovered_models,
                index=default_index,
                help="Model IDs are loaded from the gateway's /v1/models endpoint.",
            )
        else:
            model_help = (
                "Use an exact ID returned by the GetQHSE models endpoint."
                if key_mode == "local"
                else provider_definition.model_help
            )
            model_id = st.text_input(
                f"{active_provider} model",
                value=configured_model,
                help=model_help,
                key=f"model_{key_mode}_{active_provider}",
            )

        if key_mode == "local":
            st.caption(
                "Requests use the protected GetQHSE gateway configuration. "
                "Refresh to discover the currently installed models."
            )
        elif active_provider == "OpenRouter":
            st.caption(
                "openrouter/free still requires an OpenRouter key. Free model "
                "availability and rate limits can vary."
            )
        else:
            st.caption(
                f"Requests go directly to {active_provider}; its own quota and "
                "billing rules apply."
            )

    test_connection = st.button(
        "Test connection",
        use_container_width=True,
        disabled=not configuration_ready or not model_id.strip(),
    )

    if test_connection:
        try:
            test_client, _ = _build_runtime_client(
                key_mode=key_mode,
                provider=active_provider,
                user_api_key=user_api_key,
                model=model_id,
                secrets=secret_values,
            )
            with st.spinner(f"Contacting {active_provider}…"):
                test_result = test_client.test_connection()
            st.success(
                f"Connected · {test_result.model} · {test_result.latency_ms} ms"
            )
        except (
            MissingGetQHSEKeyError,
            ProviderConfigurationError,
            LLMRequestError,
        ) as exc:
            st.error(str(exc))

    st.divider()
    st.markdown(
        '<div class="dp-section-label">Developer profile</div>',
        unsafe_allow_html=True,
    )
    with st.expander("Personalize relevance", expanded=True):
        technologies_text = st.text_input(
            "Technologies",
            value="Python, JavaScript",
            help="Comma-separated technologies you currently use.",
        )
        interests_text = st.text_input(
            "Interests",
            value="AI agents, LLM tooling",
            help="Comma-separated topics you want DevPulseAI to prioritize.",
        )
        experience_level = st.selectbox(
            "Experience level",
            ["beginner", "intermediate", "advanced"],
            index=1,
        )

    profile_error = None
    developer_profile = None
    try:
        developer_profile = validate_contract(
            DeveloperProfile,
            {
                "technologies": _parse_profile_items(technologies_text),
                "interests": _parse_profile_items(interests_text),
                "experience_level": experience_level,
            },
            context="Developer profile",
        )
    except ContractValidationError as exc:
        profile_error = str(exc)
        st.error(profile_error)
    profile_ready = developer_profile is not None

    st.markdown('<div class="dp-section-label">Signal feed</div>', unsafe_allow_html=True)
    sources = st.multiselect(
        "Sources",
        ["GitHub", "ArXiv", "Hacker News", "Medium", "Hugging Face"],
        default=["GitHub", "ArXiv", "Hacker News"],
    )
    signal_count = st.slider(
        "Signals per source",
        min_value=1,
        max_value=10,
        value=min(DEFAULT_SIGNAL_LIMIT, 5),
    )

    run_button = st.button(
        "Run intelligence pipeline",
        type="primary",
        use_container_width=True,
        disabled=(
            not configuration_ready
            or not profile_ready
            or not sources
            or not model_id.strip()
        ),
    )
    if not configuration_ready:
        st.caption("The pipeline unlocks after a matching API key is available.")
    elif not profile_ready:
        st.caption("Add at least one technology or interest to run the pipeline.")


st.markdown(
    """
    <section class="dp-hero">
        <div class="dp-eyebrow">Developer signal intelligence</div>
        <h1 class="dp-title">Know what changed.<br><span>Know what matters.</span></h1>
        <p class="dp-subtitle">
            DevPulseAI gathers engineering signals, scores their relevance,
            checks for risk, and turns the noise into a focused technical brief.
        </p>
        <div class="dp-chips">
            <span class="dp-chip">Multi-provider AI</span>
            <span class="dp-chip">5 technical sources</span>
            <span class="dp-chip">Deterministic collection</span>
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)


if not run_button:
    st.markdown("### How the baseline works")
    first, second, third = st.columns(3)
    with first:
        st.markdown(
            """
            <div class="dp-card">
                <div class="dp-card-number">01 · COLLECT</div>
                <div class="dp-card-title">Gather technical signals</div>
                <div class="dp-card-copy">Adapters fetch GitHub, arXiv, Hacker News, Medium, and Hugging Face data.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with second:
        st.markdown(
            """
            <div class="dp-card">
                <div class="dp-card-number">02 · ANALYZE</div>
                <div class="dp-card-title">Score relevance and risk</div>
                <div class="dp-card-copy">Your selected AI provider powers the reasoning stages while collection stays deterministic.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with third:
        st.markdown(
            """
            <div class="dp-card">
                <div class="dp-card-number">03 · PRIORITIZE</div>
                <div class="dp-card-title">Build a focused digest</div>
                <div class="dp-card-copy">Signals are ranked so urgent, relevant developments appear first.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if key_available:
        st.info("Configuration is ready. Test the connection or run the pipeline from the sidebar.")
    else:
        st.info("Choose a key source in the sidebar to enable AI analysis.")

else:
    try:
        base_llm_client, runtime_key_source = _build_runtime_client(
            key_mode=key_mode,
            provider=active_provider,
            user_api_key=user_api_key,
            model=model_id,
            secrets=secret_values,
        )
    except (MissingGetQHSEKeyError, ProviderConfigurationError) as exc:
        st.error(str(exc))
        st.stop()

    llm_client = ObservabilityLLMClient(base_llm_client)
    pipeline_started = perf_counter()
    timeline: List[Dict[str, Any]] = []
    collector = SignalCollector()
    relevance = RelevanceAgent(
        model_id=model_id,
        llm_client=llm_client,
    )
    risk = RiskAgent(
        model_id=model_id,
        llm_client=llm_client,
    )
    synthesis = SynthesisAgent(
        model_id=model_id,
        llm_client=llm_client,
    )
    critic = CriticAgent(
        model_id=model_id,
        llm_client=llm_client,
    )
    review_workflow = ReviewedSynthesisWorkflow(synthesis, critic)

    collection_started = perf_counter()
    try:
        with st.status("Collecting technical signals", expanded=True) as status:
            raw_signals = _collect_selected_signals(sources, signal_count)
            st.write(
                f"Fetched {len(raw_signals)} raw signals from {len(sources)} sources."
            )
            normalized = collector.collect(raw_signals)
            st.write(f"Normalized and deduplicated to {len(normalized)} signals.")
            status.update(label="Signal collection complete", state="complete")
        timeline.append(
            {
                "stage": "collection",
                "status": "complete",
                "duration_ms": round(
                    (perf_counter() - collection_started) * 1000
                ),
                "items": len(normalized),
            }
        )
    except ContractValidationError as exc:
        st.error("Signal collection stopped because source data was invalid.")
        st.warning(str(exc))
        st.stop()

    try:
        analysis_left, analysis_right = st.columns(2)
        with analysis_left:
            relevance_started = perf_counter()
            with st.status("Scoring developer relevance") as status:
                scored = relevance.score_batch(normalized, developer_profile)
                status.update(label="Relevance scoring complete", state="complete")
            timeline.append(
                {
                    "stage": "relevance",
                    "status": "complete",
                    "duration_ms": round(
                        (perf_counter() - relevance_started) * 1000
                    ),
                    "items": len(scored),
                }
            )

        with analysis_right:
            risk_started = perf_counter()
            with st.status("Assessing technical risk") as status:
                assessed = risk.assess_batch(scored)
                status.update(label="Risk assessment complete", state="complete")
            timeline.append(
                {
                    "stage": "risk",
                    "status": "complete",
                    "duration_ms": round((perf_counter() - risk_started) * 1000),
                    "items": len(assessed),
                }
            )
    except (LLMRequestError, AnalysisFailedError, ContractValidationError) as exc:
        st.error("AI analysis stopped — no heuristic result was substituted.")
        st.warning(str(exc))
        st.info(
            f"Check the {active_provider} key and model, or wait if the provider "
            "is rate-limiting requests. Then run the pipeline again."
        )
        st.stop()

    synthesis_started = perf_counter()
    try:
        with st.status("Building intelligence digest") as status:
            digest = synthesis.synthesize(assessed)
            status.update(label="Digest complete", state="complete")
        timeline.append(
            {
                "stage": "synthesis",
                "status": "complete",
                "duration_ms": round(
                    (perf_counter() - synthesis_started) * 1000
                ),
                "items": len(digest.get("priority_signals", [])),
            }
        )
    except (LLMRequestError, AnalysisFailedError, ContractValidationError) as exc:
        st.error("AI digest generation stopped — no fallback was substituted.")
        st.warning(str(exc))
        st.info(
            "The synthesis model must return valid JSON and cite only evidence "
            "IDs generated from this run. Check the provider/model and retry."
        )
        st.stop()

    review_started = perf_counter()
    try:
        with st.status("Reviewing and revising digest quality") as status:
            pipeline_result = review_workflow.run(
                assessed,
                developer_profile,
                initial_digest=digest,
            )
            digest = pipeline_result["digest"]
            critic_result = pipeline_result["critic"]
            revision_history = pipeline_result["revision_history"]
            revisions_used = pipeline_result["revision_attempts"]
            review_label = (
                f"Review passed after {revisions_used} revision(s)"
                if pipeline_result["status"] == "completed"
                else "Review still needs attention after 2 revisions"
            )
            status.update(label=review_label, state="complete")
        timeline.append(
            {
                "stage": "critic_and_revisions",
                "status": pipeline_result["status"],
                "duration_ms": round((perf_counter() - review_started) * 1000),
                "items": revisions_used,
            }
        )
    except (LLMRequestError, AnalysisFailedError, ContractValidationError) as exc:
        st.error("Independent review stopped — no review score was fabricated.")
        st.warning(str(exc))
        st.info("Check the selected provider/model, then run the pipeline again.")
        st.stop()

    priority_signals = digest.get("priority_signals", [])
    elevated_risks = sum(
        1
        for signal in assessed
        if signal.get("risk", {}).get("risk_level") in {"HIGH", "CRITICAL"}
    )
    high_relevance = sum(
        1
        for signal in assessed
        if signal.get("relevance", {}).get("score", 0) >= 70
    )

    st.markdown('<div class="dp-result-title">Intelligence digest</div>', unsafe_allow_html=True)
    st.caption(
        f"Provider: {active_provider} · Model: {model_id} · "
        f"Key source: {runtime_key_source}"
    )
    st.caption(
        "Profile: "
        f"{developer_profile.experience_level} · "
        f"Technologies: {', '.join(developer_profile.technologies) or 'none'} · "
        f"Interests: {', '.join(developer_profile.interests) or 'none'}"
    )

    metric_one, metric_two, metric_three = st.columns(3)
    metric_one.metric("Signals analyzed", len(assessed))
    metric_two.metric("High relevance", high_relevance)
    metric_three.metric("Elevated risk", elevated_risks)

    _render_pipeline_observability(
        observed_client=llm_client,
        timeline=timeline,
        raw_signals=raw_signals,
        normalized_signals=normalized,
        analyzed_signals=assessed,
        provider=active_provider,
        model=model_id,
        key_source=runtime_key_source,
        total_duration_ms=round((perf_counter() - pipeline_started) * 1000),
    )

    st.markdown("#### Independent quality review")
    if critic_result["passed"]:
        st.success("Passed — the digest met the evidence and quality thresholds.")
    else:
        st.warning(
            "Needs review — the digest still did not meet every quality "
            "threshold after the bounded revision process."
        )
    st.caption(f"Revisions used: {revisions_used}/2")
    if pipeline_result.get("revision_failure"):
        st.warning(
            "A revision response was invalid, so the last validated digest "
            "was preserved and marked for review."
        )
        st.caption(pipeline_result["revision_failure"])

    critic_one, critic_two, critic_three = st.columns(3)
    critic_one.metric("Groundedness", f"{critic_result['groundedness_score']:.0f}/100")
    critic_two.metric(
        "Profile relevance",
        f"{critic_result['profile_relevance_score']:.0f}/100",
    )
    critic_three.metric(
        "Completeness",
        f"{critic_result['completeness_score']:.0f}/100",
    )

    if critic_result.get("unsupported_claims"):
        st.markdown("**Unsupported claims**")
        for claim in critic_result["unsupported_claims"]:
            st.write(f"• {claim}")
    if critic_result.get("feedback"):
        with st.expander("Critic feedback"):
            for feedback_item in critic_result["feedback"]:
                st.write(f"• {feedback_item}")

    with st.expander("Revision history"):
        for record in revision_history:
            record_critic = record["critic"]
            verdict = "Passed" if record_critic["passed"] else "Needs review"
            st.markdown(
                f"**Revision {record['revision_number']} · {verdict}**  \n"
                f"Groundedness {record_critic['groundedness_score']:.0f} · "
                f"Profile relevance "
                f"{record_critic['profile_relevance_score']:.0f} · "
                f"Completeness {record_critic['completeness_score']:.0f}"
            )
            st.write(record["digest_summary"])
            for feedback_item in record_critic.get("feedback", []):
                st.caption(feedback_item)
            st.divider()

    st.info(digest["executive_summary"])
    st.caption(
        "Summary evidence: "
        + ", ".join(digest.get("summary_evidence_ids", []))
    )

    st.markdown("#### Recommended actions")
    for recommendation in digest.get("recommendations", []):
        st.write(f"• {recommendation['text']}")
        st.caption("Evidence: " + ", ".join(recommendation["evidence_ids"]))

    with st.expander("Evidence pack used by the synthesis model"):
        for evidence in digest.get("evidence_pack", []):
            st.markdown(
                f"**{evidence['evidence_id']} · {evidence['title']}**  \n"
                f"Source: {evidence['source']} · Relevance: "
                f"{evidence['relevance_score']} · Risk: {evidence['risk_level']}"
            )
            st.write(evidence["claim"])
            if evidence.get("source_url"):
                st.link_button("Open source", evidence["source_url"])
            st.divider()

    st.markdown("#### Priority signals")
    if priority_signals:
        for priority_signal in priority_signals:
            _render_signal(priority_signal)
    else:
        st.info("No priority signals were produced for this run.")
