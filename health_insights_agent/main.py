# |---------------------------------------------------------|
# |                                                         |
# |                 Give Feedback / Get Help                |
# | https://github.com/getbindu/Bindu/issues/new/choose    |
# |                                                         |
# |---------------------------------------------------------|
#
#  Thank you users! We ❤️ you! - 🌻

"""health-insights-agent - HIA Medical Analysis Agent using Bindu Framework."""

import argparse
import asyncio
import json
import os
from pathlib import Path
from textwrap import dedent
from typing import Any

from agno.agent import Agent
from agno.models.openrouter import OpenRouter
from agno.tools import Toolkit
from agno.tools.mem0 import Mem0Tools
from bindu.penguin.bindufy import bindufy
from dotenv import load_dotenv

from health_insights_agent.tools import (
    analyze_medical_report,
    extract_text_from_pdf,
    extract_health_indicators,
    generate_health_insights,
    assess_health_risks,
    get_medical_reference_ranges,
    validate_medical_content,
    generate_recommendations,
)

# Load environment variables from .env file
load_dotenv()

# Global tools instances
health_tools: Toolkit | None = None
agent: Agent | None = None
model_name: str | None = None
openrouter_api_key: str | None = None
mem0_api_key: str | None = None
_initialized = False
_init_lock = asyncio.Lock()


class HealthAnalysisTools(Toolkit):
    """Custom toolkit for health analysis functions."""

    def __init__(self):
        super().__init__(name="health_analysis_tools")
        self.register(analyze_medical_report)
        self.register(extract_text_from_pdf)
        self.register(extract_health_indicators)
        self.register(generate_health_insights)
        self.register(assess_health_risks)
        self.register(get_medical_reference_ranges)
        self.register(validate_medical_content)
        self.register(generate_recommendations)


def initialize_health_tools() -> None:
    """Initialize all health analysis tools as a Toolkit instance."""
    global health_tools

    health_tools = HealthAnalysisTools()
    print("✅ Health analysis tools initialized")


def load_config() -> dict:
    """Load agent configuration from project root."""
    # Get path to agent_config.json in project root
    config_path = Path(__file__).parent / "agent_config.json"

    with open(config_path) as f:
        return json.load(f)


# Create the agent instance
async def initialize_agent() -> None:
    """Initialize the agent once."""
    global agent, model_name, health_tools

    if not model_name:
        msg = "model_name must be set before initializing agent"
        raise ValueError(msg)

    # Initialize health tools if not already done
    if not health_tools:
        initialize_health_tools()

    agent = Agent(
        name="Health Insights Agent (HIA)",
        model=OpenRouter(
            id=model_name,
            api_key=openrouter_api_key,
            cache_response=True,
            supports_native_structured_outputs=True,
        ),
        tools=[
            tool
            for tool in [
                health_tools,  # Health analysis toolkit
                Mem0Tools(api_key=mem0_api_key) if mem0_api_key else None,
            ]
            if tool is not None
        ],
        description=dedent("""\
            You are an expert medical analyst with comprehensive knowledge of laboratory medicine, 
            hematology, and clinical diagnostics. Your expertise encompasses: 🩺

            - Complete Blood Count (CBC) analysis and interpretation
            - Liver function tests and hepatic health assessment
            - Kidney function and metabolic panel analysis
            - Lipid profile and cardiovascular risk assessment
            - Thyroid function and endocrine evaluation
            - Electrolyte balance and fluid status
            - Medical report analysis and health insights
            - Risk assessment and preventive care recommendations\
        """),
        instructions=dedent("""\
            1. Report Analysis Phase 📋
               - Use health analysis tools to process medical reports
               - Extract and validate health indicators from blood test results
               - Analyze complete blood count, metabolic panel, and organ function tests
               - Identify abnormalities and reference range deviations

            2. Clinical Interpretation Phase 🔍
               - Interpret abnormal values in clinical context
               - Assess organ system function (liver, kidney, thyroid, etc.)
               - Evaluate metabolic and cardiovascular health indicators
               - Consider patient age and gender in interpretation

            3. Risk Assessment Phase ⚠️
               - Assess health risks based on abnormal findings
               - Identify potential underlying conditions
               - Evaluate urgency of medical follow-up
               - Consider age and gender-specific risk factors

            4. Recommendations Phase 💡
               - Provide specific lifestyle and dietary recommendations
               - Suggest appropriate medical follow-up
               - Recommend preventive measures
               - Provide monitoring guidelines

            Always:
            - Use structured health analysis tools for accurate interpretation
            - Cite specific test values and reference ranges
            - Include appropriate medical disclaimers
            - Emphasize that analysis is not a substitute for professional medical advice
            - Provide clear, actionable health insights\
        """),
        expected_output=dedent("""\
            # Health Insights Analysis Report 🩺

            > **Disclaimer**: This analysis is generated by AI and should not be considered as a replacement for professional medical advice. Please consult with a healthcare provider for proper medical diagnosis and treatment.

            ## Patient Summary
            {Patient demographics and analysis overview}
            {Overall health status assessment}

            ## Test Results Analysis
            {Complete blood count findings}
            {Metabolic panel results}
            {Liver and kidney function}
            {Lipid profile and cardiovascular assessment}
            {Thyroid and endocrine evaluation}

            ## Abnormal Findings
            {List of abnormal test values}
            {Clinical significance of abnormalities}
            {Potential underlying conditions}

            ## Health Risk Assessment
            {Overall risk level (Low/Moderate/High)}
            {Specific health risk factors}
            {Age and gender considerations}

            ## Recommendations
            {Lifestyle modifications}
            {Dietary recommendations}
            {Medical follow-up suggestions}
            {Preventive measures}
            {Monitoring guidelines}

            ## Next Steps
            {Urgency of medical consultation}
            {Specific tests to consider}
            {Timeline for follow-up}

            ---
            Analysis by AI Health Insights Agent
            Based on provided medical report data
            Generated: {current_date}
            Analysis valid as of: {current_time}\
        """),
        add_datetime_to_context=True,
        markdown=True,
    )
    print("✅ Health Insights Agent initialized")


async def cleanup_tools() -> None:
    """Clean up any resources."""
    global health_tools

    if health_tools:
        print("🔌 Health analysis tools cleaned up")


async def run_agent(messages: list[dict[str, str]]) -> Any:
    """Run the agent with the given messages.

    Args:
        messages: List of message dicts with 'role' and 'content' keys

    Returns:
        Agent response
    """
    global agent

    # Run the agent and get response
    if agent is None:
        raise ValueError("Agent not initialized")
    response = await agent.arun(messages)
    return response


async def handler(messages: list[dict[str, str]]) -> Any:
    """Handle incoming agent messages.

    Args:
        messages: List of message dicts with 'role' and 'content' keys
                  e.g., [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]

    Returns:
        Agent response (ManifestWorker will handle extraction)
    """

    # Run agent with messages
    global _initialized

    # Lazy initialization on first call (in bindufy's event loop)
    async with _init_lock:
        if not _initialized:
            print("🔧 Initializing Health Insights Agent...")
            await initialize_agent()
            _initialized = True

    # Run the async agent
    result = await run_agent(messages)
    return result


async def initialize_all(env: dict[str, str] | None = None):
    """Initialize agent and tools.

    Args:
        env: Environment variables dict (not used for integrated tools)
    """
    await initialize_agent()


def main():
    """Run the Health Insights Agent."""
    global model_name, openrouter_api_key, mem0_api_key

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="HIA Health Insights Agent with Medical Analysis Tools")
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("MODEL_NAME", "anthropic/claude-3.5-sonnet"),
        help="Model ID to use (default: anthropic/claude-3.5-sonnet, env: MODEL_NAME), if you want you can use any free model: https://openrouter.ai/models?q=free",
    )

    parser.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("OPENROUTER_API_KEY"),
        help="OpenRouter API key (env: OPENROUTER_API_KEY)",
    )
    parser.add_argument(
        "--mem0-api-key",
        type=str,
        default=os.getenv("MEM0_API_KEY"),
        help="Mem0 API key (env: MEM0_API_KEY)",
    )
    args = parser.parse_args()

    # Set global model name and API keys
    model_name = args.model
    openrouter_api_key = args.api_key
    mem0_api_key = args.mem0_api_key

    if not openrouter_api_key:
        raise ValueError("OPENROUTER_API_KEY required")
    if not mem0_api_key:
        raise ValueError("MEM0_API_KEY required. Get your API key from: https://app.mem0.ai/dashboard/api-keys")

    print(f"🤖 Health Insights Agent (HIA) using model: {model_name}")
    print("🩺 Comprehensive medical report analysis and health insights")
    print("🧠 Mem0 memory enabled")

    # Load configuration
    config = load_config()

    try:
        # Bindufy and start the agent server
        # Note: Agent will be initialized lazily on first request
        print("🚀 Starting Health Insights Agent server...")
        bindufy(config, handler)
    finally:
        # Cleanup on exit
        print("\n🧹 Cleaning up...")
        asyncio.run(cleanup_tools())


# Bindufy and start the agent server
if __name__ == "__main__":
    main()
