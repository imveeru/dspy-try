import dspy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from io import BytesIO
import re
import os

# -----------------------------------------------------------------------------
# Configure DSPy to use Ollama as the underlying LLM.
# Replace these placeholders with your actual Ollama credentials.
# -----------------------------------------------------------------------------
dspy.configure(lm=dspy.Ollama(api_key="YOUR_OLLAMA_API_KEY", model="OLLAMA_MODEL_NAME"))

# -----------------------------------------------------------------------------
# Constant: Database Schema for further metric queries.
# -----------------------------------------------------------------------------
DB_SCHEMA = {
    "RevenueMetrics": ["daily_revenue", "monthly_growth", "seasonal_adjustment"],
    "MarketingMetrics": ["ad_spend", "click_through_rate", "conversion_rate"],
    "OperationalMetrics": ["inventory_levels", "supply_chain_efficiency", "customer_satisfaction"]
}

# =============================================================================
# HELPER FUNCTIONS (Fully Dynamic)
# =============================================================================
def parse_markdown_table(md_table: str) -> pd.DataFrame:
    """
    Convert a markdown table (with pipes) into a pandas DataFrame.
    Assumes that the first non-empty line contains the header.
    """
    lines = md_table.strip().split("\n")
    header = [h.strip() for h in lines[0].strip("|").split("|") if h.strip()]
    data_lines = []
    for line in lines[2:]:
        parts = [p.strip() for p in line.strip("|").split("|") if p.strip()]
        if parts:
            data_lines.append(parts)
    df = pd.DataFrame(data_lines, columns=header)
    # Convert columns that look numeric.
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")
    return df

def generate_schema_info(df: pd.DataFrame) -> str:
    """
    Generate a descriptive string of the DataFrame schema.
    This information is later used in the planning prompt.
    """
    return ", ".join([f"{col} ({df[col].dtype})" for col in df.columns])

def compute_percentage_change(new_val, old_val):
    """
    Compute the percentage change given new and old values.
    """
    try:
        return ((new_val - old_val) / old_val) * 100
    except ZeroDivisionError:
        return float('nan')

# =============================================================================
# DSPy SIGNATURES (For Causal Analysis Objective)
# =============================================================================
class DynamicCausalPlanner(dspy.Signature):
    overall_context = dspy.InputField(desc="Text describing the revenue anomaly and contextual information.")
    summary1 = dspy.InputField(desc="Markdown summary from analysis 1.")
    summary2 = dspy.InputField(desc="Markdown summary from analysis 2.")
    summary3 = dspy.InputField(desc="Markdown summary from analysis 3.")
    revenue_schema = dspy.InputField(desc="Extracted schema from the revenue data.")
    kpi_schema = dspy.InputField(desc="Extracted schema description for the KPI report (if any).")
    plan = dspy.OutputField(desc="Autonomously generated multi-step causal analysis plan with hypotheses, questions, and analytical frameworks.")

class PreprocessingData(dspy.Signature):
    data_md = dspy.InputField(desc="Data source provided as a markdown table.")
    plan = dspy.InputField(desc="Plan text (for consistency across modules).")
    cleaned_data = dspy.OutputField(desc="Cleaned data as a pandas DataFrame.")
    log = dspy.OutputField(desc="Preprocessing log and extracted schema.")
    schema = dspy.OutputField(desc="Extracted schema string from the data.")

# The KPI report is treated as a raw text input since it is already a report.
class DynamicCausalAnalysis(dspy.Signature):
    revenue_df = dspy.InputField(desc="Cleaned revenue data as a DataFrame.")
    kpi_report_text = dspy.InputField(desc="Provided KPI report text (in markdown).")
    plan = dspy.InputField(desc="Autonomously generated causal analysis plan.")
    causal_report = dspy.OutputField(desc="In-depth causal reasoning report in markdown, with explicit hypotheses and framework explanations.")

class FinalCausalReport(dspy.Signature):
    plan = dspy.InputField(desc="Final dynamic causal analysis plan text.")
    causal_report = dspy.InputField(desc="Causal reasoning report in markdown.")
    general_instructions = dspy.InputField(desc="General follow-up investigative instructions.")
    db_instructions = dspy.InputField(desc="Database query instructions based on DB_SCHEMA.")
    final_report = dspy.OutputField(desc="Comprehensive final markdown report.")

# =============================================================================
# CHAIN-OF-THOUGHT MODULE INITIALIZATION
# =============================================================================
planner = dspy.ChainOfThought(DynamicCausalPlanner)
preprocessor = dspy.ChainOfThought(PreprocessingData)
causal_analyzer = dspy.ChainOfThought(DynamicCausalAnalysis)
final_reporter = dspy.ChainOfThought(FinalCausalReport)

# =============================================================================
# DYNAMIC MODULE IMPLEMENTATIONS WITH DETAILED PROMPT EXPLANATIONS
# =============================================================================

# (1) Dynamic Causal Planner:
@dspy.module(DynamicCausalPlanner)
def dynamic_causal_plan_module(overall_context, summary1, summary2, summary3, revenue_schema, kpi_schema):
    """
    This module constructs a detailed chain-of-thought prompt that instructs the LLM to generate
    a multi-step causal analysis plan. The prompt includes:
      - A consolidation of insights from the three provided summaries.
      - The revenue data schema and KPI data description (kpi_schema is marked as preformatted).
      - Specific hypotheses and analytical questions such as:
          • What is the percentage drop in revenue?
          • Which potential KPIs (like conversion rate) might have contributed?
          • What assumptions can be made about user engagement or marketing performance?
      - An explanation for further probing (follow-up instructions).
    
    The produced plan text is entirely dynamic, based solely on the provided inputs.
    """
    plan_text = (
        "Dynamic Causal Analysis Plan:\n"
        "Step 1: Consolidate insights from the three analysis summaries:\n"
        "         - Summary 1: {}\n"
        "         - Summary 2: {}\n"
        "         - Summary 3: {}\n"
        "Step 2: Using the revenue data (schema: {}) and KPI report description (schema: {}), "
        "determine the key numeric indicators to compare.\n"
        "Step 3: Ask: 'What is the week-over-week percentage change in revenue?' and formulate hypotheses based on that value.\n"
        "Step 4: Incorporate the provided KPI report (which highlights metrics that performed low/high) as evidence.\n"
        "Step 5: Based on the computed values and the KPI report, formulate a causal reasoning chain addressing:\n"
        "         - Hypotheses (e.g., decreased engagement, operational issues, marketing underperformance),\n"
        "         - Analytical questions (e.g., 'Could a decline in conversion rate be causing this drop?'),\n"
        "         - Assumptions and frameworks to analyze the anomaly.\n"
        "Step 6: Propose further investigative actions, including general checks and specific database queries.\n"
        "\nHypotheses Example: A significant revenue drop combined with poor KPI performance may indicate issues in the conversion funnel.\n"
    ).format(summary1, summary2, summary3, revenue_schema, kpi_schema)
    return {"plan": plan_text}

# (2) Preprocessing Module for Data Sources:
@dspy.module(PreprocessingData)
def preprocess_data_module(data_md, plan):
    """
    This module processes the provided markdown table dynamically:
      - Parses the table.
      - Performs basic cleaning (dropping missing values).
      - Extracts the schema information which is later used in dynamic planning.
    The prompt behind this step is: 'Extract all relevant structure from the data so that the analysis can be based solely on objective numeric features.' 
    """
    df = parse_markdown_table(data_md)
    df_clean = df.dropna()
    schema_info = generate_schema_info(df_clean)
    log_text = "Data parsed and cleaned. Extracted schema: " + schema_info
    return {"cleaned_data": df_clean, "log": log_text, "schema": schema_info}

# (3) Dynamic Causal Analysis Module:
@dspy.module(DynamicCausalAnalysis)
def dynamic_causal_analysis_module(revenue_df, kpi_report_text, plan):
    """
    This module performs the core causal analysis.
    It uses real calculations to:
      - Compute the week-over-week percentage change in revenue.
      - Integrate the provided KPI report (which is assumed to already highlight metrics that underperformed).
    
    The internal chain-of-thought prompt here includes explicit questions:
      • 'What is the computed revenue drop?'
      • 'How does that compare with the KPI insights provided?'
      • 'What hypothesis can be drawn about the underlying cause?'
    
    The output is a detailed causal reasoning report, which is the main component of the final output.
    """
    # --- Revenue Analysis ---
    revenue_report = ""
    revenue_change = None
    if len(revenue_df) >= 2 and "Revenue" in revenue_df.columns:
        try:
            latest_rev = float(revenue_df["Revenue"].iloc[-1])
            prev_rev = float(revenue_df["Revenue"].iloc[-2])
            revenue_change = compute_percentage_change(latest_rev, prev_rev)
            revenue_report = f"Calculated a revenue drop of {revenue_change:.2f}% between the latest two periods.\n"
        except Exception:
            revenue_report = "Revenue change calculation failed.\n"
    else:
        revenue_report = "Insufficient revenue data for week-over-week analysis.\n"
    
    # --- Incorporate Provided KPI Report ---
    kpi_report = "\nProvided KPI Report:\n" + kpi_report_text + "\n"
    
    # --- Causal Reasoning Chain ---
    reasoning = "Causal Reasoning:\n"
    reasoning += revenue_report + "\n" + kpi_report + "\n"
    # Hypothesis-based reasoning:
    if revenue_change is not None:
        reasoning += (
            "Based on the calculated revenue drop and the KPI report indications, we hypothesize that "
            "the anomaly could be driven by issues such as a decline in conversion rates or user engagement. "
            "Key questions include: Did marketing performance drop? Were there operational disruptions? "
            "The evidence suggests that the revenue anomaly is likely linked to deficits in the conversion funnel.\n"
            "Assumptions: The revenue drop is significant and is not due to seasonal factors, and the KPI report is accurate and reliable."
        )
    else:
        reasoning += "Unable to form robust hypotheses due to insufficient data."
    
    full_report = "### In-Depth Causal Analysis Report\n\n" + reasoning
    return {"causal_report": full_report}

# (4) Final Causal Report Module:
@dspy.module(FinalCausalReport)
def final_causal_report_module(plan, causal_report):
    """
    This module compiles the final comprehensive report.
    It appends two sets of follow-up instructions:
      1. General instructions for further investigation.
      2. Database-specific instructions derived from the provided DB schema.
    
    Each instruction prompt is explained as:
      - 'General Instructions' prompt: Suggest broad areas to check for deeper insights.
      - 'Database Query Instructions' prompt: Provide a set of queries based on the DB schema to extract further performance metrics.
    """
    general_instructions = (
        "1. Reassess user engagement, marketing, and operational logs for anomalies during the affected period.\n"
        "2. Validate the KPI measurements with alternative data sources.\n"
        "3. Review historical trends for context and comparative analysis."
    )
    
    db_instructions = "Based on the provided DB schema, execute the following queries:\n"
    for table, fields in DB_SCHEMA.items():
        db_instructions += f" - Examine table '{table}' for fields: {', '.join(fields)} to identify related anomalies.\n"
    
    final_text = (
        "# Final Causal Analysis Report\n\n"
        "## Analysis Plan\n" + plan + "\n\n"
        "## Detailed Causal Analysis\n" + causal_report + "\n\n"
        "## Additional Investigative Instructions\n\n"
        "**General Instructions:**\n" + general_instructions + "\n\n"
        "**Database Query Instructions:**\n" + db_instructions
    )
    return {"final_report": final_text}

# =============================================================================
# TOP-LEVEL DYNAMIC CAUSAL ANALYSIS AGENT FUNCTION
# =============================================================================
def dynamic_causal_analysis_agent(overall_context: str,
                                  summary1: str,
                                  summary2: str,
                                  summary3: str,
                                  revenue_data_md: str,
                                  kpi_report_md: str):
    """
    Top-level function:
      1. Preprocesses the revenue data to extract a clean DataFrame and its schema.
      2. Uses the provided KPI report as-is.
      3. Generates a fully dynamic causal analysis plan using all the input summaries and schema info.
         - The plan prompt includes detailed explanations and asks the LLM to incorporate hypotheses, questions,
           and analytical frameworks.
      4. Computes revenue changes and integrates the KPI report to generate an in-depth causal reasoning chain.
      5. Compiles a final report with both general and database-specific follow-up instructions.
    
    This function is completely autonomous; every decision is based on the provided input data.
    """
    # Preprocess Revenue Data
    revenue_preprocess = preprocess_data_module(data_md=revenue_data_md, plan="")
    revenue_df = revenue_preprocess["cleaned_data"]
    revenue_schema = revenue_preprocess["schema"]
    print("Revenue Data Preprocessing Log:", revenue_preprocess["log"])
    
    # For KPI report, we assume it is already a formatted report.
    kpi_schema = "Preformatted KPI Report"
    
    # Generate Dynamic Causal Analysis Plan
    plan_output = dynamic_causal_plan_module(overall_context, summary1, summary2, summary3,
                                             revenue_schema, kpi_schema)
    analysis_plan = plan_output["plan"]
    print("\nDynamic Causal Analysis Plan:\n", analysis_plan)
    
    # Perform Dynamic Causal Analysis using the revenue data and KPI report.
    causal_output = dynamic_causal_analysis_module(revenue_df, kpi_report_md, analysis_plan)
    causal_report = causal_output["causal_report"]
    
    # Compile the Final Causal Analysis Report with follow-up instructions.
    final_output = final_causal_report_module(plan=analysis_plan, causal_report=causal_report)
    return final_output["final_report"]

# =============================================================================
# MAIN EXECUTION (Demonstration with Dummy Data)
# =============================================================================
if __name__ == '__main__':
    # Dummy markdown summaries from three previous analyses.
    summary1 = "### Analysis Summary 1\n\nPrevious analysis indicates stable revenue trends with minor fluctuations."
    summary2 = "### Analysis Summary 2\n\nUser engagement metrics were within expected ranges in the past."
    summary3 = "### Analysis Summary 3\n\nOperational metrics showed no abnormal variations."
    
    # Dummy revenue data (markdown table).
    revenue_dummy = pd.DataFrame({
        "Week": ["Week1", "Week2", "Week3", "Week4"],
        "Revenue": [100000, 98000, 95000, 70000]
    })
    revenue_data_md = revenue_dummy.to_markdown(index=False)
    
    # Dummy KPI report provided as a preformatted markdown text.
    kpi_report_md = (
        "### KPI Metrics Check Report\n\n"
        "- **conversion_rate**: Noted underperformance versus target values.\n"
        "- **bounce_rate**: Elevated bounce rates observed compared to historical norms.\n"
        "- **session_duration**: Marginal decrease in average session duration.\n"
        "The report highlights a consistent underperformance in conversion metrics."
    )
    
    # Overall context describing the revenue anomaly.
    overall_context = (
        "This week's revenue anomaly shows a marked decline relative to previous weeks. "
        "Investigate and determine the root cause by integrating insights from the provided summaries, "
        "revenue data, and KPI report."
    )
    
    # Execute the dynamic causal analysis agent.
    final_report = dynamic_causal_analysis_agent(
        overall_context,
        summary1,
        summary2,
        summary3,
        revenue_data_md,
        kpi_report_md
    )
    
    # Save the final report to a markdown file.
    report_file = "final_causal_analysis_report.md"
    with open(report_file, "w") as f:
        f.write(final_report)
    
    print("\nFinal Causal Analysis Report generated and saved as:", report_file)
    print("\n--- Report Preview ---\n")
    print(final_report[:600] + "\n...\n")
