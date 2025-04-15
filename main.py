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
# Replace 'YOUR_OLLAMA_API_KEY' and 'OLLAMA_MODEL_NAME' with your actual credentials.
# -----------------------------------------------------------------------------
dspy.configure(lm=dspy.Ollama(api_key="YOUR_OLLAMA_API_KEY", model="OLLAMA_MODEL_NAME"))

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def parse_markdown_table(md_table: str) -> pd.DataFrame:
    """
    Convert a markdown table (with pipes) into a pandas DataFrame.
    Assumes the first non-empty line is the header.
    """
    lines = md_table.strip().split("\n")
    header = [h.strip() for h in lines[0].strip("|").split("|") if h.strip()]
    data_lines = []
    for line in lines[2:]:
        parts = [p.strip() for p in line.strip("|").split("|") if p.strip()]
        if parts:
            data_lines.append(parts)
    df = pd.DataFrame(data_lines, columns=header)
    # Attempt numeric conversion for each column.
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")
    return df

def generate_schema_info(df: pd.DataFrame) -> str:
    """
    Generate a descriptive string of the DataFrame schema.
    """
    schema_parts = [f"{col} ({df[col].dtype})" for col in df.columns]
    return ", ".join(schema_parts)

def choose_numeric_columns(schema_info: str) -> list:
    """
    From the schema information, extract and return all numeric columns.
    """
    numeric_cols = []
    for part in schema_info.split(","):
        part = part.strip()
        # Check if the dtype description contains 'int' or 'float'
        if "int" in part.lower() or "float" in part.lower():
            col_name = part.split(" ")[0]
            numeric_cols.append(col_name)
    return numeric_cols

# =============================================================================
# DSPy SIGNATURES (Dynamic Versions)
# =============================================================================
class DynamicAnalyticalPlanner(dspy.Signature):
    overall_context = dspy.InputField(desc="Overall analysis goal and context.")
    main_context = dspy.InputField(desc="Context of the main dataset.")
    support_context = dspy.InputField(desc="Context of the support dataset (if any).")
    schema_info = dspy.InputField(desc="Extracted dataset schema (column names and types).")
    plan = dspy.OutputField(desc="Multi-step dynamic analysis plan with hypothesis and strategy.")

class PreprocessingAgent(dspy.Signature):
    dataset = dspy.InputField(desc="Dataset as a markdown table.")
    plan = dspy.InputField(desc="Initial plan text (for consistency).")
    cleaned_data = dspy.OutputField(desc="Cleaned dataset (pandas DataFrame).")
    log = dspy.OutputField(desc="Preprocessing log and details.")
    schema = dspy.OutputField(desc="Extracted schema information.")

class DynamicStatisticalAnalysisAgent(dspy.Signature):
    dataset = dspy.InputField(desc="Cleaned dataset (DataFrame).")
    plan = dspy.InputField(desc="Multi-step analysis plan (including chosen columns and analysis decisions).")
    report = dspy.OutputField(desc="Detailed statistical analysis report in markdown.")

class DynamicVisualizationAgent(dspy.Signature):
    dataset = dspy.InputField(desc="Cleaned dataset (DataFrame).")
    plan = dspy.InputField(desc="Multi-step analysis plan.")
    image = dspy.OutputField(desc="Visualization output as image bytes.")

class FinalReportModule(dspy.Signature):
    plan = dspy.InputField(desc="Final analysis plan text.")
    stat_report = dspy.InputField(desc="Statistical analysis report in markdown.")
    viz_path = dspy.InputField(desc="Path to the saved visualization image file.")
    final_report = dspy.OutputField(desc="Comprehensive final markdown report.")

# =============================================================================
# CHAIN-OF-THOUGHT MODULE INITIALIZATION
# =============================================================================
planner = dspy.ChainOfThought(DynamicAnalyticalPlanner)
preprocessor = dspy.ChainOfThought(PreprocessingAgent)
statistical_reporter = dspy.ChainOfThought(DynamicStatisticalAnalysisAgent)
visualizer = dspy.ChainOfThought(DynamicVisualizationAgent)

# =============================================================================
# DYNAMIC MODULE IMPLEMENTATIONS
# =============================================================================

# (1) Dynamic Analytical Planner: Produces a multi-step plan based on the extracted schema.
@dspy.module(DynamicAnalyticalPlanner)
def dynamic_plan_module(overall_context, main_context, support_context, schema_info):
    numeric_cols = choose_numeric_columns(schema_info)
    if len(numeric_cols) < 2:
        plan_text = (
            "Dynamic Analysis Plan:\n"
            "Insufficient numeric columns detected. Please provide a dataset with at least two numeric columns."
        )
    elif len(numeric_cols) == 2:
        plan_text = (
            "Dynamic Analysis Plan:\n"
            "Identified numeric columns: " + ", ".join(numeric_cols) + "\n"
            "Plan: Perform descriptive statistics and pairwise analysis between the two columns.\n"
            "If their Pearson correlation coefficient is high, perform OLS regression and visualize with a scatter plot with a regression line.\n"
            "Otherwise, report the correlation and descriptive statistics."
        )
    else:
        plan_text = (
            "Dynamic Analysis Plan:\n"
            "Identified numeric columns: " + ", ".join(numeric_cols) + "\n"
            "Plan: Compute descriptive statistics for all numeric columns and calculate the pairwise Pearson correlation matrix.\n"
            "Visualize the relationships by generating a correlation heatmap."
        )
    return {"plan": plan_text}

# (2) Preprocessing Module: Parses the markdown, cleans the data, and extracts schema.
@dspy.module(PreprocessingAgent)
def preprocess_module(dataset, plan):
    df = parse_markdown_table(dataset)
    df_clean = df.dropna()  # Basic cleaning: drop rows with missing values.
    schema_info = generate_schema_info(df_clean)
    log_text = "Data parsed and cleaned. Detected schema: " + schema_info
    return {"cleaned_data": df_clean, "log": log_text, "schema": schema_info}

# (3) Dynamic Statistical Analysis Agent:
@dspy.module(DynamicStatisticalAnalysisAgent)
def dynamic_statistical_module(dataset, plan):
    # Extract numeric columns from plan using regex.
    match = re.search(r"Identified numeric columns:\s*(.*)", plan)
    if match:
        cols_str = match.group(1).split("\n")[0]
        numeric_cols = [col.strip() for col in cols_str.split(",") if col.strip()]
    else:
        numeric_cols = choose_numeric_columns(generate_schema_info(dataset))
    
    if len(numeric_cols) < 2:
        report = "Insufficient numeric columns for statistical analysis."
        return {"report": report}
    
    df = dataset.copy()
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=numeric_cols)
    
    if len(numeric_cols) == 2:
        # For exactly two numeric columns, perform detailed pair analysis.
        predictor, response = numeric_cols[0], numeric_cols[1]
        desc_stats = df[[predictor, response]].describe().to_markdown()
        corr = df[[predictor, response]].corr().iloc[0, 1]
        if abs(corr) >= 0.5:
            X = sm.add_constant(df[predictor])
            y = df[response]
            model = sm.OLS(y, X).fit()
            summary_text = model.summary().as_text()
            analysis_detail = (
                f"Performed OLS Regression Analysis between {predictor} and {response}:\n\n```\n{summary_text}\n```\n"
                "This regression model quantifies their linear relationship."
            )
        else:
            analysis_detail = f"Pearson correlation coefficient between {predictor} and {response} is {corr:.2f}."
        report = (
            "### Statistical Analysis Report\n\n"
            "**Descriptive Statistics:**\n" + desc_stats + "\n\n"
            "**Relationship Evaluation:**\n" + analysis_detail
        )
        return {"report": report}
    else:
        # For more than two numeric columns, report overall descriptive stats and correlation matrix.
        desc_stats = df[numeric_cols].describe().to_markdown()
        corr_matrix = df[numeric_cols].corr().round(2)
        corr_md = corr_matrix.to_markdown()
        report = (
            "### Statistical Analysis Report\n\n"
            "**Descriptive Statistics for all numeric columns:**\n" + desc_stats + "\n\n"
            "**Correlation Matrix:**\n" + corr_md
        )
        return {"report": report}

# (4) Dynamic Visualization Agent:
@dspy.module(DynamicVisualizationAgent)
def dynamic_visualization_module(dataset, plan):
    # Extract numeric columns from the plan.
    match = re.search(r"Identified numeric columns:\s*(.*)", plan)
    if match:
        cols_str = match.group(1).split("\n")[0]
        numeric_cols = [col.strip() for col in cols_str.split(",") if col.strip()]
    else:
        numeric_cols = choose_numeric_columns(generate_schema_info(dataset))
    
    df = dataset.copy()
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=numeric_cols)
    
    plt.figure(figsize=(8, 6))
    if len(numeric_cols) == 2:
        predictor, response = numeric_cols[0], numeric_cols[1]
        corr = df[[predictor, response]].corr().iloc[0, 1]
        if abs(corr) >= 0.5:
            sns.scatterplot(data=df, x=predictor, y=response, color="blue", label="Data Points")
            X = sm.add_constant(df[predictor])
            model = sm.OLS(df[response], X).fit()
            pred = model.predict(X)
            plt.plot(df[predictor], pred, color="red", label="Regression Line")
            plt.title(f"{predictor} vs {response} (OLS Regression)")
        else:
            sns.scatterplot(data=df, x=predictor, y=response, color="green", label="Data Points")
            plt.title(f"{predictor} vs {response} (Correlation)")
        plt.xlabel(predictor)
        plt.ylabel(response)
        plt.legend()
    else:
        # Generate a heatmap for the correlation matrix.
        corr_matrix = df[numeric_cols].corr()
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Correlation Heatmap")
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    image_bytes = buf.getvalue()
    plt.close()
    return {"image": image_bytes}

# (5) Final Report Module: Combines the analysis plan, statistical report, and visualization.
@dspy.module(FinalReportModule)
def final_report_module(plan, stat_report, viz_path):
    final_text = (
        "# Final Dynamic Analysis Report\n\n"
        "## Analysis Plan\n" + plan + "\n\n"
        "## Statistical Analysis\n" + stat_report + "\n\n"
        "## Visualization\n"
        f"![Visualization]({viz_path})\n\n"
        "The analysis was dynamically designed and executed based on the provided dataset and context."
    )
    return {"final_report": final_text}

# =============================================================================
# TOP‑LEVEL DYNAMIC ANALYSIS AGENT FUNCTION
# =============================================================================
def dynamic_data_analysis_agent(overall_context: str, main_context: str, main_data_md: str,
                                support_context: str = "", support_data_md: str = ""):
    # Step 1: Preprocessing – parse, clean, and extract schema.
    preprocess_output = preprocess_module(dataset=main_data_md, plan="")
    cleaned_data = preprocess_output["cleaned_data"]
    schema_info = preprocess_output["schema"]
    print("Preprocessing Log:", preprocess_output["log"])
    print("Detected Schema:", schema_info)
    
    # Step 2: Dynamic Planning – generate multi-step plan.
    plan_output = dynamic_plan_module(overall_context=overall_context,
                                      main_context=main_context,
                                      support_context=support_context,
                                      schema_info=schema_info)
    analysis_plan = plan_output["plan"]
    print("\nDynamic Analysis Plan:\n", analysis_plan)
    
    # Step 3: Dynamic Statistical Analysis.
    stat_output = dynamic_statistical_module(dataset=cleaned_data, plan=analysis_plan)
    stat_report = stat_output["report"]
    
    # Step 4: Dynamic Visualization.
    viz_output = dynamic_visualization_module(dataset=cleaned_data, plan=analysis_plan)
    viz_image_path = "dynamic_visualization.png"
    with open(viz_image_path, "wb") as f:
        f.write(viz_output["image"])
    
    # Step 5: Final Report – combine plan, analysis report, and visualization.
    final_output = final_report_module(plan=analysis_plan,
                                       stat_report=stat_report,
                                       viz_path=viz_image_path)
    return final_output["final_report"]

# =============================================================================
# MAIN EXECUTION (Demonstration with Dummy Data)
# =============================================================================
if __name__ == '__main__':
    # Create dummy data as a markdown table.
    np.random.seed(0)
    dummy_df = pd.DataFrame({
        "Experience": np.random.randint(1, 10, 100),
        "Education_Level": np.random.choice(["High School", "Bachelors", "Masters", "PhD"], 100),
        "Region": np.random.choice(["North", "South", "East", "West"], 100),
        "Salary": np.random.randint(30000, 100000, 100)
    })
    main_data_markdown = dummy_df.to_markdown(index=False)
    
    # Define input contexts.
    overall_context = (
        "Investigate the overall statistical characteristics of the employee dataset. "
        "Dynamically analyze the relationships among numeric columns without a predefined predictor and response."
    )
    main_context = (
        "The dataset consists of employee information including Experience, Education_Level, Region, and Salary."
    )
    support_context = "Additional qualitative insights are not provided in this example."
    
    # Execute the dynamic analysis agent.
    final_report = dynamic_data_analysis_agent(overall_context, main_context, main_data_markdown, support_context)
    
    # Save the final report to a markdown file.
    report_file = "dynamic_analysis_report.md"
    with open(report_file, "w") as f:
        f.write(final_report)
    
    print("\nFinal Report generated and saved as:", report_file)
    print("\n--- Report Preview ---\n")
    print(final_report[:600] + "\n...\n")
