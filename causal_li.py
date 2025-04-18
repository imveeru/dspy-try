# File: causal_agent_llamaindex.py
from typing import List, Dict, Any
from scipy import stats
import numpy as np
# Updated imports for Ollama per LlamaIndex docs\#from llama_index.llms.ollama import Ollama - corrected below
from llama_index.llms.ollama import Ollama
from llama_index.tools import BaseTool, ToolMetadata
from llama_index.agent import Agent

# 1. Define custom statistical tools as BaseTool subclasses
class PearsonTool(BaseTool):
    def __init__(self):
        metadata = ToolMetadata(
            name="pearson",
            description="Compute Pearson correlation coefficient between two numeric lists."
        )
        super().__init__(metadata=metadata)
    def _run(self, x: List[float], y: List[float]) -> float:
        return stats.pearsonr(x, y)[0]
    async def _arun(self, *args, **kwargs):
        raise NotImplementedError

class ZTestTool(BaseTool):
    def __init__(self):
        metadata = ToolMetadata(
            name="z_test",
            description="Compute the z-test statistic between a sample list and a population list."
        )
        super().__init__(metadata=metadata)
    def _run(self, sample: List[float], population: List[float]) -> float:
        return (np.mean(sample) - np.mean(population)) / (np.std(population) / np.sqrt(len(sample)))
    async def _arun(self, *args, **kwargs):
        raise NotImplementedError

class ANOVATool(BaseTool):
    def __init__(self):
        metadata = ToolMetadata(
            name="anova",
            description="Perform one-way ANOVA returning the F-statistic."
        )
        super().__init__(metadata=metadata)
    def _run(self, groups: List[List[float]]) -> float:
        return stats.f_oneway(*groups)[0]
    async def _arun(self, *args, **kwargs):
        raise NotImplementedError

class ContributionTool(BaseTool):
    def __init__(self):
        metadata = ToolMetadata(
            name="contribution",
            description="Compute contribution analysis as covariance(x,y)/variance(y)."
        )
        super().__init__(metadata=metadata)
    def _run(self, x: List[float], y: List[float]) -> float:
        cov = np.cov(x, y)[0,1]
        return cov / np.var(y)
    async def _arun(self, *args, **kwargs):
        raise NotImplementedError

class TTestTool(BaseTool):
    def __init__(self):
        metadata = ToolMetadata(
            name="t_test",
            description="Compute two-sample t-test statistic (unequal variance)."
        )
        super().__init__(metadata=metadata)
    def _run(self, a: List[float], b: List[float]) -> float:
        return stats.ttest_ind(a, b, equal_var=False).statistic
    async def _arun(self, *args, **kwargs):
        raise NotImplementedError

class ChiSquareTool(BaseTool):
    def __init__(self):
        metadata = ToolMetadata(
            name="chi_square",
            description="Compute chi-square test statistic for observed vs expected counts."
        )
        super().__init__(metadata=metadata)
    def _run(self, observed: List[float], expected: List[float]) -> float:
        return float(stats.chisquare(observed, expected).statistic)
    async def _arun(self, *args, **kwargs):
        raise NotImplementedError

class MannWhitneyUTool(BaseTool):
    def __init__(self):
        metadata = ToolMetadata(
            name="mannwhitneyu",
            description="Compute Mann-Whitney U test statistic (two-sided)."
        )
        super().__init__(metadata=metadata)
    def _run(self, a: List[float], b: List[float]) -> float:
        return float(stats.mannwhitneyu(a, b, alternative='two-sided').statistic)
    async def _arun(self, *args, **kwargs):
        raise NotImplementedError

class LinearRegressionTool(BaseTool):
    def __init__(self):
        metadata = ToolMetadata(
            name="linear_regression",
            description="Compute linear regression slope and intercept between two numeric lists."
        )
        super().__init__(metadata=metadata)
    def _run(self, x: List[float], y: List[float]) -> Dict[str, float]:
        slope, intercept, _, _, _ = stats.linregress(x, y)
        return {"slope": slope, "intercept": intercept}
    async def _arun(self, *args, **kwargs):
        raise NotImplementedError

class CausalAnalysisAgent:
    def __init__(
        self,
        primary_model: str = "llama3.1:latest",
        validation_model: str = "deepseek",
        request_timeout: float = 120.0
    ):
        # 2. LLMs using updated Ollama init
        self.primary_llm = Ollama(model=primary_model, request_timeout=request_timeout)
        self.validation_llm = Ollama(model=validation_model, request_timeout=request_timeout)

        # 3. Register tools with LlamaIndex Agent using llm param
        self.agent = Agent.from_tools(
            tools=[
                PearsonTool(), ZTestTool(), ANOVATool(), ContributionTool(),
                TTestTool(), ChiSquareTool(), MannWhitneyUTool(), LinearRegressionTool()
            ],
            llm=self.validation_llm,
            verbose=True
        )

        # 4. Prompt templates
        self.prompts = {
            "input_analysis": (
                """
You are an expert data analyst. Given these reports:\n{reports}\nand this data summary:\n{data_stats}\nThink step by step and provide a concise input analysis.
"""
            ),
            "frameworks": (
                """
As a research framework advisor, review:\n{reports}\nand this analysis:\n{input_analysis}\nThink step by step, then list appropriate analytical frameworks separated by semicolons.
"""
            ),
            "hypotheses": (
                """
You are a hypothesis generator. Based on the analysis:\n{input_analysis}\nThink step by step and generate multiple hypotheses separated by semicolons.
"""
            ),
            "insights": (
                """
Given these validation results:\n{validations}\nThink step by step and provide general insights separated by semicolons.
"""
            ),
            "recommendations": (
                """
Based on insights:\n{insights}\nand this schema:\n{schema}\nThink step by step and generate actionable recommendations. Prefix database-specific actions with 'Database:'. Separate items with semicolons.
"""
            ),
        }

    def run(self, reports: List[str], data: Dict[str, Any], schema: str) -> Dict[str, Any]:
        reports_str = "\n\n".join(reports)
        data_str = str(data)
        # 1. Input analysis
        analysis = self.primary_llm.complete(
            self.prompts["input_analysis"].format(reports=reports_str, data_stats=data_str)
        )
        # 2. Framework selection
        frameworks = self.primary_llm.complete(
            self.prompts["frameworks"].format(reports=reports_str, input_analysis=analysis)
        )
        # 3. Hypotheses
        hyps_text = self.primary_llm.complete(
            self.prompts["hypotheses"].format(input_analysis=analysis)
        )
        hypotheses = [h.strip() for h in hyps_text.split(';') if h.strip()]
        # 4. Validate via agent and tools
        validations = []
        for hyp in hypotheses:
            result = self.agent.invoke(
                f"Validate hypothesis: {hyp}\nUsing data: {data_str}"
            )
            validations.append(result)
        # 5. Insights generation
        insights_text = self.validation_llm.complete(
            self.prompts["insights"].format(validations="\n".join(validations))
        )
        insights = [i.strip() for i in insights_text.split(';') if i.strip()]
        # 6. Recommendations
        recs_text = self.primary_llm.complete(
            self.prompts["recommendations"].format(insights="; ".join(insights), schema=schema)
        )
        recommendations = [r.strip() for r in recs_text.split(';') if r.strip()]

        return {
            "frameworks": frameworks,
            "root_causes": validations,
            "insights": insights,
            "recommendations": recommendations
        }

    def format_report(self, result: Dict[str, Any], schema: str) -> str:
        report = "# Root Cause Analysis Report\n\n"
        report += "## Frameworks Proposed\n"
        for fw in result.get("frameworks", "").split(';'):
            report += f"- {fw.strip()}\n"
        report += "\n## Root Cause Analysis\n"
        for i, cause in enumerate(result.get("root_causes", []), 1):
            report += f"{i}. {cause}\n"
        report += "\n## General Insights\n"
        for ins in result.get("insights", []):
            report += f"- {ins}\n"
        report += "\n## Recommendations\n"
        for r in result.get("recommendations", []):
            report += f"- {r}\n"
        report += f"\n## Database Schema Reference\n{schema}\n"
        return report

# Example usage
if __name__ == "__main__":
    agent = CausalAnalysisAgent()
    sample_reports = [
        "### Marketing Report\n- 25% ad spend reduction in Week 23\n- CTR drop from 1.8% to 1.2%",
        "### Technical Report\n- 4hr payment outage on 2024-05-20\n- API latency spikes to 1200ms",
        "### Operations Report\n- Stockouts for 3 top SKUs\n- West region shipping delays"
    ]
    sample_data = {
        "revenue": {"week22": 450000, "week23": 382500, "week24": 441000},
        "kpis": {"acquisitions": [1200, 720, 1100], "conversion_rate": [3.2, 2.8, 3.1]}
    }
    sample_schema = """
    Database Schema:
    1. marketing (campaign_id, spend, impressions, week)
    2. sales (transaction_id, amount, timestamp, region)
    3. inventory (sku_id, stock_level, week)
    4. errors (error_id, type, timestamp)
    """
    result = agent.run(sample_reports, sample_data, sample_schema)
    print(agent.format_report(result, sample_schema))

# Dependencies:
# pip install llama-index-llms-ollama scipy numpy
# Ensure Ollama service is running locally (`ollama serve`)
