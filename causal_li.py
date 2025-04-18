# File: causal_agent_llamaindex.py
from typing import List, Dict, Any
from scipy import stats
import numpy as np

# LlamaIndex imports
from llama_index.llms.ollama import Ollama  # Ollama LLM client
from llama_index.core.tools import BaseTool, ToolMetadata  # Custom tool base classes
from llama_index.core.agent.workflow import ReActAgent  # ReAct agent implementation
from llama_index.core.workflow import Context  # Agent execution context

# 1. Define custom statistical tools
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
            description="Compute z-test statistic between sample and population."
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
            description="Perform one-way ANOVA, returning the F-statistic."
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
            description="Compute covariance(x,y)/variance(y) contribution metric."
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
            description="Compute linear regression slope and intercept."
        )
        super().__init__(metadata=metadata)
    def _run(self, x: List[float], y: List[float]) -> Dict[str, float]:
        slope, intercept, _, _, _ = stats.linregress(x, y)
        return {"slope": slope, "intercept": intercept}
    async def _arun(self, *args, **kwargs):
        raise NotImplementedError

# 2. Agent class
class CausalAnalysisAgent:
    def __init__(
        self,
        primary_model: str = "llama3.1:latest",
        validation_model: str = "deepseek",
        request_timeout: float = 120.0
    ):
        # Initialize LLMs
        self.primary_llm = Ollama(model=primary_model, request_timeout=request_timeout)
        self.validation_llm = Ollama(model=validation_model, request_timeout=request_timeout)

        # Setup validation agent with tools
        tools = [
            PearsonTool(), ZTestTool(), ANOVATool(), ContributionTool(),
            TTestTool(), ChiSquareTool(), MannWhitneyUTool(), LinearRegressionTool()
        ]
        self.validation_agent = ReActAgent(
            tools=tools,
            llm=self.validation_llm
        )

        # Define prompt templates
        self.prompts = {
            "analysis": (
                """
You are a data analyst. Given these reports:\n{reports}\nand this data summary:\n{data}\nThink step by step and provide a concise analysis.
"""
            ),
            "frameworks": (
                """
As a framework advisor, review these reports and analysis.\n{reports}\nAnalysis:\n{analysis}\nThink step by step and list appropriate frameworks separated by semicolons.
"""
            ),
            "hypotheses": (
                """
You are a hypothesis generator. Based on analysis:\n{analysis}\nThink step by step and generate hypotheses separated by semicolons.
"""
            ),
            "insights": (
                """
Given these validation results:\n{validations}\nThink step by step and provide general insights separated by semicolons.
"""
            ),
            "recommendations": (
                """
Based on insights:\n{insights}\nand schema:\n{schema}\nThink step by step and generate actionable recommendations; prefix database actions with 'Database:'.
"""
            )
        }

    def run(self, reports: List[str], data: Dict[str, Any], schema: str) -> Dict[str, Any]:
        # 1. Analysis
        rpt = "\n\n".join(reports)
        data_str = str(data)
        analysis = self.primary_llm.complete(self.prompts["analysis"].format(reports=rpt, data=data_str))

        # 2. Framework selection
        frameworks = self.primary_llm.complete(
            self.prompts["frameworks"].format(reports=rpt, analysis=analysis)
        )

        # 3. Hypotheses
        hyps = self.primary_llm.complete(
            self.prompts["hypotheses"].format(analysis=analysis)
        )
        hypotheses = [h.strip() for h in hyps.split(';') if h.strip()]

        # 4. Validation via ReActAgent
        ctx = Context(self.validation_agent)
        validations = []
        for hyp in hypotheses:
            out = self.validation_agent.run(f"Validate hypothesis: {hyp} using data: {data_str}", ctx=ctx)
            validations.append(str(out))

        # 5. Insights
        insights = self.validation_llm.complete(
            self.prompts["insights"].format(validations="\n".join(validations))
        )
        insights_list = [i.strip() for i in insights.split(';') if i.strip()]

        # 6. Recommendations
        recs = self.primary_llm.complete(
            self.prompts["recommendations"].format(insights=insights, schema=schema)
        )
        recommendations = [r.strip() for r in recs.split(';') if r.strip()]

        return {
            "frameworks": frameworks,
            "root_causes": validations,
            "insights": insights_list,
            "recommendations": recommendations
        }

    def format_report(self, result: Dict[str, Any], schema: str) -> str:
        report = "# Causal Analysis Report\n\n"
        report += "## Frameworks\n"
        for fw in result.get("frameworks", "").split(';'):
            report += f"- {fw.strip()}\n"
        report += "\n## Root Causes\n"
        for i, rc in enumerate(result.get("root_causes", []), 1):
            report += f"{i}. {rc}\n"
        report += "\n## Insights\n"
        for ins in result.get("insights", []):
            report += f"- {ins}\n"
        report += "\n## Recommendations\n"
        for rec in result.get("recommendations", []):
            report += f"- {rec}\n"
        report += f"\n## Schema\n{schema}\n"
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
    sample_schema = (
        "Database Schema:\n"
        "1. marketing(campaign_id,spend,impressions,week)\n"
        "2. sales(transaction_id,amount,timestamp,region)\n"
        "3. inventory(sku_id,stock_level,week)\n"
        "4. errors(error_id,type,timestamp)"
    )
    result = agent.run(sample_reports, sample_data, sample_schema)
    print(agent.format_report(result, sample_schema))

# Requirements:
# pip install llama-index-llms-ollama scipy numpy
# Ensure Ollama is running: `ollama serve`
