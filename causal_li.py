# File: causal_agent_llamaindex.py
from typing import List, Dict, Any
import numpy as np
from scipy import stats

# LlamaIndex imports
from llama_index.tools import FunctionTool        # Function tool wrapper ([llamaindexxx.readthedocs.io](https://llamaindexxx.readthedocs.io/en/latest/examples/agent/react_agent.html?utm_source=chatgpt.com))
from llama_index.llms.ollama import Ollama            # Ollama LLM client ([llamaindexxx.readthedocs.io](https://llamaindexxx.readthedocs.io/en/latest/examples/agent/react_agent.html?utm_source=chatgpt.com))
from llama_index.agent import ReActAgent               # ReAct agent implementation ([llamaindexxx.readthedocs.io](https://llamaindexxx.readthedocs.io/en/latest/examples/agent/react_agent.html?utm_source=chatgpt.com))

# 1. Define statistical functions and wrap them as FunctionTools

def pearson(x: List[float], y: List[float]) -> float:
    """Compute Pearson correlation coefficient."""
    return stats.pearsonr(x, y)[0]
pearson_tool = FunctionTool.from_defaults(fn=pearson)

def z_test(sample: List[float], population: List[float]) -> float:
    """Compute z-test statistic between sample and population."""
    return (np.mean(sample) - np.mean(population)) / (np.std(population) / np.sqrt(len(sample)))
z_test_tool = FunctionTool.from_defaults(fn=z_test)

def anova(groups: List[List[float]]) -> float:
    """Perform one-way ANOVA, returning F-statistic."""
    return stats.f_oneway(*groups)[0]
anova_tool = FunctionTool.from_defaults(fn=anova)

def contribution(x: List[float], y: List[float]) -> float:
    """Compute covariance(x,y)/variance(y)."""
    cov = np.cov(x, y)[0,1]
    return cov / np.var(y)
contribution_tool = FunctionTool.from_defaults(fn=contribution)

def t_test(a: List[float], b: List[float]) -> float:
    """Compute two-sample t-test statistic."""
    return stats.ttest_ind(a, b, equal_var=False).statistic
t_test_tool = FunctionTool.from_defaults(fn=t_test)

def chi_square(observed: List[float], expected: List[float]) -> float:
    """Compute chi-square statistic for observed vs expected."""
    return float(stats.chisquare(observed, expected).statistic)
chi_square_tool = FunctionTool.from_defaults(fn=chi_square)

def mannwhitneyu(a: List[float], b: List[float]) -> float:
    """Compute Mann-Whitney U statistic."""
    return float(stats.mannwhitneyu(a, b, alternative='two-sided').statistic)
mwu_tool = FunctionTool.from_defaults(fn=mannwhitneyu)

def linear_regression(x: List[float], y: List[float]) -> Dict[str, float]:
    """Compute linear regression slope and intercept."""
    slope, intercept, _, _, _ = stats.linregress(x, y)
    return {"slope": slope, "intercept": intercept}
linreg_tool = FunctionTool.from_defaults(fn=linear_regression)

# Aggregate all tools
tools = [
    pearson_tool, z_test_tool, anova_tool, contribution_tool,
    t_test_tool, chi_square_tool, mwu_tool, linreg_tool
]

# 2. Define the causal analysis agent
class CausalAnalysisAgent:
    def __init__(
        self,
        primary_model: str = "llama3.1:latest",
        validation_model: str = "deepseek",
        request_timeout: float = 120.0
    ):
        # Initialize Ollama LLMs
        self.primary_llm = Ollama(model=primary_model, request_timeout=request_timeout)
        self.validation_llm = Ollama(model=validation_model, request_timeout=request_timeout)
        # Configure ReAct agent for hypothesis validation
        self.validation_agent = ReActAgent.from_tools(
            tools=tools,
            llm=self.validation_llm,
            verbose=True
        )
        # Prompt templates
        self.prompts = {
            "analysis": (
                """
You are a data analyst. Given these reports:\n{reports}\nand data summary:\n{data}\nThink step by step and provide a concise analysis.
"""
            ),
            "frameworks": (
                """
As a framework advisor, review reports and analysis:\n{reports}\nAnalysis:\n{analysis}\nThink step by step and list frameworks separated by semicolons.
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
        rpt = "\n\n".join(reports)
        data_str = str(data)
        # 1. Analysis
        resp1 = self.primary_llm.complete(
            self.prompts["analysis"].format(reports=rpt, data=data_str)
        )
        analysis = resp1.text
        # 2. Frameworks
        resp2 = self.primary_llm.complete(
            self.prompts["frameworks"].format(reports=rpt, analysis=analysis)
        )
        frameworks = resp2.text
        # 3. Hypotheses
        resp3 = self.primary_llm.complete(
            self.prompts["hypotheses"].format(analysis=analysis)
        )
        hyps = resp3.text
        hypotheses = [h.strip() for h in hyps.split(';') if h.strip()]
        # 4. Validation via chat()
        validations = []
        for hyp in hypotheses:
            chat_resp = self.validation_agent.chat(
                message=f"Validate hypothesis: {hyp} using data: {data_str}"
            )
            validations.append(chat_resp.response)
        # 5. Insights
        resp5 = self.validation_agent.chat(
            message=self.prompts["insights"].format(validations="\n".join(validations))
        )
        insights = [i.strip() for i in resp5.response.split(';') if i.strip()]
        # 6. Recommendations
        resp6 = self.primary_llm.complete(
            self.prompts["recommendations"].format(insights=resp5.response, schema=schema)
        )
        recommendations = [r.strip() for r in resp6.text.split(';') if r.strip()]
        return {
            "frameworks": frameworks,
            "root_causes": validations,
            "insights": insights,
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
