# File: causal_agent_langchain.py
from typing import List, Dict, Any
from scipy import stats
import numpy as np
from langchain.llms import Ollama
from langchain import LLMChain, PromptTemplate
from langchain.agents import Tool, initialize_agent, AgentType

class CausalAnalysisAgent:
    def __init__(self,
                 primary_model: str = "llama3.1:latest",
                 validation_model: str = "deepseek",
                 api_base: str = "http://localhost:11434",
                 api_key: str = ""):
        # 1. Statistical Tools
        def pearson_tool(x: List[float], y: List[float]) -> float:
            return stats.pearsonr(x, y)[0]

        def z_test_tool(sample: List[float], population: List[float]) -> float:
            return (np.mean(sample) - np.mean(population)) / (np.std(population) / np.sqrt(len(sample)))

        def anova_tool(groups: List[List[float]]) -> float:
            return stats.f_oneway(*groups)[0]

        def contribution_tool(x: List[float], y: List[float]) -> float:
            cov = np.cov(x, y)[0,1]
            return cov / np.var(y)

        def t_test_tool(a: List[float], b: List[float]) -> float:
            return stats.ttest_ind(a, b, equal_var=False).statistic

        def chi_square_tool(observed: List[float], expected: List[float]) -> float:
            return float(stats.chisquare(observed, expected).statistic)

        def mannwhitneyu_tool(a: List[float], b: List[float]) -> float:
            return float(stats.mannwhitneyu(a, b, alternative='two-sided').statistic)

        def linear_regression_tool(x: List[float], y: List[float]) -> Dict[str, float]:
            slope, intercept, _, _, _ = stats.linregress(x, y)
            return {"slope": slope, "intercept": intercept}

        self.tools = [
            Tool(name="pearson", func=pearson_tool, description="Pearson correlation between two lists."),
            Tool(name="z_test", func=z_test_tool, description="Z-test between sample and population."),
            Tool(name="anova", func=anova_tool, description="One-way ANOVA F-statistic."),
            Tool(name="contribution", func=contribution_tool, description="Cov(x,y)/Var(y)."),
            Tool(name="t_test", func=t_test_tool, description="Two-sample t-test statistic."),
            Tool(name="chi_square", func=chi_square_tool, description="Chi-square test statistic."),
            Tool(name="mannwhitneyu", func=mannwhitneyu_tool, description="Mann-Whitney U test statistic."),
            Tool(name="linear_regression", func=linear_regression_tool, description="Linear regression slope and intercept."),
        ]

        # 2. Configure LLMs
        # Primary Chains use Llama3.1:latest
        self.primary_llm = Ollama(model=primary_model, api_base=api_base, api_key=api_key, temperature=0)
        # Validation Agent & Insights Chain use Deepseek
        self.validation_llm = Ollama(model=validation_model, api_base=api_base, api_key=api_key, temperature=0)

        # 3. Chains (Chain-of-Thought)
        self.input_analysis_chain = LLMChain(
            llm=self.primary_llm,
            prompt=PromptTemplate(
                input_variables=["reports","data_stats"],
                template="""
You are a data analyst. Given these reports:
{reports}
and data summary:
{data_stats}
Think step by step and then provide a concise input analysis.
"""
            ),
            output_key="input_analysis"
        )

        self.framework_chain = LLMChain(
            llm=self.primary_llm,
            prompt=PromptTemplate(
                input_variables=["reports","input_analysis"],
                template="""
As a framework advisor, review:
{reports}
and the analysis:
{input_analysis}
Think step by step, then list suitable analytical frameworks separated by semicolons.
"""
            ),
            output_key="frameworks"
        )

        self.hypotheses_chain = LLMChain(
            llm=self.primary_llm,
            prompt=PromptTemplate(
                input_variables=["input_analysis"],
                template="""
You are a hypothesis generator. Based on analysis:
{input_analysis}
Think step by step and generate multiple hypotheses separated by semicolons.
"""
            ),
            output_key="hypotheses"
        )

        # 4. Validation Agent (ReAct with Deepseek)
        self.validation_agent = initialize_agent(
            tools=self.tools,
            llm=self.validation_llm,
            agent=AgentType.REACT_DESCRIPTION,
            verbose=True
        )

        # 5. Insights Chain (Chain-of-Thought with Deepseek)
        self.insights_chain = LLMChain(
            llm=self.validation_llm,
            prompt=PromptTemplate(
                input_variables=["validations"],
                template="""
Given validation results:
{validations}
Think step by step and then provide general insights separated by semicolons.
"""
            ),
            output_key="insights"
        )

        # 6. Recommendations Chain (Chain-of-Thought with Primary LLM)
        self.recommendations_chain = LLMChain(
            llm=self.primary_llm,
            prompt=PromptTemplate(
                input_variables=["insights","schema"],
                template="""
Based on insights:
{insights}
and schema:
{schema}
Think step by step, then generate actionable recommendations. Prefix database actions with 'Database:'.
"""
            ),
            output_key="recommendations"
        )

    def run(self, reports: List[str], data: Dict[str, Any], schema: str) -> Dict[str, Any]:
        # Prepare inputs
        reports_str = "\n\n".join(reports)
        data_str = str(data)

        # 1. Input Analysis
        analysis = self.input_analysis_chain.run(reports=reports_str, data_stats=data_str)

        # 2. Framework Selection
        frameworks = self.framework_chain.run(reports=reports_str, input_analysis=analysis)

        # 3. Hypotheses Generation
        hyps_text = self.hypotheses_chain.run(input_analysis=analysis)
        hypotheses = [h.strip() for h in hyps_text.split(";") if h.strip()]

        # 4. Hypotheses Validation
        validations = []
        for hyp in hypotheses:
            res = self.validation_agent.run(f"Validate hypothesis: {hyp}\nUsing data: {data_str}")
            validations.append(res)

        # 5. Insights Generation
        insights_text = self.insights_chain.run(validations="\n".join(validations))
        insights = [i.strip() for i in insights_text.split(";") if i.strip()]

        # 6. Recommendations Generation
        recs_text = self.recommendations_chain.run(insights=insights_text, schema=schema)
        recommendations = [r.strip() for r in recs_text.split(";") if r.strip()]

        return {
            "frameworks": frameworks,
            "root_causes": validations,
            "insights": insights,
            "recommendations": recommendations
        }

    def format_report(self, result: Dict[str, Any], schema: str) -> str:
        report = "# Root Cause Analysis Report

"
        report += "## Frameworks Proposed
"
        for fw in result.get("frameworks", "").split(";"):
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

# Example Usage
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

# Requirements:
# pip install langchain scipy numpy
# Run Ollama service: ollama serve
