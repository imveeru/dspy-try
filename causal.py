# File: causal_agent_ollama.py
import dspy
import numpy as np
from scipy import stats
from typing import List, Dict, Any

# Configure Ollama model
lm = dspy.LM('ollama_chat/llama3.1', api_base='http://localhost:11434', api_key='')
dspy.configure(lm=lm)

class StatisticalValidator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.tools = [
            dspy.Tool(
                name="pearson",
                func=lambda x, y: stats.pearsonr(x, y)[0]
            ),
            dspy.Tool(
                name="ztest",
                func=self.z_test
            ),
            dspy.Tool(
                name="anova",
                func=self.anova
            ),
            dspy.Tool(
                name="contribution",
                func=self.contribution_analysis
            )
        ]

    def z_test(self, sample, population):
        return (np.mean(sample) - np.mean(population)) / (np.std(population)/np.sqrt(len(sample)))

    def anova(self, groups):
        return stats.f_oneway(*groups)[0]

    def contribution_analysis(self, x, y):
        return np.cov(x, y)[0,1] / np.var(y)

class FrameworkSelector(dspy.Module):
    def __init__(self):
        super().__init__()
        self.choose_frameworks = dspy.ChainOfThought(
            "reports, data_stats -> frameworks"
        )
    
    def forward(self, reports: str, data_stats: str):
        frameworks = self.choose_frameworks(
            reports=reports,
            data_stats=data_stats
        ).frameworks
        return dspy.Prediction(frameworks=frameworks)

class AutonomousAnalyst(dspy.Module):
    def __init__(self):
        super().__init__()
        self.stats = StatisticalValidator()
        self.framework_selector = FrameworkSelector()
        
        self.analyze_inputs = dspy.ChainOfThought("reports, data -> input_analysis")
        self.generate_hypotheses = dspy.ChainOfThought("input_analysis -> hypotheses")
        self.validate_hypotheses = dspy.ReAct("hypothesis, data -> validation", tools=self.stats.tools)
        self.generate_insights = dspy.ChainOfThought("validations -> insights")
        self.create_recommendations = dspy.ChainOfThought("insights, schema -> recommendations")

    def forward(self, reports: List[str], data: Dict, schema: str):
        reports_str = "\n\n".join(reports)
        data_str = str(data)
        
        input_analysis = self.analyze_inputs(reports=reports_str, data=data_str).input_analysis
        frameworks = self.framework_selector(reports=reports_str, data_stats=input_analysis).frameworks
        hypotheses = self.generate_hypotheses(input_analysis=input_analysis).hypotheses.split("; ")
        
        validations = []
        for hypothesis in hypotheses:
            validation = self.validate_hypotheses(
                hypothesis=hypothesis,
                data=data_str
            ).validation
            validations.append(validation)
        
        insights = self.generate_insights(validations="\n".join(validations)).insights
        recommendations = self.create_recommendations(
            insights=insights,
            schema=schema
        ).recommendations
        
        return dspy.Prediction(
            root_causes=validations,
            insights=insights.split("; "),
            recommendations=recommendations.split("; ")
        )

def format_report(prediction, schema: str) -> str:
    report = "# Root Cause Analysis Report\n\n"
    
    # Root Causes section
    report += "## Root Cause Analysis\n"
    for i, cause in enumerate(prediction.root_causes, 1):
        report += f"{i}. {cause}\n"
    
    # General Insights section
    report += "\n## General Insights\n"
    for insight in prediction.insights:
        report += f"- {insight}\n"
    
    # Recommendations section
    report += "\n## Recommendations\n### General Actions\n"
    report += "\n".join([f"- {a}" for a in prediction.recommendations if "database" not in a.lower()])
    
    report += "\n\n### Database Investigations\n"
    report += "\n".join([f"- {a}" for a in prediction.recommendations if "database" in a.lower()])
    
    # Schema documentation
    report += f"\n\n## Database Schema Reference\n{schema}"
    return report

# Installation instructions
"""
1. Install required packages:
pip install dspy numpy scipy

2. Install Ollama:
- Download from https://ollama.ai/
- Follow installation instructions for your OS

3. Pull model:
ollama pull llama2

4. Start Ollama service:
ollama serve
"""

# Example execution
if __name__ == "__main__":
    # Sample data
    reports = [
        "### Marketing Report\n- 25% ad spend reduction in Week 23\n- CTR drop from 1.8% to 1.2%",
        "### Technical Report\n- 4hr payment outage on 2024-05-20\n- API latency spikes to 1200ms",
        "### Operations Report\n- Stockouts for 3 top SKUs\n- West region shipping delays"
    ]
    
    data = {
        "revenue": {
            "week22": 450000,
            "week23": 382500,
            "week24": 441000
        },
        "kpis": {
            "acquisitions": [1200, 720, 1100],
            "conversion_rate": [3.2, 2.8, 3.1]
        }
    }
    
    schema = """
    Database Schema:
    1. marketing (campaign_id, spend, impressions, week)
    2. sales (transaction_id, amount, timestamp, region)
    3. inventory (sku_id, stock_level, week)
    4. errors (error_id, type, timestamp)
    """
    
    analyst = AutonomousAnalyst()
    result = analyst(reports=reports, data=data, schema=schema)
    print(format_report(result, schema))
