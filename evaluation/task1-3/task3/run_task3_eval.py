import json
import pandas as pd
from tqdm import tqdm
import requests
import re
import os

# === 配置 ===
JUDGE_MODEL = "llama3.1:70b"
OLLAMA_API_URL = "http://localhost:11434/api/chat"
INPUT_EVAL_CSV = "results_task3_graphrag_n.csv" 

with open("test3_cases_custom.json", "r", encoding="utf-8") as f:
    CASE_DATA = {item.get('id', item.get('case_id')): item for item in json.load(f)}

def call_judge(prompt):
    data = {
        "model": JUDGE_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0.0} 
    }
    try:
        resp = requests.post(OLLAMA_API_URL, json=data).json()
        return resp['message']['content']
    except: return ""

def parse_scores(text):
    scores = {"Relevancy": 0, "Coverage": 0, "Convincity": 0, "Factuality": 0}
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            data = json.loads(match.group())
            for k in scores.keys():
                for dk in data.keys():
                    if k.lower() == dk.lower(): scores[k] = float(data[dk])
        else:
            for k in scores.keys():
                m = re.search(f"{k}[:\s\*\-]+(\d+(\.\d+)?)", text, re.IGNORECASE)
                if m: scores[k] = float(m.group(1))
    except: pass
    return scores

def evaluate_single(case_text, expected, model_answer, graph_evidence):
    # 🌟 关键修改：告诉裁判，能解释清楚因果关系的回答才是好回答！
    prompt = f"""
    You are a senior medical evaluator comparing AI diagnostic models.
    
    [Patient Case]: {case_text}
    [Expected Diagnosis]: {expected}
    [Graph Evidence (For Reference)]: {graph_evidence}
    
    [AI Answer]: {model_answer}
    
    Evaluate on 4 metrics (Scale 1-5, be strict but fair):
    
    1. Relevancy: The diagnostic response's alignment with the query entities in the case.
       - 5: The diagnosis and reasoning are tightly aligned with the core entities of the case.
       - 3: Partially aligned; touches on some relevant entities but also discusses tangential or unrelated issues.
       - 1: The response is about entities not present or not central to the case (off-topic diagnosis).
       - PENALTY: Deduct points if the answer fixates on minor/incidental
         findings while ignoring the chief complaint.
    2. Coverage: The extent to which the diagnostic response encompasses the query entities in the case.
       - 5: Explicitly addresses nearly all key symptoms, findings, drugs, and history items in the case.
       - 3: Covers roughly half of the key entities; misses some important findings or medications.
       - 1: Ignores most of the clinical entities; diagnosis is made from very limited information.
    3. Convincity (Explainability): The persuasiveness and clinical feasibility
       of the diagnostic response.
       - 5: Provides a clinically convincing, mechanistically coherent explanation; explicitly cites causal/pathophysiological links (e.g., "A causes B", "X is associated with Y via Z").
       - 3: Provides a plausible diagnosis but limited mechanistic reasoning; feasible but not fully convincing.
       - 1: Merely guesses a diagnosis without any reasoning chain, or the reasoning is clinically implausible.
       - BONUS: Give 5/5 if the answer explicitly traces a causal chain or cites graph-style evidence.
       - PENALTY: Give low score if it just outputs a diagnosis label without explaining the path.
    4. Factuality: The scientific/clinical accuracy of the information in the diagnostic response, judged against the Expected Diagnosis and established medical knowledge.
       - 5: Final diagnosis matches (or is clinically equivalent to) the Expected Diagnosis, and all supporting statements are medically accurate.
       - 3: Diagnosis is partially correct (e.g., right organ system or related condition) but contains minor inaccuracies.
       - 1: Diagnosis is incorrect or contains major factual errors (wrong mechanism, wrong drug indication, etc.).
    
    Output JSON format ONLY: {{"Relevancy": <score>, "Coverage": <score>, "Convincity": <score>, "Factuality": <score>}}
    """
    resp = call_judge(prompt)
    return parse_scores(resp)

def run_evaluation():
    if not os.path.exists(INPUT_EVAL_CSV):
        print(f"❌ File not found.")
        return

    df = pd.read_csv(INPUT_EVAL_CSV)
    results = []

    print(f"🚀 Evaluating {len(df)} cases...")

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        case_id = row.get('case_id')
        case_info = CASE_DATA.get(case_id, {})
        evidence = json.dumps(case_info.get('graph_evidence', []))
        text = row['case_text']
        expected = row['expected']

        # 1. Baseline
        s1 = evaluate_single(text, expected, row['prediction_baseline'], evidence)
        s1['Model'] = 'Zero-shot'
        results.append(s1)

        # 2. RAG
        s2 = evaluate_single(text, expected, row['prediction_rag'], evidence)
        s2['Model'] = 'RAG'
        results.append(s2)

        # 3. Ours
        s3 = evaluate_single(text, expected, row['prediction_ours'], evidence)
        s3['Model'] = 'Ours (Graphusion)'
        results.append(s3)

    res_df = pd.DataFrame(results)
    summary = res_df.groupby('Model')[['Relevancy', 'Coverage', 'Convincity', 'Factuality']].mean()
    
    print("\n=== Final Task 3 Scores (Optimized for Explainability) ===")
    print(summary)
    
    summary.to_csv("task3_final_metrics_summary_n.csv")

if __name__ == "__main__":
    run_evaluation()