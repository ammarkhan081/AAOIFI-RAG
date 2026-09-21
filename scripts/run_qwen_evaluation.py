import json
import os
import sys
from pathlib import Path

# Add src to pythonpath
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aaoifi_rag.reporting.evaluation import build_evaluation_label, EXPECTED_BEHAVIOURS
from aaoifi_rag.reporting.metrics import compute_routing_metrics

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_qwen_evaluation.py <path_to_qwen_results.json>")
        sys.exit(1)
        
    results_path = Path(sys.argv[1])
    if not results_path.exists():
        print(f"File not found: {results_path}")
        sys.exit(1)
        
    hard_set_path = REPO_ROOT / "data" / "private" / "hard_set.jsonl"
    if not hard_set_path.exists():
        print(f"Hard set not found at {hard_set_path}")
        sys.exit(1)
        
    # 1. Load ground truth
    ground_truth = {}
    with hard_set_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                ground_truth[item["item_id"]] = item
                
    # 2. Load results
    with results_path.open("r", encoding="utf-8") as f:
        results_data = json.load(f)
        
    # Determine item array
    if isinstance(results_data, dict) and "items" in results_data:
        items = results_data["items"]
    elif isinstance(results_data, list):
        items = results_data
    else:
        print("Unrecognized JSON structure.")
        sys.exit(1)
        
    # 3. Build Evaluation Labels
    labels = []
    gates = []
    
    for res in items:
        item_id = res.get("item_id")
        if not item_id or item_id not in ground_truth:
            continue
            
        gt_item = ground_truth[item_id]
        
        # Get final routing decision (ANSWER, ABSTAIN, ESCALATE)
        decision = res.get("decision")
        if not decision:
            # Fallback if parsing an older ablation file
            status = res.get("status")
            decision = "abstain" if status == "abstained" else "answer"
            
        # Get triggering gate from stages
        gate = None
        for stage in res.get("stages", []):
            if stage["stage"] == "route":
                detail = stage.get("detail", "")
                if "gate=" in detail:
                    gate_str = detail.split("gate=")[1].split()[0]
                    if gate_str != "None":
                        gate = gate_str
                        
        labels.append(build_evaluation_label(gt_item, decision))
        gates.append(gate)
        
    # 4. Compute Metrics
    metrics = compute_routing_metrics(labels, triggering_gates=gates)
    
    # 5. Print Scorecard
    print(f"## Evaluation Metrics for {results_path.name}\n")
    print("| Metric | Value | Definition |")
    print("|--------|-------|------------|")
    print(f"| **Items Evaluated** | {metrics.n_items} | Total n in evaluation set |")
    print(f"| **Answer Rate** | {metrics.coverage:.1%} ({metrics.n_answer}/{metrics.n_items}) | % of items producing a verified ANSWER |")
    print(f"| **Abstention Rate** | {metrics.abstention_rate:.1%} ({metrics.n_abstain}/{metrics.n_items}) | % of items where model abstained |")
    print(f"| **Escalation Rate** | {metrics.escalation_rate:.1%} ({metrics.n_escalate}/{metrics.n_items}) | % of items caught by safety gates |")
    print(f"| **Routing Agreement** | {metrics.routing_agreement:.1%} ({metrics.n_decision_matches_expected}/{metrics.n_items}) | % of decisions matching Ground Truth |")
    
    if metrics.triggering_gates:
        print("\n### Gate Interventions (Escalations)")
        for g, count in metrics.triggering_gates.items():
            if g != "none":
                print(f"- **{g}**: {count}")
                
    print(f"\n*Note: All hard-set items are answerable by definition, so expected_behavior = 'answer' for all items. Abstentions are conservative misses, Escalations are caught errors.*")

if __name__ == "__main__":
    main()
