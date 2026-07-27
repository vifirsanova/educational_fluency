"""
Human-in-the-Loop Dashboard for reviewing uncertain responses.
"""

import streamlit as st
import json
import os
import sys
from datetime import datetime

# Add src/agents to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src", "agents"))

from orchestrator import Orchestrator


# Page configuration
st.set_page_config(
    page_title="HITL Review Dashboard",
    page_icon="👁️",
    layout="wide"
)

st.title("Human-in-the-Loop Review Dashboard")
st.markdown("Review and adjudicate uncertain responses from the multi-agent system.")


def load_pipeline():
    """Load the pipeline orchestrator."""
    # Create a simple queue storage
    hitl_queue_file = "results/hitl_queue.json"
    if os.path.exists(hitl_queue_file):
        with open(hitl_queue_file, "r") as f:
            return json.load(f)
    return []


def save_review(review_data):
    """Save a review and update the queue."""
    hitl_queue_file = "results/hitl_queue.json"
    
    # Load existing queue
    if os.path.exists(hitl_queue_file):
        with open(hitl_queue_file, "r") as f:
            hitl_queue = json.load(f)
    else:
        hitl_queue = []
    
    # Update the reviewed item
    for i, item in enumerate(hitl_queue):
        if item.get("query_id") == review_data["query_id"]:
            hitl_queue[i]["status"] = "reviewed"
            hitl_queue[i]["reviewed_at"] = datetime.now().isoformat()
            hitl_queue[i]["human_decision"] = review_data["decision"]
            hitl_queue[i]["human_response"] = review_data.get("response", "")
            hitl_queue[i]["reviewer_notes"] = review_data.get("notes", "")
            break
    
    # Save updated queue
    with open(hitl_queue_file, "w") as f:
        json.dump(hitl_queue, f, indent=2)
    
    # Also save to reviewed file
    reviewed_file = "results/hitl_reviewed.json"
    if os.path.exists(reviewed_file):
        with open(reviewed_file, 'r') as f:
            reviewed = json.load(f)
    else:
        reviewed = []
    
    reviewed.append({
        "query_id": review_data["query_id"],
        "reviewed_at": datetime.now().isoformat(),
        "decision": review_data["decision"],
        "human_response": review_data.get("response", ""),
        "notes": review_data.get("notes", "")
    })
    
    with open(reviewed_file, 'w') as f:
        json.dump(reviewed, f, indent=2)
    
    st.success(f"Review saved for query {review_data['query_id']}")


def get_pending_reviews():
    """Get pending reviews from the queue."""
    hitl_queue_file = "results/hitl_queue.json"
    if os.path.exists(hitl_queue_file):
        with open(hitl_queue_file, "r") as f:
            queue = json.load(f)
            return [item for item in queue if item.get("status") != "reviewed"]
    return []


def main():
    pending = get_pending_reviews()
    
    st.sidebar.header("Dashboard Stats")
    st.sidebar.metric("Pending Reviews", len(pending))
    
    reviewed_count = 0
    reviewed_file = "results/hitl_reviewed.json"
    if os.path.exists(reviewed_file):
        with open(reviewed_file, 'r') as f:
            reviewed_count = len(json.load(f))
    st.sidebar.metric("Completed Reviews", reviewed_count)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Review Triggers")
    st.sidebar.markdown("""
    - **Confidence < 0.5**: Human provides answer directly
    - **Confidence 0.5-0.75 with a repeated query**: Human reviews the Gatekeeper decision
    - **Faithfulness < 0.70**: Human adjudicates correctness
    - **Editor removal > 50%**: Human checks for information loss
    - **Agent evaluation failed**: System error requiring human review
    """)
    
    if not pending:
        st.info("No pending reviews. All responses have been processed automatically.")
        return
    
    # Display pending reviews
    for idx, item in enumerate(pending):
        with st.expander(f"Query {idx + 1}: {item.get('query', '')[:100]}...", expanded=idx == 0):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Query")
                st.write(item.get('query', ''))
                
                st.markdown("#### Candidate Answer")
                st.text_area("Generated answer", item.get('candidate_answer', ''), height=150, key=f"candidate_{idx}")
                
                st.markdown("#### Edited Answer")
                st.text_area("After editor", item.get('edited_answer', ''), height=150, key=f"edited_{idx}")
            
            with col2:
                st.markdown("#### Review Context")
                
                trigger_reason = item.get('review_reason', 'unknown')
                trigger_colors = {
                    'low_confidence_below_0.5': '🔴',
                    'medium_confidence_repeated_query': '🟡',
                    'low_faithfulness': '🟠',
                    'excessive_removal': '🔵',
                    'agent_evaluation_failed': '⚫',
                }
                st.markdown(f"**Trigger:** {trigger_colors.get(trigger_reason, '⚪')} {trigger_reason.replace('_', ' ').title()}")
                
                st.metric("Gatekeeper Confidence", f"{item.get('gatekeeper_confidence', 0):.2f}")
                st.metric("Verifier Faithfulness", f"{item.get('verifier_faithfulness', 0):.2f}")
                st.metric("Editor Removal %", f"{item.get('removal_percentage', 0)*100:.1f}%")
                
                # Display retrieved evidence
                st.markdown("#### Retrieved Evidence")
                passages = item.get("retrieved_passages", [])
                if passages:
                    for passage_index, passage in enumerate(passages, start=1):
                        with st.expander(f"Passage {passage_index}"):
                            st.write(passage)
                else:
                    st.warning("No retrieved evidence is available.")
                
                # Display agent explanations
                gatekeeper_result = item.get("gatekeeper_result", {})
                verifier_result = item.get("verifier_result", {})
                editor_metadata = item.get("editor_metadata", {})
                
                st.markdown("#### Gatekeeper Assessment")
                st.write(gatekeeper_result.get("reason", "No reason recorded"))
                
                knowledge_gaps = gatekeeper_result.get("knowledge_gaps", [])
                if knowledge_gaps:
                    st.markdown("**Knowledge gaps:**")
                    for gap in knowledge_gaps:
                        st.markdown(f"- {gap}")
                
                st.markdown("#### Verifier Assessment")
                st.write(verifier_result.get("reason", "No reason recorded"))
                
                unsupported_claims = verifier_result.get("unsupported_claims", [])
                if unsupported_claims:
                    st.markdown("**Unsupported claims:**")
                    for claim in unsupported_claims:
                        st.markdown(f"- {claim}")
                
                st.markdown("#### Editor Metadata")
                st.json(editor_metadata)
            
            st.markdown("---")
            st.markdown("#### Human Adjudication")
            
            col_a, col_b = st.columns([1, 2])
            
            with col_a:
                if trigger_reason == "low_confidence_below_0.5":
                    decision_options = ["Provide answer", "Abstain"]
                else:
                    decision_options = ["Accept", "Reject", "Modify"]
                
                decision = st.radio(
                    "Decision",
                    decision_options,
                    key=f"decision_{idx}"
                )
            
            with col_b:
                if decision in {"Provide answer", "Modify"}:
                    default_response = (
                        ""
                        if decision == "Provide answer"
                        else item.get("edited_answer", "")
                    )
                    
                    human_response = st.text_area(
                        "Human-reviewed response:" if decision == "Modify" else "Provide answer:",
                        value=default_response,
                        height=120,
                        key=f"response_{idx}"
                    )
                elif decision == "Accept":
                    human_response = item.get("edited_answer", "")
                else:
                    human_response = ""
            
            notes = st.text_input("Reviewer notes (optional)", key=f"notes_{idx}")
            
            if st.button(f"Submit Review for Query {idx + 1}", key=f"submit_{idx}"):
                review_data = {
                    "query_id": item.get("query_id", f"query_{idx}"),
                    "decision": decision.lower(),
                    "response": human_response,
                    "notes": notes
                }
                save_review(review_data)
                st.rerun()


if __name__ == "__main__":
    main()
