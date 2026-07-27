"""
Human-in-the-Loop Dashboard for reviewing uncertain responses.
"""

import streamlit as st
import json
import os
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator.pipeline import PipelineOrchestrator


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
    pipeline = PipelineOrchestrator()
    pipeline.load_hitl_queue()
    return pipeline


def save_review(pipeline, review_data):
    """Save a review and update the queue."""
    hitl_queue = pipeline.hitl_queue
    
    for i, item in enumerate(hitl_queue):
        if item["query_id"] == review_data["query_id"]:
            hitl_queue[i]["status"] = "reviewed"
            hitl_queue[i]["reviewed_at"] = datetime.now().isoformat()
            hitl_queue[i]["human_decision"] = review_data["decision"]
            hitl_queue[i]["human_response"] = review_data.get("response", "")
            hitl_queue[i]["reviewer_notes"] = review_data.get("notes", "")
            break
    
    # Save updated queue
    with open("results/hitl_queue.json", "w") as f:
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


def main():
    pipeline = load_pipeline()
    pending = pipeline.get_pending_reviews()
    
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
    - **Confidence 0.5-0.75 with repeated query**: Human reviews Gatekeeper decision
    - **Faithfulness < 0.70**: Human adjudicates correctness
    - **Editor removal > 50%**: Human checks for information loss
    - **Agent evaluation failed**: System error requiring human review
    """)
    
    if not pending:
        st.info("No pending reviews. All responses have been processed automatically.")
        return
    
    # Display pending reviews
    for idx, item in enumerate(pending):
        with st.expander(f"Query {idx + 1}: {item['query'][:100]}...", expanded=idx == 0):
            
            trigger_reason = item.get('review_reason', 'unknown')
            trigger_colors = {
                'low_confidence_below_0.5': '🔴',
                'medium_confidence_repeated_query': '🟡',
                'low_faithfulness': '🟠',
                'excessive_removal': '🔵',
                'agent_evaluation_failed': '⚫',
            }
            
            st.markdown(f"**Trigger:** {trigger_colors.get(trigger_reason, '⚪')} {trigger_reason.replace('_', ' ').title()}")
            
            # Display query and context
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Query")
                st.write(item['query'])
            
            with col2:
                st.metric("Gatekeeper Confidence", f"{item.get('gatekeeper_confidence', 0):.2f}")
                st.metric("Verifier Faithfulness", f"{item.get('verifier_faithfulness', 0):.2f}")
                st.metric("Editor Removal %", f"{item.get('removal_percentage', 0)*100:.1f}%")
            
            # Display evidence
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
            
            # --- HUMAN ADJUDICATION SECTION ---
            st.markdown("---")
            st.markdown("#### Human Adjudication")
            
            # Show different content based on trigger type
            if trigger_reason == "low_confidence_below_0.5":
                # Case 1: Confidence < 0.5 - Human provides answer directly
                st.info("**Low confidence**: The system could not find sufficient evidence. Please provide an answer directly or abstain.")
                
                # Do NOT show candidate or edited answers
                human_response = st.text_area(
                    "Provide answer:",
                    value="",
                    height=150,
                    key=f"response_{idx}",
                    help="Write your answer based on your knowledge and the provided evidence."
                )
                
                decision = st.radio(
                    "Decision",
                    ["Provide answer", "Abstain"],
                    key=f"decision_{idx}"
                )
                
                if decision == "Abstain":
                    human_response = ""
                    
            elif trigger_reason == "medium_confidence_repeated_query":
                # Case 2: Confidence 0.5-0.75 with repeated query - Human reviews gatekeeper decision
                st.info("**Medium confidence with repeated query**: Review whether the Gatekeeper's abstention decision was correct.")
                
                st.markdown("#### Gatekeeper Decision")
                st.write(f"**Confidence:** {item.get('gatekeeper_confidence', 0):.2f}")
                st.write(f"**Reason:** {gatekeeper_result.get('reason', 'No reason recorded')}")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("**Candidate Answer (generated)**")
                    st.text_area("", item.get('candidate_answer', ''), height=100, key=f"candidate_{idx}")
                with col_b:
                    st.markdown("**Edited Answer (if applicable)**")
                    st.text_area("", item.get('edited_answer', ''), height=100, key=f"edited_{idx}")
                
                decision = st.radio(
                    "Decision",
                    ["Accept abstention", "Provide answer", "Request human review"],
                    key=f"decision_{idx}"
                )
                
                if decision == "Accept abstention":
                    human_response = "Abstention accepted - no answer provided"
                elif decision == "Provide answer":
                    human_response = st.text_area(
                        "Provide answer:",
                        value=item.get('edited_answer', ''),
                        height=120,
                        key=f"response_override_{idx}"
                    )
                else:  # Request human review
                    human_response = "Requesting additional human review"
                    
            elif trigger_reason == "low_faithfulness":
                # Case 3: Faithfulness < 0.70 - Human adjudicates correctness
                st.info("**Low faithfulness**: The Verifier found unsupported claims. Adjudicate whether the answer is factually correct.")
                
                # Show the answer being evaluated
                st.markdown("#### Answer to Evaluate")
                st.text_area("", item.get('edited_answer', item.get('candidate_answer', '')), height=120, key=f"answer_eval_{idx}")
                
                # Show unsupported claims
                if unsupported_claims:
                    st.markdown("**Claims identified as unsupported:**")
                    for claim in unsupported_claims:
                        st.markdown(f"- {claim}")
                
                decision = st.radio(
                    "Decision",
                    ["Accept (answer is correct)", "Reject (answer is incorrect)", "Modify"],
                    key=f"decision_{idx}"
                )
                
                if decision == "Modify":
                    human_response = st.text_area(
                        "Provide corrected answer:",
                        value=item.get('edited_answer', ''),
                        height=120,
                        key=f"response_corrected_{idx}"
                    )
                elif decision == "Accept (answer is correct)":
                    human_response = item.get('edited_answer', item.get('candidate_answer', ''))
                else:  # Reject
                    human_response = ""
                    
            elif trigger_reason == "excessive_removal":
                # Case 4: Editor removal > 50% - Human checks for information loss
                st.info("**Excessive removal**: The Editor removed more than 50% of content. Check if important information was lost.")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("**Original Candidate**")
                    st.text_area("", item.get('candidate_answer', ''), height=150, key=f"original_{idx}")
                with col_b:
                    st.markdown("**Edited Version**")
                    st.text_area("", item.get('edited_answer', ''), height=150, key=f"edited_compare_{idx}")
                
                st.metric("Removal Percentage", f"{item.get('removal_percentage', 0)*100:.1f}%")
                
                decision = st.radio(
                    "Decision",
                    ["Accept (no information loss)", "Reject (information lost)", "Modify"],
                    key=f"decision_{idx}"
                )
                
                if decision == "Modify":
                    human_response = st.text_area(
                        "Provide revised version:",
                        value=item.get('edited_answer', ''),
                        height=120,
                        key=f"response_revised_{idx}"
                    )
                elif decision == "Accept (no information loss)":
                    human_response = item.get('edited_answer', '')
                else:  # Reject
                    human_response = item.get('candidate_answer', '')  # Keep original
                    
            else:  # agent_evaluation_failed or unknown
                # Case 5: System error - Human review required
                st.error("**System error**: The agent evaluation failed. Please review manually.")
                
                st.markdown("#### Candidate Answer")
                st.text_area("", item.get('candidate_answer', ''), height=120, key=f"error_candidate_{idx}")
                
                st.markdown("#### Error Information")
                st.json(item.get('error_info', {}))
                
                human_response = st.text_area(
                    "Provide corrected response:",
                    value=item.get('edited_answer', ''),
                    height=120,
                    key=f"response_error_{idx}"
                )
                
                decision = st.radio(
                    "Decision",
                    ["Accept", "Reject", "Modify"],
                    key=f"decision_{idx}"
                )
            
            notes = st.text_input("Reviewer notes (optional)", key=f"notes_{idx}")
            
            if st.button(f"Submit Review for Query {idx + 1}", key=f"submit_{idx}"):
                review_data = {
                    "query_id": item["query_id"],
                    "decision": decision.lower(),
                    "response": human_response,
                    "notes": notes
                }
                save_review(pipeline, review_data)
                st.rerun()


if __name__ == "__main__":
    main()
