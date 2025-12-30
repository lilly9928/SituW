import gradio as gr
import json
import os
import random
from datetime import datetime

DATA_PATH = "/data3/KJE/code/SituW/situW/utils/Exp_SituationModeling_LLM/data/questions_shuffled.json"
SAVE_PATH = "/data3/KJE/code/SituW/situW/utils/Exp_SituationModeling_LLM/outputs/"

LABELS = ["a", "b", "c", "d"]

# -------------------------
# Load Dataset
# -------------------------
with open(DATA_PATH) as f:
    DATASET = json.load(f)

TOTAL = len(DATASET)

# -------------------------
# Helper
# -------------------------
def get_gt(options):
    return sorted(
        LABELS[i] for i, opt in enumerate(options)
        if opt["label"] == 0
    )

# -------------------------
# Consent Logic
# -------------------------
def agree_and_start():
    return (
        gr.update(visible=False),  # consent box
        gr.update(visible=True)    # evaluation box
    )

# -------------------------
# Core Logic
# -------------------------
def next_question(idx, correct_cnt, results, saved, perm, a, b, c, d):

    # -------------------------
    # Selection validation
    # -------------------------
    selected_cnt = sum([a, b, c, d])

    if idx > 0:
        if selected_cnt != 2:
            gr.Warning("⚠️ You must select exactly two options.")
            return (
                gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(),
                idx, correct_cnt, results, saved, perm
            )

    # -------------------------
    # Save previous answer
    # -------------------------
    if idx > 0 and idx <= TOTAL:
        prev_item = DATASET[idx - 1]

        selected = sorted(
            LABELS[perm[i]]
            for i, flag in enumerate([a, b, c, d]) if flag
        )

        gt = get_gt(prev_item["options"])
        is_correct = selected == gt

        if is_correct:
            correct_cnt += 1

        results.append({
            "question_id": prev_item["question_id"],
            "prediction": selected,
            "ground_truth": gt,
            "correct": is_correct,
            "raw_output": ",".join(selected)
        })

    # -------------------------
    # End Condition
    # -------------------------
    if idx >= TOTAL:
        acc = correct_cnt / TOTAL * 100

        if not saved:
            os.makedirs(SAVE_PATH, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            save_file = os.path.join(
                SAVE_PATH,
                f"human_eval_{timestamp}.json"
            )

            save_data = {
                "summary": {
                    "total": TOTAL,
                    "correct": correct_cnt,
                    "accuracy": round(acc, 2),
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                },
                "results": results
            }

            with open(save_file, "w", encoding="utf-8") as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)

            saved = True

        final_md = f"""
## ✅ Evaluation Finished

- **Correct:** {correct_cnt} / {TOTAL}
- **Accuracy:** {acc:.2f}%
"""

        return (
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            final_md,
            idx,
            correct_cnt,
            results,
            saved,
            perm
        )

    # -------------------------
    # Load Next Question
    # -------------------------
    item = DATASET[idx]
    options = item["options"]

    perm = list(range(4))
    random.shuffle(perm)
    shuffled_options = [options[i] for i in perm]

    question_md = f"""
### Question {idx + 1} / {TOTAL}
**Question ID:** {item['question_id']}
"""

    return (
        gr.update(label=f"(a) {shuffled_options[0]['text']}", value=False, visible=True),
        gr.update(label=f"(b) {shuffled_options[1]['text']}", value=False, visible=True),
        gr.update(label=f"(c) {shuffled_options[2]['text']}", value=False, visible=True),
        gr.update(label=f"(d) {shuffled_options[3]['text']}", value=False, visible=True),
        question_md,
        idx + 1,
        correct_cnt,
        results,
        saved,
        perm
    )

# -------------------------
# Gradio UI
# -------------------------
with gr.Blocks() as demo:

    # =========================
    # Consent Page
    # =========================
    with gr.Column(visible=True) as consent_box:
        gr.Markdown("""
            # 🧠 Human Language Understanding Study – Consent Form

            Thank you for participating in this study.

            ### Purpose of the Study
            The purpose of this research is to investigate whether humans recognize and construct **situational understanding** while reading sentences.

            ### What You Will Do
            You will be asked to read short sentences and select **two options** that are most similar in meaning.

            ### Data & Privacy
            - All responses are collected **anonymously**
            - No personally identifiable information is stored
            - Your answers and performance data **will be saved for research purposes only**

            ### Voluntary Participation
            Your participation is voluntary, and you may stop at any time by closing the page.

            By clicking **“I Agree”**, you confirm that you understand the information above and agree to participate in this study.
            """)

        agree_btn = gr.Button("I Agree and Start")

    # =========================
    # Evaluation Page
    # =========================
    with gr.Column(visible=False) as eval_box:

        gr.Markdown("# Confusable Pair Human Evaluation")
        gr.Markdown("### From the four options, select only two options that can be considered similar in meaning.")
        gr.Markdown("### You may translate the options into any language you are comfortable with before answering.")

        state_idx = gr.State(0)
        state_correct = gr.State(0)
        state_results = gr.State([])
        state_saved = gr.State(False)
        state_perm = gr.State([])

        question_text = gr.Markdown()

        cb_a = gr.Checkbox(label="(a)")
        cb_b = gr.Checkbox(label="(b)")
        cb_c = gr.Checkbox(label="(c)")
        cb_d = gr.Checkbox(label="(d)")

        next_btn = gr.Button("Next")

        next_btn.click(
            fn=next_question,
            inputs=[
                state_idx, state_correct, state_results, state_saved, state_perm,
                cb_a, cb_b, cb_c, cb_d
            ],
            outputs=[
                cb_a, cb_b, cb_c, cb_d,
                question_text,
                state_idx, state_correct, state_results, state_saved, state_perm
            ]
        )

        demo.load(
            fn=next_question,
            inputs=[
                state_idx, state_correct, state_results, state_saved, state_perm,
                cb_a, cb_b, cb_c, cb_d
            ],
            outputs=[
                cb_a, cb_b, cb_c, cb_d,
                question_text,
                state_idx, state_correct, state_results, state_saved, state_perm
            ]
        )

    # =========================
    # Consent Button Action
    # =========================
    agree_btn.click(
        fn=agree_and_start,
        inputs=[],
        outputs=[consent_box, eval_box]
    )

demo.launch(share=True)
