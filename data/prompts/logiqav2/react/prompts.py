REACT_PROMPT_V1 = ("Solve a question answering task by having a Thought, then Finish with your answer. Thought can reason about the current situation. "
                   "Finish[answer] returns the answer and finishes the task. You will be given context that you should use to help you answer the question.\n"
                   "Here are some examples:")

REACT_PROMPT_V2 = ("Solve a question answering task by having a Thought, then Finish with your answer. Thought can reason about the current situation. "
                   "Finish[answer] returns the answer and finishes the task. You will be given context that you should use to help you answer the question.")

prompts = {
    "react_v1": REACT_PROMPT_V1,
    "react_v2": REACT_PROMPT_V2,
    "react_v2_mistral": "[INST] " + REACT_PROMPT_V2 + " [/INST]",
}


def get_prompt(prompt_name: str) -> str:
    return prompts[prompt_name]
