import random


def _predict_second_passage(
    text: list[str], 
    no_instruction_variation: bool = False, 
    no_instruction: bool = False,
) -> tuple:
    """
    Generate prompt and completion for predicting the second passage.
    
    Args:
        text: List of sentences.
        no_instruction_variation: If True, the instruction will not be varied.
        no_instruction: If True, the instruction will not be included.
        
    Returns:
        Tuple of prompt and completion.
        
    Example:
        >>> result = _predict_second_passage(["This is the first passage.", "This is the second passage.", "This is the third passage."])
        >>> result
        ("Given the following text, predict the second passage.\n\n[1] This is the first passage.\n[3] This is the third passage.\n\n", "This is the second passage.")
    """
    instruction_variants = (
        "Predict the missing second passage.",
        "Complete the text by filling in the second passage.",
        "Write the missing middle passage.",
        "Fill in the second part of the text.",
        "What is the missing second passage?",
        "Provide the text for the second passage.",
        "What should go between these passages?",
        "Insert the missing second passage.",
        "Continue the text and connect these passages.",
        "What's the missing middle part?",
        "Complete the middle section.",
        "What text belongs in the middle?",
        "Bridge these two passages.",
        "Fill in the gap between passages.",
        "Write the second section.",
        "What connects these two parts?",
        "Provide the missing middle text.",
        "What's the second passage?",
        "Create the connecting passage.",
        "Supply the missing middle paragraph.",
    )
    if no_instruction_variation:
        instruction = "Predict the missing second passage."
    else:
        instruction = random.choice(instruction_variants)
    anchor_variants = (
        ("[1] ", "[3] "),
        ("[A] ", "[C] "),
        ("[a] ", "[c] "),
        ("[i] ", "[iii] "),
        ("[First] ", "[Third] "),
        ("[Opening] ", "[Closing] "),
        ("1. ", "3. "),
        ("A. ", "C. "),
        ("a. ", "c. "),
        ("i. ", "iii. "),
        ("1: ", "3: "),
        ("A: ", "C: "),
        ("a: ", "c: "),
        ("i: ", "iii: "),
        ("First: ", "Third: "),
        ("Opening: ", "Closing: "),
        ("(1) ", "(3) "),
        ("(A) ", "(C) "),
        ("(a) ", "(c) "),
        ("(i) ", "(iii) "),
        ("1) ", "3) "),
        ("A) ", "C) "),
        ("a) ", "c) "),
        ("i) ", "iii) "),
    )
    # 50% chance not to use an anchor
    if random.random() < 0.5:
        anchor = ("", "")
    else:
        anchor = random.choice(anchor_variants)
    completion = text[1]
    if no_instruction:
        return f"{anchor[0]}{text[0]}\n{anchor[1]}{text[2]}", completion
    # 50% chance to swap the instruction and context
    if random.random() < 0.5:
        prompt = f"{instruction}\n\n{anchor[0]}{text[0]}\n{anchor[1]}{text[2]}"
    else:
        prompt = f"{anchor[0]}{text[0]}\n{anchor[1]}{text[2]}\n\n{instruction}"
    return prompt, completion


def generate_passage_completion_instruction_data(
    example: dict,
    batched: bool = False,
    min_num_words: int = 5,
    **kwargs
) -> dict:
    """
    Generate instruction data for passage completion task.
    """
    task_funcs = {
        "predict_second_passage": _predict_second_passage,
    }
    task_variants = list(task_funcs.keys())
    if batched:
        results = [
            task_funcs[random.choice(task_variants)](text, kwargs.get("no_instruction_variation", False), kwargs.get("no_instruction", False))
            for text in example["text"]
            if all(len(sentence.split()) >= min_num_words for sentence in text)
        ]
        if results:
            prompts, completions = zip(*results)
        else:
            prompts, completions = [], []
        return {"prompt": list(prompts), "completion": list(completions)}
    else:
        raise ValueError("Batched mode is required for passage completion task.")
