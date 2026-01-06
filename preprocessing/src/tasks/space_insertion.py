import random


def _space_insertion(
    text: str,
    no_instruction_variation: bool = False,
    no_instruction: bool = False,
    **kwargs
) -> tuple:
    """
    Generate instruction data for space insertion task.
    The task involves inserting spaces into a given text to make it readable.
    
    Args:
        text: The input text.
        no_instruction_variation: A boolean indicating whether to use a single instruction variation (default is False).
        no_instruction: A boolean indicating whether to exclude the instruction (default is False).
        **kwargs: Additional keyword arguments.
        
    Returns:
        A dictionary containing the prompt and completion.
        
    Example:
        Input: "The quick brown fox jumps over the lazy dog."
        Output: (
            "Insert spaces into the text to make it readable.\n\nThequickbrownfoxjumpsoverthelazydog.",
            "The quick brown fox jumps over the lazy dog."
        )
    """
    instruction_variants = (
        "Insert whitespaces into the text to make it readable.",
        "Add spaces to make this text readable.",
        "Insert spaces between words in this text.",
        "Put spaces back into this text.",
        "Restore spaces to make this text legible.",
        "Add whitespace to separate words.",
        "Fix this text by adding spaces.",
        "Separate the words with spaces.",
        "Make this text readable by inserting spaces.",
        "Add appropriate spacing to this text.",
        "Insert spaces where needed in this text.",
        "Put whitespace between words.",
        "Restore proper spacing to this text.",
        "Space out the words in this text.",
        "Add spaces to separate the words.",
        "Put in spaces to make this readable.",
        "Insert whitespace to form proper words.",
        "Separate this text into readable words.",
        "Break up this text with proper spacing.",
        "Add spaces to form proper sentences."
    )
    if no_instruction_variation:
        instruction = "Insert whitespaces into the text to make it readable."
    else:
        instruction = random.choice(instruction_variants)
    # Remove all spaces
    context = text.replace(" ", "")
    if no_instruction:
        prompt = context
    else:
        # 50% chance to swap instruction and context
        if random.random() < 0.5:
            prompt = f"{instruction}\n\n{context}"
        else:
            prompt = f"{context}\n\n{instruction}"
    return prompt, text
        

def generate_space_insertion_instruction_data(
    example: dict,
    batched: bool = False,
    min_num_words: int = 5,
    use_only_primary: bool = False,
    **kwargs
) -> dict:
    """
    Generate instruction data for space insertion task.
    
    Args:
        example: A dictionary containing the input text. If batched is True, it should contain a list of texts under the key "text".
        batched: A boolean indicating whether the input is batched (default is False).
        min_num_words: An integer specifying the minimum number of words required in the text (default is 5).
        use_only_primary: A boolean indicating whether to use only the primary task variant (default is False).
        **kwargs: Additional keyword arguments.
        
    Returns:
        A dictionary containing the prompt and completion. If batched is True, it returns lists of prompts and completions.
    """
    task_funcs = {
        "space_insertion": _space_insertion,
    }
    task_variants = list(task_funcs.keys())
    if batched:
        if use_only_primary:
            task_variant = "space_insertion"
            results = [
                task_funcs[task_variant](text, kwargs.get("no_instruction_variation", False), kwargs.get("no_instruction", False))
                for text in example["text"]
                if len(text.split()) >= min_num_words
            ]
        else:
            results = [
                task_funcs[random.choice(task_variants)](text)
                for text in example["text"]
                if len(text.split()) >= min_num_words
            ]
        if results:
            prompts, completions = zip(*results)
        else:
            prompts, completions = [], []
        return {"prompt": list(prompts), "completion": list(completions)}
    else:
        if use_only_primary:
            task_name = "space_insertion"
        else:
            task_name = random.choice(task_variants)
        text = example["text"]
        prompt, completion = task_funcs[task_name](text)
        return {"prompt": prompt, "completion": completion}
