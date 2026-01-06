import random

MASK_TOKEN_VARIATIONS = (
    "[MASK]",
    "___",
    "@@@",
    "###",
    "+++",
    "<<>>",
    "(())",
    "$$$",
)

def _single_character_identification(
    text: str
) -> tuple:
    """
    Generate instruction data for single character identification task.
    
    Args:
        text: The input text.
        
    Returns:
        A tuple containing the prompt and completion.
        
    Example:
        Input: "The quick brown fox jumps over the lazy dog."
        Output: (
            "Identify the masked character in the following text: \n\n[MASK]he quick brown fox jumps over the lazy dog.",
            "T"
        )
    """
    instruction_variants = (
        "What letter should replace the [MASK] in this text?",
        "Which character is hidden behind the [MASK] in the following text?",
        "What is the missing letter marked by [MASK] in this passage?",
        "Can you figure out what character belongs in place of the [MASK]?",
        "What letter would complete this text where you see [MASK]?",
        "Fill in the [MASK] with the correct missing character:",
        "Guess the letter that has been replaced with [MASK]:",
        "What character should go where you see [MASK] in this text?",
        "Identify which letter belongs in the [MASK] position:",
        "What's the original character that was replaced by [MASK]?",
    )
    instruction = random.choice(instruction_variants)
    valid_indices = [i for i in range(len(text)) if text[i].isalnum() and not text[i].isdigit()]
    if not valid_indices:
        return f"{instruction}\n\n{text}", "No character to identify."
    char_idx = random.choice(valid_indices)
    char = text[char_idx]
    masked_text = text[:char_idx] + "[MASK]" + text[char_idx + 1:]
    completion = char
    prompt = f"{instruction}\n\n{masked_text}"
    return prompt, completion


def _multi_character_identification(
    text: str,
) -> tuple:
    """
    Generate instruction data for multi-character identification task. 
    The task involves identifying multiple characters in the text and returns a list of characters as the completion.
    
    Args:
        text: The input text.
        
    Returns:
        A tuple containing the prompt and completion.
        
    Example:
        Input: "The quick brown fox jumps over the lazy dog."
        Output: (
            "Identify the masked characters in the following text and provide a list of them in the order of appearance: \n\n[MASK]he quick brown fox jumps over the lazy dog.",
            "['T', 'q', 'j']"
        )
    """
    instruction_variants = (
        "What are the hidden characters in this text? List them in order:",
        "Can you reveal the characters behind each [MASK]? List them in sequence:",
        "Name each masked character in this text in order of appearance:",
        "What letters have been replaced by [MASK]? List them in order:",
        "Identify each hidden character in sequence:",
        "Uncover the masked characters and list them in order:",
        "What characters should replace the [MASK]s? List them in order:",
        "Reveal the original characters hidden by [MASK] in sequence:",
        "List the missing characters in order of appearance:",
        "What letters belong in place of each [MASK]? List them in sequence:",
    )
    instruction = random.choice(instruction_variants)
    valid_indices = [i for i in range(len(text)) if text[i].isalnum() and not text[i].isdigit()]
    if not valid_indices:
        return f"{instruction}\n\n{text}", "No characters to identify."
    num_chars_to_identify = random.randint(1, min(3, len(valid_indices)))
    char_indices = random.sample(valid_indices, num_chars_to_identify)
    char_indices.sort()
    masked_text = list(text)
    completion = []
    for idx in char_indices:
        char = text[idx]
        masked_text[idx] = "[MASK]"
        completion.append(char)
    masked_text = "".join(masked_text)
    prompt = f"{instruction}\n\n{masked_text}"
    return prompt, str(completion)


def _multi_character_identification_and_replacement(
    text: str,
    no_instruction_variation: bool = False,
    no_instruction: bool = False,
    **kwargs
) -> tuple:
    """
    Generate instruction data for multi-character identification and replacement task. 
    The task involves identifying multiple characters in the text and returns the correct text as the completion.
    
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
            "Identify the masked characters in the following text and replace them with the correct characters to restore the original text: \n\n[MASK]he quick
            brown [MASK]ox j[MASK]mps over the lazy dog.",
            "The quick brown fox jumps over the lazy dog."
        )
    """
    mask_variant = random.choice(MASK_TOKEN_VARIATIONS)
    instruction_variants = (
        f"Replace all {mask_variant} with the correct characters.",
        f"Fill in all {mask_variant} tokens to restore the original text.",
        f"What characters should replace the {mask_variant} in this text?",
        f"Restore this text by replacing each {mask_variant} with the correct character.",
        f"Unmask this text by filling in all {mask_variant} with the right letters.",
        f"Complete this text by substituting appropriate characters for all {mask_variant}.",
        f"Identify what letters should go in place of each {mask_variant}.",
        f"Fix this text by replacing each {mask_variant} with its original character.",
        f"What was the original text before characters were replaced with {mask_variant}?",
        f"Reconstruct the original text by figuring out what each {mask_variant} represents.",
        f"What belongs in place of each {mask_variant} to reveal the original text?",
        f"Decode this message by replacing all {mask_variant} with the right characters.",
        f"What character goes in each {mask_variant}?",
        f"Recover the hidden characters marked by {mask_variant} in this text.",
        f"Convert this masked text back to normal by replacing each {mask_variant}.",
        f"Guess each character that was substituted with {mask_variant}.",
        f"Show the complete text by filling in all {mask_variant} tokens.",
        f"Replace {mask_variant} symbols with their original characters.",
        f"Reveal the text hidden behind these {mask_variant} markers.",
        f"What's the text with all {mask_variant} properly replaced?",
    )
    if no_instruction_variation:
        instruction = f"Replace all {mask_variant} with the correct characters."
    else:
        instruction = random.choice(instruction_variants)
    valid_indices = [i for i in range(len(text)) if text[i].isalnum() and not text[i].isdigit()]
    if not valid_indices:
        if no_instruction:
            return text, text
        return f"{instruction}\n\n{text}", text
    char_indices = random.sample(valid_indices, round(len(valid_indices) * kwargs.get("mask_ratio", 0.15)))
    char_indices.sort()
    masked_text = list(text)
    completion = text
    for idx in char_indices:
        masked_text[idx] = mask_variant
    masked_text = "".join(masked_text)
    if no_instruction:
        return masked_text, completion
    # 50% chance to swap the instruction and context
    if random.random() < 0.5:
        instruction, masked_text = masked_text, instruction
    prompt = f"{instruction}\n\n{masked_text}"
    return prompt, completion


def generate_char_identification_instruction_data(
    example: dict,
    batched: bool = False,
    min_num_words: int = 5,
    use_only_primary: bool = False,
    **kwargs
) -> dict:
    """
    Generate instruction data for character identification task.
    
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
        "single_character_identification": _single_character_identification,
        "multi_character_identification": _multi_character_identification,
        "multi_character_identification_and_replacement": _multi_character_identification_and_replacement,
    }
    task_variants = list(task_funcs.keys())
    if batched:
        if use_only_primary:
            task_variant = "multi_character_identification_and_replacement"
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
            task_name = "multi_character_identification_and_replacement"
        else:
            task_name = random.choice(task_variants)
        text = example["text"]
        prompt, completion = task_funcs[task_name](text)
        return {"prompt": prompt, "completion": completion}
