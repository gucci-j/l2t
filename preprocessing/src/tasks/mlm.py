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

def _fill_in_the_blank(
    text: str,
    **kwargs
) -> tuple:
    """
    Generates a fill-in-the-blank prompt along with the corresponding completion using a masked version of the input text.
    The function randomly selects an instruction from a predefined set of variants and replaces one word in the input text with the "[MASK]" 
    token according to the specified mask ratio. It returns a tuple where the first element is the prompt (instruction followed by the masked text) 
    and the second element is the original word.
    
    Parameters:
        text (str): The input text to be processed.
        
    Returns:
        tuple: A tuple containing:
            - prompt (str): The constructed prompt with instruction and masked text.
            - completion (str): The original, unmodified word.
            
    Example:
        Input: "The quick brown fox jumps over the lazy dog."
        Output: (
            "Fill in the blank in the following text: \n\nThe quick brown [MASK] jumps over the lazy dog.",
            "fox"
        )
    """
    instruction_variants = (
        "What word fits in the blank in this text?",
        "Complete this sentence by filling in the missing word:",
        "Which word belongs in the blank space?",
        "Fill in the missing word in this sentence:",
        "What's the correct word for this blank?",
        "Choose the most appropriate word for the blank:",
        "What word would make this sentence complete?",
        "Find the right word to fill the gap:",
        "Insert the missing word in this text:",
        "What word should go in place of the blank?",
    )
    instruction = random.choice(instruction_variants)
    words = text.split()
    masked_text = words.copy()
    masked_index = random.choice(range(len(words)))
    masked_word = masked_text[masked_index]
    masked_text[masked_index] = random.choice(MASK_TOKEN_VARIATIONS)
    masked_text = " ".join(masked_text)
    prompt = f"{instruction}\n\n{masked_text}"
    return prompt, masked_word


def _generate_missing_word_list(
    text: str,
    **kwargs
) -> tuple:
    """
    Generates a prompt for generating a list of missing words in the input text.
    The function randomly selects an instruction from a predefined set of variants and replaces a percentage
    of the words in the input text with the "[MASK]" token according to the specified mask ratio. It returns a tuple
    where the first element is the prompt (instruction followed by the masked text) and the second element is a list
    of the missing words.
    
    Parameters:
        text (str): The input text to be processed.
        **kwargs: Optional keyword arguments.
            mask_ratio (float, optional): The fraction of words to mask in the text. Defaults to 0.15 if not provided.
            
    Returns:
        tuple: A tuple containing:
            - prompt (str): The constructed prompt with instruction and masked text.
            - missing_words (str): A list of the missing words in the text.
            
    Example:
        Input: "The quick brown fox jumps over the lazy dog."
        Output: (
            "Return a list of the missing words in the following text: \n\nThe quick brown [MASK]
            fox jumps over the [MASK] dog.",
                "['fox', 'lazy']"
            )
        )
    """
    instruction_variants = (
        "What are all the missing words in this text? List them in order.",
        "List the words that should fill each blank, in order.",
        "What words belong in the blank spaces? List them all.",
        "Give me all the missing words in order.",
        "What words were removed from this text? List them in sequence.",
        "Tell me all the words that should go in the blanks, in order.",
        "What's the complete list of missing words in order?",
        "List each word that belongs in the blanks.",
        "What words complete this text? List them in order.",
        "Tell me which words are missing, in sequence."
    )
    instruction = random.choice(instruction_variants)
    words = text.split()
    masked_text = words.copy()
    masked_indices = random.sample(range(len(words)), round(len(words) * kwargs.get("mask_ratio", 0.15)))
    missing_words = str([masked_text[idx] for idx in masked_indices])
    for idx in masked_indices:
        masked_text[idx] = random.choice(MASK_TOKEN_VARIATIONS)
    masked_text = " ".join(masked_text)
    prompt = f"{instruction}\n\n{masked_text}"
    return prompt, missing_words


def _fill_in_the_blanks(
    text: str,
    no_instruction_variation: bool = False,
    no_instruction: bool = False,
    **kwargs
) -> tuple:
    """
    Generates a fill-in-the-blank prompt along with the corresponding completion using a masked version of the input text.
    The function randomly selects an instruction from a predefined set of variants and replaces a percentage
    of the words in the input text with the "[MASK]" token according to the specified mask ratio. It returns a tuple
    where the first element is the prompt (instruction followed by the masked text) and the second element is the original text.
    
    Parameters:
        text (str): The input text to be processed.
        no_instruction_variation (bool, optional): A boolean indicating whether to use a single instruction variation. Defaults to False.
        no_instruction (bool, optional): A boolean indicating whether to exclude the instruction. Defaults to False.
        **kwargs: Optional keyword arguments.
            mask_ratio (float, optional): The fraction of words to mask in the text. Defaults to 0.15 if not provided.
            
    Returns:
        tuple: A tuple containing:
            - prompt (str): The constructed prompt with instruction and masked text.
            - completion (str): The original, unmodified text.
    
    Example:
        Input: "The quick brown fox jumps over the lazy dog."
        Output: (
            "Fill in the blank in the following text: \n\nThe quick brown [MASK]
            fox jumps over the lazy dog.",
                "The quick brown fox jumps over the lazy dog."
            )
        )
    """
    instruction_variants = (
        "Replace all blanks with the correct words.",
        "Complete the text by filling in all the blanks.",
        "Fill in all the missing words in this passage.",
        "Provide the correct words for each blank in the text.",
        "Restore the original text by filling in the blanks.",
        "Insert appropriate words in place of each blank.",
        "What words should replace the blanks in this text?",
        "Complete the following text by adding the missing words.",
        "Fill in each blank with the most suitable word.",
        "Reconstruct the complete text by filling in all blanks.",
        "Supply the missing words in the blanks below.",
        "Add the appropriate words to complete this text.",
        "What are the words that should fill each gap?",
        "Identify and insert the missing words in this passage.",
        "Write the correct word for each blank space.",
        "Find the right words to complete the text.",
        "Fill the blanks to restore the complete passage.",
        "Put the proper words in place of each blank.",
        "Complete this passage by filling all blanks.",
        "Determine what words belong in the blanks."
    )
    if no_instruction_variation:
        instruction = "Replace all blanks with the correct words."
    else:
        instruction = random.choice(instruction_variants)
    words = text.split()
    masked_text = words.copy()
    masked_indices = random.sample(range(len(words)), round(len(words) * kwargs.get("mask_ratio", 0.15)))
    for idx in masked_indices:
        masked_text[idx] = random.choice(MASK_TOKEN_VARIATIONS)
    masked_text = " ".join(masked_text)
    if no_instruction:
        return masked_text, text
    # 50% chance to swap the instruction and context
    if random.random() < 0.5:
        instruction, masked_text = masked_text, instruction
    prompt = f"{instruction}\n\n{masked_text}"
    return prompt, " ".join(text.split())


def generate_mlm_instruction_data(
    example: dict,
    batched: bool = False,
    min_num_words: int = 5,
    use_only_primary: bool = False,
    **kwargs
) -> dict:
    """
    Generate instruction data for masked language modeling task.
    """
    task_funcs = {
        "fill_in_the_blank": _fill_in_the_blank,
        "generate_missing_word_list": _generate_missing_word_list,
        "fill_in_the_blanks": _fill_in_the_blanks,
    }
    task_variants = list(task_funcs.keys())
    if batched:
        if use_only_primary:
            task_variant = "fill_in_the_blanks"
            results = [
                task_funcs[task_variant](text, **kwargs)
                for text in example["text"]
                if len(text.split()) >= min_num_words
            ]
        else:
            results = [
                task_funcs[random.choice(task_variants)](text, **kwargs)
                for text in example["text"]
                if len(text.split()) >= min_num_words
            ]
        if results:
            prompts, completions = zip(*results)
        else:
            prompts, completions = [], []
        return {"prompt": list(prompts), "completion": list(completions)}
    else:
        # Get the input text
        text = example["text"]
        if use_only_primary:
            task_variant = "fill_in_the_blanks"
        else:
            # Randomly select a task variant
            task_variant = random.choice(task_variants)
        # Generate instruction data based on the task variant
        prompt, completion = task_funcs[task_variant](text, **kwargs)
        return {"prompt": prompt, "completion": completion}
