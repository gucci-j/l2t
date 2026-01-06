import random


def _generate_tuples_of_shuffled_words(
    text: str,
    **kwargs
) -> tuple:
    """
    Generate instruction data for the task of generating tuples of shuffled words.
    
    Args:
        text: The input text.
        
    Returns:
        A tuple containing the prompt and a list of tuples representing the swapped word pairs.
        
    Example:
        Input: "The quick brown fox jumps over the lazy dog."
        Possible Output: (
            "Identify the pairs of shuffled words in the following text: \n\nThe quick fox brown jumps over the lazy dog.",
            [('fox', 'brown')]
        )
    """
    instruction_variants = (
        "Please provide a list of tuples containing the pairs of words that were swapped in this text:",
        "Return a list of tuples showing which words were exchanged in this text:",
        "Identify and list as tuples the pairs of words that were shuffled in this text:",
        "Express as a list of tuples the word pairs that were swapped in the following text:", 
        "Output a list of tuples representing the pairs of words that changed positions in this text:",
        "Using a list of tuples, show which words were interchanged in the following text:",
        "Present the shuffled word pairs as a list of tuples for this text:",
        "Find and return as a list of tuples the words that switched positions in this text:",
        "Indicate with a list of tuples which words were swapped in the following text:",
        "Analyze this text and provide a list of tuples containing the shuffled word pairs:"
    )
    words = text.split()
    num_pairs_to_shuffle = max(round(len(words) * random.uniform(0.05, 0.1)), 1)
    # Ensure we have enough words to swap
    if num_pairs_to_shuffle * 2 > len(words):
        num_pairs_to_shuffle = len(words) // 2
    indices_pool = list(range(len(words)))
    shuffled_indices = random.sample(indices_pool, num_pairs_to_shuffle * 2)
    swapped_pairs = []
    # Swap only the selected pairs and record them
    for i in range(0, len(shuffled_indices), 2):
        idx1 = shuffled_indices[i]
        idx2 = shuffled_indices[i + 1]
        # Perform the swap and record the resulting pair (order matters)
        words[idx1], words[idx2] = words[idx2], words[idx1]
        swapped_pairs.append((words[idx1], words[idx2]))
    shuffled_text = " ".join(words)
    instruction = random.choice(instruction_variants)
    prompt = f"{instruction}\n\n{shuffled_text}"
    return prompt, str(swapped_pairs)


def _predict_the_number_of_shuffles(
    text: str,
    **kwargs
) -> tuple:
    """
    Generate instruction data for the task of predicting the number of shuffles.
    
    Args:
        text: The input text.
        
    Returns:
        A tuple containing the prompt and the number of shuffles.
        
    Example:
        Input: "The quick brown fox jumps over the lazy dog."
        Possible Output: (
            "Predict the number of shuffles performed on the following text: \n\nThe brown quick fox over jumps the lazy dog.",
            "2"
        )
    """
    instruction_variants = (
        "How many word swaps were performed in this text?",
        "Count the number of word shuffles in this text:",
        "This text has been shuffled several times. How many?",
        "Can you tell how many word pairs were swapped in this text?", 
        "Determine how many times words were exchanged in this text:",
        "How many word exchanges occurred in this text?",
        "Calculate the number of word swaps in this text:",
        "What is the total number of word shuffles in this passage?",
        "How many word pairs were rearranged in this text?",
        "Count the total number of word exchanges in this text:"
    )
    words = text.split()
    num_pairs_to_shuffle = max(round(len(words) * random.uniform(0.05, 0.1)), 1)
    # Ensure we have enough words to swap
    if num_pairs_to_shuffle * 2 > len(words):
        num_pairs_to_shuffle = len(words) // 2
    indices_pool = list(range(len(words)))
    shuffled_indices = random.sample(indices_pool, num_pairs_to_shuffle * 2)
    # Swap only the selected pairs
    for i in range(0, len(shuffled_indices), 2):
        words[shuffled_indices[i]], words[shuffled_indices[i + 1]] = words[shuffled_indices[i + 1]], words[shuffled_indices[i]]
    shuffled_text = " ".join(words)
    instruction = random.choice(instruction_variants)
    prompt = f"{instruction}\n\n{shuffled_text}"
    return prompt, str(num_pairs_to_shuffle)


def _predict_the_number_of_shuffles_with_reasoning(
    text: str,
    **kwargs
) -> tuple:
    """
    Generate instruction data for the task of predicting the number of shuffles with reasoning.
    
    Args:
        text: The input text.
        
    Returns:
        A tuple containing the prompt, the number of shuffles, and the reasoning.
        
    Example:
        Input: "The quick brown fox jumps over the lazy dog."
        Possible Output: (
            "Estimate the number of shuffles performed on the following text and provide a brief explanation: \n\nThe brown quick fox over jumps the lazy dog.",
            "The words 'quick' and 'brown' have swapped positions, as well as 'over' and 'jumps'. Therefore, there were 2 shuffles."
        )
    """
    instruction_variants = (
        "Estimate the number of shuffles performed on the following text and provide a brief explanation:",
        "Predict the number of shuffles that occurred in this text and explain your reasoning:",
        "How many times was this text shuffled? Provide a concise explanation:",
        "Determine the number of shuffles in this text and justify your answer:", 
        "Estimate the total number of word swaps in this text and give a reason:",
        "Predict the number of word exchanges in this text and provide a rationale:",
        "Calculate the number of shuffles in this text and explain your reasoning:",
        "What is the total number of word shuffles in this passage? Explain your answer:",
        "Estimate the number of word pairs rearranged in this text and explain your reasoning:",
        "Predict the total number of word exchanges in this text and justify your answer:"
    )
    words = text.split()
    num_pairs_to_shuffle = max(round(len(words) * random.uniform(0.05, 0.1)), 1)
    # Ensure we have enough words to swap
    if num_pairs_to_shuffle * 2 > len(words):
        num_pairs_to_shuffle = len(words) // 2
    indices_pool = list(range(len(words)))
    shuffled_indices = random.sample(indices_pool, num_pairs_to_shuffle * 2)
    # Swap only the selected pairs
    for i in range(0, len(shuffled_indices), 2):
        words[shuffled_indices[i]], words[shuffled_indices[i + 1]] = words[shuffled_indices[i + 1]], words[shuffled_indices[i]]
    shuffled_text = " ".join(words)
    instruction = random.choice(instruction_variants)
    prompt = f"{instruction} \n\n{shuffled_text}"
    # Build explanation string for each pair of shuffled words
    pair_explanations = []
    for i in range(0, len(shuffled_indices), 2):
        pair_explanations.append(f"'{words[shuffled_indices[i]]}' and '{words[shuffled_indices[i+1]]}'")
    # Join the pairs with commas and 'and' for the last pair
    if len(pair_explanations) == 1:
        pairs_text = pair_explanations[0]
    else:
        pairs_text = ", ".join(pair_explanations[:-1]) + ", and " + pair_explanations[-1]
    completion_variants = (
        f"I observed that {pairs_text} have different positions compared to their original order. Therefore, there were {num_pairs_to_shuffle} shuffles.",
        f"Based on the word order changes, {pairs_text} were moved from their original positions, indicating {num_pairs_to_shuffle} shuffles.",
        f"By analyzing the text structure, I found that {pairs_text} were repositioned, which means {num_pairs_to_shuffle} shuffles occurred.",
        f"Looking at the word arrangements, {pairs_text} appear in different positions, suggesting {num_pairs_to_shuffle} shuffles took place.",
        f"After examining the text carefully, I noticed that {pairs_text} changed positions, confirming {num_pairs_to_shuffle} shuffles.",
        f"The word order reveals that {pairs_text} are not in their original positions. This indicates {num_pairs_to_shuffle} shuffles.",
        f"When comparing word positions, {pairs_text} show clear movement, proving there were {num_pairs_to_shuffle} shuffles.",
        f"By tracking word movements, I found that {pairs_text} switched places, resulting in {num_pairs_to_shuffle} shuffles.",
        f"The text structure shows that {pairs_text} were moved around, demonstrating {num_pairs_to_shuffle} shuffles.",
        f"After careful analysis, I found that {pairs_text} were rearranged, confirming {num_pairs_to_shuffle} shuffles."
    )
    return prompt, random.choice(completion_variants)


def _generate_tuples_of_shuffled_words_with_hint(
    text: str,
    **kwargs
) -> tuple:
    """
    Generate instruction data for the task of generating tuples of shuffled words with a hint (i.e. the number of shuffles).
    
    Args:
        text: The input text.
        
    Returns:
        A tuple containing the prompt and a list of tuples representing the swapped word pairs.
        
    Example:
        Input: "The quick brown fox jumps over the lazy dog."
        Output: (
            "Identify the pairs of shuffled words in the following text. The text was shuffled 2 times: \n\nThe quick fox brown over jumps the lazy dog.",
            "[('fox', 'brown'), ('over', 'jumps')]"
        )
    """
    instruction_variants = (
        "The following text has been shuffled {num_shuffles} times. List the pairs of swapped words as a list of tuples:",
        "This text underwent {num_shuffles} word swaps. Return a list of tuples showing which words were exchanged:", 
        "{num_shuffles} word pairs were shuffled in this text. Output them as a list of tuples:",
        "After {num_shuffles} word swaps, provide a list of tuples showing which pairs of words changed positions:",
        "Given that {num_shuffles} shuffles occurred, return a list of tuples containing the swapped word pairs:",
        "With {num_shuffles} word exchanges performed, list the word pairs as tuples:",
        "Knowing that {num_shuffles} shuffles took place, output a list of tuples containing the swapped word pairs:",
        "In this text, {num_shuffles} pairs of words were shuffled. Present them as a list of tuples:",
        "The text contains {num_shuffles} word swaps. Express the exchanged pairs as a list of tuples:",
        "Find the {num_shuffles} pairs of shuffled words and return them as a list of tuples:"
    )
    words = text.split()
    num_pairs_to_shuffle = max(round(len(words) * random.uniform(0.05, 0.1)), 1)
    # Ensure we have enough words to swap
    if num_pairs_to_shuffle * 2 > len(words):
        num_pairs_to_shuffle = len(words) // 2
    indices_pool = list(range(len(words)))
    shuffled_indices = random.sample(indices_pool, num_pairs_to_shuffle * 2)
    swapped_pairs = []
    # Swap only the selected pairs and record them
    for i in range(0, len(shuffled_indices), 2):
        idx1 = shuffled_indices[i]
        idx2 = shuffled_indices[i + 1]
        # Perform the swap and record the resulting pair (order matters)
        words[idx1], words[idx2] = words[idx2], words[idx1]
        swapped_pairs.append((words[idx1], words[idx2]))
    shuffled_text = " ".join(words)
    instruction = random.choice(instruction_variants).format(num_shuffles=num_pairs_to_shuffle)
    prompt = f"{instruction}\n\n{shuffled_text}"
    return prompt, str(swapped_pairs)


def _generate_correctly_ordered_text(
    text: str,
    no_instruction_variation: bool = False,
    no_instruction: bool = False,
    **kwargs
) -> tuple:
    """
    Generate instruction data for the task of generating correctly ordered text.
    
    Args:
        text: The input text.
        no_instruction_variation: A boolean flag indicating whether to use a fixed instruction variant.
        no_instruction: A boolean flag indicating whether to exclude the instruction.
        
    Returns:
        A tuple containing the prompt and the correctly ordered text.
        
    Example:
        Input: "The quick brown fox jumps over the lazy dog."
        Possible Output: (
            "Recover the original order of the following shuffled text: \n\nThe quick brown fox jumps over the lazy dog.",
            "The quick brown fox jumps over the lazy dog."
        )
    """
    instruction_variants = (
        "Fix this shuffled text.",
        "Restore the original word order.",
        "Rearrange the words correctly.",
        "Put the shuffled words back in order.",
        "Correct the word sequence.",
        "Unscramble this text.",
        "Fix the word order.",
        "Reorder these shuffled words.",
        "Return the text to its proper order.",
        "Arrange the words in the correct sequence.",
        "Correct this jumbled text.",
        "Reorganize these shuffled words.",
        "Fix the word arrangement.",
        "Put these words in proper order.",
        "Revert to the original word order.",
        "Sort out this scrambled text.",
        "Restore the correct word sequence.",
        "Arrange the text properly.",
        "Correct the word positions.",
        "Reorganize this mixed-up text."
    )
    words = text.split()
    num_pairs_to_shuffle = max(round(len(words) * random.uniform(0.05, 0.1)), 1)
    # Ensure we have enough words to swap
    if num_pairs_to_shuffle * 2 > len(words):
        num_pairs_to_shuffle = len(words) // 2
    indices_pool = list(range(len(words)))
    shuffled_indices = random.sample(indices_pool, num_pairs_to_shuffle * 2)
    # Swap only the selected pairs
    for i in range(0, len(shuffled_indices), 2):
        words[shuffled_indices[i]], words[shuffled_indices[i + 1]] = words[shuffled_indices[i + 1]], words[shuffled_indices[i]]
    shuffled_text = " ".join(words)
    if no_instruction:
        return shuffled_text, " ".join(text.split())
    if no_instruction_variation:
        instruction = "Fix this shuffled text."
    else:
        instruction = random.choice(instruction_variants)
    # 50% chance to swap the instruction and context
    if random.random() < 0.5:
        instruction, shuffled_text = shuffled_text, instruction
    prompt = f"{instruction}\n\n{shuffled_text}"
    return prompt, " ".join(text.split())


def _generate_correctly_ordered_text_with_hint(
    text: str,
    **kwargs
) -> tuple:
    """
    Generate instruction data for the task of generating correctly ordered text with a hint (i.e. the number of shuffles).
    
    Args:
        text: The input text.
        
    Returns:
        A tuple containing the prompt and the correctly ordered text.
        
    Example:
        Input: "The quick brown fox jumps over the lazy dog."
        Output: (
            "Recover the original order of the following shuffled text. The text was shuffled 2 times: \n\nThe quick brown fox jumps over the lazy dog.",
            "The quick brown fox jumps over the lazy dog."
        )
    """
    instruction_variants = (
        "This text has been shuffled {num_shuffles} times. Can you restore it to its original order?",
        "After {num_shuffles} word swaps, what would be the original text?",
        "The word order was changed {num_shuffles} times. What's the correct sequence?",
        "Knowing that {num_shuffles} shuffles occurred, can you reconstruct the original text?",
        "The text underwent {num_shuffles} word exchanges. What was the initial version?",
        "With {num_shuffles} word pairs swapped, please recover the original text:",
        "Given that {num_shuffles} shuffles took place, what's the proper word order?",
        "Help restore this text that was shuffled {num_shuffles} times to its original form:",
        "This passage had {num_shuffles} word swaps. What did it look like originally?",
        "Considering {num_shuffles} word exchanges occurred, what was the original text?"
    )
    words = text.split()
    num_pairs_to_shuffle = max(round(len(words) * random.uniform(0.05, 0.1)), 1)
    # Ensure we have enough words to swap
    if num_pairs_to_shuffle * 2 > len(words):
        num_pairs_to_shuffle = len(words) // 2
    indices_pool = list(range(len(words)))
    shuffled_indices = random.sample(indices_pool, num_pairs_to_shuffle * 2)
    # Swap only the selected pairs
    for i in range(0, len(shuffled_indices), 2):
        words[shuffled_indices[i]], words[shuffled_indices[i + 1]] = words[shuffled_indices[i + 1]], words[shuffled_indices[i]]
    shuffled_text = " ".join(words)
    instruction = random.choice(instruction_variants).format(num_shuffles=num_pairs_to_shuffle)
    prompt = f"{instruction}\n\n{shuffled_text}"
    return prompt, " ".join(text.split())


def generate_shuffle_instruction_data(
    example: dict,
    batched: bool = False,
    min_num_words: int = 5,
    use_only_primary: bool = False,
    **kwargs
) -> dict:
    """
    Generate instruction data for text shuffling task.
    """
    task_funcs = {
        "generate_tuples_of_shuffled_words": _generate_tuples_of_shuffled_words,
        "predict_the_number_of_shuffles": _predict_the_number_of_shuffles,
        "predict_the_number_of_shuffles_with_reasoning": _predict_the_number_of_shuffles_with_reasoning,
        "generate_tuples_of_shuffled_words_with_hint": _generate_tuples_of_shuffled_words_with_hint,
        "generate_correctly_ordered_text": _generate_correctly_ordered_text,
        "generate_correctly_ordered_text_with_hint": _generate_correctly_ordered_text_with_hint,
    }
    task_variants = list(task_funcs.keys())
    if batched:
        if use_only_primary:
            task_variant = "generate_correctly_ordered_text"
            results = [
                task_funcs[task_variant](text, kwargs.get("no_instruction_variation", False), kwargs.get("no_instruction", False))
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
            task_variant = "generate_correctly_ordered_text"
        else:
            # Randomly select a task variant
            task_variant = random.choice(task_variants)
        # Generate instruction data based on the task variant
        prompt, completion = task_funcs[task_variant](text, **kwargs)
        return {"prompt": prompt, "completion": completion}
    