import random
import re


def _generate_list_of_corrected_words(
    text: str
) -> tuple:
    """
    Generate prompt-completion pair for the task of generating a list of corrected words.

    This implementation introduces typos at the word-level. It randomly selects a few words,
    changes one character within each selected word to simulate a typo, and returns a list of
    corrected words.

    Args:
        text (str): The input text to be manipulated.

    Returns:
        tuple: A tuple containing the prompt and completion strings.

    Example:
        >>> text = "The cat sat on the mat"
        >>> result = _generate_list_of_corrected_words(text)
        >>> print(result)
        ('Review the text below for typographical errors and provide a list of the corrected words: \n\nThe cAt sat on the mat', "['cat']")
    """
    instruction_variants = (
        "Please identify and list the correct spellings of any misspelled words in this text:",
        "What are the proper spellings of the incorrectly typed words in the following text?",
        "Could you help me identify and list the correct versions of any misspelled words below?",
        "Please find the typos in this text and provide a list of their correct spellings:",
        "I need help identifying spelling mistakes - what are the correct forms of any misspelled words?",
        "Can you spot any typos and list how these words should be correctly spelled?",
        "Please review this text and list the proper spellings of any words with typographical errors:",
        "What would be the correct spellings of the words that contain typing mistakes in this text?",
        "Kindly examine this text for spelling errors and list the proper forms of misspelled words:",
        "Could you point out any typing errors by listing the correct spellings of the affected words?",
    )
    instruction = random.choice(instruction_variants)
    # Find word matches using regex to capture word boundaries.
    words = list(re.finditer(r'\b\w+\b', text))
    if not words:
        return f"{instruction} \n\n{text}", "[]"
    num_typos = max(1, round(len(words) * random.uniform(0.01, 0.08)))
    typo_indices = random.sample(range(len(words)), num_typos)
    typo_text = list(text)
    corrected_words = []
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    for idx in typo_indices:
        match = words[idx]
        start, end = match.span()
        original_word = text[start:end]
        # Introduce a typo by changing one character within the word.
        if original_word:
            char_idx = random.randrange(len(original_word))
            original_char = original_word[char_idx]
            typo_char = random.choice(alphabet)
            if original_char.isupper():
                typo_char = typo_char.upper()
            typo_word = original_word[:char_idx] + typo_char + original_word[char_idx+1:]
            corrected_words.append(original_word)
            typo_text[start:end] = list(typo_word)
    prompt = f"{instruction}\n\n{''.join(typo_text)}"
    return prompt, str(corrected_words)


def _generate_list_of_typos(
    text: str
) -> tuple:
    """
    Generate prompt-completion pair for the task of generating a list of typos.

    This implementation introduces typos at the word-level. It randomly selects a few words,
    changes one character within each selected word to simulate a typo, and returns a list of
    typo words.

    Args:
        text (str): The input text to be manipulated.

    Returns:
        tuple: A tuple containing the prompt and completion strings.

    Example:
        >>> text = "The cat sat on the mat"
        >>> result = _generate_list_of_typos(text)
        >>> print(result)
        ('Identify the typos in the text below and provide a list of the words with typos: \n\nThe cAt sat on the mat', "['cAt']")
    """
    instruction_variants = (
        "Here's a text with some typos - could you list out the misspelled words?",
        "I need help finding the typos - which words are spelled incorrectly in this text?",
        "Please identify all the words containing typing mistakes in this passage:",
        "Can you make a list of the misspelled words you spot in this text?",
        "Review this text and point out which words have typing errors:",
        "What words in this text contain typographical errors?",
        "Please list all the words that have spelling mistakes in the following text:",
        "Could you identify and list the words with typos in this passage?",
        "Which words in this text are not spelled correctly?",
        "Help me find all the typos - what words are misspelled in this text?",
    )
    instruction = random.choice(instruction_variants)
    # Find word matches using regex to capture word boundaries.
    words = list(re.finditer(r'\b\w+\b', text))
    if not words:
        return f"{instruction} \n\n{text}", "[]"
    num_typos = max(1, round(len(words) * random.uniform(0.01, 0.08)))
    typo_indices = random.sample(range(len(words)), num_typos)
    typo_text = list(text)
    words_with_typos = []
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    for idx in typo_indices:
        match = words[idx]
        start, end = match.span()
        original_word = text[start:end]
        # Introduce a typo by changing one character within the word.
        if original_word:
            char_idx = random.randrange(len(original_word))
            original_char = original_word[char_idx]
            typo_char = random.choice(alphabet)
            if original_char.isupper():
                typo_char = typo_char.upper()
            typo_word = original_word
            if typo_char != original_char:
                typo_word = original_word[:char_idx] + typo_char + original_word[char_idx+1:]
                words_with_typos.append(typo_word)
            typo_text[start:end] = list(typo_word)
    prompt = f"{instruction}\n\n{''.join(typo_text)}"
    return prompt, str(words_with_typos)


def _predict_number_of_typos(
    text: str
) -> tuple:
    """
    Generate prompt-completion pair for the task of predicting the number of typos.

    This implementation introduces typos at the word-level. It randomly selects a few words,
    changes one character within each selected word to simulate a typo, and returns the number
    of typos.

    Args:
        text (str): The input text to be manipulated.

    Returns:
        tuple: A tuple containing the prompt and completion strings.

    Example:
        >>> text = "The cat sat on the mat"
        >>> result = _predict_number_of_typos(text)
        >>> print(result)
        ('How many typographical errors can you find in the text below? \n\nThe cAt sat on the mat', '1')
    """
    instruction_variants = (
        "Can you count how many spelling mistakes appear in this text?",
        "Please tell me the total number of typos in the following passage:",
        "How many words are misspelled in the text below?",
        "Count the number of typing errors in this text:",
        "What's the total count of typographical errors in this passage?",
        "How many spelling errors can you spot in the following text?",
        "Could you determine the number of typos present in this text?",
        "Please count all typing mistakes in the passage below:",
        "Tell me how many words are incorrectly typed in this text:",
        "What is the exact number of spelling mistakes in the following passage?",
    )
    instruction = random.choice(instruction_variants)
    typo_text = list(text)
    # Only consider indices of alphanumeric characters
    valid_indices = [i for i in range(len(text)) if text[i].isalnum()]
    if not valid_indices:
        return f"{instruction}\n\n{text}", "0"
    num_typos = max(1, round(len(valid_indices) * random.uniform(0.03, 0.08)))
    indices = random.sample(valid_indices, num_typos)
    alphabet = [c for c in "abcdefghijklmnopqrstuvwxyz"]
    for idx in indices:
        typo_char = random.choice(alphabet)
        typo_text[idx] = typo_char.upper() if text[idx].isupper() else typo_char.lower()
    prompt = f"{instruction}\n\n{''.join(typo_text)}"
    return prompt, str(num_typos)


def _generate_list_of_corrected_words_with_hint(
    text: str
) -> tuple:
    """
    Generate prompt-completion pair for the task of generating a list of corrected words with hint.

    This implementation introduces typos at the word-level. It randomly selects a few words,
    changes one character within each selected word to simulate a typo, and returns a list of
    corrected words along with a hint.

    Args:
        text (str): The input text to be manipulated.

    Returns:
        tuple: A tuple containing the prompt and completion strings.

    Example:
        >>> text = "The cat sat on the mat"
        >>> result = _generate_list_of_corrected_words_with_hint(text)
        >>> print(result)
        ('Review the text below for typographical errors and provide a list of the corrected words: \n\nThe cAt sat on the mat\n\nHint: There is/are 1 word(s) with a typo in the text.', "['cat']")
    """
    instruction_variants = (
        "Could you review this text and list the correct versions of any misspelled words?",
        "Please identify any typing mistakes and show me their proper spellings.",
        "What would be the correct spellings of the words that contain typos below?",
        "Can you spot the misspelled words and provide their correct forms?",
        "Help me fix these typos by listing the proper spellings of incorrect words.",
        "I need the correct spellings for any words with typing errors in this text.",
        "Please find and list the proper spellings of any mistyped words.",
        "Could you show me the correct forms of words that have typing mistakes?",
        "What are the proper spellings for the words that contain typographical errors?",
        "Please examine this text and list how the misspelled words should be written.",
    )
    instruction = random.choice(instruction_variants)
    # Find word matches using regex to capture word boundaries.
    words = list(re.finditer(r'\b\w+\b', text))
    if not words:
        return f"{instruction}\n\n{text}", "[]"
    num_typos = max(1, round(len(words) * random.uniform(0.01, 0.08)))
    typo_indices = random.sample(range(len(words)), num_typos)
    typo_text = list(text)
    corrected_words = []
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    for idx in typo_indices:
        match = words[idx]
        start, end = match.span()
        original_word = text[start:end]
        # Introduce a typo by changing one character within the word.
        if original_word:
            char_idx = random.randrange(len(original_word))
            original_char = original_word[char_idx]
            typo_char = random.choice(alphabet)
            if original_char.isupper():
                typo_char = typo_char.upper()
            typo_word = original_word[:char_idx] + typo_char + original_word[char_idx+1:]
            corrected_words.append(original_word)
            typo_text[start:end] = list(typo_word)
    typo_text_str = ''.join(typo_text)
    if corrected_words !=[]:
        hint = f"\n\nHint: There {'is' if num_typos == 1 else 'are'} {num_typos} word{'s' if num_typos != 1 else ''} with a typo in the text."
    else:
        return f"{instruction}\n\n{text}", "[]"
    prompt = f"{instruction}\n\n{typo_text_str}{hint}"
    return prompt, str(corrected_words)


def _generate_list_of_typos_with_hint(
    text: str
) -> tuple:
    """
    Generate prompt-completion pair for the task of generating a list of typos with hint.

    This implementation introduces typos at the word-level. It randomly selects a few words,
    changes one character within each selected word to simulate a typo, and returns a list of
    typo words along with a hint.

    Args:
        text (str): The input text to be manipulated.

    Returns:
        tuple: A tuple containing the prompt and completion strings.

    Example:
        >>> text = "The cat sat on the mat"
        >>> result = _generate_list_of_typos_with_hint(text)
        >>> print(result)
        ('Identify the typos in the text below and provide a list of the words with typos: \n\nThe cAt sat on the mat\n\nHint: There is/are 1 word(s) with a typo in the text.', "['cAt']")
    """
    instruction_variants = (
        "Which words appear to be misspelled in the following text?",
        "Could you list all the words that contain typing errors in this passage?",
        "Please identify and list any words that are spelled incorrectly below:",
        "Make a list of all words that have typographical errors in this text:",
        "What words in this passage contain spelling mistakes?",
        "Can you spot and list the misspelled words in the following text?",
        "Please show me which words have typing errors in this passage:",
        "Examine the text and list any words that aren't spelled correctly:",
        "Find and list all words containing typographical errors below:",
        "Which words need correction in the following text?"
    )
    instruction = random.choice(instruction_variants)
    # Find word matches using regex to capture word boundaries.
    words = list(re.finditer(r'\b\w+\b', text))
    if not words:
        return f"{instruction}\n\n{text}", "[]"
    num_typos = max(1, round(len(words) * random.uniform(0.01, 0.08)))
    typo_indices = random.sample(range(len(words)), num_typos)
    typo_text = list(text)
    words_with_typos = []
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    for idx in typo_indices:
        match = words[idx]
        start, end = match.span()
        original_word = text[start:end]
        # Introduce a typo by changing one character within the word.
        if original_word:
            char_idx = random.randrange(len(original_word))
            original_char = original_word[char_idx]
            typo_char = random.choice(alphabet)
            if original_char.isupper():
                typo_char = typo_char.upper()
            typo_word = original_word
            if typo_char != original_char:
                typo_word = original_word[:char_idx] + typo_char + original_word[char_idx+1:]
                words_with_typos.append(typo_word)
            typo_text[start:end] = list(typo_word)
    typo_text_str = ''.join(typo_text)
    if words_with_typos !=[]:
        hint = f"\n\nHint: There {'is' if num_typos == 1 else 'are'} {num_typos} word{'s' if num_typos != 1 else ''} with a typo in the text."
    else:
        return f"{instruction}\n\n{text}", "[]"
    prompt = f"{instruction}\n\n{typo_text_str}{hint}"
    return prompt, str(words_with_typos)


def _generate_corrected_text(
    text: str,
    no_instruction_variation: bool = False,
    no_instruction: bool = False
) -> tuple:
    """
    Generate prompt-completion pair for the task of generating corrected text.

    This implementation introduces typos at the word-level. It randomly selects a few words,
    changes one character within each selected word to simulate a typo, and returns the corrected text.

    Args:
        text (str): The input text to be manipulated.
        no_instruction_variation (bool): A flag to disable random selection of instruction variants.
        no_instruction (bool): A flag to disable the instruction.

    Returns:
        tuple: A tuple containing the prompt and completion strings.

    Example:
        >>> text = "The cat sat on the mat"
        >>> result = _generate_corrected_text(text)
        >>> print(result)
        ('Correct the typographical errors in the text below: \n\nThe cAt sat on the mat', 'The cat sat on the mat')
    """
    instruction_variants = (
        "Fix the typos in this text.",
        "Please correct any spelling errors in this text.",
        "Proofread and fix any typos in the text.",
        "Correct all the typing mistakes in this passage.",
        "Fix all spelling mistakes in the following text.",
        "Edit this text to remove all typographical errors.",
        "Correct the misspelled words in this text.",
        "Please identify and fix any typos below.",
        "Revise this text to correct all typing errors.",
        "Find and correct all the spelling mistakes.",
        "Rewrite this text without any typos.",
        "Eliminate all spelling errors in this passage.",
        "Check and fix the typos in this text.",
        "Clean up the spelling mistakes below.",
        "Please correct the misspelled words.",
        "Fix the typing errors in this passage.",
        "Repair any spelling mistakes in this text.",
        "Remove all typos from the following text.",
        "Make spelling corrections to this text.",
        "Fix any typing mistakes you find.",
    )
    if no_instruction_variation:
        instruction = "Fix the typos in this text."
    else:
        instruction = random.choice(instruction_variants)
    typo_text = list(text)
    # Only consider indices of alphanumeric characters
    valid_indices = [i for i in range(len(text)) if text[i].isalnum()]
    if not valid_indices:
        # 50% chance to swap the instruction and context
        if random.random() < 0.5:
            return f"{instruction}\n\n{text}", text
        else:
            return f"{text}\n\n{instruction}", text
    num_typos = max(1, round(len(valid_indices) * random.uniform(0.03, 0.08)))
    indices = random.sample(valid_indices, num_typos)
    alphabet = [c for c in "abcdefghijklmnopqrstuvwxyz"]
    for idx in indices:
        typo_char = random.choice(alphabet)
        typo_text[idx] = typo_char.upper() if text[idx].isupper() else typo_char.lower()
    typo_text = ''.join(typo_text)
    if no_instruction:
        return typo_text, text
    # 50% chance to swap the instruction and context
    if random.random() < 0.5:
        instruction, typo_text = typo_text, instruction
    prompt = f"{instruction}\n\n{typo_text}"
    return prompt, text


def _generate_corrected_text_with_hint(
    text: str
) -> tuple:
    """
    Generate prompt-completion pair for the task of generating corrected text with hint.

    This implementation introduces typos at the word-level. It randomly selects a few words,
    changes one character within each selected word to simulate a typo, and returns the corrected text
    along with a hint.

    Args:
        text (str): The input text to be manipulated.

    Returns:
        tuple: A tuple containing the prompt and completion strings.

    Example:
        >>> text = "The cat sat on the mat"
        >>> result = _generate_corrected_text_with_hint(text)
        >>> print(result)
        ('Correct the typographical errors in the text below: \n\nThe cAt sat on the mat\n\nHint: There is/are 1 typo(s) in the text.', 'The cat sat on the mat')
    """
    instruction_variants = (
        "Could you please fix any spelling mistakes in this passage?",
        "Would you mind correcting the typos in the following text?",
        "This text contains some typing errors - could you correct them?",
        "I need help fixing the spelling errors in this passage:",
        "Please proofread and correct any typing mistakes below:",
        "Can you help me identify and fix the typos in this text?",
        "I'd appreciate your help correcting any misspellings here:",
        "Please review and fix any typographical errors in this passage:",
        "Could you clean up the spelling mistakes in the text below?",
        "Would you please correct any typing errors you find here?"
    )
    instruction = random.choice(instruction_variants)
    typo_text = list(text)
    # Only consider indices of alphanumeric characters
    valid_indices = [i for i in range(len(text)) if text[i].isalnum()]
    if not valid_indices:
        return f"{instruction}\n\n{text}", "0"
    num_typos = max(1, round(len(valid_indices) * random.uniform(0.03, 0.08)))
    indices = random.sample(valid_indices, num_typos)
    alphabet = [c for c in "abcdefghijklmnopqrstuvwxyz"]
    for idx in indices:
        typo_char = random.choice(alphabet)
        typo_text[idx] = typo_char.upper() if text[idx].isupper() else typo_char.lower()
    hint = f"\n\nHint: There {'is' if num_typos == 1 else 'are'} {num_typos} typo{'s' if num_typos != 1 else ''} in the text."
    prompt = f"{instruction}\n\n{''.join(typo_text)}{hint}"
    return prompt, text


def _generate_tuples(
    text: str
) -> tuple:
    """
    Generate prompt-completion pair for the task of generating tuples.

    This implementation introduces typos at the word-level. It randomly selects a few words,
    changes one character within each selected word to simulate a typo, and returns a list of
    tuples containing the original and corrected words.

    Args:
        text (str): The input text to be manipulated.

    Returns:
        tuple: A tuple containing the prompt and completion strings.

    Example:
        >>> text = "The cat sat on the mat"
        >>> result = _generate_tuples(text)
        >>> print(result)
        ('Identify the typos in the text below and provide a list of tuples containing the original and corrected words: \n\nThe cAt sat on the mat', "[('cAt', 'cat')]")
    """
    instruction_variants = (
        "Please identify any typos and list them as pairs of (incorrect, correct) words:",
        "For each typo in the text, provide the incorrect and correct spellings as tuples:",
        "Show me the misspelled words and their correct forms as (typo, correct) pairs:",
        "Create a list of tuples showing both the typos and their proper spellings:",
        "Please list all typing errors as (misspelled, correct) word pairs:",
        "Match each typo with its correct spelling in a list of tuples:",
        "Generate pairs of (incorrect, correct) spellings for all typos found:",
        "Present the typing mistakes and their corrections as (wrong, right) pairs:",
        "List any misspellings alongside their correct forms as tuples:",
        "Format the typos and their corrections as pairs of (error, fixed) words:",
    )
    instruction = random.choice(instruction_variants)
    # Find word matches using regex to capture word boundaries.
    words = list(re.finditer(r'\b\w+\b', text))
    if not words:
        return f"{instruction}\n\n{text}", "[]"
    num_typos = max(1, round(len(words) * random.uniform(0.01, 0.08)))
    typo_indices = random.sample(range(len(words)), num_typos)
    typo_text = list(text)
    typo_tuples = []
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    for idx in typo_indices:
        match = words[idx]
        start, end = match.span()
        original_word = text[start:end]
        # Introduce a typo by changing one character within the word.
        if original_word:
            char_idx = random.randrange(len(original_word))
            original_char = original_word[char_idx]
            typo_char = random.choice(alphabet)
            if original_char.isupper():
                typo_char = typo_char.upper()
            typo_word = original_word[:char_idx] + typo_char + original_word[char_idx+1:]
            typo_tuples.append((typo_word, original_word))
            typo_text[start:end] = list(typo_word)
    prompt = f"{instruction}\n\n{''.join(typo_text)}"
    return prompt, str(typo_tuples)


def _generate_tuples_with_hint(
    text: str
) -> tuple:
    """
    Generate prompt-completion pair for the task of generating tuples with hint.

    This implementation introduces typos at the word-level. It randomly selects a few words,
    changes one character within each selected word to simulate a typo, and returns a list of
    tuples containing the original and corrected words along with a hint.

    Args:
        text (str): The input text to be manipulated.

    Returns:
        tuple: A tuple containing the prompt and completion strings.

    Example:
        >>> text = "The cat sat on the mat"
        >>> result = _generate_tuples_with_hint(text)
        >>> print(result)
        ('Identify the typos in the text below and provide a list of tuples containing the original and corrected words: \n\nThe cAt sat on the mat\n\nHint: There is/are 1 tuple(s) with a typo in the text.', "[('cAt', 'cat')]")
    """
    instruction_variants = (
        "For each typo found, create a tuple with the misspelled and correct version:",
        "List all typos as (incorrect, correct) word pairs in the following text:",
        "Please show the before and after of each typo as (original, fixed) pairs:",
        "Create a list of (misspelled, correct) tuples for each error below:",
        "Match each typo with its correct spelling using (wrong, right) pairs:",
        "Identify spelling mistakes and show them as (error, correction) tuples:",
        "Present each typo and its fix as (incorrect, proper) word pairs:",
        "Format the typing errors as (mistake, correction) pairs from this text:",
        "Show me the typos and their corrections as (before, after) tuples:",
        "Make a list of (typo, correct) pairs for each spelling error found:",
    )
    instruction = random.choice(instruction_variants)
    # Find word matches using regex to capture word boundaries.
    words = list(re.finditer(r'\b\w+\b', text))
    if not words:
        return f"{instruction}\n\n{text}", "[]"
    num_typos = max(1, round(len(words) * random.uniform(0.01, 0.08)))
    typo_indices = random.sample(range(len(words)), num_typos)
    typo_text = list(text)
    typo_tuples = []
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    for idx in typo_indices:
        match = words[idx]
        start, end = match.span()
        original_word = text[start:end]
        # Introduce a typo by changing one character within the word.
        if original_word:
            char_idx = random.randrange(len(original_word))
            original_char = original_word[char_idx]
            typo_char = random.choice(alphabet)
            if original_char.isupper():
                typo_char = typo_char.upper()
            typo_word = original_word[:char_idx] + typo_char + original_word[char_idx+1:]
            typo_tuples.append((typo_word, original_word))
            typo_text[start:end] = list(typo_word)
    typo_text_str = ''.join(typo_text)
    if typo_tuples !=[]:
        hint = f"\n\nHint: There {'is' if num_typos == 1 else 'are'} {num_typos} tuple{'s' if num_typos != 1 else ''} with a typo in the text."
    else:
        return f"{instruction}\n\n{text}", "[]"
    prompt = f"{instruction}\n\n{typo_text_str}{hint}"
    return prompt, str(typo_tuples)


def generate_typo_correction_instruction_data(
    example: dict,
    batched: bool = False,
    min_num_words: int = 5,
    use_only_primary: bool = False,
    **kwargs
) -> dict:
    """
    Generate instruction data for typo correction task.
    """
    task_funcs = {
        "generate_list_of_corrected_words": _generate_list_of_corrected_words,
        "generate_list_of_typos": _generate_list_of_typos,
        "predict_number_of_typos": _predict_number_of_typos,
        "generate_list_of_corrected_words_with_hint": _generate_list_of_corrected_words_with_hint,
        "generate_list_of_typos_with_hint": _generate_list_of_typos_with_hint,
        "generate_corrected_text": _generate_corrected_text,
        "generate_corrected_text_with_hint": _generate_corrected_text_with_hint,
        "generate_tuples": _generate_tuples,
        "generate_tuples_with_hint": _generate_tuples_with_hint,
    }
    task_variants = list(task_funcs.keys())
    if batched:
        if use_only_primary:
            task_variant = "generate_corrected_text"
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
            task_variant = "generate_corrected_text"
        else:
            task_variant = random.choice(task_variants)
        prompt, completion = task_funcs[task_variant](example["text"])
        return {"prompt": prompt, "completion": completion}