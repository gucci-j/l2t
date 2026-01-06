import random
import re
from collections import Counter


def _generate_char_count(
    text: str,
    no_instruction_variation: bool = False,
    no_instruction: bool = False
) -> tuple:
    """
    Generate prompt and completion for char count task.

    Args:
        text (str): Input text.
        no_instruction_variation (bool): If True, do not use instruction variation.
        no_instruction (bool): If True, do not use instruction.

    Returns:
        tuple: Prompt and completion.

    Example:
        >>> _generate_char_count(["This is a test."])
        >>> ("Count the number of characters.", "15")
    """
    instruction_variants = (
        "Count the number of characters.",
        "How many characters are there in the given text?",
        "Calculate the total number of characters.",
        "Find the total number of characters.",
        "Determine the character count of this text.",
        "What is the total character count?",
        "Count all characters in the text.",
        "How many characters does this text contain?",
        "Provide the character count for this text.",
        "Tell me the number of characters.",
        "Count the total characters in this passage.",
        "Give me the character count.",
        "How many characters are in this text?",
        "Calculate the character count.",
        "What's the character count of this text?",
        "Report the total number of characters.",
        "Count characters in the following text.",
        "Determine how many characters are present.",
        "Count the letters and symbols in this text.",
        "Find out how many characters this text has.",
    )
    if no_instruction_variation:
        instruction = "Count the number of characters."
    else:
        instruction = random.choice(instruction_variants)
    completion = str(len(text))
    # 50% chance to swap the instruction and context
    if random.random() < 0.5:
        instruction, text = text, instruction
    prompt = f"{instruction}\n\n{text}"
    return prompt, completion
    

def _generate_char_count_by_dict(
    text: str,
    no_instruction_variation: bool = False,
    no_instruction: bool = False
) -> tuple:
    """
    Generate prompt and completion for char count task using dictionary.

    Args:
        text (str): Input text.
        no_instruction_variation (bool): If True, do not use instruction variation.
        no_instruction (bool): If True, do not use instruction.

    Returns:
        tuple: Prompt and completion.

    Example:
        >>> _generate_char_count_by_dict(["This is a test."])
        >>> ("Create a dictionary of characters and their counts.", "'a': 1, 'e': 1, 'h': 1, 'i': 2, 's': 3, 't': 3, 'T': 1, ' ': 3, '.': 1")
    """
    instruction_variants = (
        "Generate a dictionary of characters and their counts.",
        "Create a dictionary of characters and their counts."
        "List down the characters and their counts.",
        "Find the characters and their counts.",
    )
    if no_instruction_variation:
        instruction = "Create a dictionary of characters and their counts."
    else:
        instruction = random.choice(instruction_variants)
    # Use Counter for efficient character counting
    char_dict = Counter(text)
    completion = ", ".join([f"'{char}': {count}" for char, count in char_dict.items()])
    if no_instruction:
        return text, completion
    # 50% chance to swap the instruction and context
    if random.random() < 0.5:
        instruction, text = text, instruction
    prompt = f"{instruction}\n\n{text}"
    return prompt, completion


def _generate_chat_count_by_sequence(
    text: str,
    no_instruction_variation: bool = False,
    no_instruction: bool = False
) -> tuple:
    """
    Generate prompt and completion for char count task using sequence.

    Args:
        text (str): Input text.
        no_instruction_variation (bool): If True, do not use instruction variation.
        no_instruction (bool): If True, do not use instruction.

    Returns:
        tuple: Prompt and completion.

    Example:
        >>> _generate_chat_count_by_sequence(["This is a test."])
        >>> ("Generate a sequence of character counts.", "4 2 1 5")
    """
    instruction_variants = (
        "Generate a sequence of character counts.",
        "List down the character counts.",
        "Find the character counts.",
        "Create a sequence of character counts.",
    )
    if no_instruction_variation:
        instruction = "Generate a sequence of character counts."
    else:
        instruction = random.choice(instruction_variants)
    # Split text by spaces and count characters in each part
    words = text.split()
    char_counts = [str(len(word)) for word in words]
    completion = " ".join(char_counts)
    if no_instruction:
        return text, completion
    # 50% chance to swap the instruction and context
    if random.random() < 0.5:
        instruction, text = text, instruction
    prompt = f"{instruction}\n\n{text}"
    return prompt, completion


def generate_char_count_instruction_data(
    example: dict,
    batched: bool = False,
    min_num_words: int = 5,
    use_only_primary: bool = False,
    **kwargs
) -> dict:
    """
    Generate instruction data for char count task.
    """
    task_funcs = {
        "generate_char_count": _generate_char_count,
        "generate_char_count_by_dict": _generate_char_count_by_dict,
        "generate_chat_count_by_sequence": _generate_chat_count_by_sequence,
    }
    task_variants = list(task_funcs.keys())
    if batched:
        if use_only_primary:
            task_variant = "generate_char_count"
            results = [
                task_funcs[task_variant](text, kwargs.get("no_instruction_variation", False), kwargs.get("no_instruction", False))
                for text in example["text"]
                if len(text.split()) >= min_num_words
            ]
        else:
            results = [
                task_funcs[random.choice(task_variants)](text, kwargs.get("no_instruction_variation", False), kwargs.get("no_instruction", False))
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
            task_variant = "generate_char_count"
        else:
            task_variant = random.choice(task_variants)
        prompt, completion = task_funcs[task_variant](example["text"], kwargs.get("no_instruction_variation", False), kwargs.get("no_instruction", False))
        return {"prompt": prompt, "completion": completion}