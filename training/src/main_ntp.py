import torch
import torch.distributed as dist
from transformers import (AutoModelForCausalLM, AutoConfig, AutoTokenizer)
import datasets
from datasets.distributed import split_dataset_by_node
import glob
import logging
import math
import os
import json

from util import CustomNTPArgumentParser, arrow_stream_generator_multi, StopOnCheckpointCallback, CustomTrainer, SaveCheckpointAtStepCallback

torch.set_float32_matmul_precision('high')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def round_to_nearest_multiple(vocabulary_size, multiple):
    rounded_size = math.ceil(vocabulary_size / multiple) * multiple
    return rounded_size


def filter_processed_files(
    train_data_files: list[str],
    rank: int,
    world_size: int,
    checkpoint_interval: int,
    batch_size: int,
    gradient_accumulation_steps: int,
) -> list[str]:
    """
    Skip files already processed during training resumption.
    
    Args:
        train_data_files: List of training file paths.
        rank: Current process rank in distributed training.
        world_size: Total number of processes.
        checkpoint_interval: Duration of each checkpoint.
        batch_size: Per-device batch size.
        gradient_accumulation_steps: Number of gradient accumulation steps.
    
    Returns:
        Filtered list of unprocessed files.
    """
    # Calculate total samples processed across all devices
    samples_per_step = batch_size * world_size * gradient_accumulation_steps
    total_samples_processed = checkpoint_interval * samples_per_step

    processed_files = []
    total_samples = 0

    for file in train_data_files:
        # Load metadata from state.json in the same directory
        state_json_path = os.path.join(os.path.dirname(file), "state.json")
        if not os.path.exists(state_json_path):
            raise FileNotFoundError(f"State file not found at {state_json_path}")

        with open(state_json_path, "r") as f:
            state = json.load(f)
            shard_lengths = state.get("_shard_lengths", [])
            data_files = state.get("_data_files", [])

            # If shard lengths are missing, skip filtering
            if shard_lengths == []:
                if rank == 0:
                    logger.info("Missing shard lengths. Skipping file filtering.")
                return train_data_files

            # Map data file paths and find sample count
            data_file_paths = [os.path.join(os.path.dirname(file), f["filename"]) for f in data_files]
            try:
                sample_count = shard_lengths[data_file_paths.index(file)]
            except ValueError:
                continue

            total_samples += sample_count
            if total_samples <= total_samples_processed:
                processed_files.append(file)
            else:
                # The last file is highly likely to be partially processed.
                # To avoid contaminating the training process, we skip it.
                processed_files.append(file)
                break

    # Remove processed files from the list
    remaining_files = [f for f in train_data_files if f not in processed_files]
    if rank == 0:
        logger.info(f"Filtered {len(processed_files)} processed files. Remaining: {len(remaining_files)} files.")
        logger.info(f"{remaining_files[:100]}")
    return remaining_files


def main(args, training_args):
    #####
    # Set up distributed training
    #####
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if not dist.is_initialized() and world_size > 1:
        dist.init_process_group(backend='nccl')
    logger.info(f"Rank: {rank}, World size: {world_size}")
        
    #####
    # Load the dataset
    #####
    if args.use_streaming:
        # Get all files
        read_size = 131072  # 128KB
        
        # Fetch the list of original training files
        train_data_files = sorted(glob.glob(args.train_dataset_path))
        if not train_data_files:
            if rank == 0:
                logger.info(f"No files found at {args.train_dataset_path}")
            exit(1)
        if rank == 0:
            logger.info(f"Found {len(train_data_files)} files: {train_data_files[:100]}")

        # Filter out already processed files if resume_from_checkpoint is provided
        if args.resume_from_checkpoint:
            for index in range(args.num_resumptions):
                train_data_files = filter_processed_files(
                    train_data_files,
                    rank,
                    world_size,
                    args.resumption_checkpoint_interval,
                    training_args.per_device_train_batch_size,
                    training_args.gradient_accumulation_steps
                )
                # Skip the corresponding number of files if num_failures is greater than 0
                if args.num_failures is not None:
                    num_failure = args.num_failures[index]
                    if num_failure > 0:
                        train_data_files = train_data_files[(num_failure + training_args.dataloader_num_workers):]
                        if rank == 0:
                            logger.info(f"Filtered {num_failure + training_args.dataloader_num_workers} processed files. Remaining: {len(train_data_files)} files.")
                            logger.info(f"{train_data_files[:5]}")
                
            # Check if training_args.ignore_data_skip is set
            if training_args.ignore_data_skip:
                if rank == 0:
                    logger.info("Ignoring data skip. Proceeding with training.")
            else:
                training_args.ignore_data_skip = True
                if rank == 0:
                    logger.info("Setting ignore_data_skip to True. Proceeding with training.")

        # Adjust the number of files for distributed training to be a multiple of world_size
        # Note that Rank 0 will dispatch files evenly to all ranks
        if world_size > 1:
            num_files = len(train_data_files)
            if num_files % world_size != 0:
                num_files -= num_files % world_size
                train_data_files = train_data_files[:num_files]
                if rank == 0:
                    logger.info(f"Adjusted number of files for distributed training: {num_files} files.")
            
        # Create the dataset
        train_dataset = datasets.IterableDataset.from_generator(
            arrow_stream_generator_multi,
            gen_kwargs={
                "filepaths": train_data_files,
                "read_size": read_size,
                "rank": rank,
            }
        )
        dev_data_files = sorted(glob.glob(args.dev_dataset_path))
        if not dev_data_files:
            if rank == 0:
                logger.info(f"No files found at {args.dev_dataset_path}")
            exit(1)
        if rank == 0:
            logger.info(f"Found {len(dev_data_files)} files: {dev_data_files[:5]}")
        dev_dataset = datasets.IterableDataset.from_generator(
            arrow_stream_generator_multi,
            gen_kwargs={"filepaths": dev_data_files, "read_size": read_size, "rank": rank}
        )
        
    else:
        train_dataset = datasets.load_from_disk(args.train_dataset_path)
        train_dataset = train_dataset.shuffle(seed=training_args.seed)
        dev_dataset = datasets.load_from_disk(args.dev_dataset_path)
    
    #####
    # Load the tokenizer
    #####
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name_or_path,
        cache_dir=args.cache_dir
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    #####
    # Load the model
    #####
    if args.resume_from_checkpoint:
        model = AutoModelForCausalLM.from_pretrained(
            args.resume_from_checkpoint,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2"
        )
        if rank == 0:
            logger.info(f"Config: {model.config}")
    else:
        config = AutoConfig.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_dir,
        )
        if args.model_size == "100m":
            config.vocab_size = round_to_nearest_multiple(len(tokenizer), 128)
            config.bos_token_id = tokenizer.bos_token_id
            config.eos_token_id = tokenizer.eos_token_id
            config.pad_token_id = tokenizer.pad_token_id
            config.max_position_embeddings = 32768
            config.max_window_layers = 12
            config.sliding_window = 32768
            config.rope_theta = 1000000.0
            config.rms_norm_eps = 1e-06
            config.use_mrope = False
            config.use_sliding_window = False
            config.use_cache = True
            config.tie_word_embeddings = True
            config.hidden_act = "silu"
            config.initializer_range = 0.02
            config.attention_dropout = 0.0
            config.num_hidden_layers = 12
            config.num_attention_heads = 12
            config.num_key_value_heads = 2
            config.intermediate_size = 2560
            config.hidden_size = 768
        elif args.model_size == "500m":
            config.vocab_size = round_to_nearest_multiple(len(tokenizer), 128)
            config.bos_token_id = tokenizer.bos_token_id
            config.eos_token_id = tokenizer.eos_token_id
            config.pad_token_id = tokenizer.pad_token_id
            config.max_position_embeddings = 32768
            config.max_window_layers = 24
            config.sliding_window = 32768
            config.rope_theta = 1000000.0
            config.rms_norm_eps = 1e-06
            config.use_mrope = False
            config.use_sliding_window = False
            config.use_cache = True
            config.tie_word_embeddings = True
            config.hidden_act = "silu"
            config.initializer_range = 0.02
            config.attention_dropout = 0.0
            config.num_hidden_layers = 24
            config.num_attention_heads = 24
            config.num_key_value_heads = 2
            config.intermediate_size = 4864
            config.hidden_size = 1024
        elif args.model_size == "1b":
            config.vocab_size = round_to_nearest_multiple(len(tokenizer), 128)
            config.bos_token_id = tokenizer.bos_token_id
            config.eos_token_id = tokenizer.eos_token_id
            config.pad_token_id = tokenizer.pad_token_id
            config.max_position_embeddings = 32768
            config.max_window_layers = 30
            config.sliding_window = 32768
            config.rope_theta = 1000000.0
            config.rms_norm_eps = 1e-06
            config.use_mrope = False
            config.use_sliding_window = False
            config.use_cache = True
            config.tie_word_embeddings = True
            config.hidden_act = "silu"
            config.initializer_range = 0.02
            config.attention_dropout = 0.0
            config.num_hidden_layers = 30
            config.num_attention_heads = 32
            config.num_key_value_heads = 4
            config.intermediate_size = 4752
            config.hidden_size = 1728
        else:
            raise ValueError("Invalid model size.")
        if rank == 0:
            logger.info(f"Config: {config}")
        model = AutoModelForCausalLM.from_config(
            config=config, 
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2"
        )
    if rank == 0:
        logger.info(model)

    #####
    # Set up the trainer
    #####
    callbacks = []
    if args.stop_on_checkpoint:
        callbacks.append(StopOnCheckpointCallback(stop_after_n_checkpoints=args.stop_after_n_checkpoints))
    if args.save_checkpoint_at_step:
        callbacks.append(
            SaveCheckpointAtStepCallback(
                save_steps=[args.save_checkpoint_at_step],
                output_dir=training_args.output_dir,
            )
        )
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        callbacks=callbacks,
    )
        
    #####
    # Train the model
    #####
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    #####
    # Save the model
    #####
    if rank == 0:
        trainer.save_model(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)
    
    #####
    # Clean up distributed training
    #####
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = CustomNTPArgumentParser()
    args, training_args = parser.parse_args()
    if int(os.environ.get("RANK", 0)) == 0:
        logger.info(args)
        logger.info(training_args)
    main(args, training_args)
