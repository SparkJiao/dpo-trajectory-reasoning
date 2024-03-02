import torch
from transformers import PreTrainedTokenizer
from fairscale.nn.model_parallel import initialize as mpu
import torch.distributed as dist


def react_process_reward(reduction: str = "none"):
    # I think this can also be implemented inside the reward model.
    # According to the definition, the reduction shouldn't be used?
    def func(prompt_input_ids: torch.LongTensor,
             seq: torch.LongTensor,
             rm_outputs,
             tokenizer: PreTrainedTokenizer
             ):
        prompt_len = prompt_input_ids.shape[1]

        from data.general import extract_react_ending_positions_v2

        full_text = tokenizer.batch_decode(seq, skip_special_tokens=True)  # Here the `eos` token is already overlooked.

        rewards = rm_outputs["values"].new_zeros(rm_outputs["values"].size())
        values = rm_outputs["values"]
        for i in range(seq.shape[0]):
            seq_left_padding = seq[i, :prompt_len].eq(tokenizer.pad_token_id).sum().item()

            ending, _ = extract_react_ending_positions_v2(tokenizer, full_text[i], seq.shape[1])
            acc = None
            for j, e in enumerate(ending):
                assert e + seq_left_padding >= prompt_len

                if reduction == "sum":
                    if j == 0:
                        acc = values[i, e + seq_left_padding]
                    else:
                        acc += values[i, e + seq_left_padding]
                    rewards[i, e + seq_left_padding] = acc
                elif reduction == "min":
                    if j == 0:
                        acc = values[i, e + seq_left_padding]
                    else:
                        acc = min(acc, values[i, e + seq_left_padding])
                    rewards[i, e + seq_left_padding] = acc
                elif reduction == "prod":
                    if j == 0:
                        acc = values[i, e + seq_left_padding]
                    else:
                        acc *= values[i, e + seq_left_padding]
                    rewards[i, e + seq_left_padding] = acc
                elif reduction == "none":
                    rewards[i, e + seq_left_padding] = values[i, e + seq_left_padding]
                else:
                    raise ValueError(f"Unknown reduction: {reduction}")

        return rewards

    return func


def react_process_reward_eos(reduction: str = "none", model_parallel: bool = False):
    # I think this can also be implemented inside the reward model.
    # According to the definition, the reduction shouldn't be used?
    # Update `eos` version: the previous version seems the last reward is not counted on the eos token, so the generation
    # process can be encouraged to non-stop.
    def func(prompt_input_ids: torch.LongTensor,
             seq: torch.Tensor,
             rm_outputs,
             tokenizer: PreTrainedTokenizer
             ):
        prompt_len = prompt_input_ids.shape[1]

        from data.general import extract_react_ending_positions_v2

        full_text = tokenizer.batch_decode(seq, skip_special_tokens=True)  # Here the `eos` token is already overlooked.

        rewards = rm_outputs["values"].new_zeros(rm_outputs["values"].size())
        values = rm_outputs["values"]
        for i in range(seq.shape[0]):
            seq_left_padding = seq[i, :prompt_len].eq(tokenizer.pad_token_id).sum().item()
            eos_index = (seq[i].eq(tokenizer.eos_token_id)).nonzero().squeeze(-1)

            ending, _ = extract_react_ending_positions_v2(tokenizer, full_text[i], seq.shape[1])
            acc = None
            for j, e in enumerate(ending):
                assert e + seq_left_padding >= prompt_len

                if j == len(ending) - 1 and len(eos_index):
                    e = eos_index[0].item()
                    seq_left_padding = 0

                if reduction == "sum":
                    if j == 0:
                        acc = values[i, e + seq_left_padding]
                    else:
                        acc += values[i, e + seq_left_padding]
                    rewards[i, e + seq_left_padding] = acc
                elif reduction == "min":
                    if j == 0:
                        acc = values[i, e + seq_left_padding]
                    else:
                        acc = min(acc, values[i, e + seq_left_padding])
                    rewards[i, e + seq_left_padding] = acc
                elif reduction == "prod":
                    if j == 0:
                        acc = values[i, e + seq_left_padding]
                    else:
                        acc *= values[i, e + seq_left_padding]
                    rewards[i, e + seq_left_padding] = acc
                elif reduction == "none":
                    rewards[i, e + seq_left_padding] = values[i, e + seq_left_padding]
                else:
                    raise ValueError(f"Unknown reduction: {reduction}")

        return rewards

    def parallel_func(prompt_input_ids: torch.LongTensor,
                      seq: torch.LongTensor,
                      rm_outputs,
                      tokenizer: PreTrainedTokenizer
                      ):
        mp_size = mpu.get_model_parallel_world_size()
        mp_rank = mpu.get_model_parallel_rank()

        # Split tensors into different rank
        seq_list = list(torch.chunk(seq, mp_size, dim=0))
        _seq = seq_list[mp_rank]
        # rm_outputs["values"] = list(torch.chunk(rm_outputs["values"], mp_size, dim=0))[mp_rank]
        values = list(torch.chunk(rm_outputs["values"], mp_size, dim=0))
        rm_outputs["values"] = values[mp_rank]

        # Run the function on each rank
        _rewards = func(prompt_input_ids, _seq, rm_outputs, tokenizer)

        # Merge the results
        tensor_list = [torch.zeros_like(values[i]) for i in range(mp_size)]
        dist.all_gather(tensor_list, _rewards)
        return torch.cat(tensor_list, dim=0)

    if model_parallel:
        return parallel_func

    return func
