import torch
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    RwkvConfig,
    RwkvForCausalLM,
    RecurrentGemmaConfig,
    RecurrentGemmaForCausalLM,
    MambaConfig,
    MambaForCausalLM,
    NystromformerConfig,
    NystromformerForCausalLM,
)
from composer.models import HuggingFaceModel
from composer.metrics.nlp import LanguagePerplexity
from composer.models import write_huggingface_pretrained_from_composer_checkpoint


def create_model(tokenizer, context_window, device, model_type):
    if model_type == "llama":
        config = LlamaConfig(
            vocab_size=tokenizer.vocab_size,
            num_attention_heads=2,
            num_key_value_heads=2,
            num_hidden_layers=15,
            hidden_size=512,
            intermediate_size=512,
            tie_word_embeddings=True,
            rms_norm_eps=1e-5,
            rope_theta=500000,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.get_vocab()["BeginSession"],
            eos_token_id=tokenizer.get_vocab()["EndSession"],
            max_position_embeddings=context_window,
            use_cache=True,
        )
        model_class = LlamaForCausalLM
    elif model_type == "rwkv":
        config = RwkvConfig(
            vocab_size=tokenizer.vocab_size,
            context_length=context_window,
            hidden_size=512,
            num_hidden_layers=15,
            layer_norm_epsilon=1e-5,
            use_cache=True,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        model_class = RwkvForCausalLM
    elif model_type == "recurrent_gemma":
        config = RecurrentGemmaConfig(
            vocab_size=tokenizer.vocab_size,
            num_hidden_layers=15,
            hidden_size=512,
            intermediate_size=512,
            num_attention_heads=2,
            num_key_value_heads=2,
            rms_norm_eps=1e-5,
            attention_window_size=context_window,
            use_cache=True,
            rope_theta=500000,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
        )
        model_class = RecurrentGemmaForCausalLM
    elif model_type == "mamba":
        config = MambaConfig(
            vocab_size=tokenizer.vocab_size,
            hidden_size=512,
            state_size=16,
            num_hidden_layers=15,
            layer_norm_epsilon=1e-5,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            expand=2,
            conv_kernel=4,
            use_bias=False,
            use_conv_bias=True,
            hidden_act="silu",
            initializer_range=0.1,
            residual_in_fp32=True,
            time_step_rank="auto",
            time_step_scale=1.0,
            time_step_min=0.001,
            time_step_max=0.1,
            time_step_init_scheme="random",
            time_step_floor=0.0001,
            rescale_prenorm_residual=False,
            use_cache=True,
        )
        model_class = MambaForCausalLM
    elif model_type == "nyst":
        config = NystromformerConfig(
            vocab_size=tokenizer.vocab_size,
            hidden_size=512,
            num_hidden_layers=15,
            num_attention_heads=2,
            intermediate_size=512,
            hidden_act="gelu_new",
            # hidden_dropout_prob=0.1,
            # attention_probs_dropout_prcob=0.1,
            max_position_embeddings=context_window,
            type_vocab_size=2,
            segment_means_seq_len=64,
            num_landmarks=64,
            conv_kernel_size=65,
            inv_coeff_init_option=False,
            layer_norm_eps=1e-5,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        model_class = NystromformerForCausalLM
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    model = HuggingFaceModel(
        model_class(config),
        tokenizer=tokenizer,
        metrics=[LanguagePerplexity(ignore_index=-100)],
        use_logits=True,
    )
    model.to(device)
    model.train()

    return model


def load_pretrained_model(checkpoint_path, device):
    model = LlamaForCausalLM.from_pretrained(checkpoint_path)
    model.to(device)
    model.eval()

    return model


def save_checkpoint_as_hf_model(checkpoint_path, run_name):
    new_foldername = checkpoint_path.replace(".pt", "-hf")
    write_huggingface_pretrained_from_composer_checkpoint(
        checkpoint_path,
        f"checkpoints/{run_name}/{new_foldername}",
    )
