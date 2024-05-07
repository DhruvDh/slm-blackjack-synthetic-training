import torch
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    RwkvConfig,
    RwkvForCausalLM,
    RecurrentGemmaConfig,
    RecurrentGemmaForCausalLM,
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
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    model = HuggingFaceModel(
        model_class(config),
        tokenizer=tokenizer,
        metrics=[LanguagePerplexity(ignore_index=-100)],
        use_logits=True,
    )
    model.to(device)
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        model.to(torch.bfloat16)
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
