"""
Interactive sampling / chat interface for NovaMind-3B.
"""
import os
import sys
import argparse

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.model_config import NovaMind3BConfig
from model.transformer import NovaMind3B
from tokenizer.tokenizer import get_tokenizer


def load_model(checkpoint_path, device="cuda"):
    config = NovaMind3BConfig()
    config.mtp_depth = 0
    model = NovaMind3B(config)
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model", ckpt)
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("mtp_module")}
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        print("WARNING: No checkpoint loaded, using random weights")
    
    model = model.to(device).eval()
    return model


@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=512, temperature=0.7, top_p=0.9, device="cuda"):
    input_ids = tokenizer.encode(prompt)
    x = torch.tensor([input_ids], dtype=torch.long, device=device)
    
    output = model.generate(
        x, max_new_tokens=max_new_tokens,
        temperature=temperature, top_p=top_p,
    )
    
    new_tokens = output[0, len(input_ids):].tolist()
    return tokenizer.decode(new_tokens)


def chat_mode(model, tokenizer, args):
    """Interactive chat mode with conversation history."""
    device = args.device
    history = []
    
    print("\nNovaMind-3B Chat (type 'quit' to exit, 'clear' to reset)")
    print("-" * 50)
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        
        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "clear":
            history = []
            print("History cleared.")
            continue
        
        # Build prompt from history
        history.append({"role": "user", "content": user_input})
        
        prompt = ""
        for msg in history:
            if msg["role"] == "user":
                prompt += f"User: {msg['content']}\n"
            else:
                prompt += f"Assistant: {msg['content']}\n"
        prompt += "Assistant:"
        
        response = generate(
            model, tokenizer, prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            device=device,
        )
        
        # Clean up response
        response = response.strip()
        for stop in ["\nUser:", "\n\nUser:", "<|endoftext|>"]:
            if stop in response:
                response = response[:response.index(stop)]
        
        history.append({"role": "assistant", "content": response})
        print(f"\nAssistant: {response}")


def completion_mode(model, tokenizer, args):
    """Single-prompt completion mode."""
    device = args.device
    
    if args.prompt:
        response = generate(
            model, tokenizer, args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            device=device,
        )
        print(response)
    else:
        print("\nNovaMind-3B Completion (type 'quit' to exit)")
        print("-" * 50)
        while True:
            try:
                prompt = input("\nPrompt: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            
            if not prompt or prompt.lower() == "quit":
                break
            
            response = generate(
                model, tokenizer, prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                device=device,
            )
            print(f"\n{response}")


def main():
    parser = argparse.ArgumentParser(description="NovaMind-3B Inference")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--mode", choices=["chat", "complete"], default="chat")
    parser.add_argument("--prompt", type=str, default=None, help="Single prompt (complete mode)")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    model = load_model(args.checkpoint, device=args.device)
    tokenizer = get_tokenizer()
    
    if args.mode == "chat":
        chat_mode(model, tokenizer, args)
    else:
        completion_mode(model, tokenizer, args)


if __name__ == "__main__":
    main()
