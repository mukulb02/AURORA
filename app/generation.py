import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)

from app.config import GEN_MODEL, device

tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL)

qa_model = (
    AutoModelForSeq2SeqLM
    .from_pretrained(GEN_MODEL)
    .to(device)
)


def truncate_context(text: str, max_words: int = 300):

    words = text.split()

    if len(words) > max_words:
        return " ".join(words[:max_words])

    return text


def hf_generate(context: str, query: str):

    prompt = (
        f"Context: {context}\n\n"
        f"Question: {query}\nAnswer:"
    )

    input_ids = tokenizer.encode(
        prompt,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():

        output = qa_model.generate(
            input_ids,
            max_new_tokens=300,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id
        )

    return tokenizer.decode(
        output[0],
        skip_special_tokens=True
    ).strip()