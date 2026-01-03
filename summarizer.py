from transformers import BartTokenizer, BartForConditionalGeneration

def load_model():
    tokenizer = BartTokenizer.from_pretrained("trained_bart")
    model = BartForConditionalGeneration.from_pretrained("trained_bart")
    return tokenizer, model

def summarize_text(text):
    tokenizer, model = load_model()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=150,
        min_length=40,
        num_beams=4
    )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
