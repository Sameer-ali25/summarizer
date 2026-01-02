from transformers import BartTokenizer, BartForConditionalGeneration

def load_model():
    model_name = "facebook/bart-large-cnn"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

def summarize_text(text, max_len=150, min_len=40):
    tokenizer, model = load_model()

    inputs = tokenizer(
        text,
        max_length=1024,
        return_tensors="pt",
        truncation=True
    )

    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=max_len,
        min_length=min_len,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
