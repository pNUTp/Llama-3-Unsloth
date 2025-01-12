# Language Model with Inference and Speech Synthesis

## Overview
This project uses a pretrained language model (LLama) and applies fine-tuning for specific tasks. It integrates a text-to-speech system to provide vocal responses to generated text. The model is fine-tuned using the Alpaca dataset and can generate responses based on user input. It also includes GPU memory management, automatic training, and inference pipelines.

## Features
- **Text Generation**: Uses a pretrained language model (LLama) fine-tuned with the Alpaca dataset to generate responses based on input instructions.
- **Text-to-Speech**: Converts generated text responses into speech using Google's Text-to-Speech (gTTS) API and plays it back automatically.
- **Fine-Tuning**: Fine-tunes the model using LoRA (Low-Rank Adaptation) to optimize for specific tasks.
- **Integration with Weights & Biases**: Tracks training metrics and visualizes results with Weights & Biases.
- **Automatic GPU Memory Management**: Optimizes GPU memory usage to handle large models effectively.

## Requirements
- Python 3.8+
- PyTorch
- TensorFlow
- transformers
- datasets
- gTTS (Google Text-to-Speech)
- SpeechRecognition
- wandb

### Installation
Install dependencies using the following commands:
```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps "xformers<0.0.26" trl peft accelerate bitsandbytes
pip install xformers[cuda] -U
pip install wandb
pip install gtts
pip install SpeechRecognition


## Setup
1. Clone the repository and install the dependencies as mentioned above.
2. Configure your `wandb_config.json` with your WandB API key to track experiments.
3. Load and fine-tune the model using the Alpaca dataset. The training process can be customized by adjusting hyperparameters in the `TrainingArguments`.

## Training
The model is fine-tuned using the Alpaca dataset, which contains task-specific instructions and responses. It uses the `SFTTrainer` class from the `trl` library for supervised fine-tuning. The trainer is set up with GPU memory optimizations and efficient batch processing.

```python
trainer_args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=5,
    max_steps=60,  # Overriding Epochs for large execution time
    learning_rate=2e-4,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=1,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    output_dir="outputs",
    report_to="wandb"
)
```

## Inference
Use the following code to generate responses based on user instructions:

```python
while True:
    instruction = input("Enter your question (or 'quit' to exit): ")
    if instruction.lower() == 'quit':
        break
    generated_output = generate_language_model_output(instruction, model, tokenizer, alpaca_prompt)
    response = generated_output[-1].split("### Response:\n")[-1].strip()
    print(f"Question: {instruction}")
    print(f"Answer: {response}")
```

## Text-to-Speech
After generating the response, the text is converted to speech and played back automatically. This is done using the `gTTS` library, which saves the speech as an MP3 file and plays it using IPython's `Audio` display.

```python
def text_to_speech(text, filename="response.mp3"):
    tts = gTTS(text)
    tts.save(filename)
    return Audio(filename, autoplay=True)
```

## GPU Memory Management
The model uses GPU memory efficiently, and you can view the current memory usage during training using the following code:

```python
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")
```

## Results
You can track the training process in WandB, visualize the results, and monitor model performance.

## Future Work
- Improve performance by experimenting with different architectures and hyperparameters.
- Extend the pipeline to support multi-turn conversations or integrate with external systems.
- Further optimize memory usage for handling larger models.

## License
This project is licensed under the MIT License.
```

This README gives a detailed overview of the project's setup, features, and how to use it, making it easy for others to understand and replicate the process.
