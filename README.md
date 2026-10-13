# Manga Transcription using Qwen2.5-VL and LoRA

A manga transcription project that explores extracting dialogue, narration, internal thoughts, and vocalisations from three consecutive manga pages while preserving reading order and speaker labels.

## About the Project

This project uses a vision-language model to extract text from manga pages. The goal is to read three consecutive pages and identify dialogue, narration, internal thoughts, and vocalisations while keeping the text in the correct reading order.

Along with the transcribed text, the model predicts speaker labels so that dialogue from the same character can be grouped consistently across pages.

The project uses Qwen2.5-VL-3B-Instruct with LoRA (Low-Rank Adaptation) to adapt a pretrained model to this task.

## Why Qwen2.5-VL-3B-Instruct?

Qwen2.5-VL is a vision-language model that can process images alongside textual instructions, making it suitable for a task involving manga pages and their text.

I chose the 3-billion-parameter model as a practical starting point for experimentation with limited GPU resources, particularly through Google Colab. Rather than training a transcription system from scratch, I used LoRA to train a small set of additional parameters while keeping the pretrained model largely unchanged.

The validation results show that transcription completeness and accuracy still need improvement.

## What the Project Does

* Processes three consecutive manga pages as one sequence.
* Attempts to extract dialogue, internal thoughts, narration, and vocalisations.
* Attempts to preserve reading order and punctuation.
* Predicts speaker labels for extracted text.
* Produces predictions in a structured JSONL format.
* Evaluates predictions against labelled reference data.

## Approach

The provided labelled dataset was split into training and validation sets:

|Split|Number of sequences|
|-|-:|
|Training|64|
|Validation|16|
|Test|15|

Each sequence contains three manga pages. Training and inference experiments were carried out in Google Colab using a GPU runtime.

## Validation Results

The model was evaluated on 16 held-out validation sequences using the provided scoring script.

|Metric|Score|
|-|-:|
|Text order score|0.2048|
|Balanced joint F1|0.0865|
|Speaker accuracy on matched text|0.6021|
|Joint F1|0.1397|
|Matched token coverage|0.1919|

These results indicate that the project still has room for improvement, particularly in transcription coverage and overall text matching. Speaker accuracy is calculated only on text matched by the evaluator, so it should not be interpreted as overall transcription accuracy.

## Test Predictions

The final test prediction file is:

`outputs/test\_predictions.jsonl`

It contains predictions for 15 test sequences. The output passed structural validation, including checks for sequence IDs, three page lists per sequence, and the required JSON structure.

Correct formatting does not guarantee that every line has been transcribed accurately. Some predictions were incomplete or contained incorrect text, so further work is needed to improve the results.

## Running the Evaluation

From the project directory, run:

```bash
python score.py \\
  --references outputs/validation\_references.jsonl \\
  --predictions outputs/validation\_predictions\_final.jsonl \\
  --output outputs/validation\_metrics.json
```

This evaluates the validation predictions against the labelled references and saves the resulting metrics.

## Current Limitations

### Transcription accuracy

The model sometimes produces incorrect or garbled text, especially when the text is small or difficult to read.

### Incomplete extraction

Some dialogue or thoughts may be missed, resulting in incomplete transcriptions across the three pages.

### Speaker identification

The model may assign incorrect speaker labels or fail to maintain consistent labels across pages.

### Limited evaluation

The current evaluation uses only 16 validation sequences, so the results may not represent performance on a wider range of manga.

### Training reproducibility

The saved LoRA adapter was trained interactively in Google Colab. The standalone training script has not yet been verified through a fresh end-to-end run.

### Limited user interface

The model and prediction workflow are primarily operated through scripts and JSONL files. The live website is a separate frontend demonstration and is not connected to the model inference pipeline.

## Future Work

* Improve transcription quality: Experiment with training settings, data preparation, and inference prompts to improve text coverage and accuracy.
* Improve the UI/UX: Develop a clean, accessible interface for uploading manga pages and reviewing transcriptions.
* Add export options: Support downloading the final transcription as JSON, JSONL, or plain text.
* Integrate the backend: Connect the frontend to the model inference pipeline.
* Improve reproducibility: Create a reliable setup process so the model and application can be installed and run without manually repeating the Colab workflow.

## Proposed UI Concept \& Live Frontend Prototype

The planned MangaTranscribe interface is intended to let users upload manga pages, review transcription results, edit speaker labels, and export output.

🌐 [Explore the Live Frontend Prototype](https://mangaprototype.netlify.app)

The deployed website demonstrates the proposed user interface and workflow. It is a frontend prototype only and is not connected to the Qwen model or the AI inference backend. Model training, inference code, and prediction files are maintained separately in this repository.

!\[MangaTranscribe UI concept](docs/mangatranscribe-ui-concept.png)

## Long-Term Goal

The long-term goal is to develop MangaTranscribe into a complete manga transcription tool where users can upload manga pages, generate page-wise transcriptions, review and correct speaker labels, and download the final output through a single interface.

