# CODE GENERATED WHILE READING HANDS-ON GENAI, LLMS AND AIAGENTS BOOK BY AMAN KHARWAL

A collection of standalone scripts exploring different AI/ML techniques: local LLM agents, image generation, embeddings, RAG, and Hugging Face pipelines.

## Files

- **main.py** — CrewAI multi-agent crew (researcher + writer) that uses a local Ollama (`llama3`) model to research a topic and write a blog post.
- **app_streamlit.py** — Streamlit UI wrapping `main.py`'s crew; lets a user type a topic and view the generated blog post.
- **MCP-main.py** — Minimal MCP (Model Context Protocol) server example exposing a `read_company_report` tool, consumed by a simulated researcher/editor agent pair using Ollama (`mistral`).
- **first_research_agent.py** / **first_research_agent.ipynb** — Simple web-research agent: searches Wikipedia for a topic and summarizes the article with a T5 model (Colab notebook + exported script).
- **fine-tuning-ai.py** — Fine-tunes GPT-2 with LoRA (PEFT) on a subset of the IMDB dataset, comparing generation output before and after training.
- **retrieval_augmented_generation.py** — RAG pipeline: chunks a PDF, embeds chunks with Sentence-Transformers, indexes them in FAISS, and answers queries using a Flan-T5 model.
- **tokenization.py** — Generates sentence embeddings with `all-MiniLM-L6-v2` and compares them via cosine similarity.
- **transformers_pipeline.py** — Basic Hugging Face `pipeline` example doing text generation with `distilgpt2`.
- **visual_question_answering.py** — Uses a BLIP model to answer a question about the contents of an image.
- **generate_images_DALLE.py** — Generates an image from a text prompt using OpenAI's DALL-E 3 API.
- **generate_images_STABLE_DIFFUSION.py** — Generates an image from a text prompt locally using Stable Diffusion (requires GPU).
- **generationg_visuals_design.py** — Stable Diffusion example tuned for minimalist vector-art style branding visuals (with a negative prompt).
