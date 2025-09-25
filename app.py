"""Gradio app for describing retail products using the LLaVA multimodal model."""
from __future__ import annotations

import argparse
import os
from functools import lru_cache
from typing import Optional

import gradio as gr
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

MODEL_ID = "llava-hf/llava-1.5-7b-hf"
DEFAULT_QUESTION = (
    "این محصول چیست؟ لطفاً نوع کالا، رنگ و جنس ظاهری، کاربرد اصلی و بهترین دسته‌بندی پیشنهادی"
    " برای فروشگاه آنلاین را بیان کن."
)
SYSTEM_PROMPT = (
    "You are an e-commerce assistant that receives a product image and produces rich,"
    " structured descriptions in Persian. Summaries must include:"
    "\n1. نوع و مدل احتمالی کالا"
    "\n2. رنگ‌ها و جنس یا متریال قابل تشخیص"
    "\n3. کاربرد یا سناریوی استفاده‌ی معمول"
    "\n4. دو یا سه دسته‌بندی پیشنهادی برای وب‌سایت فروشگاهی"
    "\nIf you are uncertain, clearly mention the uncertainty and provide the closest guess."
)

def _device_config() -> tuple[dict, torch.dtype]:
    """Return keyword arguments for model loading based on the available hardware."""
    if torch.cuda.is_available():
        return {"device_map": "auto", "torch_dtype": torch.float16}, torch.float16

    if os.environ.get("LLAVA_USE_8BIT") == "1":
        try:
            import bitsandbytes  # type: ignore # noqa: F401
        except ImportError as exc:  # pragma: no cover - defensive branch
            raise RuntimeError(
                "برای استفاده از حالت 8-بیتی باید کتابخانهٔ bitsandbytes نصب شده باشد."
            ) from exc

        return {"load_in_8bit": True, "device_map": "auto"}, torch.float16

    return {"torch_dtype": torch.float32}, torch.float32


@lru_cache(maxsize=1)
def load_model() -> tuple[LlavaForConditionalGeneration, AutoProcessor, torch.device]:
    """Load the LLaVA model and processor once and cache them for reuse."""
    model_kwargs, dtype = _device_config()

    processor = AutoProcessor.from_pretrained(MODEL_ID, use_fast=False)
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        low_cpu_mem_usage=True,
        **model_kwargs,
    )

    if not torch.cuda.is_available() and "load_in_8bit" not in model_kwargs:
        model.to(dtype=dtype, device=torch.device("cpu"))

    device = next(model.parameters()).device
    return model, processor, device


def build_prompt(question: str) -> str:
    """Compose a single prompt string following the chat template expected by LLaVA."""
    sanitized_question = question.strip() or DEFAULT_QUESTION
    return (
        "[INST] <<SYS>>" + SYSTEM_PROMPT + "<</SYS>>\n"
        "<image>\n"
        + sanitized_question
        + "\nپاسخ را به صورت متنی ساختارمند در چند جمله ارائه کن. [/INST]"
    )


def generate_description(
    image: Optional[Image.Image],
    question: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
) -> str:
    if image is None:
        return "لطفاً ابتدا تصویر محصول را بارگذاری کنید."

    model, processor, device = load_model()
    prompt = build_prompt(question)

    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
        )

    output = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    if "[/INST]" in output:
        output = output.split("[/INST]")[-1]

    return output.strip()


def build_interface() -> gr.Blocks:
    with gr.Blocks(title="LLava Retail Product Describer") as demo:
        gr.Markdown(
            """
            # تشخیص و توصیف کالا با LLaVA
            تصویر محصول خود را بارگذاری کرده و توصیف خودکار شامل نوع کالا، رنگ و جنس، کاربرد و دسته‌بندی پیشنهادی دریافت کنید.
            """
        )

        with gr.Row():
            image_input = gr.Image(type="pil", label="تصویر محصول")
            description_output = gr.Textbox(
                label="خروجی مدل",
                lines=12,
                show_copy_button=True,
            )

        question_input = gr.Textbox(
            label="پرسش (در صورت خالی بودن، متن پیش‌فرض استفاده می‌شود)",
            value=DEFAULT_QUESTION,
            lines=3,
        )

        with gr.Accordion("تنظیمات پیشرفته", open=False):
            temperature_slider = gr.Slider(
                minimum=0.0, maximum=1.0, value=0.2, step=0.05, label="Temperature"
            )
            top_p_slider = gr.Slider(minimum=0.1, maximum=1.0, value=0.7, step=0.05, label="Top-p")
            max_tokens_slider = gr.Slider(
                minimum=32, maximum=1024, value=256, step=32, label="حداکثر توکن خروجی"
            )

        analyze_button = gr.Button("تولید توضیحات", variant="primary")

        analyze_button.click(
            fn=generate_description,
            inputs=[image_input, question_input, temperature_slider, top_p_slider, max_tokens_slider],
            outputs=description_output,
        )

        gr.Examples(
            examples=[
                ["https://huggingface.co/datasets/hf-internal-testing/fixtures_image_utils/resolve/main/food_mixed.jpg", DEFAULT_QUESTION],
                ["https://huggingface.co/datasets/mishig/sample_images/resolve/main/bicycle.png", DEFAULT_QUESTION],
            ],
            inputs=[image_input, question_input],
            label="نمونه تصاویر برای تست سریع",
        )

        gr.Markdown(
            """
            ### نکات مهم
            - اولین اجرای مدل ممکن است چند دقیقه طول بکشد زیرا باید وزن‌ها دانلود شوند.
            - برای سرعت بالاتر و مصرف حافظه کمتر می‌توانید از GPU استفاده کنید.
            - خروجی مدل ممکن است نیاز به بازبینی انسانی داشته باشد، مخصوصاً برای دسته‌بندی‌های حساس.
            """
        )

    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the LLaVA Gradio demo.")
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a publicly shareable Gradio link (useful for Colab or remote demos).",
    )
    parser.add_argument(
        "--server-port",
        type=int,
        default=None,
        help="Optional port for the Gradio server. Defaults to Gradio's standard port.",
    )
    parser.add_argument(
        "--server-name",
        type=str,
        default=None,
        help="Bind address for the Gradio server (e.g., 0.0.0.0 for remote access).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    interface = build_interface()
    interface.queue().launch(share=args.share, server_port=args.server_port, server_name=args.server_name)
