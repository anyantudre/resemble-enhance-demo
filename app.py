import argparse
import logging

import gradio as gr
import soundfile as sf
import torch

from resemble_enhance.enhancer.inference import denoise, enhance, load_enhancer

logging.basicConfig(level=logging.INFO)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# The free Space has 2 vCPUs, so long clips would block the queue for minutes.
MAX_SECONDS = 30 if DEVICE == "cpu" else 120


def load_audio(path: str) -> tuple[torch.Tensor, int]:
    try:
        wav, sr = sf.read(path, dtype="float32", always_2d=True)
    except sf.LibsndfileError as e:
        raise gr.Error("Could not read this file. Please upload WAV, FLAC, MP3 or OGG audio.") from e
    # Mix down to mono: the models work on a single channel.
    return torch.from_numpy(wav.mean(axis=1)), sr


def run(path, solver, nfe, tau, denoise_first, max_seconds):
    if path is None:
        raise gr.Error("Please upload or record an audio clip.")

    dwav, sr = load_audio(path)
    duration = dwav.shape[-1] / sr
    if max_seconds and duration > max_seconds:
        raise gr.Error(f"This clip is {duration:.0f} s long. The demo accepts up to {max_seconds} s.")

    lambd = 0.9 if denoise_first else 0.1

    denoised, out_sr = denoise(dwav, sr, DEVICE)
    enhanced, out_sr = enhance(dwav, sr, DEVICE, nfe=int(nfe), solver=solver.lower(), lambd=lambd, tau=tau)

    return (out_sr, denoised.cpu().numpy()), (out_sr, enhanced.cpu().numpy())


def build_demo(max_seconds: int | None) -> gr.Blocks:
    limit = f"Clips up to {max_seconds} seconds. " if max_seconds else ""
    hardware = "a free CPU, so processing takes a while" if DEVICE == "cpu" else "a GPU"

    # delete_cache clears uploads and results older than an hour, every 30 minutes,
    # so the Space disk can't fill up over time.
    with gr.Blocks(title="Resemble Enhance", delete_cache=(1800, 3600)) as demo:
        gr.Markdown(
            "# Resemble Enhance\n"
            "Speech denoising and enhancement with "
            "[Resemble Enhance](https://github.com/resemble-ai/resemble-enhance). "
            "The denoiser separates speech from background noise. The enhancer restores distortions "
            f"and extends the bandwidth to 44.1 kHz. {limit}This Space runs on {hardware}."
        )
        with gr.Row():
            with gr.Column():
                audio_in = gr.Audio(type="filepath", label="Input audio", sources=["upload", "microphone"])
                denoise_first = gr.Checkbox(value=False, label="Denoise before enhancement (for heavy background noise)")
                with gr.Accordion("Enhancer settings", open=False):
                    solver = gr.Dropdown(choices=["Midpoint", "RK4", "Euler"], value="Midpoint", label="CFM ODE solver")
                    nfe = gr.Slider(
                        minimum=1,
                        maximum=128,
                        # On the free CPU each step adds about 5 s per 10 s of audio.
                        value=16 if DEVICE == "cpu" else 64,
                        step=1,
                        label="Function evaluations (higher is usually better but slower)",
                    )
                    tau = gr.Slider(
                        minimum=0,
                        maximum=1,
                        value=0.5,
                        step=0.01,
                        label="Prior temperature (higher can improve quality but reduce stability)",
                    )
                button = gr.Button("Enhance", variant="primary")
            with gr.Column():
                denoised_out = gr.Audio(label="Denoised", format="wav")
                enhanced_out = gr.Audio(label="Enhanced", format="wav")

        button.click(
            lambda *args: run(*args, max_seconds=max_seconds),
            inputs=[audio_in, solver, nfe, tau, denoise_first],
            outputs=[denoised_out, enhanced_out],
            concurrency_limit=1,
        )

    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unlimited", action="store_true", help="accept clips of any length")
    args = parser.parse_args()

    # Download (about 700 MB on the first run) and load the weights before taking requests.
    load_enhancer(None, DEVICE)

    demo = build_demo(None if args.unlimited else MAX_SECONDS)
    demo.queue(max_size=10).launch()


if __name__ == "__main__":
    main()
