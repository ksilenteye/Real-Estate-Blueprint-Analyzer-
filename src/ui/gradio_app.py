"""
Gradio Blocks interface with three tabs:
  Tab 1 - Upload: Image upload + style selection
  Tab 2 - Segmentation: Shows segmentation overlay + room list
  Tab 3 - Furnished: Shows furnished output + comparison view
"""
import gradio as gr
from PIL import Image
from typing import Optional

from config import AppConfig
from src.pipeline.blueprint_pipeline import BlueprintPipeline


def create_app(config: AppConfig) -> gr.Blocks:
    pipeline = BlueprintPipeline(config)

    with gr.Blocks(
        title="Blueprint to Furnished Plan",
        theme=gr.themes.Soft(),
        css="""
        .main-title { text-align: center; margin-bottom: 0.5em; }
        .subtitle { text-align: center; color: #666; margin-bottom: 1.5em; }
        """,
    ) as app:

        gr.Markdown(
            "# Blueprint to Furnished Plan",
            elem_classes=["main-title"],
        )
        gr.Markdown(
            "Upload a house blueprint/floorplan. The system detects rooms using "
            "SAM2 + OpenCV, classifies them, and generates a furnished version "
            "using Stable Diffusion + ControlNet.",
            elem_classes=["subtitle"],
        )

        # Shared state across tabs
        state_classified = gr.State(value=None)
        state_original_img = gr.State(value=None)

        with gr.Tabs() as tabs:

            # ---- TAB 1: Upload ----
            with gr.TabItem("1. Upload Blueprint", id=0):
                with gr.Row():
                    with gr.Column(scale=2):
                        input_image = gr.Image(
                            label="Upload Blueprint",
                            type="pil",
                            sources=["upload"],
                            height=500,
                        )
                    with gr.Column(scale=1):
                        gr.Markdown("### Settings")
                        style_dropdown = gr.Dropdown(
                            choices=list(config.style.styles.keys()),
                            value="modern",
                            label="Furnishing Style",
                            info="Choose the interior design style",
                        )
                        seed_input = gr.Number(
                            label="Seed (0 = random)",
                            value=0,
                            precision=0,
                            info="Set a seed for reproducible results",
                        )
                        gr.Markdown("---")
                        with gr.Row():
                            segment_btn = gr.Button(
                                "Detect Rooms",
                                variant="primary",
                                size="lg",
                            )
                        with gr.Row():
                            full_btn = gr.Button(
                                "Full Pipeline (Detect + Furnish)",
                                variant="secondary",
                                size="lg",
                            )
                        gr.Markdown(
                            "*Detect Rooms* runs segmentation only. "
                            "*Full Pipeline* runs segmentation + generation."
                        )

            # ---- TAB 2: Segmentation ----
            with gr.TabItem("2. Room Segmentation", id=1):
                with gr.Row():
                    original_display = gr.Image(
                        label="Original Blueprint",
                        height=450,
                        interactive=False,
                    )
                    segmentation_display = gr.Image(
                        label="Room Segmentation",
                        height=450,
                        interactive=False,
                    )
                room_summary_text = gr.Textbox(
                    label="Detected Rooms",
                    lines=8,
                    interactive=False,
                )
                with gr.Row():
                    generate_btn = gr.Button(
                        "Generate Furnished Plan",
                        variant="primary",
                        size="lg",
                    )
                    style_dropdown_seg = gr.Dropdown(
                        choices=list(config.style.styles.keys()),
                        value="modern",
                        label="Style",
                    )
                    seed_input_seg = gr.Number(
                        label="Seed (0 = random)",
                        value=0,
                        precision=0,
                    )

            # ---- TAB 3: Furnished Output ----
            with gr.TabItem("3. Furnished Output", id=2):
                with gr.Row():
                    seg_compare = gr.Image(
                        label="Segmentation",
                        height=450,
                        interactive=False,
                    )
                    furnished_display = gr.Image(
                        label="Furnished Plan",
                        height=450,
                        interactive=False,
                    )
                with gr.Row():
                    regen_style = gr.Dropdown(
                        choices=list(config.style.styles.keys()),
                        value="modern",
                        label="Style for Regeneration",
                    )
                    regen_seed = gr.Number(
                        label="Seed",
                        value=0,
                        precision=0,
                    )
                    regen_btn = gr.Button(
                        "Regenerate with New Style",
                        variant="secondary",
                    )

        # ---- EVENT HANDLERS ----

        def on_segment(image, progress=gr.Progress()):
            """Handler for 'Detect Rooms' button."""
            if image is None:
                gr.Warning("Please upload a blueprint image first.")
                return [None, None, "No image uploaded.", None, None, gr.Tabs(selected=0)]

            overlay, classified, summary = pipeline.run_segmentation_only(
                image, progress_callback=progress
            )
            return [
                image,           # original_display
                overlay,         # segmentation_display
                summary,         # room_summary_text
                classified,      # state_classified
                image,           # state_original_img
                gr.Tabs(selected=1),
            ]

        def on_generate(orig_img, classified, style, seed, progress=gr.Progress()):
            """Handler for 'Generate Furnished Plan' button."""
            if orig_img is None or classified is None:
                gr.Warning("Please run room detection first.")
                return [None, gr.Tabs(selected=1)]

            actual_seed = int(seed) if seed and seed > 0 else None
            furnished = pipeline.run_generation_only(
                orig_img, classified, style, actual_seed, progress
            )
            return [
                furnished,       # furnished_display
                gr.Tabs(selected=2),
            ]

        def on_full_pipeline(image, style, seed, progress=gr.Progress()):
            """Handler for 'Full Pipeline' button."""
            if image is None:
                gr.Warning("Please upload a blueprint image first.")
                return [None, None, "", None, None, None, None, gr.Tabs(selected=0)]

            actual_seed = int(seed) if seed and seed > 0 else None
            result = pipeline.run(image, style, actual_seed, progress)

            return [
                result.original,             # original_display
                result.segmentation_overlay,  # segmentation_display
                result.room_summary,          # room_summary_text
                result.rooms,                 # state_classified
                result.original,             # state_original_img
                result.segmentation_overlay,  # seg_compare
                result.furnished,             # furnished_display
                gr.Tabs(selected=2),
            ]

        def on_regenerate(orig_img, classified, style, seed, progress=gr.Progress()):
            """Handler for 'Regenerate' button on the furnished tab."""
            if orig_img is None or classified is None:
                gr.Warning("No segmentation data available. Run detection first.")
                return None

            actual_seed = int(seed) if seed and seed > 0 else None
            furnished = pipeline.run_generation_only(
                orig_img, classified, style, actual_seed, progress
            )
            return furnished

        # Wire up buttons
        segment_btn.click(
            fn=on_segment,
            inputs=[input_image],
            outputs=[
                original_display,
                segmentation_display,
                room_summary_text,
                state_classified,
                state_original_img,
                tabs,
            ],
        )

        generate_btn.click(
            fn=on_generate,
            inputs=[
                state_original_img,
                state_classified,
                style_dropdown_seg,
                seed_input_seg,
            ],
            outputs=[furnished_display, tabs],
        )

        full_btn.click(
            fn=on_full_pipeline,
            inputs=[input_image, style_dropdown, seed_input],
            outputs=[
                original_display,
                segmentation_display,
                room_summary_text,
                state_classified,
                state_original_img,
                seg_compare,
                furnished_display,
                tabs,
            ],
        )

        regen_btn.click(
            fn=on_regenerate,
            inputs=[
                state_original_img,
                state_classified,
                regen_style,
                regen_seed,
            ],
            outputs=[furnished_display],
        )

    return app
