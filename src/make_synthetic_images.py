"""Generate a synthetic image dataset for the Smelter Super Kings capstone."""

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageOps

from src.model_utils import (
    CLASS_NAMES,
    DATASET_NOTICE_FILENAME,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_IMAGES_PER_CLASS,
    METADATA_FILENAME,
    PRODUCT_TYPES,
    ensure_directory,
)


SYNTHETIC_NOTE = "SYNTHETIC classroom data generated for image-track model development."


@dataclass(frozen=True)
class GenerationConfig:
    """Settings used to generate a synthetic defect dataset."""

    out_dir: Path
    per_class: int
    seed: int
    size: tuple[int, int]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for dataset generation."""
    parser = argparse.ArgumentParser(description="Generate synthetic surface-defect images.")
    parser.add_argument("--out", default="data/images", help="Output directory for generated images.")
    parser.add_argument(
        "--per_class",
        type=int,
        default=DEFAULT_IMAGES_PER_CLASS,
        help="Number of images to generate per class.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible generation.")
    parser.add_argument(
        "--width",
        type=int,
        default=DEFAULT_IMAGE_SIZE[0],
        help="Image width in pixels.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=DEFAULT_IMAGE_SIZE[1],
        help="Image height in pixels.",
    )
    return parser.parse_args()


def create_base_surface(size: tuple[int, int], rng: random.Random, product_type: str) -> Image.Image:
    """Create a brushed-metal surface with lighting and texture variation."""
    width, height = size
    base_tones = {
        "coil": (188, 191, 196),
        "sheet": (197, 199, 203),
        "plate": (180, 183, 188),
        "strip": (191, 194, 199),
    }
    base = Image.new("RGB", size, base_tones.get(product_type, (190, 193, 198)))
    draw = ImageDraw.Draw(base)

    for y in range(height):
        delta = int((y / max(height - 1, 1)) * 18) - 9
        jitter = rng.randint(-3, 3)
        color = tuple(max(80, min(240, channel + delta + jitter)) for channel in base.getpixel((0, y)))
        draw.line((0, y, width, y), fill=color)

    noise = Image.effect_noise(size, rng.uniform(8.0, 18.0)).convert("L")
    noise_colored = ImageOps.colorize(noise, black=(105, 108, 112), white=(225, 228, 232))
    base = Image.blend(base, noise_colored, 0.18)

    draw = ImageDraw.Draw(base)
    spacing = rng.randint(4, 8)
    for y in range(0, height, spacing):
        line_intensity = rng.randint(-10, 10)
        tone = tuple(max(60, min(245, channel + line_intensity)) for channel in base.getpixel((0, y)))
        draw.line((0, y, width, y), fill=tone, width=1)

    highlight = Image.new("RGBA", size, (0, 0, 0, 0))
    highlight_draw = ImageDraw.Draw(highlight)
    band_y = rng.randint(int(height * 0.15), int(height * 0.75))
    highlight_draw.rectangle(
        (0, band_y, width, min(height, band_y + rng.randint(18, 34))),
        fill=(255, 255, 255, rng.randint(18, 42)),
    )
    highlight = highlight.filter(ImageFilter.GaussianBlur(radius=rng.uniform(4.0, 9.0)))
    base = Image.alpha_composite(base.convert("RGBA"), highlight).convert("RGB")

    return base


def draw_soft_line(
    overlay: Image.Image,
    rng: random.Random,
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int, int],
    width: int,
    blur_radius: float,
) -> None:
    """Draw a blurred line on an RGBA overlay."""
    line_image = Image.new("RGBA", overlay.size, (0, 0, 0, 0))
    line_draw = ImageDraw.Draw(line_image)
    line_draw.line((start, end), fill=color, width=width)
    softened = line_image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    overlay.alpha_composite(softened)


def add_acceptable_pattern(image: Image.Image, rng: random.Random) -> Image.Image:
    """Keep the surface mostly clean with faint non-defect variation."""
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    width, height = image.size
    draw = ImageDraw.Draw(overlay)

    for _ in range(rng.randint(1, 2)):
        x0 = rng.randint(0, width // 2)
        y0 = rng.randint(0, height - 1)
        x1 = min(width, x0 + rng.randint(width // 4, width // 2))
        y1 = min(height, y0 + rng.randint(-12, 12))
        draw.line((x0, y0, x1, y1), fill=(255, 255, 255, rng.randint(18, 28)), width=1)

    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=1.8))
    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def add_scratch_pattern(image: Image.Image, rng: random.Random) -> Image.Image:
    """Add elongated scratch-like marks to the surface."""
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    width, height = image.size

    for _ in range(rng.randint(2, 4)):
        start = (rng.randint(0, width // 2), rng.randint(0, height - 1))
        end = (
            min(width - 1, start[0] + rng.randint(width // 3, width - 10)),
            max(0, min(height - 1, start[1] + rng.randint(-25, 25))),
        )
        draw_soft_line(
            overlay,
            rng,
            start,
            end,
            color=(70, 72, 78, rng.randint(130, 180)),
            width=rng.randint(2, 4),
            blur_radius=rng.uniform(0.6, 1.4),
        )
        draw_soft_line(
            overlay,
            rng,
            (start[0], max(0, start[1] - 1)),
            (end[0], max(0, end[1] - 1)),
            color=(245, 245, 245, rng.randint(70, 110)),
            width=1,
            blur_radius=0.8,
        )

    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def add_dent_pattern(image: Image.Image, rng: random.Random) -> Image.Image:
    """Add dent-like depressions using soft shadow and highlight."""
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    width, height = image.size
    draw = ImageDraw.Draw(overlay)

    for _ in range(rng.randint(1, 2)):
        cx = rng.randint(width // 4, width - width // 4)
        cy = rng.randint(height // 4, height - height // 4)
        rx = rng.randint(16, 28)
        ry = rng.randint(12, 24)

        draw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=(40, 45, 50, rng.randint(50, 80)))
        draw.ellipse(
            (cx - rx + 5, cy - ry + 3, cx + rx + 6, cy + ry + 6),
            fill=(255, 255, 255, rng.randint(32, 54)),
        )

    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=rng.uniform(6.0, 8.5)))
    blended = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    return ImageEnhance.Contrast(blended).enhance(rng.uniform(0.96, 1.04))


def add_stain_pattern(image: Image.Image, rng: random.Random) -> Image.Image:
    """Add stain-like discoloration with diffuse irregular blobs."""
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    width, height = image.size
    draw = ImageDraw.Draw(overlay)
    stain_palette = [
        (163, 132, 84, 95),
        (145, 120, 73, 115),
        (120, 134, 152, 80),
        (178, 150, 98, 88),
    ]

    for _ in range(rng.randint(2, 4)):
        cx = rng.randint(width // 5, width - width // 5)
        cy = rng.randint(height // 5, height - height // 5)
        rx = rng.randint(18, 34)
        ry = rng.randint(14, 30)
        fill = stain_palette[rng.randrange(len(stain_palette))]
        draw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=fill)

    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=rng.uniform(7.0, 11.0)))

    if rng.random() < 0.45:
        secondary = Image.new("RGBA", image.size, (0, 0, 0, 0))
        secondary_draw = ImageDraw.Draw(secondary)
        secondary_draw.rectangle(
            (
                rng.randint(0, width // 3),
                rng.randint(0, height // 3),
                rng.randint(width // 2, width),
                rng.randint(height // 2, height),
            ),
            fill=(120, 98, 60, rng.randint(12, 22)),
        )
        overlay.alpha_composite(secondary.filter(ImageFilter.GaussianBlur(radius=10.0)))

    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def add_other_pattern(image: Image.Image, rng: random.Random) -> Image.Image:
    """Add mixed surface anomalies that do not neatly match the named classes."""
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    width, height = image.size
    draw = ImageDraw.Draw(overlay)

    for _ in range(rng.randint(8, 16)):
        radius = rng.randint(2, 6)
        x = rng.randint(radius, width - radius)
        y = rng.randint(radius, height - radius)
        fill = (rng.randint(65, 130), rng.randint(65, 130), rng.randint(65, 130), rng.randint(80, 135))
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill)

    if rng.random() < 0.6:
        jagged_points = []
        start_x = rng.randint(10, width // 3)
        start_y = rng.randint(10, height - 10)
        jagged_points.append((start_x, start_y))
        for _ in range(rng.randint(3, 5)):
            start_x += rng.randint(12, 28)
            start_y += rng.randint(-18, 18)
            jagged_points.append((min(width - 5, start_x), max(5, min(height - 5, start_y))))
        draw.line(jagged_points, fill=(70, 70, 76, rng.randint(120, 160)), width=rng.randint(2, 3))

    if rng.random() < 0.45:
        draw.rectangle(
            (
                rng.randint(0, width // 2),
                rng.randint(0, height // 2),
                rng.randint(width // 2, width - 1),
                rng.randint(height // 2, height - 1),
            ),
            fill=(122, 112, 86, rng.randint(18, 28)),
        )

    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=rng.uniform(1.6, 3.6)))
    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def add_shared_variation(image: Image.Image, rng: random.Random) -> Image.Image:
    """Apply shared lighting and sensor variation to avoid perfect separation."""
    width, height = image.size
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for _ in range(rng.randint(8, 16)):
        x = rng.randint(0, width - 1)
        y = rng.randint(0, height - 1)
        alpha = rng.randint(10, 24)
        draw.ellipse((x, y, x + 2, y + 2), fill=(255, 255, 255, alpha))

    image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")

    if rng.random() < 0.5:
        image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.25, 0.8)))

    image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.94, 1.06))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.94, 1.08))

    if rng.random() < 0.35:
        shifted = ImageChops.offset(image, rng.randint(-2, 2), rng.randint(-2, 2))
        image = Image.blend(image, shifted, 0.08)

    return image


def generate_image(label: str, product_type: str, size: tuple[int, int], rng: random.Random) -> Image.Image:
    """Generate a single synthetic image for the requested class."""
    image = create_base_surface(size, rng, product_type)

    if label == "acceptable":
        image = add_acceptable_pattern(image, rng)
    elif label == "scratch":
        image = add_scratch_pattern(image, rng)
    elif label == "dent":
        image = add_dent_pattern(image, rng)
    elif label == "stain":
        image = add_stain_pattern(image, rng)
    elif label == "other":
        image = add_other_pattern(image, rng)
    else:  # pragma: no cover - guarded by controlled class list
        raise ValueError(f"Unsupported label: {label}")

    return add_shared_variation(image, rng)


def write_dataset_notice(out_dir: Path) -> None:
    """Write a clear notice that the dataset is synthetic."""
    notice = out_dir / DATASET_NOTICE_FILENAME
    notice.write_text(
        "\n".join(
            [
                "SYNTHETIC DATASET NOTICE",
                "",
                "All images in this directory were programmatically generated for classroom use.",
                "They do not represent real plant inspection images or validated production distributions.",
                "Use this dataset only for local development, testing, and demonstration of the image-track capstone.",
            ]
        ),
        encoding="utf-8",
    )


def write_metadata(out_dir: Path, rows: list[dict[str, str]]) -> None:
    """Persist metadata for the generated dataset."""
    metadata_path = out_dir / METADATA_FILENAME
    fieldnames = [
        "sample_id",
        "label",
        "product_type",
        "image_path",
        "is_synthetic",
        "synthetic_note",
        "width",
        "height",
    ]
    with metadata_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def generate_dataset(config: GenerationConfig) -> None:
    """Generate a full synthetic dataset across all configured classes."""
    out_dir = ensure_directory(config.out_dir)
    rows: list[dict[str, str]] = []
    master_rng = random.Random(config.seed)

    for class_name in CLASS_NAMES:
        class_dir = ensure_directory(out_dir / class_name)

        for index in range(config.per_class):
            sample_rng = random.Random(master_rng.randint(0, 10_000_000))
            product_type = sample_rng.choice(PRODUCT_TYPES)
            image = generate_image(class_name, product_type, config.size, sample_rng)

            filename = f"{class_name}_{index:03d}.png"
            relative_path = Path(class_name) / filename
            image.save(out_dir / relative_path)

            rows.append(
                {
                    "sample_id": f"{class_name}-{index:03d}",
                    "label": class_name,
                    "product_type": product_type,
                    "image_path": relative_path.as_posix(),
                    "is_synthetic": "true",
                    "synthetic_note": SYNTHETIC_NOTE,
                    "width": str(config.size[0]),
                    "height": str(config.size[1]),
                }
            )

    write_dataset_notice(out_dir)
    write_metadata(out_dir, rows)


def main() -> None:
    """Generate the dataset from command-line arguments."""
    args = parse_args()
    config = GenerationConfig(
        out_dir=Path(args.out),
        per_class=args.per_class,
        seed=args.seed,
        size=(args.width, args.height),
    )
    generate_dataset(config)
    print(f"Generated SYNTHETIC dataset in: {config.out_dir}")
    print(f"Classes: {', '.join(CLASS_NAMES)}")
    print(f"Images per class: {config.per_class}")


if __name__ == "__main__":
    main()
