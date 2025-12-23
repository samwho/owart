#!/usr/bin/env python3
"""
OpenWeights.art - Artistic visualization of neural network weights.

Creates poster-style visualizations of model weights where each pixel
represents a single weight value, with intensity based on value range
and color based on the type of layer.
"""

import argparse
import concurrent.futures
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


@dataclass
class WeightBlock:
    """A block of weights to be placed in the visualization."""

    name: str
    weights: np.ndarray  # Flattened weights, already normalized to [0, 1]
    category: str
    width: int
    height: int
    x: int = 0
    y: int = 0
    row_lengths: list[int] | None = field(default=None)


def _poly_fits_at(occupancy: list[int], x: int, y: int, row_lengths: list[int]) -> bool:
    for i, row_len in enumerate(row_lengths):
        if row_len <= 0:
            continue
        mask = ((1 << row_len) - 1) << x
        row_idx = y + i
        if row_idx < len(occupancy) and (occupancy[row_idx] & mask):
            return False
    return True


def _poly_find_in_range(
    occupancy: list[int],
    row_lengths: list[int],
    canvas_width: int,
    shape_w: int,
    y_start: int,
    y_end: int,
) -> tuple[int, int] | None:
    x_limit = canvas_width - shape_w + 1
    for y in range(y_start, y_end):
        for x in range(0, x_limit):
            if _poly_fits_at(occupancy, x, y, row_lengths):
                return (y, x)
    return None


# Palette definitions (RGB), sourced from Matplotlib's categorical palettes.
PALETTES = {
    "tab20": [
        (31, 119, 180),
        (255, 127, 14),
        (44, 160, 44),
        (214, 39, 40),
        (148, 103, 189),
        (140, 86, 75),
        (227, 119, 194),
        (127, 127, 127),
        (188, 189, 34),
        (23, 190, 207),
        (174, 199, 232),
        (255, 187, 120),
        (152, 223, 138),
        (255, 152, 150),
        (197, 176, 213),
        (196, 156, 148),
        (247, 182, 210),
        (199, 199, 199),
        (219, 219, 141),
        (158, 218, 229),
    ],
    "tab20b": [
        (57, 59, 121),
        (82, 84, 163),
        (107, 110, 207),
        (156, 158, 222),
        (99, 121, 57),
        (140, 162, 82),
        (181, 207, 107),
        (206, 219, 156),
        (140, 109, 49),
        (189, 158, 57),
        (231, 186, 82),
        (231, 203, 148),
        (132, 60, 57),
        (189, 72, 65),
        (231, 97, 97),
        (231, 150, 156),
        (109, 68, 121),
        (150, 98, 165),
        (189, 143, 198),
        (210, 185, 208),
    ],
    "tab20c": [
        (49, 130, 189),
        (107, 174, 214),
        (158, 202, 225),
        (198, 219, 239),
        (230, 85, 13),
        (253, 141, 60),
        (253, 174, 107),
        (253, 208, 162),
        (49, 163, 84),
        (116, 196, 118),
        (161, 217, 155),
        (199, 233, 192),
        (132, 60, 57),
        (173, 73, 74),
        (214, 97, 107),
        (231, 150, 156),
        (123, 65, 115),
        (165, 81, 148),
        (206, 109, 189),
        (222, 158, 214),
    ],
}
DEFAULT_PALETTE_NAME = "tab20"


def normalize_weight_name(name: str) -> str:
    """
    Normalize a weight name by removing layer-specific indices.

    This ensures that weights like 'model.layers.0.self_attn.q_proj' and
    'model.layers.15.self_attn.q_proj' get the same category and thus the same color.
    """
    # Replace numeric indices (e.g., layers.0, layers.15) with a placeholder
    # This handles patterns like .0., .123., [0], [123]
    normalized = re.sub(r'\.\d+\.', '.N.', name)
    normalized = re.sub(r'\.\d+$', '.N', normalized)
    normalized = re.sub(r'\[\d+\]', '[N]', normalized)
    return normalized


class ColorAssigner:
    """Assigns colors to weight categories dynamically using the selected palette."""

    def __init__(self, palette: list[tuple[int, int, int]]):
        self.category_to_color: dict[str, tuple[int, int, int]] = {}
        self.next_color_idx = 0
        self.palette = palette

    def set_palette(self, palette: list[tuple[int, int, int]]) -> None:
        self.palette = palette
        self.category_to_color.clear()
        self.next_color_idx = 0

    def get_color(self, category: str) -> tuple[int, int, int]:
        """Get a color for a category, assigning a new one if needed."""
        if category not in self.category_to_color:
            color = self.palette[self.next_color_idx % len(self.palette)]
            self.category_to_color[category] = color
            self.next_color_idx += 1
        return self.category_to_color[category]


# Global color assigner instance
color_assigner = ColorAssigner(PALETTES[DEFAULT_PALETTE_NAME])


def categorize_weight(name: str) -> str:
    """Categorize a weight tensor based on its normalized name."""
    return normalize_weight_name(name)


def detect_quant_mode(scales: mx.array, biases) -> str:
    """Detect the quantization mode based on scales and biases."""
    # MXFP4/MXFP8 use uint8 scales and no biases
    if biases is None and scales.dtype == mx.uint8:
        return "mxfp4"
    # Standard affine quantization has biases
    return "affine"


def extract_weights(model) -> list[tuple[str, mx.array]]:
    """Extract all weight tensors from a model with their names."""
    weights = []
    seen = set()

    def add_weight(name: str, value):
        if name in seen:
            return
        seen.add(name)

        if isinstance(value, mx.array):
            weights.append((name, value))
        elif isinstance(value, dict):
            if "weight" in value and "scales" in value:
                try:
                    w = value["weight"]
                    s = value["scales"]
                    b = value.get("biases")
                    bits = value.get("bits", 4)
                    group_size = value.get("group_size", 64)
                    if isinstance(bits, mx.array):
                        bits = int(bits.item())
                    if isinstance(group_size, mx.array):
                        group_size = int(group_size.item())

                    mode = value.get("mode") or detect_quant_mode(s, b)
                    dequantized = mx.dequantize(w, s, b, group_size, bits, mode=mode)
                    weights.append((name, dequantized))
                except Exception as e:
                    print(f"  Warning: Could not dequantize {name}: {e}")
                    for k, v in value.items():
                        add_weight(f"{name}.{k}", v)
            else:
                for k, v in value.items():
                    add_weight(f"{name}.{k}", v)

    def traverse(obj, prefix=""):
        # First, check if this is a quantized layer (has weight and scales in parameters)
        if hasattr(obj, "parameters"):
            params = obj.parameters()
            if "weight" in params and "scales" in params:
                # This is a quantized layer - get metadata from the layer object
                if hasattr(obj, "group_size"):
                    params["group_size"] = obj.group_size
                if hasattr(obj, "bits"):
                    params["bits"] = obj.bits
                if hasattr(obj, "mode"):
                    params["mode"] = obj.mode
                add_weight(prefix, params)
                return  # Don't recurse into children for quantized layers

        # Recurse into children
        if hasattr(obj, "children"):
            for name, child in obj.children().items():
                full_name = f"{prefix}.{name}" if prefix else name
                traverse(child, full_name)

        # Handle lists/sequences of modules
        if hasattr(obj, "__iter__") and not isinstance(obj, (str, dict, mx.array)):
            try:
                for i, item in enumerate(obj):
                    if hasattr(item, "parameters") or hasattr(item, "children"):
                        traverse(item, f"{prefix}.{i}" if prefix else str(i))
            except TypeError:
                pass

        # For non-quantized layers with parameters, add them directly
        if hasattr(obj, "parameters"):
            params = obj.parameters()
            if "weight" not in params or "scales" not in params:
                # Not a quantized layer - add individual parameters
                for name, param in params.items():
                    if isinstance(param, mx.array):
                        full_name = f"{prefix}.{name}" if prefix else name
                        add_weight(full_name, param)

    if hasattr(model, "model"):
        traverse(model.model, "model")
    traverse(model, "")

    return weights


def calculate_block_dimensions(size: int) -> tuple[int, int]:
    """Calculate width and height for a block to be roughly square."""
    width = int(np.sqrt(size))
    if width == 0:
        width = 1
    height = (size + width - 1) // width
    return width, height


def tensor_to_float32(tensor: mx.array) -> np.ndarray:
    """Convert a tensor to a flattened float32 numpy array."""
    # Convert to float32 in MLX first to handle bfloat16 and other types
    # that numpy doesn't support directly
    tensor_f32 = tensor.astype(mx.float32)
    arr = np.array(tensor_f32).flatten()
    return arr


def print_histogram(samples: np.ndarray, title: str, bins: int = 40, width: int = 50):
    """Print an ASCII histogram to the terminal."""
    hist, bin_edges = np.histogram(samples, bins=bins)
    max_count = hist.max()

    print(f"\n{title}")
    print(f"Range: [{samples.min():.4f}, {samples.max():.4f}]")
    print()

    for i, count in enumerate(hist):
        bar_len = int((count / max_count) * width) if max_count > 0 else 0
        bar = "█" * bar_len
        left = bin_edges[i]
        print(f"{left:8.3f} |{bar}")
    print(f"{bin_edges[-1]:8.3f} |")


def normalize_array(
    arr: np.ndarray,
    global_min: float,
    global_max: float,
    clip_percent: float = 0.0,
) -> np.ndarray:
    """Normalize a numpy array to [0, 1] using linear scaling with optional percentile clipping."""
    if clip_percent > 0:
        pct = min(49.0, max(0.0, clip_percent))
        low = np.percentile(arr, pct)
        high = np.percentile(arr, 100.0 - pct)
        global_min = max(global_min, float(low))
        global_max = min(global_max, float(high))

    if global_max <= global_min:
        return np.full_like(arr, 0.5)

    normalized = (arr - global_min) / (global_max - global_min)
    normalized = np.clip(normalized, 0.0, 1.0)
    return normalized


def reduce_weights(weights: np.ndarray, scale: int, method: str = "mean") -> np.ndarray:
    """Reduce weights by combining scale x scale blocks into single pixels.

    Methods:
        mean: Average values (smooth, but loses contrast)
        median: Median value (balances smoothness and detail)
        max: Maximum absolute value (preserves extremes, can be noisy)
        sample: Take first value in each block (fast, preserves original values)
    """
    if scale <= 1:
        return weights

    # Calculate original 2D shape
    width, height = calculate_block_dimensions(len(weights))

    # Pad to make dimensions divisible by scale
    new_width = ((width + scale - 1) // scale) * scale
    new_height = ((height + scale - 1) // scale) * scale

    # Create padded array and fill with original weights
    padded = np.zeros(new_width * new_height, dtype=weights.dtype)
    padded[: len(weights)] = weights

    # Reshape to 2D
    grid = padded.reshape(new_height, new_width)

    # Reshape to group scale x scale blocks
    reduced_height = new_height // scale
    reduced_width = new_width // scale
    blocks = grid.reshape(reduced_height, scale, reduced_width, scale)

    if method == "mean":
        reduced = blocks.mean(axis=(1, 3))
    elif method == "median":
        # Median preserves more detail than mean while being smoother than max
        blocks_flat = blocks.reshape(reduced_height, reduced_width, scale * scale)
        reduced = np.median(blocks_flat, axis=2)
    elif method == "max":
        # Take the value with maximum absolute value (preserves sign)
        blocks_flat = blocks.reshape(reduced_height, reduced_width, scale * scale)
        max_idx = np.abs(blocks_flat).argmax(axis=2)
        reduced = np.take_along_axis(blocks_flat, max_idx[:, :, np.newaxis], axis=2).squeeze(axis=2)
    elif method == "sample":
        # Just take the first value in each block
        reduced = blocks[:, 0, :, 0]
    else:
        raise ValueError(f"Unknown reduction method: {method}")

    return reduced.flatten()


def apply_intensity_curve(
    values: np.ndarray,
    method: str,
    gamma: float,
    sigmoid_k: float,
) -> np.ndarray:
    """Adjust normalized values to improve visual contrast."""
    if method == "linear":
        return values
    if method == "gamma":
        return np.power(values, gamma)
    if method == "smoothstep":
        return values * values * (3.0 - 2.0 * values)
    if method == "sigmoid":
        k = max(0.1, sigmoid_k)
        y = 1.0 / (1.0 + np.exp(-k * (values - 0.5)))
        y0 = 1.0 / (1.0 + np.exp(0.5 * k))
        y1 = 1.0 / (1.0 + np.exp(-0.5 * k))
        return (y - y0) / (y1 - y0)
    raise ValueError(f"Unknown intensity method: {method}")


def create_weight_blocks(
    weights: list[tuple[str, mx.array]],
    scale: int = 1,
    scale_method: str = "mean",
    intensity_method: str = "sigmoid",
    intensity_gamma: float = 0.75,
    intensity_sigmoid_k: float = 6.0,
    intensity_clip_percent: float = 0.0,
    show_histogram: bool = False,
) -> list[WeightBlock]:
    """Create WeightBlock objects from weight tensors."""
    # First pass: reduce all weights
    reduced_weights = []

    for name, tensor in tqdm(weights, desc="Reducing weights"):
        raw = tensor_to_float32(tensor)
        reduced = reduce_weights(raw, scale, method=scale_method)
        reduced_weights.append((name, reduced))

    # Show histogram if requested
    if show_histogram:
        all_samples = []
        max_samples = 10000
        for name, reduced in reduced_weights:
            if len(reduced) > max_samples:
                indices = np.random.choice(len(reduced), max_samples, replace=False)
                all_samples.append(reduced[indices])
            else:
                all_samples.append(reduced)
        samples = np.concatenate(all_samples)

        print_histogram(samples, "Raw Weight Distribution (after reduction)")

    # Second pass: normalize each block individually and create blocks
    blocks = []

    for name, reduced in tqdm(reduced_weights, desc="Creating blocks"):
        # Normalize each block to its own range for maximum contrast
        block_min, block_max = float(reduced.min()), float(reduced.max())
        normalized = normalize_array(
            reduced,
            block_min,
            block_max,
            clip_percent=intensity_clip_percent,
        )
        normalized = apply_intensity_curve(
            normalized,
            method=intensity_method,
            gamma=intensity_gamma,
            sigmoid_k=intensity_sigmoid_k,
        )

        category = categorize_weight(name)
        # Dimensions will be set during packing, so use placeholder values
        width, height = 1, len(reduced)

        blocks.append(
            WeightBlock(
                name=name,
                weights=normalized,
                category=category,
                width=width,
                height=height,
            )
        )

    return blocks


@dataclass
class FreeRectangle:
    """A free rectangle in the packing area."""
    x: int
    y: int
    width: int
    height: int


def pack_blocks_maxrects(blocks: list[WeightBlock], padding: int = 2) -> tuple[int, int, list[WeightBlock]]:
    """Pack blocks using MaxRects algorithm for optimal space utilization."""
    if not blocks:
        return 0, 0, []

    # Calculate total number of weights across all blocks
    total_weights = sum(len(b.weights) for b in blocks)

    # Start with a square canvas
    canvas_width = int(np.sqrt(total_weights * 1.05))
    canvas_height = canvas_width

    # Sort blocks by area (largest first)
    sorted_blocks = sorted(blocks, key=lambda b: len(b.weights), reverse=True)

    # Track free rectangles
    free_rects = [FreeRectangle(0, 0, canvas_width, canvas_height)]
    positioned = []

    for block in tqdm(sorted_blocks, desc="Packing blocks"):
        num_weights = len(block.weights)

        best_rect_idx = None
        best_score = (float('inf'), float('inf'))
        best_dims = None

        # Try each free rectangle
        for idx, rect in enumerate(free_rects):
            # Try various aspect ratios
            for aspect_ratio in [1.0, 0.9, 1.11, 0.8, 1.25, 0.75, 1.33, 0.67, 1.5, 0.5, 2.0]:
                width = int(np.sqrt(num_weights * aspect_ratio))
                width = max(1, min(width, rect.width))
                height = (num_weights + width - 1) // width

                if width <= rect.width and height <= rect.height:
                    # Best Short Side Fit (BSSF) heuristic
                    leftover_x = rect.width - width
                    leftover_y = rect.height - height
                    short_side = min(leftover_x, leftover_y)
                    long_side = max(leftover_x, leftover_y)

                    # Score: prefer small short side, then small long side
                    score = (short_side, long_side)

                    if score < best_score:
                        best_score = score
                        best_rect_idx = idx
                        best_dims = (width, height)

        if best_rect_idx is None:
            # Expand canvas - prefer to keep it squarish
            if canvas_height <= canvas_width:
                expansion = max(canvas_height // 2, 100)
                free_rects.append(FreeRectangle(0, canvas_height, canvas_width, expansion))
                canvas_height += expansion
            else:
                expansion = max(canvas_width // 2, 100)
                free_rects.append(FreeRectangle(canvas_width, 0, expansion, canvas_height))
                canvas_width += expansion

            # Retry with new space
            width = int(np.sqrt(num_weights))
            height = (num_weights + width - 1) // width
            best_rect_idx = len(free_rects) - 1
            best_dims = (width, height)

        # Place the block
        rect = free_rects[best_rect_idx]
        width, height = best_dims

        block.x = rect.x
        block.y = rect.y
        block.width = width
        block.height = height
        positioned.append(block)

        # MaxRects: subdivide and update all free rectangles
        new_rects = []

        for fr in free_rects:
            if fr == free_rects[best_rect_idx]:
                continue

            # Check if this free rect intersects with placed block
            if (block.x < fr.x + fr.width and block.x + block.width > fr.x and
                block.y < fr.y + fr.height and block.y + block.height > fr.y):

                # Split this rectangle around the placed block
                # Left side
                if block.x > fr.x:
                    new_rects.append(FreeRectangle(
                        fr.x, fr.y,
                        block.x - fr.x - padding,
                        fr.height
                    ))

                # Right side
                if block.x + block.width + padding < fr.x + fr.width:
                    new_rects.append(FreeRectangle(
                        block.x + block.width + padding, fr.y,
                        fr.x + fr.width - (block.x + block.width + padding),
                        fr.height
                    ))

                # Bottom side
                if block.y > fr.y:
                    new_rects.append(FreeRectangle(
                        fr.x, fr.y,
                        fr.width,
                        block.y - fr.y - padding
                    ))

                # Top side
                if block.y + block.height + padding < fr.y + fr.height:
                    new_rects.append(FreeRectangle(
                        fr.x, block.y + block.height + padding,
                        fr.width,
                        fr.y + fr.height - (block.y + block.height + padding)
                    ))
            else:
                new_rects.append(fr)

        # Add new rectangles from the placed block's rectangle
        rect = free_rects[best_rect_idx]

        # Right remainder
        if rect.width > width + padding:
            new_rects.append(FreeRectangle(
                rect.x + width + padding, rect.y,
                rect.width - width - padding, rect.height
            ))

        # Bottom remainder
        if rect.height > height + padding:
            new_rects.append(FreeRectangle(
                rect.x, rect.y + height + padding,
                rect.width, rect.height - height - padding
            ))

        # Remove degenerate rectangles
        free_rects = [r for r in new_rects if r.width > 0 and r.height > 0]

        # Remove redundant rectangles (those fully contained in others)
        filtered = []
        for i, r1 in enumerate(free_rects):
            contained = False
            for j, r2 in enumerate(free_rects):
                if i != j and (r2.x <= r1.x and r2.y <= r1.y and
                              r2.x + r2.width >= r1.x + r1.width and
                              r2.y + r2.height >= r1.y + r1.height):
                    contained = True
                    break
            if not contained:
                filtered.append(r1)
        free_rects = filtered

    # Calculate actual used dimensions
    total_width = max(b.x + b.width for b in positioned) if positioned else 0
    total_height = max(b.y + b.height for b in positioned) if positioned else 0

    return total_width, total_height, positioned


def pack_blocks_guillotine(blocks: list[WeightBlock], padding: int = 2) -> tuple[int, int, list[WeightBlock]]:
    """Pack blocks using Guillotine algorithm with aspect ratio optimization."""
    if not blocks:
        return 0, 0, []

    # Calculate total number of weights across all blocks
    total_weights = sum(len(b.weights) for b in blocks)

    # Target a square canvas initially
    canvas_width = int(np.sqrt(total_weights * 1.2))  # Add 20% buffer for padding
    canvas_height = canvas_width

    # Sort blocks by area (largest first) for better packing
    sorted_blocks = sorted(blocks, key=lambda b: len(b.weights), reverse=True)

    # Initialize with one large free rectangle
    free_rects = [FreeRectangle(0, 0, canvas_width, canvas_height)]
    positioned = []

    for block in tqdm(sorted_blocks, desc="Packing blocks"):
        num_weights = len(block.weights)

        # Try to find the best free rectangle for this block
        best_rect_idx = None
        best_fit = None
        best_dims = None

        for idx, rect in enumerate(free_rects):
            # Skip rects that are too small
            available_area = (rect.width - padding) * (rect.height - padding)
            if available_area < num_weights:
                continue

            # Try different aspect ratios for this block
            for aspect_ratio in [1.0, 0.75, 1.33, 0.5, 2.0]:
                width = int(np.sqrt(num_weights * aspect_ratio))
                width = max(1, min(width, rect.width - padding))
                height = (num_weights + width - 1) // width

                if height <= rect.height - padding:
                    # Calculate fitness (prefer tight fits)
                    waste = (rect.width * rect.height) - (width * height)

                    if best_fit is None or waste < best_fit:
                        best_fit = waste
                        best_rect_idx = idx
                        best_dims = (width, height)

        if best_rect_idx is None:
            # No space found, expand canvas vertically
            new_rect = FreeRectangle(0, canvas_height, canvas_width, canvas_width)
            free_rects.append(new_rect)
            canvas_height += canvas_width

            # Try again with new space
            width = int(np.sqrt(num_weights))
            width = max(1, min(width, canvas_width))
            height = (num_weights + width - 1) // width
            best_rect_idx = len(free_rects) - 1
            best_dims = (width, height)

        # Place the block
        rect = free_rects[best_rect_idx]
        width, height = best_dims

        block.x = rect.x
        block.y = rect.y
        block.width = width
        block.height = height
        positioned.append(block)

        # Remove the used rectangle
        free_rects.pop(best_rect_idx)

        # Split the remaining space (Guillotine split)
        # Horizontal split
        if rect.width > width + padding:
            free_rects.append(FreeRectangle(
                rect.x + width + padding,
                rect.y,
                rect.width - width - padding,
                height
            ))

        # Vertical split
        if rect.height > height + padding:
            free_rects.append(FreeRectangle(
                rect.x,
                rect.y + height + padding,
                rect.width,
                rect.height - height - padding
            ))

    # Calculate actual used dimensions
    total_width = max(b.x + b.width for b in positioned) if positioned else 0
    total_height = max(b.y + b.height for b in positioned) if positioned else 0

    return total_width, total_height, positioned


def pack_blocks_shelf(blocks: list[WeightBlock], padding: int = 2) -> tuple[int, int, list[WeightBlock]]:
    """Pack blocks to completely fill canvas by resizing blocks proportionally."""
    if not blocks:
        return 0, 0, []

    # Calculate total number of weights across all blocks
    total_weights = sum(len(b.weights) for b in blocks)

    # Target a roughly square canvas
    target_size = int(np.sqrt(total_weights))

    # Sort blocks by size (largest first) for better visual organization
    sorted_blocks = sorted(blocks, key=lambda b: len(b.weights), reverse=True)

    # Use a simple row-by-row layout
    current_x = 0
    current_y = 0
    row_height = 0
    positioned = []

    for block in tqdm(sorted_blocks, desc="Packing blocks"):
        # Calculate how many weights this block should occupy
        num_weights = len(block.weights)

        # Calculate ideal square dimensions
        ideal_width = int(np.sqrt(num_weights))
        if ideal_width == 0:
            ideal_width = 1

        # Calculate remaining space in current row
        remaining_width = target_size - current_x

        # Try to maintain roughly square proportions but fit within target width
        block_width = min(ideal_width, remaining_width)

        # Check if this block fits in current row (need positive width)
        if block_width <= 0 or (block_width < ideal_width and current_x > 0):
            # Move to next row
            current_y += row_height + padding
            current_x = 0
            row_height = 0

            # Recalculate width for new row with full target_size available
            block_width = min(ideal_width, target_size)

        # Ensure width is always positive
        if block_width <= 0:
            block_width = 1

        block_height = (num_weights + block_width - 1) // block_width

        # Position the block
        block.x = current_x
        block.y = current_y
        block.width = block_width
        block.height = block_height

        positioned.append(block)

        # Update position tracking
        current_x += block_width + padding
        row_height = max(row_height, block_height)

    # Calculate final canvas dimensions
    total_width = max(b.x + b.width for b in positioned)
    total_height = max(b.y + b.height for b in positioned)

    return total_width, total_height, positioned


def build_row_lengths(num_weights: int, width: int) -> list[int]:
    """Build row lengths for a left-justified polyomino of size num_weights."""
    width = max(1, int(width))
    height = (num_weights + width - 1) // width
    if height <= 1:
        return [num_weights]
    last_row = num_weights - width * (height - 1)
    return [width] * (height - 1) + [last_row]


def rotate_row_lengths(row_lengths: list[int]) -> tuple[list[int], int, int]:
    """Rotate a left-justified polyomino 90 degrees clockwise."""
    height = len(row_lengths)
    width = max(row_lengths) if row_lengths else 0
    rotated = []
    for y in range(width):
        rotated.append(sum(1 for r in row_lengths if r > y))
    return rotated, height, width


def pack_blocks_polyomino(
    blocks: list[WeightBlock],
    padding: int = 0,
    workers: int = 1,
) -> tuple[int, int, list[WeightBlock]]:
    """Pack blocks as left-justified polyominoes using a bottom-left heuristic."""
    if not blocks:
        return 0, 0, []

    total_weights = sum(len(b.weights) for b in blocks)
    sorted_blocks = sorted(blocks, key=lambda b: len(b.weights), reverse=True)

    # Choose a reasonable canvas width to start with.
    canvas_width = int(np.sqrt(total_weights) * 1.05)
    if canvas_width <= 0:
        canvas_width = 1

    # Ensure the width can accommodate the widest candidate shape.
    max_shape_width = 1
    for b in sorted_blocks:
        base_width = max(1, int(np.sqrt(len(b.weights))))
        row_lengths = build_row_lengths(len(b.weights), base_width)
        max_shape_width = max(max_shape_width, max(row_lengths))
        rotated, _, _ = rotate_row_lengths(row_lengths)
        max_shape_width = max(max_shape_width, max(rotated))
    canvas_width = max(canvas_width, max_shape_width)

    occupancy: list[int] = []
    positioned: list[WeightBlock] = []

    def ensure_height(h: int) -> None:
        while len(occupancy) < h:
            occupancy.append(0)

    executor = None
    if workers > 1:
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=workers)

    try:
        for block in tqdm(sorted_blocks, desc="Packing polyominoes"):
            num_weights = len(block.weights)
            base_width = max(1, int(np.sqrt(num_weights)))
            row_lengths = build_row_lengths(num_weights, base_width)
            rotated_rows, rotated_w, rotated_h = rotate_row_lengths(row_lengths)

            candidates = [
                (row_lengths, max(row_lengths), len(row_lengths)),
                (rotated_rows, rotated_w, rotated_h),
            ]

            placed = False
            best = None

            for rows, shape_w, shape_h in candidates:
                if shape_w > canvas_width:
                    continue
                current_height = len(occupancy)
                if executor is not None and current_height > 0:
                    y_end = current_height + 1
                    chunk = max(1, (y_end + workers - 1) // workers)
                    tasks = []
                    for w in range(workers):
                        y_start = w * chunk
                        y_stop = min(y_end, (w + 1) * chunk)
                        if y_start >= y_stop:
                            continue
                        tasks.append((y_start, y_stop))

                    futures = [
                        executor.submit(
                            _poly_find_in_range,
                            occupancy,
                            rows,
                            canvas_width,
                            shape_w,
                            y_start,
                            y_stop,
                        )
                        for (y_start, y_stop) in tasks
                    ]
                    results = []
                    for fut in concurrent.futures.as_completed(futures):
                        res = fut.result()
                        if res is not None:
                            results.append(res)

                    if results:
                        y, x = min(results, key=lambda t: (t[0], t[1]))
                        best = (x, y, rows, shape_w, shape_h)
                        placed = True
                else:
                    for y in range(0, current_height + 1):
                        for x in range(0, canvas_width - shape_w + 1):
                            if _poly_fits_at(occupancy, x, y, rows):
                                best = (x, y, rows, shape_w, shape_h)
                                placed = True
                                break
                        if placed:
                            break
                if placed:
                    break

            if not placed:
                # Place at bottom-left with height expansion.
                rows, shape_w, shape_h = candidates[0]
                if shape_w > canvas_width:
                    canvas_width = shape_w
                y = len(occupancy)
                x = 0
                ensure_height(y + shape_h)
                best = (x, y, rows, shape_w, shape_h)

            x, y, rows, shape_w, shape_h = best
            ensure_height(y + shape_h)
            for i, row_len in enumerate(rows):
                if row_len <= 0:
                    continue
                mask = ((1 << row_len) - 1) << x
                occupancy[y + i] |= mask

            block.x = x
            block.y = y
            block.width = shape_w
            block.height = shape_h
            block.row_lengths = rows
            positioned.append(block)
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    while occupancy and occupancy[-1] == 0:
        occupancy.pop()

    total_width = max((row.bit_length() for row in occupancy), default=0)
    total_width = max(total_width, 1) if total_weights > 0 else 0
    total_height = len(occupancy)
    return total_width, total_height, positioned


def render_image(
    blocks: list[WeightBlock],
    width: int,
    height: int,
    background: tuple[int, int, int] = (20, 20, 20),
    color_method: str = "blackwhite",
) -> Image.Image:
    """Render the weight blocks to an image using vectorized numpy operations."""
    # Create RGB array
    canvas = np.full((height, width, 3), background, dtype=np.uint8)

    for block in tqdm(blocks, desc="Rendering blocks"):
        color = np.array(color_assigner.get_color(block.category), dtype=np.float32)

        # Create a 2D array for this block
        # Pad or truncate weights to fill the exact block dimensions
        full_size = block.width * block.height

        # Debug: check dimensions
        if full_size <= 0:
            print(f"Warning: block {block.name} has invalid dimensions: {block.width}x{block.height}")
            continue

        if len(block.weights) < full_size:
            padded = np.zeros(full_size, dtype=np.float32)
            padded[:len(block.weights)] = block.weights
        elif len(block.weights) > full_size:
            padded = np.asarray(block.weights[:full_size], dtype=np.float32)
        else:
            padded = np.asarray(block.weights, dtype=np.float32)

        # Ensure padded has the right size before reshape
        if padded.size != full_size:
            print(f"Error: padded size {padded.size} != full_size {full_size} for block {block.name}")
            print(f"Block dimensions: {block.width}x{block.height}, weights: {len(block.weights)}")
            continue

        # Check for invalid dimensions
        if block.height < 0 or block.width < 0:
            print(f"Error: block {block.name} has negative dimensions: {block.width}x{block.height}")
            continue

        # Reshape to 2D - use int() to ensure proper types
        try:
            block_2d = padded.reshape(int(block.height), int(block.width))
        except ValueError as e:
            print(f"Reshape error for block {block.name}: {e}")
            print(f"  padded.shape: {padded.shape}, padded.size: {padded.size}")
            print(f"  target shape: ({block.height}, {block.width}) = {block.height * block.width}")
            print(f"  block.height type: {type(block.height)}, block.width type: {type(block.width)}")
            raise

        t = block_2d[:, :, np.newaxis]  # Shape: (H, W, 1)

        if color_method == "opacity":
            # Opacity: 0 -> black, 1 -> full palette color.
            block_rgb = (t * color).astype(np.uint8)
        else:
            # Blackwhite: 0.5 = palette color, 0 = black, 1 = white.
            below_mask = t < 0.5

            # For below 0.5: t=0 -> black, t=0.5 -> color
            t_below = t * 2.0
            color_below = t_below * color

            # For above 0.5: t=0.5 -> color, t=1 -> white
            t_above = (t - 0.5) * 2.0
            color_above = color + t_above * (255.0 - color)

            block_rgb = np.where(below_mask, color_below, color_above).astype(np.uint8)

        # Place in canvas
        if block.row_lengths is None:
            y1, y2 = block.y, block.y + block.height
            x1, x2 = block.x, block.x + block.width

            # Clip to canvas bounds
            cy1, cy2 = max(0, y1), min(height, y2)
            cx1, cx2 = max(0, x1), min(width, x2)

            # Corresponding block region
            by1, by2 = cy1 - y1, block.height - (y2 - cy2)
            bx1, bx2 = cx1 - x1, block.width - (x2 - cx2)

            canvas[cy1:cy2, cx1:cx2] = block_rgb[by1:by2, bx1:bx2]
        else:
            for row_idx, row_len in enumerate(block.row_lengths):
                if row_len <= 0:
                    continue
                y = block.y + row_idx
                if y < 0 or y >= height:
                    continue
                x1 = block.x
                x2 = block.x + row_len
                cx1, cx2 = max(0, x1), min(width, x2)
                if cx1 >= cx2:
                    continue
                bx1 = cx1 - x1
                bx2 = bx1 + (cx2 - cx1)
                canvas[y, cx1:cx2] = block_rgb[row_idx, bx1:bx2]

    return Image.fromarray(canvas)


def render_framed_image(
    blocks: list[WeightBlock],
    width: int,
    height: int,
    model_name: str,
    total_params: int,
    scale: int,
    scale_method: str,
    padding: int,
    packing: str,
    frame_padding: int = 40,
    section_gap: int = 24,
    bottom_padding: int = 40,
    font_size: int = 20,
    line_spacing: int | None = None,
    inner_border_width: int = 0,
    inner_border_color: tuple[int, int, int] = (120, 120, 120),
    background: tuple[int, int, int] = (20, 20, 20),
    frame_color: tuple[int, int, int] = (40, 40, 40),
    text_color: tuple[int, int, int] = (200, 200, 200),
    color_method: str = "blackwhite",
) -> Image.Image:
    """Render the weight visualization with a frame, info text, and legend."""

    def font_for_size(size: int) -> ImageFont.ImageFont:
        for path in (
            # Prefer Fira Code (monospace) for all text.
            "/Library/Fonts/FiraCode-Regular.ttf",
            str(Path.home() / "Library/Fonts/FiraCode-Regular.ttf"),
            "/usr/share/fonts/truetype/firacode/FiraCode-Regular.ttf",
            "/usr/share/fonts/truetype/fira-code/FiraCode-Regular.ttf",
            # Fall back to other common monospace fonts.
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        ):
            try:
                if Path(path).exists():
                    return ImageFont.truetype(path, size)
            except (OSError, IOError):
                continue
        return ImageFont.load_default()

    def text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> int:
        left, _, right, _ = draw.textbbox((0, 0), text, font=font)
        return int(right - left)

    def text_height(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> int:
        _, top, _, bottom = draw.textbbox((0, 0), text, font=font)
        return int(bottom - top)

    def fit_font(
        draw: ImageDraw.ImageDraw,
        text: str,
        max_width: int,
        base_size: int,
        min_size: int,
    ) -> ImageFont.ImageFont:
        for size in range(base_size, min_size - 1, -1):
            font = font_for_size(size)
            if text_width(draw, text, font) <= max_width:
                return font
        return font_for_size(min_size)

    def ellipsize(
        draw: ImageDraw.ImageDraw,
        text: str,
        font: ImageFont.ImageFont,
        max_width: int,
    ) -> str:
        if text_width(draw, text, font) <= max_width:
            return text
        suffix = "…"
        lo, hi = 0, len(text)
        while lo < hi:
            mid = (lo + hi) // 2
            candidate = text[:mid].rstrip() + suffix
            if text_width(draw, candidate, font) <= max_width:
                lo = mid + 1
            else:
                hi = mid
        return (text[: max(0, lo - 1)].rstrip() + suffix) if lo > 0 else suffix

    def draw_centered_text(
        draw: ImageDraw.ImageDraw,
        y: int,
        text: str,
        font: ImageFont.ImageFont,
    ) -> None:
        w = text_width(draw, text, font)
        x = frame_padding + max(0, (content_width - w) // 2)
        draw.text((x, y), text, fill=text_color, font=font)

    def layout_legend_lines(
        draw: ImageDraw.ImageDraw,
        items: list[tuple[str, tuple[int, int, int]]],
        font: ImageFont.ImageFont,
        max_width: int,
        swatch: int,
        swatch_text_gap: int,
        gap: int,
    ) -> tuple[list[list[tuple[str, tuple[int, int, int], int]]], list[int], int]:
        if not items:
            return [], [], 0

        def entry_width(label: str) -> int:
            return swatch + swatch_text_gap + text_width(draw, label, font)

        n = len(items)

        def try_columns(cols: int) -> tuple[list[list[tuple[str, tuple[int, int, int], int]]], list[int], int] | None:
            cols = max(1, cols)
            rows = (n + cols - 1) // cols

            prepared: list[tuple[str, tuple[int, int, int], int]] = [(label, color, entry_width(label)) for label, color in items]

            col_widths = [0] * cols
            grid: list[list[tuple[str, tuple[int, int, int], int]]] = [[] for _ in range(rows)]
            for idx, entry in enumerate(prepared):
                r = idx // cols
                c = idx % cols
                grid[r].append(entry)
                col_widths[c] = max(col_widths[c], entry[2])

            total_w = sum(col_widths) + gap * (cols - 1)
            if total_w <= max_width:
                return grid, col_widths, total_w
            return None

        for cols in range(n, 0, -1):
            candidate = try_columns(cols)
            if candidate is not None:
                return candidate

        # Should be unreachable because cols=1 must fit after ellipsizing, but keep a safe fallback.
        prepared = [(label, color, entry_width(label)) for label, color in items]
        col_w = max((p[2] for p in prepared), default=0)
        return [[p] for p in prepared], [col_w], col_w

    # Render the main visualization
    main_image = render_image(blocks, width, height, background, color_method=color_method)

    # Calculate dimensions for frame, text, and legend
    frame_padding = max(0, int(frame_padding))
    section_gap = max(0, int(section_gap))
    bottom_padding = max(0, int(bottom_padding))
    font_size = max(6, int(font_size))
    if line_spacing is None:
        line_spacing = max(4, int(round(font_size * 0.35)))
    else:
        line_spacing = max(0, int(line_spacing))
    inner_border_width = max(0, int(inner_border_width))

    total_width = width + (frame_padding * 2)

    # Measure text and compute dynamic section heights before allocating final canvas
    measure_img = Image.new("RGB", (total_width, 10), frame_color)
    measure_draw = ImageDraw.Draw(measure_img)
    content_width = total_width - (frame_padding * 2)

    title_base_size = max(font_size + 2, int(round(font_size * 1.6)))
    title_min_size = max(10, int(round(font_size * 0.9)))
    legend_size = max(6, int(round(font_size * 0.8)))

    title_font = fit_font(measure_draw, model_name, content_width, base_size=title_base_size, min_size=title_min_size)
    info_font = font_for_size(font_size)
    legend_font = font_for_size(legend_size)

    title_h = text_height(measure_draw, "Ag", title_font)
    info_h = text_height(measure_draw, "Ag", info_font)

    params_str = f"{total_params:,} parameters"
    params_str = ellipsize(measure_draw, params_str, info_font, content_width)

    info_height = title_h + line_spacing + info_h

    # Get unique categories and their colors (for legend sizing)
    categories: dict[str, tuple[int, int, int]] = {}
    for block in blocks:
        if block.category not in categories:
            categories[block.category] = color_assigner.get_color(block.category)

    legend_items = []
    for category, color in sorted(categories.items()):
        display_name = category.replace("model.layers.N.", "").replace("model.", "")
        legend_items.append((display_name, color))

    legend_line_h = text_height(measure_draw, "Ag", legend_font)
    swatch_size = max(12, int(round(legend_line_h * 0.95)))
    swatch_text_gap = 12

    # Never truncate legend entries: if a single label doesn't fit even alone, shrink legend font.
    if legend_items:
        longest_label = max((label for label, _c in legend_items), key=len)
        legend_font = fit_font(
            measure_draw,
            longest_label,
            max(10, content_width - (swatch_size + swatch_text_gap)),
            base_size=legend_size,
            min_size=6,
        )
        legend_line_h = text_height(measure_draw, "Ag", legend_font)
        swatch_size = max(12, int(round(legend_line_h * 0.95)))

    legend_row_h = max(swatch_size, legend_line_h) + 8
    legend_gap = max(16, int(round(font_size * 1.2)))
    legend_bottom_padding = 6

    legend_lines, legend_col_widths, legend_total_w = layout_legend_lines(
        measure_draw,
        legend_items,
        legend_font,
        content_width,
        swatch=swatch_size,
        swatch_text_gap=swatch_text_gap,
        gap=legend_gap,
    )
    legend_rows = len(legend_lines)
    legend_height = (legend_rows * legend_row_h) + legend_bottom_padding

    total_height = (
        frame_padding
        + height
        + section_gap
        + info_height
        + section_gap
        + legend_height
        + bottom_padding
    )

    # Create the framed image
    framed = Image.new("RGB", (total_width, total_height), frame_color)

    # Paste the main visualization
    framed.paste(main_image, (frame_padding, frame_padding))

    # Draw text and legend
    draw = ImageDraw.Draw(framed)

    # Optional border around the main visualization
    if inner_border_width > 0:
        x1 = frame_padding
        y1 = frame_padding
        x2 = frame_padding + width - 1
        y2 = frame_padding + height - 1
        for i in range(inner_border_width):
            draw.rectangle([x1 + i, y1 + i, x2 - i, y2 - i], outline=inner_border_color, width=1)

    # Model info section (below the visualization)
    info_y = frame_padding + height + section_gap

    # Model name
    draw_centered_text(draw, info_y, ellipsize(draw, model_name, title_font, content_width), title_font)

    # Parameters and generation info
    info_line_y = info_y + title_h + line_spacing
    draw_centered_text(draw, info_line_y, params_str, info_font)

    # Legend section
    legend_y = info_y + info_height + section_gap

    def draw_legend_entry(x: int, y: int, label: str, color: tuple[int, int, int]) -> None:
        draw.rectangle([x, y, x + swatch_size, y + swatch_size], fill=color)
        bbox = draw.textbbox((0, 0), label, font=legend_font)
        text_h = bbox[3] - bbox[1]
        text_y = y + (swatch_size - text_h) // 2 - bbox[1]
        draw.text((x + swatch_size + swatch_text_gap, text_y), label, fill=text_color, font=legend_font)

    # Legend grid (centered as a block); columns align across rows and last row is not re-centered.
    legend_start_x = frame_padding + max(0, (content_width - legend_total_w) // 2)
    col_starts = []
    x = legend_start_x
    for i, w in enumerate(legend_col_widths):
        col_starts.append(x)
        x += w + (legend_gap if i < len(legend_col_widths) - 1 else 0)

    for row, line in enumerate(legend_lines):
        y = legend_y + (row * legend_row_h)
        for col, (label, color, _w) in enumerate(line):
            draw_legend_entry(col_starts[col], y, label, color)

    return framed


def main():
    def parse_rgb(value: str) -> tuple[int, int, int]:
        v = value.strip()
        if v.startswith("#"):
            v = v[1:]
        if len(v) == 6 and all(c in "0123456789abcdefABCDEF" for c in v):
            return (int(v[0:2], 16), int(v[2:4], 16), int(v[4:6], 16))
        parts = [p.strip() for p in value.split(",")]
        if len(parts) == 3 and all(p.isdigit() for p in parts):
            r, g, b = (int(p) for p in parts)
            if all(0 <= x <= 255 for x in (r, g, b)):
                return (r, g, b)
        raise argparse.ArgumentTypeError("Expected '#RRGGBB' or 'R,G,B' with 0-255 components")

    parser = argparse.ArgumentParser(
        description="Generate artistic visualizations of neural network weights."
    )
    parser.add_argument(
        "model",
        type=str,
        help="Model name or path (e.g., 'mlx-community/Llama-3.2-1B-Instruct-4bit')",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output PNG file path (default: derived from model name)",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=2,
        help="Padding between weight blocks (default: 2)",
    )
    parser.add_argument(
        "--shrink",
        type=int,
        default=None,
        help="Reduce image size by averaging NxN weight blocks into single pixels (default: auto for ~3000x3000)",
    )
    parser.add_argument(
        "--histogram",
        action="store_true",
        help="Show ASCII histogram of weight distribution",
    )
    parser.add_argument(
        "--scale-method",
        type=str,
        choices=["mean", "median", "max", "sample"],
        default="sample",
        help="Method for reducing pixels: mean (average), median (balanced), max (extremes), sample (first value)",
    )
    parser.add_argument(
        "--intensity-method",
        type=str,
        choices=["linear", "gamma", "smoothstep", "sigmoid"],
        default="linear",
        help="Intensity curve to spread normalized values (default: smoothstep)",
    )
    parser.add_argument(
        "--intensity-gamma",
        type=float,
        default=0.75,
        help="Gamma value for intensity curve when using --intensity-method gamma (default: 0.75)",
    )
    parser.add_argument(
        "--intensity-sigmoid-k",
        type=float,
        default=6.0,
        help="Steepness for sigmoid intensity curve (default: 6.0)",
    )
    parser.add_argument(
        "--intensity-clip-percent",
        type=float,
        default=0.01,
        help="Clip top/bottom N percentile before normalization (default: 0.01)",
    )
    parser.add_argument(
        "--palette",
        type=str,
        choices=sorted(PALETTES.keys()),
        default=DEFAULT_PALETTE_NAME,
        help=f"Color palette for layer categories (default: {DEFAULT_PALETTE_NAME})",
    )
    parser.add_argument(
        "--color-method",
        type=str,
        choices=["blackwhite", "opacity"],
        default="blackwhite",
        help="Color intensity mapping (default: blackwhite)",
    )
    parser.add_argument(
        "--polyomino-workers",
        type=int,
        default=os.cpu_count() or 1,
        help="Number of worker processes for polyomino packing (default: CPU count)",
    )
    parser.add_argument(
        "--packing",
        type=str,
        choices=["maxrects", "guillotine", "shelf", "polyomino"],
        default="polyomino",
        help="Block packing algorithm (default: polyomino)",
    )
    parser.add_argument(
        "--no-frame",
        action="store_true",
        help="Output just the visualization without frame/info/legend",
    )
    parser.add_argument(
        "--frame-padding",
        type=int,
        default=80,
        help="Padding around visualization inside the frame (default: 80)",
    )
    parser.add_argument(
        "--frame-section-gap",
        type=int,
        default=80,
        help="Vertical gap between footer sections (default: 80)",
    )
    parser.add_argument(
        "--frame-bottom-padding",
        type=int,
        default=40,
        help="Bottom padding below the legend (default: 40)",
    )
    parser.add_argument(
        "--frame-font-size",
        type=int,
        default=60,
        help="Base font size for all frame text (default: 60)",
    )
    parser.add_argument(
        "--frame-line-spacing",
        type=int,
        default=None,
        help="Vertical spacing between footer text lines (default: 0.35 * font size)",
    )
    parser.add_argument(
        "--frame-bg",
        type=parse_rgb,
        default=(0, 0, 0),
        help="Frame background color, '#RRGGBB' or 'R,G,B' (default: 0,0,0)",
    )
    parser.add_argument(
        "--inner-border-width",
        type=int,
        default=5,
        help="Border width around the weights visualization in pixels (default: 5)",
    )
    parser.add_argument(
        "--inner-border-color",
        type=parse_rgb,
        default=(255, 255, 255),
        help="Border color around the weights visualization, '#RRGGBB' or 'R,G,B' (default: 255,255,255)",
    )

    args = parser.parse_args()

    if args.output is None:
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", args.model).strip("._-")
        if not safe_name:
            safe_name = "weights"
        args.output = f"{safe_name}.png"

    palette = PALETTES[args.palette]
    if len(palette) < 16:
        raise SystemExit("Selected palette must have at least 16 colors.")
    color_assigner.set_palette(palette)

    print(f"Loading model: {args.model}")
    model, tokenizer = load(args.model)

    print("Extracting weights...")
    weights = extract_weights(model)
    print(f"Found {len(weights)} weight tensors")

    total_params = sum(np.prod(w.shape) for _, w in weights)
    print(f"Total parameters: {total_params:,}")

    if args.shrink is None:
        target_pixels = 3000 * 3000
        args.shrink = max(1, int(round(np.sqrt(total_params / target_pixels))))
        print(f"Auto shrink: {args.shrink} (target ~3000x3000)")

    blocks = create_weight_blocks(
        weights,
        scale=args.shrink,
        scale_method=args.scale_method,
        intensity_method=args.intensity_method,
        intensity_gamma=args.intensity_gamma,
        intensity_sigmoid_k=args.intensity_sigmoid_k,
        intensity_clip_percent=args.intensity_clip_percent,
        show_histogram=args.histogram,
    )

    packers = {
        "maxrects": pack_blocks_maxrects,
        "guillotine": pack_blocks_guillotine,
        "shelf": pack_blocks_shelf,
        "polyomino": lambda b, padding: pack_blocks_polyomino(
            b,
            padding=padding,
            workers=max(1, args.polyomino_workers),
        ),
    }
    width, height, positioned_blocks = packers[args.packing](blocks, padding=args.padding)
    print(f"Canvas size: {width} x {height} pixels")

    if args.no_frame:
        img = render_image(positioned_blocks, width, height)
    else:
        img = render_framed_image(
            positioned_blocks,
            width,
            height,
            model_name=args.model,
            total_params=total_params,
            scale=args.shrink,
            scale_method=args.scale_method,
            padding=args.padding,
            packing=args.packing,
            frame_padding=args.frame_padding,
            section_gap=args.frame_section_gap,
            bottom_padding=args.frame_bottom_padding,
            font_size=args.frame_font_size,
            line_spacing=args.frame_line_spacing,
            frame_color=args.frame_bg,
            inner_border_width=args.inner_border_width,
            inner_border_color=args.inner_border_color,
            color_method=args.color_method,
        )

    output_path = Path(args.output)
    img.save(output_path)
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
