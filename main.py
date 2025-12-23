#!/usr/bin/env python3
"""
OpenWeights.art - Artistic visualization of neural network weights.

Creates poster-style visualizations of model weights where each pixel
represents a single weight value, with intensity based on value range
and color based on the type of layer.
"""

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load
from PIL import Image
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


# Wong colorblind-friendly palette (RGB)
# From Bang Wong, Nature Methods 8, 441 (2011)
WONG_PALETTE = [
    (230, 159, 0),    # Orange
    (86, 180, 233),   # Sky blue
    (0, 158, 115),    # Bluish green
    (240, 228, 66),   # Yellow
    (0, 114, 178),    # Blue
    (213, 94, 0),     # Vermillion
    (204, 121, 167),  # Reddish purple
]


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
    """Assigns colors to weight categories dynamically using the Wong palette."""

    def __init__(self):
        self.category_to_color: dict[str, tuple[int, int, int]] = {}
        self.next_color_idx = 0

    def get_color(self, category: str) -> tuple[int, int, int]:
        """Get a color for a category, assigning a new one if needed."""
        if category not in self.category_to_color:
            color = WONG_PALETTE[self.next_color_idx % len(WONG_PALETTE)]
            self.category_to_color[category] = color
            self.next_color_idx += 1
        return self.category_to_color[category]


# Global color assigner instance
color_assigner = ColorAssigner()


def categorize_weight(name: str) -> str:
    """Categorize a weight tensor based on its normalized name."""
    return normalize_weight_name(name)


def dequantize_weight(weight: mx.array, scales: mx.array, biases: mx.array, bits: int = 4) -> mx.array:
    """Dequantize a packed weight tensor."""
    values_per_int = 32 // bits
    mask = (1 << bits) - 1

    out_features = weight.shape[0]
    packed_in = weight.shape[1]
    in_features = packed_in * values_per_int

    # Unpack all values at once using vectorized operations
    shifts = mx.array([i * bits for i in range(values_per_int)])
    # Broadcast: weight is (out, packed), shifts is (values_per_int,)
    # Result: (out, packed, values_per_int)
    unpacked = (weight[:, :, None] >> shifts) & mask

    # Convert to signed
    mid = 1 << (bits - 1)
    unpacked = unpacked.astype(mx.float32) - mid

    # Reshape to (out_features, in_features)
    unpacked = unpacked.reshape(out_features, in_features)

    # Apply scales and biases
    group_size = in_features // scales.shape[1]
    scales_expanded = mx.repeat(scales, group_size, axis=1)
    biases_expanded = mx.repeat(biases, group_size, axis=1)

    return unpacked * scales_expanded + biases_expanded


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
            if "weight" in value and "scales" in value and "biases" in value:
                try:
                    w = value["weight"]
                    s = value["scales"]
                    b = value["biases"]
                    bits = value.get("bits", 4)
                    if isinstance(bits, mx.array):
                        bits = int(bits.item())
                    dequantized = dequantize_weight(w, s, b, bits)
                    weights.append((name, dequantized))
                except Exception as e:
                    print(f"  Warning: Could not dequantize {name}: {e}")
                    for k, v in value.items():
                        add_weight(f"{name}.{k}", v)
            else:
                for k, v in value.items():
                    add_weight(f"{name}.{k}", v)

    def traverse(obj, prefix=""):
        if hasattr(obj, "parameters"):
            params = obj.parameters()
            # Check if this is a quantized layer (has weight, scales, biases at top level)
            if "weight" in params and "scales" in params and "biases" in params:
                add_weight(prefix, params)
            else:
                for name, param in params.items():
                    full_name = f"{prefix}.{name}" if prefix else name
                    add_weight(full_name, param)

        if hasattr(obj, "children"):
            for name, child in obj.children().items():
                full_name = f"{prefix}.{name}" if prefix else name
                traverse(child, full_name)

        if hasattr(obj, "__iter__") and not isinstance(obj, (str, dict, mx.array)):
            try:
                for i, item in enumerate(obj):
                    if hasattr(item, "parameters") or hasattr(item, "children"):
                        traverse(item, f"{prefix}.{i}" if prefix else str(i))
            except TypeError:
                pass

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
    arr = np.array(tensor).flatten().astype(np.float32)
    return arr


def compute_global_min_max(weights: list[tuple[str, mx.array]]) -> tuple[float, float]:
    """Compute the global min and max across all weight tensors."""
    global_min = float('inf')
    global_max = float('-inf')

    for name, tensor in tqdm(weights, desc="Computing global range"):
        arr = tensor_to_float32(tensor)
        tensor_min, tensor_max = float(arr.min()), float(arr.max())
        global_min = min(global_min, tensor_min)
        global_max = max(global_max, tensor_max)

    return global_min, global_max


def normalize_tensor(tensor: mx.array, global_min: float, global_max: float) -> np.ndarray:
    """Normalize a tensor to [0, 1] range using global min/max."""
    arr = tensor_to_float32(tensor)

    if global_max > global_min:
        return (arr - global_min) / (global_max - global_min)
    return np.zeros_like(arr)


def reduce_weights(weights: np.ndarray, scale: int) -> np.ndarray:
    """Reduce weights by averaging scale x scale blocks into single pixels."""
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

    # Reshape to group scale x scale blocks and take mean
    reduced_height = new_height // scale
    reduced_width = new_width // scale
    reduced = grid.reshape(reduced_height, scale, reduced_width, scale).mean(axis=(1, 3))

    return reduced.flatten()


def create_weight_blocks(weights: list[tuple[str, mx.array]], scale: int = 1) -> list[WeightBlock]:
    """Create WeightBlock objects from weight tensors."""
    # Compute global min/max for normalization
    global_min, global_max = compute_global_min_max(weights)
    print(f"Global weight range: [{global_min:.6f}, {global_max:.6f}]")

    blocks = []

    for name, tensor in tqdm(weights, desc="Creating blocks"):
        normalized = normalize_tensor(tensor, global_min, global_max)
        reduced = reduce_weights(normalized, scale)

        category = categorize_weight(name)
        # Dimensions will be set during packing, so use placeholder values
        width, height = 1, len(reduced)

        blocks.append(
            WeightBlock(
                name=name,
                weights=reduced,
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


def render_image(
    blocks: list[WeightBlock],
    width: int,
    height: int,
    background: tuple[int, int, int] = (20, 20, 20),
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

        # Apply color: multiply intensity by color
        block_rgb = (block_2d[:, :, np.newaxis] * color).astype(np.uint8)

        # Place in canvas
        y1, y2 = block.y, block.y + block.height
        x1, x2 = block.x, block.x + block.width

        # Clip to canvas bounds
        cy1, cy2 = max(0, y1), min(height, y2)
        cx1, cx2 = max(0, x1), min(width, x2)

        # Corresponding block region
        by1, by2 = cy1 - y1, block.height - (y2 - cy2)
        bx1, bx2 = cx1 - x1, block.width - (x2 - cx2)

        canvas[cy1:cy2, cx1:cx2] = block_rgb[by1:by2, bx1:bx2]

    return Image.fromarray(canvas)


def main():
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
        default="weights.png",
        help="Output PNG file path (default: weights.png)",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=2,
        help="Padding between weight blocks (default: 2)",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=1,
        help="Reduce image size by averaging NxN weight blocks into single pixels (default: 1, no reduction)",
    )

    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model, tokenizer = load(args.model)

    print("Extracting weights...")
    weights = extract_weights(model)
    print(f"Found {len(weights)} weight tensors")

    total_params = sum(np.prod(np.array(w).shape) for _, w in weights)
    print(f"Total parameters: {total_params:,}")

    blocks = create_weight_blocks(weights, scale=args.scale)

    width, height, positioned_blocks = pack_blocks_maxrects(blocks, padding=args.padding)
    print(f"Canvas size: {width} x {height} pixels")

    img = render_image(positioned_blocks, width, height)

    output_path = Path(args.output)
    img.save(output_path)
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
