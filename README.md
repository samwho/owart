# owart

Generates posters based on neural network weights.

Pick an MLX-compatible model and this script will iterate over those weights
and do the following:

- Find the global minimum and maximum weight values.
- Normalize weights to [0, 1].
- Trim the upper and lower percentiles to reduce outlier impact.
- Map normalized weights to colors based on intensity.
- Pack weight matrices into a visually appealing layout.
- Render the packed layout into a high-resolution image with a legend and metadata.

## Requirements

- Python 3.13+
- MLX on Apple Silicon (for `mlx` / `mlx-lm`)

Dependencies are declared in `pyproject.toml`.

## Usage

Basic:

```bash
uvx owart mlx-community/Llama-3.2-1B-Instruct-4bit
```

## Options

- `--packing {polyomino,maxrects,guillotine,shelf}`: packing algorithm (default: polyomino).
- `--polyomino-workers N`: parallel search workers for polyomino packing (default: CPU count).
- `--palette {tab20,tab20b,tab20c}`: color palette (20 colors).
- `--color-method {blackwhite,opacity}`: how intensity maps to color.
- `--shrink N` / `--scale-method {mean,median,max,sample}`: image reduction (shrink auto-computed if omitted).
- `--intensity-method {linear,gamma,smoothstep,sigmoid}`: intensity curve (default: smoothstep).
- `--intensity-clip-percent P`: clamp top/bottom P percentile before normalization (default: 0.05).
- `--frame-padding N`: padding around the visualization.
- `--frame-section-gap N`: gap between footer sections.
- `--frame-bottom-padding N`: space below the legend.
- `--frame-font-size N`: base font size for footer/legend.
- `--frame-line-spacing N`: line spacing (default: 0.35 * font size).
- `--frame-bg #RRGGBB` or `R,G,B`: frame background color.
- `--inner-border-width N`: border width around the central visual.
- `--inner-border-color #RRGGBB` or `R,G,B`: border color.
- `--no-frame`: output just the visualization.

## Example

```bash
uvx owart mlx-community/Llama-3.2-1B-Instruct-4bit \
  --packing polyomino --polyomino-workers 8 \
  --palette tab20 --color-method blackwhite \
  --intensity-method smoothstep --intensity-clip-percent 0.05 \
  --frame-padding 80 --frame-font-size 60 --frame-section-gap 80 \
  --frame-bg "#000000" \
  --inner-border-width 5 --inner-border-color "#FFFFFF"
```
