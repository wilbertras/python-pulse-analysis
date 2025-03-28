import matplotlib.colors as mcolors

def redefine_colors():
    # Redefine color shortcuts
    mcolors.ColorConverter.colors['b'] = '#0072BD'  # Change 'b' (blue) to a new color
    mcolors.ColorConverter.colors['r'] = '#D95319'  # Change 'r' (red) to a new color
    mcolors.ColorConverter.colors['g'] = '#77AC30'  # Change 'g' (green) to a new color
    mcolors.ColorConverter.colors['y'] = '#EDB120'  # Change 'y' (yellow) to a new color
    mcolors.ColorConverter.colors['m'] = '#7E2F8E'  # Change 'm' (magenta) to a new color
    mcolors.ColorConverter.colors['c'] = '#4DBEEE'  # Change 'c' (cyan) to a new color
    mcolors.ColorConverter.colors['k'] = '#A2142F'  # Change 'k' (black) to a new color
    mcolors.ColorConverter.colors['w'] = '#FFFFFF'  # Change 'w' (white) to a new color

if __name__ != "__main__":
    redefine_colors()