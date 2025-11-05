"""
Interactive example for creating marbling art.

This script demonstrates how to use the marbling simulation interactively,
allowing users to add dyes and apply comb tools with different patterns.
"""

import numpy as np
import matplotlib.pyplot as plt
from marbling import MarblingSimulation


def example_basic():
    """Basic example: Create a simple marbling pattern."""
    print("Example 1: Basic marbling pattern")
    print("-" * 50)
    
    # Create simulation
    sim = MarblingSimulation(width=128, height=128, viscosity=0.00005)
    
    # Add some dye drops
    sim.add_dye_drop(0.3, 0.5, color=(1.0, 0.0, 0.0), num_particles=1000, radius=0.08)  # Red
    sim.add_dye_drop(0.5, 0.5, color=(0.0, 0.0, 1.0), num_particles=1000, radius=0.08)  # Blue
    sim.add_dye_drop(0.7, 0.5, color=(1.0, 1.0, 0.0), num_particles=1000, radius=0.08)  # Yellow
    
    # Let dyes spread
    print("Letting dyes spread...")
    for _ in range(20):
        sim.step()
    
    # Apply horizontal comb stroke
    print("Applying horizontal comb stroke...")
    sim.apply_comb((0.1, 0.5), (0.9, 0.5), strength=10.0)
    
    # Continue simulation
    for _ in range(30):
        sim.step()
    
    # Visualize result
    fig, ax = plt.subplots(figsize=(8, 8))
    sim.visualize(ax)
    plt.title("Basic Marbling Pattern")
    plt.savefig('example_basic.png', dpi=150, bbox_inches='tight')
    print("Saved to example_basic.png\n")
    

def example_spiral():
    """Create a spiral marbling pattern."""
    print("Example 2: Spiral marbling pattern")
    print("-" * 50)
    
    sim = MarblingSimulation(width=128, height=128, viscosity=0.00005)
    
    # Add concentric rings of different colors
    colors = [
        (0.9, 0.1, 0.1),  # Red
        (0.1, 0.9, 0.1),  # Green
        (0.1, 0.1, 0.9),  # Blue
        (0.9, 0.9, 0.1),  # Yellow
    ]
    
    for i, color in enumerate(colors):
        radius = 0.05 + i * 0.05
        angle_offset = i * np.pi / 4
        for angle in np.linspace(0, 2 * np.pi, 8):
            x = 0.5 + radius * np.cos(angle + angle_offset)
            y = 0.5 + radius * np.sin(angle + angle_offset)
            sim.add_dye_drop(x, y, color, num_particles=500, radius=0.04)
    
    # Let dyes spread
    print("Letting dyes spread...")
    for _ in range(15):
        sim.step()
    
    # Apply spiral comb strokes
    print("Applying spiral comb strokes...")
    num_points = 20
    for i in range(num_points - 1):
        t1 = i / num_points * 2 * np.pi
        t2 = (i + 1) / num_points * 2 * np.pi
        r1 = 0.1 + 0.3 * (i / num_points)
        r2 = 0.1 + 0.3 * ((i + 1) / num_points)
        
        x1 = 0.5 + r1 * np.cos(t1)
        y1 = 0.5 + r1 * np.sin(t1)
        x2 = 0.5 + r2 * np.cos(t2)
        y2 = 0.5 + r2 * np.sin(t2)
        
        sim.apply_comb((x1, y1), (x2, y2), strength=5.0)
        for _ in range(2):
            sim.step()
    
    # Continue simulation
    for _ in range(20):
        sim.step()
    
    # Visualize
    fig, ax = plt.subplots(figsize=(8, 8))
    sim.visualize(ax)
    plt.title("Spiral Marbling Pattern")
    plt.savefig('example_spiral.png', dpi=150, bbox_inches='tight')
    print("Saved to example_spiral.png\n")


def example_chevron():
    """Create a chevron/zigzag marbling pattern."""
    print("Example 3: Chevron marbling pattern")
    print("-" * 50)
    
    sim = MarblingSimulation(width=128, height=128, viscosity=0.00005)
    
    # Add stripes of dye
    colors = [
        (0.8, 0.2, 0.3),  # Red-pink
        (0.2, 0.5, 0.8),  # Blue
        (0.9, 0.8, 0.2),  # Gold
        (0.3, 0.7, 0.4),  # Green
    ]
    
    for i, color in enumerate(colors):
        y = 0.2 + i * 0.15
        for x in np.linspace(0.1, 0.9, 7):
            sim.add_dye_drop(x, y, color, num_particles=600, radius=0.05)
    
    # Let dyes spread
    print("Letting dyes spread...")
    for _ in range(15):
        sim.step()
    
    # Apply zigzag comb strokes
    print("Applying zigzag comb strokes...")
    num_zigs = 8
    for i in range(num_zigs):
        y1 = 0.1 + (i / num_zigs) * 0.8
        y2 = 0.1 + ((i + 1) / num_zigs) * 0.8
        
        if i % 2 == 0:
            sim.apply_comb((0.2, y1), (0.8, y2), strength=8.0)
        else:
            sim.apply_comb((0.8, y1), (0.2, y2), strength=8.0)
        
        for _ in range(3):
            sim.step()
    
    # Continue simulation
    for _ in range(25):
        sim.step()
    
    # Visualize
    fig, ax = plt.subplots(figsize=(8, 8))
    sim.visualize(ax)
    plt.title("Chevron Marbling Pattern")
    plt.savefig('example_chevron.png', dpi=150, bbox_inches='tight')
    print("Saved to example_chevron.png\n")


def example_custom():
    """Example showing how to create custom patterns programmatically."""
    print("Example 4: Custom pattern with multiple comb passes")
    print("-" * 50)
    
    sim = MarblingSimulation(width=128, height=128, viscosity=0.00005)
    
    # Create a rainbow gradient
    colors = [
        (0.9, 0.0, 0.0),  # Red
        (0.9, 0.5, 0.0),  # Orange
        (0.9, 0.9, 0.0),  # Yellow
        (0.0, 0.9, 0.0),  # Green
        (0.0, 0.5, 0.9),  # Blue
        (0.5, 0.0, 0.9),  # Purple
    ]
    
    for i, color in enumerate(colors):
        x = 0.15 + i * 0.12
        for y in np.linspace(0.2, 0.8, 5):
            sim.add_dye_drop(x, y, color, num_particles=500, radius=0.04)
    
    # Spread
    print("Letting dyes spread...")
    for _ in range(10):
        sim.step()
    
    # First pass: horizontal combs
    print("First comb pass (horizontal)...")
    for y in np.linspace(0.2, 0.8, 5):
        sim.apply_comb((0.1, y), (0.9, y), strength=7.0)
        for _ in range(3):
            sim.step()
    
    # Second pass: vertical combs
    print("Second comb pass (vertical)...")
    for x in np.linspace(0.25, 0.75, 4):
        sim.apply_comb((x, 0.1), (x, 0.9), strength=5.0)
        for _ in range(3):
            sim.step()
    
    # Third pass: diagonal
    print("Third comb pass (diagonal)...")
    sim.apply_comb((0.1, 0.9), (0.9, 0.1), strength=3.0)
    
    # Final simulation
    for _ in range(30):
        sim.step()
    
    # Visualize
    fig, ax = plt.subplots(figsize=(8, 8))
    sim.visualize(ax)
    plt.title("Custom Multi-Pass Marbling")
    plt.savefig('example_custom.png', dpi=150, bbox_inches='tight')
    print("Saved to example_custom.png\n")


if __name__ == "__main__":
    print("=" * 50)
    print("Marbling Art Examples")
    print("=" * 50)
    print()
    
    # Run all examples
    example_basic()
    example_spiral()
    example_chevron()
    example_custom()
    
    print("=" * 50)
    print("All examples complete!")
    print("Images saved: example_basic.png, example_spiral.png,")
    print("              example_chevron.png, example_custom.png")
    print("=" * 50)
    
    # Show all figures
    plt.show()
