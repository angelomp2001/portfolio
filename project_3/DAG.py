import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_dag():
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Define Nodes
    nodes = {
        'Load Data': (0.5, 0.9),
        'Inspect Initial Data': (0.5, 0.75),
        'Type Conversion': (0.5, 0.6),
        'Fill Missing': (0.5, 0.45),
        'Deduplicate': (0.5, 0.3),
        'Enrich Data': (0.5, 0.15),
        'Aggregate Revenue': (0.8, 0.45),
        'Hypothesis Testing': (0.8, 0.3),
        'Visualize Results': (0.8, 0.15),
    }

    # Draw Nodes
    for label, (x, y) in nodes.items():
        box = patches.FancyBboxPatch((x - 0.12, y - 0.05), 0.24, 0.1, boxstyle="round,pad=0.02", ec="black", fc="lightblue")
        ax.add_patch(box)
        plt.text(x, y, label, ha="center", va="center", fontsize=10, weight='bold')

    # Define Edges as pairs of connected nodes
    edges = [
        ('Load Data', 'Inspect Initial Data'),
        ('Inspect Initial Data', 'Type Conversion'),
        ('Type Conversion', 'Fill Missing'),
        ('Fill Missing', 'Deduplicate'),
        ('Deduplicate', 'Enrich Data'),
        ('Enrich Data', 'Aggregate Revenue'),
        ('Aggregate Revenue', 'Hypothesis Testing'),
        ('Aggregate Revenue', 'Visualize Results')
    ]

    # Draw Edges
    for start, end in edges:
        start_pos = nodes[start]
        end_pos = nodes[end]
        plt.annotate("",
                     xy=(end_pos[0], end_pos[1] + 0.06 if end_pos[1] < start_pos[1] else end_pos[1] - 0.06),
                     xytext=(start_pos[0], start_pos[1] - 0.06 if start_pos[1] > end_pos[1] else start_pos[1] + 0.06),
                     arrowprops=dict(arrowstyle="->", lw=1.5, color='gray'))

    ax.axis('off')
    plt.title("Project 3 Component DAG Diagram", fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig('docs/component_diagram.png', dpi=300)
    print("Saved DAG diagram to 'docs/component_diagram.png'")

if __name__ == '__main__':
    draw_dag()
