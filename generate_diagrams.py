
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines

def draw_box(ax, x, y, width, height, text, color='#E3F2FD', edge='#1565C0', fontsize=10):
    # Shadow
    shadow = patches.FancyBboxPatch((x+0.05, y-0.05), width, height, boxstyle="round,pad=0.2", 
                                   facecolor='gray', alpha=0.3, zorder=1)
    ax.add_patch(shadow)
    # Box
    box = patches.FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.2", 
                                facecolor=color, edgecolor=edge, linewidth=2, zorder=2)
    ax.add_patch(box)
    # Text
    ax.text(x + width/2, y + height/2, text, ha='center', va='center', 
            fontsize=fontsize, color='#333333', fontweight='bold', zorder=3)
    return x + width/2, y, x + width/2, y + height

def draw_arrow(ax, x1, y1, x2, y2, color='#333333'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1), 
                arrowprops=dict(arrowstyle='->', linewidth=2, color=color, shrinkA=0, shrinkB=0), zorder=1)

def create_single_agent_diagram():
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Title
    ax.text(5, 11.5, "Single Agent Workflow (Linear)", ha='center', fontsize=16, fontweight='bold', color='#1565C0')

    # Dimensions
    w, h = 4, 1.2
    x = 3
    
    # Nodes
    _, _, b1_out_x, b1_out_y = draw_box(ax, x, 10, w, h, "1. Input Topic", color='#E1F5FE', edge='#0277BD')
    _, b2_in_y, b2_out_x, b2_out_y = draw_box(ax, x, 8, w, h, "2. Retrieval (ArXiv)\n(Fetches Top 5)", color='#B3E5FC', edge='#0277BD')
    _, b3_in_y, b3_out_x, b3_out_y = draw_box(ax, x, 6, w, h, "3. LLM Generation\n(Single Prompt)", color='#81D4FA', edge='#0277BD')
    _, b4_in_y, b4_out_x, b4_out_y = draw_box(ax, x, 4, w, h, "4. Self-Correction\n(Basic Loop)", color='#4FC3F7', edge='#0277BD')
    _, b5_in_y, _, _ = draw_box(ax, x, 1, w, h, "5. Final Output", color='#29B6F6', edge='#01579B')

    # Arrows
    draw_arrow(ax, 5, 10, 5, 9.4)  # 1 -> 2
    draw_arrow(ax, 5, 8, 5, 7.4)   # 2 -> 3
    draw_arrow(ax, 5, 6, 5, 5.4)   # 3 -> 4
    draw_arrow(ax, 5, 4, 5, 2.4)   # 4 -> 5
    
    # Feedback Loop Arrow
    ax.annotate('', xy=(7.2, 6.6), xytext=(7.2, 4.6), 
                arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=-1.5", color='#D32F2F', linewidth=2, linestyle='dashed'))
    ax.text(8.5, 5.6, "Refine Draft", ha='center', color='#D32F2F', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig('single_agent_workflow.png', dpi=300, bbox_inches='tight')
    print("Redesigned single_agent_workflow.png")

def create_multi_agent_diagram():
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(7, 9.5, "Multi-Agent System Architecture (MAS)", ha='center', fontsize=18, fontweight='bold', color='#2E7D32')

    # Styles
    planner_style = {'color': '#C8E6C9', 'edge': '#2E7D32'}
    writer_style = {'color': '#FFF9C4', 'edge': '#FBC02D'}
    critic_style = {'color': '#FFCCBC', 'edge': '#D84315'}
    
    # 1. Retrieval
    draw_box(ax, 0.5, 4, 2, 2, "Retrieval\n(10+ Papers)", **planner_style)
    
    # 2. Planner & Outliner
    draw_box(ax, 3.5, 6, 2.5, 1.5, "Planner Agent\n(Themes)", **planner_style)
    draw_box(ax, 3.5, 2.5, 2.5, 1.5, "Outliner Agent\n(Structure)", **planner_style)
    
    # 3. Parallel Writers (Stacked)
    ax.text(8, 7.8, "Parallel Execution", ha='center', fontsize=10, fontweight='bold', color='#F57F17')
    # Draw background box to group them
    bg = patches.FancyBboxPatch((6.8, 1.8), 2.4, 5.4, boxstyle="round,pad=0.1", 
                               facecolor='#FFFDE7', edgecolor='#FBC02D', linestyle='--', linewidth=1, zorder=0)
    ax.add_patch(bg)
    
    draw_box(ax, 7, 6, 2, 1, "Writer 1", **writer_style)
    draw_box(ax, 7, 4, 2, 1, "Writer 2", **writer_style)
    draw_box(ax, 7, 2, 2, 1, "Writer 3", **writer_style)
    
    # 4. Critic
    draw_box(ax, 10.5, 4, 2, 2, "Critic Agent\n(Strict Grading)", **critic_style)
    
    # 5. Output
    draw_box(ax, 10.5, 0.5, 2, 1.5, "Final Assembly", color='#E0F2F1', edge='#00695C')

    # Arrows
    # Retrieval -> Planner
    draw_arrow(ax, 2.5, 5, 3.5, 6.75)
    # Planner -> Outliner
    draw_arrow(ax, 4.75, 6, 4.75, 4)
    
    # Outliner -> Writers
    draw_arrow(ax, 6, 3.25, 7, 6.5)
    draw_arrow(ax, 6, 3.25, 7, 4.5)
    draw_arrow(ax, 6, 3.25, 7, 2.5)
    
    # Writers -> Critic
    draw_arrow(ax, 9, 6.5, 10.5, 5)
    draw_arrow(ax, 9, 4.5, 10.5, 5)
    draw_arrow(ax, 9, 2.5, 10.5, 5)
    
    # Critic Loop (Reject)
    ax.annotate('', xy=(8, 8.2), xytext=(11.5, 6.2), 
                arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=-0.5", color='#D32F2F', linewidth=2, linestyle='dashed'))
    ax.text(9.5, 8.5, "Reject (< 8/10)\n(Feedback Loop)", ha='center', color='#D32F2F', fontweight='bold')
    
    # Critic -> Output (Accept)
    draw_arrow(ax, 11.5, 4, 11.5, 2)
    ax.text(11.8, 3, "Accept", ha='left', color='#2E7D32', fontweight='bold')

    plt.tight_layout()
    plt.savefig('multi_agent_workflow.png', dpi=300, bbox_inches='tight')
    print("Redesigned multi_agent_workflow.png")

if __name__ == "__main__":
    create_single_agent_diagram()
    create_multi_agent_diagram()
