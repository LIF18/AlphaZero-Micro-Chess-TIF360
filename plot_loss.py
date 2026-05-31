import matplotlib.pyplot as plt
import seaborn as sns

# Real training data extracted from slurm-204956.out log
steps = [
    50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 
    550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050, 1100
]
losses = [
    17.92, 16.60, 15.37, 14.16, 13.06, 11.99, 10.99, 10.07, 9.279, 8.62,
    8.056, 7.595, 7.320, 7.048, 6.853, 6.736, 6.633, 6.562, 6.502, 6.510, 6.498, 6.470
]

sns.set_theme(style="whitegrid")
plt.figure(figsize=(8, 5), dpi=300) # 300 dpi

plt.plot(steps, losses, marker='o', markersize=5, linewidth=2.5, color='#d62728', label='Training Loss')

plt.title('Value Head & Policy Distillation Loss', fontsize=16, fontweight='bold', pad=15)
plt.xlabel('Training Steps', fontsize=14)
plt.ylabel('Multi-task Loss (MSE + CE)', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.legend(fontsize=12)

plt.tight_layout()
plt.savefig('loss_curve_for_poster.png', format='png', bbox_inches='tight')

print("High-resolution loss curve plot generated: loss_curve_for_poster.png")