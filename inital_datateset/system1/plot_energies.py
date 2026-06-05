import re
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator

font_axis_publish = {
    'color': 'black',
    'size': 20,
}

filename = "silica-pos-1.xyz"

steps = []
energies = []

pattern = re.compile(r"i\s*=\s*(\d+).*E\s*=\s*([-+]?\d*\.\d+|\d+)")

with open(filename, "r") as f:
    for line in f:
        match = pattern.search(line)
        if match:
            steps.append(int(match.group(1)))
            energies.append(float(match.group(2)))

steps = np.array(steps)
energies = np.array(energies)

# Create figure
plt.figure(figsize=(12, 10))

# Plot data
plt.plot(steps, energies, color='blue', linewidth=2.5)

# Labels
plt.xlabel("Step", fontdict=font_axis_publish)
plt.ylabel("Energy (Hartree)", fontdict=font_axis_publish)

# Tick styling
plt.tick_params(axis='both', which='major', direction='in', labelsize=18)
plt.tick_params(which='minor', length=4, width=1, direction='in',
                top=True, labeltop=False, bottom=True,
                right=True, labelright=False)
plt.tick_params(which='major', length=8, width=2, direction='in',
                top=True, labeltop=False, bottom=True,
                right=True, labelright=False)

# Spine thickness
for axis in ['top', 'bottom', 'left', 'right']:
    plt.gca().spines[axis].set_linewidth(2)

# Minor ticks
plt.gca().xaxis.set_minor_locator(AutoMinorLocator())
plt.gca().yaxis.set_minor_locator(AutoMinorLocator())

# Optional axis limits
plt.xlim(np.min(steps), np.max(steps))
# plt.ylim(np.min(energies)-0.1, np.max(energies)+0.1)

# Save plot
plt.savefig("energy_vs_step2.pdf", format='pdf', dpi=600, bbox_inches='tight')
plt.show()