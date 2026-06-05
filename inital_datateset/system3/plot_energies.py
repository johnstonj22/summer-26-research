import re
import matplotlib.pyplot as plt

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

# Plot
plt.figure()
plt.plot(steps, energies)
plt.xlabel("Step")
plt.ylabel("Energy (a.u.)")
plt.title("Energy vs Step")
#plt.xlim(2000, 10000)
plt.grid()

#Save the plot
plt.savefig("energy_vs_step.png", dpi=300)
#plt.show()
