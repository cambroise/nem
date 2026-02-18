"""Generate a Potts image dataset: 30x30 grid, 3 classes, bivariate Gaussian emissions."""

from generate import PottsImageData
import matplotlib.pyplot as plt

potts = PottsImageData(
    nl=30,
    nc=30,
    k=3,
    beta=0.8,
    centers=[[0, 0], [4, 4], [0, 4]],
    sigma=1.0,
    seed=42,
)

potts.export("potts_30x30_3")
potts.plot()

plt.savefig("potts_30x30_3.png", dpi=150, bbox_inches="tight")
plt.show()
