import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

class MultiVisitMRIViewer:
    """
    Interactive MRI viewer for multiple visits of a single subject.
    """

    def __init__(self, dataset, subj_id, normalize=True, transpose=True):
        self.dataset = dataset
        self.subj_id = subj_id
        self.normalize = normalize
        self.transpose = transpose

        # Load subject
        self.subj = dataset[subj_id]
        self.visits = sorted(self.subj.visits_available())

        if len(self.visits) == 0:
            raise ValueError(f"No visits found for subject {subj_id}")

        # Load volumes
        self.volumes = [self.subj[v]["image"] for v in self.visits]

        # Basic sanity check
        shapes = [v.shape for v in self.volumes]
        if len(set(shapes)) != 1:
            raise ValueError("All visits must have the same shape")

        self.H, self.W, self.D = self.volumes[0].shape
        self.init_slice = self.D // 2

        if self.normalize:
            self._normalize_volumes()

        # Placeholders
        self.fig = None
        self.axes = None
        self.imgs = None
        self.slider = None

    def _normalize_volumes(self):
        normed = []
        for vol in self.volumes:
            v = vol.astype(np.float32)
            v = (v - v.min()) / (v.max() - v.min() + 1e-8)
            normed.append(v)
        self.volumes = normed

    def _get_slice(self, vol, z):
        sl = vol[:, :, z]
        if self.transpose:
            sl = sl.T
        return sl

    def _update(self, val):
        z = int(self.slider.val)
        for img, vol in zip(self.imgs, self.volumes):
            img.set_data(self._get_slice(vol, z))
        self.fig.canvas.draw_idle()

    def show(self):
        n = len(self.volumes)

        self.fig, self.axes = plt.subplots(
            1, n,
            figsize=(4 * n, 4),
            squeeze=False
        )
        self.axes = self.axes[0]

        plt.subplots_adjust(bottom=0.25)

        self.imgs = []
        for ax, vol, visit in zip(self.axes, self.volumes, self.visits):
            img = ax.imshow(
                self._get_slice(vol, self.init_slice),
                cmap="gray",
                origin="lower"
            )
            ax.set_title(f"Visit {visit}")
            ax.axis("off")
            self.imgs.append(img)

        # Slider
        ax_slider = plt.axes([0.15, 0.1, 0.7, 0.03])
        self.slider = Slider(
            ax=ax_slider,
            label="Slice",
            valmin=0,
            valmax=self.D - 1,
            valinit=self.init_slice,
            valstep=1
        )

        self.slider.on_changed(self._update)
        plt.show()