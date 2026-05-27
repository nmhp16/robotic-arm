"""Train the green-plant detector: a small CNN that regresses the plant's xy
(rel env origin) from the table_cam RGBD image. Supervised MSE on the dataset
from collect_detector_data.py. Saves the model + normalization to
/tmp/plant_detector.pt and reports held-out localization error in mm.

  env -u VIRTUAL_ENV -u CONDA_PREFIX ~/IsaacLab/isaaclab.sh -p scripts/train_detector.py
(plain `python3` with torch also works — no Isaac needed.)
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn


class PlantDetector(nn.Module):
    def __init__(self, in_ch=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 5, 2, 2), nn.ReLU(),   # 84->42
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),      # 42->21
            nn.Conv2d(64, 64, 3, 2, 1), nn.ReLU(),      # 21->11
            nn.Flatten(),
        )
        self.head = nn.Sequential(nn.Linear(64 * 11 * 11, 128), nn.ReLU(), nn.Linear(128, 2))

    def forward(self, x):
        return self.head(self.conv(x))


def main():
    d = np.load("/tmp/detector_data.npz")
    rgb = d["rgb"].astype("float32") / 255.0                       # (N,84,84,3)
    depth = d["depth"].astype("float32")                           # (N,84,84)
    dmu, dsd = float(depth.mean()), float(depth.std() + 1e-6)
    depth = (depth - dmu) / dsd
    X = np.concatenate([rgb, depth[..., None]], -1).transpose(0, 3, 1, 2)  # (N,4,84,84)
    Y = d["xy"].astype("float32")                                  # (N,2) meters
    ymu, ysd = Y.mean(0), Y.std(0) + 1e-6
    Yn = (Y - ymu) / ysd

    n = len(X); rng = np.random.default_rng(0); idx = rng.permutation(n); ntr = int(0.9 * n)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xt = torch.tensor(X, device=dev); Yt = torch.tensor(Yn, device=dev)
    tr, va = idx[:ntr], idx[ntr:]
    net = PlantDetector().to(dev)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    lossf = nn.MSELoss()
    bs = 256
    for epoch in range(40):
        net.train(); perm = rng.permutation(len(tr))
        for i in range(0, len(tr), bs):
            b = tr[perm[i:i + bs]]
            opt.zero_grad(); loss = lossf(net(Xt[b]), Yt[b]); loss.backward(); opt.step()
        if (epoch + 1) % 10 == 0:
            net.eval()
            with torch.no_grad():
                pv = net(Xt[va]).cpu().numpy() * ysd + ymu          # un-normalize -> meters
            err_mm = np.linalg.norm(pv - Y[va], axis=1) * 1000.0
            print(f"RESULT epoch {epoch+1}: val median err = {np.median(err_mm):.1f}mm  "
                  f"mean = {err_mm.mean():.1f}mm  90th = {np.percentile(err_mm,90):.1f}mm", flush=True)
    torch.save({"state_dict": net.state_dict(), "ymu": ymu, "ysd": ysd,
                "dmu": dmu, "dsd": dsd}, "/tmp/plant_detector.pt")
    print(f"RESULT saved detector to /tmp/plant_detector.pt (trained on {ntr}, val {n-ntr})", flush=True)


if __name__ == "__main__":
    main()
