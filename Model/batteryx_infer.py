import json
import os

import numpy as np
import torch
import batteryx_core as bc


class BatteryXInferenceEngine:
    """
    Standalone BatteryX prediction engine.

    Required input:
        A validated BatteryX Cell object.

    Output:
        Strict JSON-serializable Python dictionary containing:
        - Raw SOH
        - Causally projected SOH
        - RUL
        - Conformal interval bounds
        - Physics compliance
        - Nearest prototype cell IDs
    """

    def __init__(self, bundle_dir, device=None):

        self.bundle_dir = os.path.abspath(bundle_dir)

        self.device = torch.device(
            device or (
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        )

        with open(
            os.path.join(self.bundle_dir, "stats.json"),
            "r",
            encoding="utf-8"
        ) as f:
            self.stats = json.load(f)

        with open(
            os.path.join(self.bundle_dir, "config.json"),
            "r",
            encoding="utf-8"
        ) as f:
            self.config = json.load(f)

        self.q_conformal = float(
            self.stats["q_conformal"]
        )

        if not np.isfinite(self.q_conformal):
            raise ValueError(
                "Deployment bundle does not contain a finite calibration quantile."
            )

        # Load compact reference bank.
        with np.load(
            os.path.join(
                self.bundle_dir,
                "compact_bank.npz"
            ),
            allow_pickle=False
        ) as data:

            self.bank = {
                "descriptors": torch.as_tensor(
                    data["descriptors"].copy(),
                    dtype=torch.float32
                ),

                "outcomes": torch.as_tensor(
                    data["outcomes"].copy(),
                    dtype=torch.float32
                ),

                "bank_features": torch.as_tensor(
                    data["bank_features"].copy(),
                    dtype=torch.float32
                ),

                "cell_ids": data["cell_ids"].astype(str).tolist(),

                "datasets": data["datasets"].astype(str).tolist(),

                "chems": data["chems"].astype(str).tolist()
            }

        # Rebuild exactly the original architecture.
        self.model = bc.BatteryX(

            cond_dim=15,
            desc_dim=21,

            d_model=int(
                self.config["D_MODEL"]
            ),

            seq_type=self.config["SEQ_TYPE"],

            knn=int(
                self.config["KNN"]
            ),

            heads=int(
                self.config["HEADS"]
            ),

            use_cnn=bool(
                self.config["USE_CNN"]
            ),

            use_seq=bool(
                self.config["USE_SEQ"]
            ),

            use_tab=bool(
                self.config["USE_TAB"]
            ),

            use_graph=bool(
                self.config["USE_GRAPH"]
            ),

            use_monotone_head=bool(
                self.config["USE_MONOTONE_HEAD"]
            ),

            stats=self.stats

        ).to(self.device)

        checkpoint_path = os.path.join(
            self.bundle_dir,
            "batteryx_best.pt"
        )

        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location=self.device,
                weights_only=True
            )
        except TypeError:
            checkpoint = torch.load(
                checkpoint_path,
                map_location=self.device
            )

        state = checkpoint["model_state_dict"]

        # Strict checkpoint validation.
        self.model.load_state_dict(
            state,
            strict=True
        )

        self.model.eval()

    @staticmethod
    def _json_list(values):

        arr = np.asarray(
            values,
            dtype=np.float64
        ).reshape(-1)

        return [
            float(x) if np.isfinite(x) else None
            for x in arr
        ]

    @torch.no_grad()
    def predict_arrays(self, cell):

        """
        Internal numerical prediction method.
        Returns NumPy arrays for parity testing.
        """

        pool = bc.BatteryPool(
            [cell],
            self.stats,
            self.bank,

            knn=int(
                self.config["KNN"]
            ),

            is_train=False,
            stride=1
        )

        if len(pool) == 0:
            raise ValueError(
                "Input cell produced no valid inference samples."
            )

        soh_raw, rul, rul_lower, rul_upper = bc.predict_pool(
            self.model,
            pool,

            q_conformal=self.q_conformal,

            batch_size=int(
                self.config.get(
                    "INFERENCE_BATCH_SIZE",
                    512
                )
            )
        )

        soh_projected = bc.causal_monotone_project(
            soh_raw
        )

        # Percentage of adjacent predictions respecting
        # the expected non-increasing SOH trajectory.
        if len(soh_raw) > 1:

            raw_compliance = float(
                np.mean(
                    np.diff(soh_raw) <= 2e-3
                ) * 100.0
            )

            projected_compliance = float(
                np.mean(
                    np.diff(soh_projected) <= 2e-3
                ) * 100.0
            )

        else:

            raw_compliance = 100.0
            projected_compliance = 100.0

        # Find nearest eligible reference prototypes
        # for the first observed cycle.
        sample = pool.samples[0]

        distances = sample["dists"].detach().cpu().clone()

        self_mask = sample["self_mask"].detach().cpu().bool()

        distances[~self_mask] = float("inf")

        finite = torch.isfinite(distances)

        neighbor_ids = []

        if finite.any():

            n_neighbors = min(
                int(self.config["KNN"]),
                int(finite.sum())
            )

            indices = torch.topk(
                distances,
                n_neighbors,
                largest=False
            ).indices.tolist()

            neighbor_ids = [
                self.bank["cell_ids"][int(i)]
                for i in indices
            ]

        cycles = [
            int(s["cycle_k"])
            for s in pool.samples
        ]

        return {
            "cycles": np.asarray(cycles),
            "soh_raw": soh_raw,
            "soh_projected": soh_projected,
            "rul": rul,
            "rul_lower": rul_lower,
            "rul_upper": rul_upper,

            "raw_physics_compliance": raw_compliance,
            "projected_physics_compliance": projected_compliance,

            "neighbor_ids": neighbor_ids
        }

    @torch.no_grad()
    def predict(self, cell):

        """
        Public prediction interface.

        Every returned value is JSON-serializable.
        Infinite upper bounds are represented as null and
        identified by interval_upper_unbounded.
        """

        result = self.predict_arrays(cell)

        upper_unbounded = [
            bool(np.isposinf(x))
            for x in result["rul_upper"]
        ]

        output = {
            "cell_id": str(cell.cell_id),

            "cycles": [
                int(x) for x in result["cycles"]
            ],

            "soh_raw": self._json_list(
                result["soh_raw"]
            ),

            "soh_projected": self._json_list(
                result["soh_projected"]
            ),

            "rul": self._json_list(
                result["rul"]
            ),

            "rul_lower": self._json_list(
                result["rul_lower"]
            ),

            "rul_upper": self._json_list(
                result["rul_upper"]
            ),

            "interval_upper_unbounded": upper_unbounded,

            "conformal_alpha": float(
                self.config["CONFORMAL_ALPHA"]
            ),

            "conformal_quantile": float(
                self.q_conformal
            ),

            "physics_compliance_percent": {
                "raw_soh": float(
                    result["raw_physics_compliance"]
                ),

                "projected_soh": float(
                    result["projected_physics_compliance"]
                )
            },

            "nearest_prototype_cell_ids": list(
                result["neighbor_ids"]
            )
        }

        # Guarantee strict JSON compatibility.
        json.dumps(
            output,
            allow_nan=False
        )

        return output
