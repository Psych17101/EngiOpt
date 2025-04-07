"""This module defines the MyPowerElecProblem class for multi-objective optimization using pymoo."""

from __future__ import annotations

from typing import Any

from model_pipeline import ModelPipeline
import numpy as np
import pandas as pd
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Real
import torch


class MyPowerElecProblem(ElementwiseProblem):
    def __init__(self, pipeline_r: ModelPipeline, pipeline_g: ModelPipeline, device: torch.device | None = None) -> None:
        """Initialize the MyPowerElecProblem.

        Args:
            pipeline_r: A ModelPipeline instance for predicting r.
            pipeline_g: A ModelPipeline instance for predicting g.
            device: The torch device to use for predictions. Defaults to None (will use CPU).
        """
        # Define the variable space:
        # All variables are now real-valued with appropriate bounds
        variables = {
            "C1": Real(bounds=(1e-6, 2e-5)),
            "C2": Real(bounds=(1e-6, 2e-5)),
            "C3": Real(bounds=(1e-6, 2e-5)),
            "C4": Real(bounds=(1e-6, 2e-5)),
            "C5": Real(bounds=(1e-6, 2e-5)),
            "C6": Real(bounds=(1e-6, 2e-5)),
            "L1": Real(bounds=(1e-6, 1e-3)),
            "L2": Real(bounds=(1e-6, 1e-3)),
            "L3": Real(bounds=(1e-6, 1e-3)),
            "T1": Real(bounds=(0.0, 0.5)),
        }
        # Two objectives:
        #  f1 = predicted r (minimize)
        #  f2 = |predicted g - 0.25| (minimize)
        super().__init__(vars=variables, n_obj=2, n_ieq_constr=0)
        self.pipeline_r = pipeline_r
        self.pipeline_g = pipeline_g
        self.device = device

    def _evaluate(self, x: dict[str, Any], out: dict[str, Any], *_args: object, **_kwargs: object) -> None:
        """Evaluate a candidate solution.

        Args:
            x: A dict containing one candidate solution.
            out: A dict to store the objective values.
            _args: Unused positional arguments.
            _kwargs: Unused keyword arguments.
        """
        # Build a one-row DataFrame.
        df = pd.DataFrame([x])

        # Get predictions using your trained pipelines.
        # (Assuming your pipelines' predict method returns a numpy array.)
        r_pred = self.pipeline_r.predict(df, batch_size=np.min([len(df), 1024]), device=self.device)[0]
        g_pred = self.pipeline_g.predict(df, batch_size=np.min([len(df), 1024]), device=self.device)[0]

        # Objectives:
        f1 = r_pred
        f2 = abs(g_pred - 0.25)

        out["F"] = [f1, f2]
