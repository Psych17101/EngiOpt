# pymoo_pe_problem.py

from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Real, Choice
import pandas as pd
from model_pipeline import ModelPipeline
import torch
import numpy as np

class MyPowerElecProblem(ElementwiseProblem):

    def __init__(self, pipeline_r: ModelPipeline, pipeline_g: ModelPipeline):
        # Define the mixed variable space:
        # For variables C1 to C6 and L1 to L3 use Choice (categorical) with two options each.
        # T1 is a real-valued variable.
        vars = {
            "C1": Choice(options=[1e-6, 2e-5]),
            "C2": Choice(options=[1e-6, 2e-5]),
            "C3": Choice(options=[1e-6, 2e-5]),
            "C4": Choice(options=[1e-6, 2e-5]),
            "C5": Choice(options=[1e-6, 2e-5]),
            "C6": Choice(options=[1e-6, 2e-5]),
            "L1": Choice(options=[1e-6, 1e-3]),
            "L2": Choice(options=[1e-6, 1e-3]),
            "L3": Choice(options=[1e-6, 1e-3]),
            "T1": Real(bounds=(0.0, 0.5))
        }
        # Two objectives:
        #  f1 = predicted r (minimize)
        #  f2 = |predicted g - 0.25| (minimize)
        super().__init__(vars=vars, n_obj=2, n_ieq_constr=0)
        self.pipeline_r = pipeline_r
        self.pipeline_g = pipeline_g

    def _evaluate(self, X, out, *args, **kwargs):
        # X is a dict containing one candidate solution.
        # Build a one-row DataFrame.
        df = pd.DataFrame([X])
        
        # Get predictions using your trained pipelines.
        # (Assuming your pipelines’ predict method returns a numpy array.)
        r_pred = self.pipeline_r.predict(df, batch_size=np.min([len(df), 1024]), device=torch.device("mps"))[0]
        g_pred = self.pipeline_g.predict(df, batch_size=np.min([len(df), 1024]), device=torch.device("mps"))[0]

        # Objectives:
        f1 = r_pred
        f2 = abs(g_pred - 0.25)

        out["F"] = [f1, f2]
