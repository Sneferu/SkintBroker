"""
Data Records

This module contains all functions and classes for recording the results of
runs.
"""

import dataclasses
import pathlib


@dataclasses.dataclass
class RunRecord:
    """
    The record of a training or validation run.  Used to shuttle information
    around the rest of the framework.

    +runtype+ is the type of the run (training or validation)
    +run_count+ is the number of runs total represented in this record
    +loss_mean is the mean loss
    +success_mean+ is the mean success
    +success_variance+ is the success variance
    """
    run_type: str
    run_count: int = 0
    loss_mean: float = 0
    success_mean: float = 0
    success_variance: float = 0

    def __add__(self, other):
        """
        Add an +other+ record to this one.
        """
        # Confirm these records are the same types
        if self.run_type != other.run_type:
            raise RuntimeError("Cannot add records with different run types!")

        # Calculate weights
        runs1 = self.run_count
        runs2 = other.run_count
        total = runs1 + runs2
        def __weighted_combine(val1, val2):
            return (val1 * runs1 + val2 * runs2) / total

        # Generate new values
        loss_mean = __weighted_combine(self.loss_mean, other.loss_mean)
        success_mean = __weighted_combine(self.success_mean,
                                          other.success_mean)
        success_variance = __weighted_combine(self.success_variance,
                                              other.success_variance)
        return RunRecord(self.run_type, run_count=total, loss_mean=loss_mean,
                         success_mean=success_mean,
                         success_variance=success_variance)
