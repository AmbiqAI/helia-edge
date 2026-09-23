"""Optional capabilities fail at use without hiding unrelated import errors."""

import subprocess
import sys

import pytest


def run_isolated(
    source: str, blocked: str, missing: str | None = None, error_type: type[ImportError] = ModuleNotFoundError
) -> None:
    guard = f"""
import importlib.abc
import sys
class AbsentPackage(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == {blocked!r}:
            raise {error_type.__name__}('dependency probe', name={missing or blocked!r})
sys.meta_path.insert(0, AbsentPackage())
"""
    result = subprocess.run([sys.executable, "-I", "-c", guard + source], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


def test_plotting_namespace_is_lazy():
    run_isolated(
        """
import helia_edge as helia
assert 'plot_history_metrics' in dir(helia.plotting)
assert {'cm', 'roc', 'history'} <= set(dir(helia.plotting))
assert 'matplotlib' not in sys.modules
""",
        "matplotlib",
    )


@pytest.mark.parametrize(
    "expression,dependency,extra",
    [
        ("helia.utils.download_s3_prefix", "boto3", "aws"),
        ("helia.utils.download_s3_prefix", "botocore", "aws"),
        ("__import__('helia_edge.utils.aws', fromlist=['download_s3_prefix'])", "boto3", "aws"),
        ("helia.plotting.confusion_matrix_plot", "matplotlib", "plotting"),
        ("helia.plotting.cm", "matplotlib", "plotting"),
        ("helia.plotting.roc_auc_plot", "matplotlib", "plotting"),
        ("helia.plotting.history", "matplotlib", "plotting"),
        ("__import__('helia_edge.plotting.history', fromlist=['plot_history_metrics'])", "matplotlib", "plotting"),
        ("__import__('helia_edge.plotting.cm', fromlist=['confusion_matrix_plot'])", "matplotlib", "plotting"),
    ],
)
def test_missing_capability_names_its_extra(expression, dependency, extra):
    run_isolated(
        f"""
import helia_edge as helia
try:
    {expression}
except ImportError as exc:
    assert 'helia-edge[{extra}]' in str(exc), str(exc)
else:
    raise AssertionError('Missing capability was unexpectedly available')
""",
        dependency,
    )


@pytest.mark.parametrize(
    "expression,dependency",
    [
        ("helia.utils.download_s3_prefix", "boto3"),
        ("helia.plotting.plot_history_metrics", "matplotlib"),
    ],
)
@pytest.mark.parametrize("error_type", [ImportError, ModuleNotFoundError])
def test_unrelated_missing_modules_are_not_relabelled(expression, dependency, error_type):
    run_isolated(
        f"""
import helia_edge as helia
try:
    {expression}
except ImportError as exc:
    assert type(exc).__name__ == {error_type.__name__!r}
    assert exc.name == 'dependency_internal_failure', str(exc)
    assert 'helia-edge[' not in str(exc)
else:
    raise AssertionError('Internal import failure was swallowed')
""",
        dependency,
        missing="dependency_internal_failure",
        error_type=error_type,
    )
