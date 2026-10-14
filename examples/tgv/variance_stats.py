#!/usr/bin/env python3
"""Post-process Neko/Nek5000 vector field files.

Reads a series of Neko field files and computes, for each timestep, the spatial
mean magnitude and variance of a vector field.

By default the vector field is the vorticity ``(omega_x, omega_y, omega_z)``
written by the ``field_writer`` simcomp. Other three-component vectors, such as
velocity ``(u, v, w)`` from the main fluid output, can be selected with
``--fields``.

Optionally, a second field series can be supplied. In that case the script
additionally computes the mean magnitude and variance of the vector difference.

Usage
-----
Single set of vorticity fields (default):
    ./vorticity_stats.py vorticity0.nek5000

Velocity from the main fluid output using a preset:
    ./vorticity_stats.py field0.nek5000 --vector velocity

Custom vector components:
    ./vorticity_stats.py vorticity0.nek5000 --fields omega_x,omega_y,omega_z

Compare two sets:
    ./vorticity_stats.py vorticity0.nek5000 --second other0.nek5000


Output
------
A space-separated table written to stdout with one line per timestep:
    time mean_magnitude variance [mean_diff variance_diff]

A progress bar is written to stderr when stdout is not a terminal; use
``--progress`` to force it or ``--no-progress`` to disable it.
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from pymech.neksuite import readnek
from tqdm import tqdm


def gll_weights_1d(order: int) -> np.ndarray:
    """Return the Gauss-Lobatto-Legendre quadrature weights for ``order``.

    The returned array has ``order + 1`` entries and sums to ``2``.

    The nodes are the roots of ``(1 - x^2) P_N'(x)``, where ``P_N`` is the
    ``N``-th Legendre polynomial. We use Newton iteration on the equivalent
    function ``f(x) = x P_N(x) - P_{N-1}(x)`` with the analytic weight formula
    ``w_i = 2 / (N(N+1) P_N(x_i)^2)``. See Canuto et al., "Spectral Methods:
    Fundamentals in Single Domains", Section 2.3.
    """
    n = order
    if n == 0:
        return np.ones(1)

    # Chebyshev-Gauss-Lobatto initial guess.
    x = np.cos(np.arange(n + 1) * np.pi / n)

    for _ in range(100):
        # Legendre polynomials up to P_n evaluated at x.
        p_prev = np.ones_like(x)
        p_curr = x.copy()
        for k in range(2, n + 1):
            p_next = ((2 * k - 1) * x * p_curr - (k - 1) * p_prev) / k
            p_prev, p_curr = p_curr, p_next

        # P_{n-1} evaluated at x.
        if n <= 1:
            p_nm1 = np.ones_like(x)
        else:
            q_prev = np.ones_like(x)
            q_curr = x.copy()
            for k in range(2, n):
                q_next = ((2 * k - 1) * x * q_curr - (k - 1) * q_prev) / k
                q_prev, q_curr = q_curr, q_next
            p_nm1 = q_curr

        # Newton update for the roots of (1 - x^2) P_n'(x).
        f = x * p_curr - p_nm1
        df = (n + 1) * p_curr
        dx = f / df
        x = x - dx
        if np.max(np.abs(dx)) < 1e-14:
            break

    weights = 2.0 / (n * (n + 1) * p_curr**2)
    return weights


def parse_nek5000_meta(meta_path: Path) -> tuple[str, int, int]:
    """Parse a ``*.nek5000`` meta file.

    Returns
    -------
    file_template :
        Template string used to construct individual field file names. The
        template contains ``%01d`` for the file-index and ``%05d`` for the
        timestep number.
    first_timestep :
        Index of the first timestep stored in the series.
    num_timesteps :
        Number of timesteps in the series.
    """
    file_template = None
    first_timestep = None
    num_timesteps = None

    with meta_path.open("r") as fh:
        for line in fh:
            key, _, value = line.partition(":")
            key = key.strip().lower()
            value = value.strip()

            if key == "filetemplate":
                # The template line contains the base name followed by the
                # format specifiers, e.g. ``vorticity%01d.f%05d``.
                file_template = value
            elif key == "firsttimestep":
                first_timestep = int(value)
            elif key == "numtimesteps":
                num_timesteps = int(value)

    if file_template is None or first_timestep is None or num_timesteps is None:
        raise ValueError(f"Could not parse meta file: {meta_path}")

    return file_template, first_timestep, num_timesteps


def build_field_filename(meta_path: Path, template: str, istep: int) -> Path:
    """Construct the absolute path to a single field file.

    The template is relative to the directory containing the meta file and
    uses ``%01d`` for the split-file index (always 0 here) and ``%05d`` for
    the timestep index.
    """
    base_dir = meta_path.parent
    # Neko writes series with a single file per timestep, so the split index
    # is always 0.
    fname = template % (0, istep)
    return base_dir / fname


def _uses_velocity_storage(field_names: tuple[str, str, str]) -> bool:
    """Return True if the requested fields live in pymech's ``elem.vel``."""
    velocity_aliases = {"u", "v", "w", "ux", "uy", "uz", "vx", "vy", "vz"}
    return all(name in velocity_aliases for name in field_names)


def read_vector_field(
    path: Path, field_names: tuple[str, str, str]
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read a vector field file and return the time, weights, and components.

    Parameters
    ----------
    path :
        Path to the Neko field file.
    field_names :
        Names of the three vector components. These are used to decide whether
        the data lives in ``elem.scal`` (scalar storage) or ``elem.vel``
        (velocity storage). For scalar storage the first three scalar slots are
        read in the order given by ``field_names``.

    Returns
    -------
    time :
        Simulation time stored in the file header.
    weights :
        A 3D array of GLL quadrature weights with shape ``(lz, ly, lx)``.
    c1, c2, c3 :
        Stacked 4-D arrays of shape ``(nelv, lz, ly, lx)`` containing the three
        vector components for all elements.
    """
    data = readnek(str(path))

    lx, ly, lz = data.lr1
    if lx != ly or ly != lz:
        raise ValueError(
            f"Only isotropic polynomial orders are supported, got {data.lr1}"
        )

    w1d = gll_weights_1d(lx - 1)
    weights = w1d[None, None, :] * w1d[None, :, None] * w1d[:, None, None]

    if _uses_velocity_storage(field_names):
        n_vel = data.var[1]
        if n_vel != 3:
            raise ValueError(
                f"Expected 3 velocity components in {path}, found {n_vel}"
            )
        vel = np.stack([elem.vel for elem in data.elem], axis=0)
        c1 = vel[:, 0, :, :, :]
        c2 = vel[:, 1, :, :, :]
        c3 = vel[:, 2, :, :, :]
    else:
        n_scalars = data.var[4]
        if n_scalars < 3:
            raise ValueError(
                f"Expected at least 3 scalar components in {path}, "
                f"found {n_scalars}"
            )
        scal = np.stack([elem.scal for elem in data.elem], axis=0)
        c1 = scal[:, 0, :, :, :]
        c2 = scal[:, 1, :, :, :]
        c3 = scal[:, 2, :, :, :]

    return float(data.time), weights, c1, c2, c3


def vector_variance(
    weights: np.ndarray,
    c1: np.ndarray,
    c2: np.ndarray,
    c3: np.ndarray,
) -> tuple[float, float]:
    """Return mean magnitude and variance of a vector field.

    The averages use GLL quadrature weights, matching the mass-matrix weighted
    volume averages used by Neko for affine elements. For non-affine meshes a
    Jacobian-weighted quadrature would be required for exact agreement.

    The mean is the magnitude of the mean vector,
    ``|<v>| = sqrt(<v1>^2 + <v2>^2 + <v3>^2)``.

    The variance is the variance of the vector components,
    ``Var[v] = <|v|^2> - |<v>|^2``,
    where ``<|v|^2> = <v1^2 + v2^2 + v3^2>``.
    """
    weights = weights[np.newaxis, :, :, :]
    norm = float(c1.shape[0] * np.sum(weights))

    def weighted_mean(values: np.ndarray) -> float:
        return float(np.sum(weights * values) / norm)

    mean_c1 = weighted_mean(c1)
    mean_c2 = weighted_mean(c2)
    mean_c3 = weighted_mean(c3)

    mean_magnitude = np.sqrt(mean_c1**2 + mean_c2**2 + mean_c3**2)

    mean_w2 = weighted_mean(c1**2 + c2**2 + c3**2)
    variance = mean_w2 - mean_magnitude**2

    return float(mean_magnitude), float(variance)


def _process_single_file(
    args: tuple[Path, int, tuple[str, str, str]]
) -> tuple[int, float, float, float]:
    """Worker: read one file and return its timestep, time, mean, and variance."""
    meta_path, istep, field_names = args
    template, first_step, _ = parse_nek5000_meta(meta_path)
    fname = build_field_filename(meta_path, template, istep)
    time, weights, c1, c2, c3 = read_vector_field(fname, field_names)
    mean, var = vector_variance(weights, c1, c2, c3)
    return istep, time, mean, var


def _process_pair(
    args: tuple[Path, Path, int, tuple[str, str, str]]
) -> tuple[int, float, float, float, float, float]:
    """Worker: read a file pair and return stats plus difference stats."""
    meta_path1, meta_path2, istep, field_names = args
    template1, _, _ = parse_nek5000_meta(meta_path1)
    template2, _, _ = parse_nek5000_meta(meta_path2)

    fname1 = build_field_filename(meta_path1, template1, istep)
    fname2 = build_field_filename(meta_path2, template2, istep)

    time1, weights1, c1a, c2a, c3a = read_vector_field(fname1, field_names)
    _time2, _weights2, c1b, c2b, c3b = read_vector_field(fname2, field_names)

    mean, var = vector_variance(weights1, c1a, c2a, c3a)
    mean_diff, var_diff = vector_variance(
        weights1, c1a - c1b, c2a - c2b, c3a - c3b
    )
    return istep, time1, mean, var, mean_diff, var_diff


def _run_parallel(
    tasks,
    worker,
    n_workers: int,
    show_progress: bool,
    total: int,
):
    """Execute tasks in parallel and return results ordered by timestep index."""
    results = []
    executor = ProcessPoolExecutor(max_workers=n_workers)
    try:
        futures = {executor.submit(worker, task): task for task in tasks}
        iterator = tqdm(
            as_completed(futures),
            total=total,
            file=sys.stderr,
            disable=not show_progress,
        )
        for future in iterator:
            results.append(future.result())
    except KeyboardInterrupt:
        # Cancel pending work and terminate workers so a single Ctrl-C is
        # enough to exit cleanly.
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    finally:
        executor.shutdown(wait=True)
    return sorted(results, key=lambda r: r[0])


def _parse_fields(field_str: str) -> tuple[str, str, str]:
    """Parse a comma-separated list of three field names."""
    names = [name.strip() for name in field_str.split(",")]
    if len(names) != 3:
        raise ValueError(
            f"Expected exactly 3 comma-separated field names, got {len(names)}"
        )
    return tuple(names)  # type: ignore[return-value]


_VECTOR_PRESETS = {
    "vorticity": ("omega_x", "omega_y", "omega_z"),
    "velocity": ("u", "v", "w"),
}


def _resolve_fields(
    vector: str | None, fields: str | None, default_vector: str
) -> tuple[str, str, str]:
    """Resolve field names from a vector preset or a custom comma-separated list.

    Exactly one of ``vector`` or ``fields`` must be supplied. The ``vector``
    preset is expanded to its component names; ``fields`` is parsed directly.
    """
    if vector is not None and fields is not None:
        raise ValueError("Only one of --vector and --fields may be given.")

    if vector is None and fields is None:
        vector = default_vector

    if vector is not None:
        preset = vector.lower()
        if preset not in _VECTOR_PRESETS:
            raise ValueError(
                f"Unknown vector preset '{vector}'. "
                f"Known presets: {', '.join(_VECTOR_PRESETS)}"
            )
        return _VECTOR_PRESETS[preset]

    return _parse_fields(fields)


def process_single_series(
    meta_path: Path,
    field_names: tuple[str, str, str],
    show_progress: bool,
    n_workers: int,
):
    """Compute and print mean/variance for one vector field series."""
    template, first_step, n_steps = parse_nek5000_meta(meta_path)

    print("# time mean_magnitude variance")
    tasks = [
        (meta_path, istep, field_names)
        for istep in range(first_step, first_step + n_steps)
    ]
    rows = _run_parallel(
        tasks, _process_single_file, n_workers, show_progress, len(tasks)
    )
    for _istep, time, mean, var in rows:
        print(f"{time:.9e} {mean:.9e} {var:.9e}")


def process_two_series(
    meta_path1: Path,
    meta_path2: Path,
    field_names: tuple[str, str, str],
    show_progress: bool,
    n_workers: int,
):
    """Compute mean/variance and mean/variance of the difference for two series."""
    template1, first1, n1 = parse_nek5000_meta(meta_path1)
    template2, first2, n2 = parse_nek5000_meta(meta_path2)

    if n1 != n2:
        raise ValueError(
            f"Timestep counts differ: {meta_path1} has {n1}, "
            f"{meta_path2} has {n2}"
        )
    if first1 != first2:
        raise ValueError(
            f"First timestep indices differ: {first1} vs {first2}"
        )

    print("# time mean_magnitude variance mean_diff variance_diff")
    tasks = [
        (meta_path1, meta_path2, istep, field_names)
        for istep in range(first1, first1 + n1)
    ]
    rows = _run_parallel(
        tasks, _process_pair, n_workers, show_progress, len(tasks)
    )
    for _istep, time, mean, var, mean_diff, var_diff in rows:
        print(
            f"{time:.9e} {mean:.9e} {var:.9e} "
            f"{mean_diff:.9e} {var_diff:.9e}"
        )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Compute vorticity statistics from Neko field files."
    )
    parser.add_argument(
        "nek5000",
        type=Path,
        help="Path to the .nek5000 meta file for the first field series.",
    )
    parser.add_argument(
        "--second",
        type=Path,
        default=None,
        help="Optional .nek5000 meta file for a second field series. "
        "When given, the mean and variance of the vector difference are added.",
    )
    parser.add_argument(
        "--vector",
        type=str,
        default=None,
        choices=list(_VECTOR_PRESETS),
        help="Select a vector preset: 'velocity' (u,v,w) or "
        "'vorticity' (omega_x, omega_y, omega_z).",
    )
    parser.add_argument(
        "--fields",
        type=str,
        default=None,
        help="Comma-separated list of three custom vector component names. "
        "Use velocity component names (u,v,w or ux,uy,uz) to read from the "
        "velocity slot of the main fluid output; otherwise the first three "
        "scalar slots are read. Mutually exclusive with --vector.",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        default=None,
        help="Show a progress bar on stderr. By default the bar is shown "
        "only when stdout is not a terminal.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Never show a progress bar.",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=1,
        help="Number of parallel worker processes (default: 1).",
    )
    args = parser.parse_args(argv)

    # Default to a progress bar when stdout is piped or redirected, so the
    # table on stdout stays clean while the user still gets feedback.
    if args.no_progress:
        show_progress = False
    elif args.progress is True:
        show_progress = True
    else:
        show_progress = not sys.stdout.isatty()

    n_workers = max(1, args.jobs)

    try:
        field_names = _resolve_fields(args.vector, args.fields, "vorticity")
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not args.nek5000.exists():
        print(f"Error: meta file not found: {args.nek5000}", file=sys.stderr)
        return 1

    if args.second is None:
        process_single_series(
            args.nek5000, field_names, show_progress, n_workers
        )
    else:
        if not args.second.exists():
            print(
                f"Error: second meta file not found: {args.second}",
                file=sys.stderr,
            )
            return 1
        process_two_series(
            args.nek5000,
            args.second,
            field_names,
            show_progress,
            n_workers,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
