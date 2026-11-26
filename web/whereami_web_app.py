import base64, io, threading, time, requests
from collections import defaultdict, deque
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np, matplotlib, matplotlib.pyplot as plt
from bidsaid.logging import setup_logger
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from whereami_web import run_pipeline

matplotlib.use("Agg")

LGR = setup_logger(__name__)

REQUESTS_LIMIT = 30
RECENT_REQUESTS: dict[str, deque] = defaultdict(deque)
RATE_LOCK = threading.Lock()
PIPELINE_LOCK = threading.Lock()
PROGRESS = threading.BoundedSemaphore(5)

app = FastAPI(title="WhereAmI Web Lookup")


MNI_BOUNDS = {"x": (-96, 96), "y": (-132, 96), "z": (-78, 114)}


class LookupParameters(BaseModel):
    x: int = Field(ge=MNI_BOUNDS["x"][0], le=MNI_BOUNDS["x"][1])
    y: int = Field(ge=MNI_BOUNDS["y"][0], le=MNI_BOUNDS["y"][1])
    z: int = Field(ge=MNI_BOUNDS["z"][0], le=MNI_BOUNDS["z"][1])
    cohort: Literal["adults", "kids"] = "adults"
    coordinate_space: Literal["MNI", "Talairach"] = "MNI"
    transform_method: Literal["Lancaster", "Brett"] = "Lancaster"

# extra ccheck
def _check_mni_bounds(coordinate):
    for value, (low, high) in zip(coordinate, MNI_BOUNDS.values()):
        if not low <= value <= high:
            raise HTTPException(
                422,
                f"Transformed MNI coordinate ({', '.join(map(str, coordinate))}) "
                "is outside the supported range.",
            )


# https://github.com/fieldtrip/fieldtrip/blob/master/private/tal2mni.m
# Matthew Brent 2021 tal -> MNI
# https://brainmap.org/training/TalairachVersusMNI.pdf
def brett_transform(coord):
    rotn = np.array(
        [
            [1.0000, 0.0000, 0.0000, 0.0000],
            [0.0000, 0.9988, 0.0500, 0.0000],
            [0.0000, -0.0500, 0.9988, 0.0000],
            [0.0000, 0.0000, 0.0000, 1.0000],
        ]
    )

    upz = np.array(
        [
            [0.9900, 0.0000, 0.0000, 0.0000],
            [0.0000, 0.9700, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.9200, 0.0000],
            [0.0000, 0.0000, 0.0000, 1.0000],
        ]
    )

    downz = np.array(
        [
            [0.9900, 0.0000, 0.0000, 0.0000],
            [0.0000, 0.9700, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.8400, 0.0000],
            [0.0000, 0.0000, 0.0000, 1.0000],
        ]
    )

    coord = np.array(coord + [1.0])
    coord = np.dot(np.linalg.inv(rotn), coord)

    # Matrix used depends on if its below or above
    # the AC
    if coord[2] < 0:
        coord = np.dot(np.linalg.inv(downz), coord)
    else:
        coord = np.dot(np.linalg.inv(upz), coord)

    return np.round(coord[:3]).astype(int).tolist()


# Affine taken from nimare
# https://pmc.ncbi.nlm.nih.gov/articles/PMC6871323/
def lancaster_transform(coord):
    icbm_other_affine = np.array(
        [
            [0.9357, 0.0029, -0.0072, -1.0423],
            [-0.0065, 0.9396, -0.0726, -1.3940],
            [0.0103, 0.0752, 0.8967, 3.6475],
            [0.0000, 0.0000, 0.0000, 1.0000],
        ]
    )

    coord = np.dot(np.array(coord + [1]), np.linalg.inv(icbm_other_affine.T))[:3]

    return np.round(coord[:3]).astype(int).tolist()


TRANSFORM_DICT = {"Brett": brett_transform, "Lancaster": lancaster_transform}

def create_transform_text(
    original_coordinate_space, original_coordinate, coordinate, transform_method
):
    if original_coordinate_space == "MNI":
        return "Placeholder"

    original_text = ", ".join(map(str, original_coordinate))
    transformed_text = ", ".join(map(str, coordinate))

    return (
        f"Transformed the original Talairach coordinates ({original_text}) to MNI space "
        f"({transformed_text}) using the {transform_method} method.\n\n "
        "IMPORTANT: Talairach to MNI transforms are approximate and can be off by several millimeters. "
        "Coordinates near the cortical surface may land at the edge of the brain or just outside it."
    )

# on arm server caddy listens to ports 443 and 80 and forward request to localhost
# via port 8000, so geting address like this should be dine
def visitor_address(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")

    return forwarded.split(",")[-1].strip() or "local"


def check_rate_limit(request: Request):
    visitor = visitor_address(request)

    current_time = time.monotonic()

    with RATE_LOCK:
        history = RECENT_REQUESTS[visitor]
        while history and current_time - history[0] > 60:
            history.popleft()

        if len(history) >= REQUESTS_LIMIT:
            raise HTTPException(429, "Too many lookups. Try again in a minute.")

        history.append(current_time)

        if len(RECENT_REQUESTS) > 5000:
            for key in [k for k, v in RECENT_REQUESTS.items() if not v]:
                del RECENT_REQUESTS[key]


def to_base64(figure) -> str:
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=100)

    return base64.b64encode(buffer.getvalue()).decode()


@lru_cache(maxsize=256)
def cached_lookup(cohort: str, x: int, y: int, z: int, coordinate_space: str, transform_method: str) -> dict:
    if not PROGRESS.acquire(blocking=False):
        raise HTTPException(503, "Server is busy, try again soon.")

    try:
        with PIPELINE_LOCK:
            try:
                original_coordinate = [x, y, z]
                if coordinate_space != "MNI":
                    coordinate = TRANSFORM_DICT[transform_method](original_coordinate)
                    _check_mni_bounds(coordinate)
                else:
                    coordinate = original_coordinate

                coordinate_transform_text = create_transform_text(
                    coordinate_space, original_coordinate, coordinate, transform_method
                )

                afni_text, output = run_pipeline(cohort, [coordinate[0], coordinate[1], coordinate[2]], None)
                atlases = []

                for content in output.values():
                    atlases.append(
                        (
                            to_base64(content.get("afni_sphere_figure"))
                            if content.get("afni_sphere_figure")
                            else None
                        )
                    )
            finally:
                plt.close("all")
    finally:
        PROGRESS.release()

    return {"afni_text": afni_text, "coordinate_transform_text": coordinate_transform_text, "atlases": atlases}


@app.post("/lookup")
def lookup(parameters: LookupParameters, request: Request):
    check_rate_limit(request)

    try:
        return cached_lookup(
            parameters.cohort, parameters.x, parameters.y, parameters.z, parameters.coordinate_space, parameters.transform_method
        )
    except HTTPException:
        raise
    except requests.HTTPError as error:
        if (status := error.response.status_code) in [429, 503, 504]:
            try:
                detail = error.response.json().get("detail")
            except ValueError:
                detail = None

            raise HTTPException(
                status, detail if isinstance(detail, str) else "Try again soon."
            )

        raise HTTPException(502, "AFNI server could not answer that lookup.")
    except requests.RequestException:
        raise HTTPException(502, "The AFNI server did not answer. Try again shortly.")
    except Exception:
        LGR.exception(
            f"Lookup failed for MNI coordinate ({parameters.x}, {parameters.y}, {parameters.z}) for {parameters.cohort} cohort"
        )

        raise HTTPException(500, "Lookup failed.")


app.mount(
    "/",
    StaticFiles(directory=Path(__file__).parent / "static", html=True),
    name="static",
)
