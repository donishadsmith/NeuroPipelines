"""
For documentation: a copy of the Python file running fastapi that is hosted on a continuously running instance
on Oracle Cloud. It's the free tier so the instance only has 1 GB ram and 1 OCPU (VM.Standard.E2.1.Micro)

https://www.freedesktop.org/software/systemd/man/latest/loginctl.html
https://man7.org/linux/man-pages/man2/fallocate.2.html
https://man7.org/linux/man-pages/man8/swapon.8.html

Uses systemd to create and manage a process that runs uvicorn in the background so it can listen to incoming
requests continuously on the cloud vm:
https://www.freedesktop.org/software/systemd/man/latest/systemd.service.html?__goaway_challenge=meta-refresh&__goaway_id=8887fcae4814db9e37888fc15975b6b4&__goaway_referer=https%3A%2F%2Fwww.google.com%2F

https://github.com/fastapi/fastapi/discussions/14783
"""

import subprocess, threading, time
from collections import defaultdict, deque
from functools import lru_cache

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

LIMIT = 30
WINDOW = 60
HISTORY = defaultdict(deque)
HISTORY_LOCK = threading.Lock()

RUNNING = threading.Lock()
PROGRESS = threading.BoundedSemaphore(5)
app = FastAPI()


class Coordinate(BaseModel):
    x: int = Field(ge=-96, le=96)
    y: int = Field(ge=-132, le=96)
    z: int = Field(ge=-78, le=114)


@lru_cache(maxsize=1024)
def run_afni(x: int, y: int, z: int) -> str:
    if not PROGRESS.acquire(blocking=False):
        raise HTTPException(503, "Server is busy, try again soon.")

    try:
        with (
            RUNNING
        ):  # one thread can call afni at a time and the lock must first be released
            output = subprocess.run(
                [
                    "podman",
                    "run",
                    "--rm",
                    "--timeout",
                    "30",
                    "localhost/afni",
                    "whereami",
                    str(x),
                    str(y),
                    str(z),
                    "-lpi",
                    "-space",
                    "MNI_2009c_asym",
                    "-atlas",
                    "FS.afni.MNI2009c_asym",
                    "-atlas",
                    "Brodmann_Pijn_AFNI",
                ],
                capture_output=True,
                text=True,
                timeout=40,
            )
    except subprocess.TimeoutExpired:
        raise HTTPException(504, "AFNI took too long. Try again.")
    finally:
        PROGRESS.release()

    if output.returncode != 0:
        raise HTTPException(503, "Sorry, AFNI failed. Try again.")

    return output.stdout


def check_rate_limit(address: str) -> None:
    current_time = time.monotonic()

    with HISTORY_LOCK:
        times = HISTORY[address]

        while times and current_time - times[0] > WINDOW:
            times.popleft()

        if len(times) >= LIMIT:
            raise HTTPException(429, "Too many requests. Try again in a minute.")

        times.append(current_time)


@app.post("/")
def whereami(coordinate: Coordinate, request: Request) -> dict[str, str]:
    check_rate_limit(request.client.host)

    return {"output": run_afni(coordinate.x, coordinate.y, coordinate.z)}
