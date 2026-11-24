"""Serve the approved Cell Lab and this workspace's trained BatteryX model."""

import sys
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, FiniteFloat, PositiveFloat, PositiveInt

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
sys.path.insert(0, str(MODEL_DIR))

import batteryx_core as core
from batteryx_infer import BatteryXInferenceEngine


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.engine = BatteryXInferenceEngine(MODEL_DIR, device="cpu")
    yield


app = FastAPI(title="BatteryX Cell Lab", version="3.0", lifespan=lifespan)


class CellInput(BaseModel):
    cell_id: str = Field(min_length=1, max_length=128)
    dataset: str = Field(default="Custom", min_length=1, max_length=128)
    chemistry: str
    form_factor: str
    temperature_c: FiniteFloat
    charge_rate_c: PositiveFloat
    discharge_rate_c: PositiveFloat
    max_voltage: PositiveFloat
    min_voltage: PositiveFloat
    nominal_capacity_ah: PositiveFloat
    cycles: list[PositiveInt] = Field(min_length=20, max_length=3000)
    discharge_capacity_ah: list[PositiveFloat] = Field(min_length=20, max_length=3000)
    voltage_grid: list[FiniteFloat] = Field(min_length=64, max_length=64)
    curves: list[list[tuple[FiniteFloat, FiniteFloat]]] = Field(
        min_length=20, max_length=3000,
        description="Measured, preprocessed Q-tilde and dQ-tilde/dV pairs for each voltage point and cycle.",
    )


@app.get("/api/ready")
def ready():
    return {"status": "ready"}


@app.post("/api/predict")
def predict(data: CellInput, request: Request):
    if data.chemistry not in core.CHEMS or data.form_factor not in core.FORMS:
        raise HTTPException(422, detail=f"Use chemistry {core.CHEMS} and form factor {core.FORMS}.")
    if len(data.cycles) != len(data.discharge_capacity_ah) or len(data.cycles) != len(data.curves):
        raise HTTPException(422, detail="Cycles, capacities, and curves must have the same length.")
    if data.cycles[-1] > 2_147_483_647 or any(b <= a for a, b in zip(data.cycles, data.cycles[1:])):
        raise HTTPException(422, detail="Cycles must increase strictly and fit in a signed 32-bit integer.")
    if any(len(curve) != 64 for curve in data.curves):
        raise HTTPException(422, detail="Every curve must have 64 voltage points.")
    voltage_grid = np.asarray(data.voltage_grid, dtype=np.float32)
    if not np.isfinite(voltage_grid).all() or not np.all(np.diff(voltage_grid) > 0):
        raise HTTPException(422, detail="The voltage grid must increase strictly.")
    conditions = np.asarray([
        data.temperature_c, data.charge_rate_c, data.discharge_rate_c,
        data.max_voltage, data.min_voltage, data.nominal_capacity_ah,
    ], dtype=np.float32)
    if not np.isfinite(conditions).all() or not 0 < conditions[4] < conditions[3]:
        raise HTTPException(422, detail="Operating conditions must be finite, with max voltage above min voltage.")
    engine = request.app.state.engine
    try:
        cell = core.make_cell(
            data.cell_id, data.dataset, data.chemistry, data.form_factor, conditions,
            data.cycles, data.discharge_capacity_ah, data.curves, voltage_grid,
            eol_soh=float(engine.config["EOL_SOH"]),
        )
    except ValueError as error:
        raise HTTPException(422, detail=str(error)) from error
    if cell is None:
        raise HTTPException(422, detail="Check nominal capacity against initial measured capacity.")
    return engine.predict(cell)


app.mount("/", StaticFiles(directory=ROOT / "frontend", html=True), name="frontend")
