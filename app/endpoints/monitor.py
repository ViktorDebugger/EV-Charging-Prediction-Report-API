from fastapi import APIRouter
from fastapi.responses import JSONResponse
import traceback
from app.services.utils.monitor import get_current_data, get_reference_data
from pathlib import Path
from datetime import datetime

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, DataQualityPreset


router = APIRouter()

@router.get("/data-drift")
async def monitor_data_drift():
    try:        
        trained_df = get_reference_data()
        current_df = get_current_data(trained_df)
        
        if trained_df.empty or current_df.empty:
            return JSONResponse(
                content={
                    "status": "error",
                    "message": "Data not found!",
                },
                status_code=404
            )
        
        report = Report(metrics=[
            DataDriftPreset(),
        ])
        
        report.run(
            reference_data=trained_df,
            current_data=current_df
        )
        
        reports_dir = Path("reports/data-drift")
        reports_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        report_filename = f"report_{timestamp}.html"
        report_path = reports_dir / report_filename
        
        report.save_html(str(report_path))

        return {
            "status": "success",
            "message": "Data Drift Report completed",
        }
    
    except Exception as e:
        error_details = traceback.format_exc()
        
        return JSONResponse(
            content={
                "status": "error",
                "message": str(e),
                "details": error_details
            },
            status_code=500
        )

@router.get("/target-drift")
async def monitor_target_drift():
    try:        
        trained_df = get_reference_data()
        current_df = get_current_data(trained_df)
        
        if trained_df.empty or current_df.empty:
            return JSONResponse(
                content={
                    "status": "error",
                    "message": "Data not found!",
                },
                status_code=404
            )
        
        if 'target' not in trained_df.columns or 'target' not in current_df.columns:
            return JSONResponse(
                content={
                    "status": "error",
                    "message": "Target value not found!"
                },
                status_code=404
            )
        
        report = Report(metrics=[
            TargetDriftPreset(columns=['target']),
        ])
        
        report.run(
            reference_data=trained_df,
            current_data=current_df
        )
        
        reports_dir = Path("reports/target-drift")
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        report_filename = f"report_{timestamp}.html"
        report_path = reports_dir / report_filename
        
        report.save_html(str(report_path))

        return {
            "status": "success",
            "message": "Target Drift Report completed",
        }
    
    except Exception as e:
        error_details = traceback.format_exc()
        
        return JSONResponse(
            content={
                "status": "error",
                "message": str(e),
                "details": error_details
            },
            status_code=500
        )

@router.get("/data-quality")
async def monitor_data_quality():
    try:        
        trained_df = get_reference_data()
        current_df = get_current_data(trained_df)
        
        if trained_df.empty or current_df.empty:
            return JSONResponse(
                content={
                    "status": "error",
                    "message": "Data not found!",
                },
                status_code=404
            )
        
        report = Report(metrics=[
            DataQualityPreset(),
        ])
        
        report.run(
            reference_data=trained_df,
            current_data=current_df
        )
        
        reports_dir = Path("reports/data-quality")
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        report_filename = f"report_{timestamp}.html"
        report_path = reports_dir / report_filename
        
        report.save_html(str(report_path))

        return {
            "status": "success",
            "message": "Data Quality Report completed",
        }
    
    except Exception as e:
        error_details = traceback.format_exc()
        
        return JSONResponse(
            content={
                "status": "error",
                "message": str(e),
                "details": error_details
            },
            status_code=500
        )