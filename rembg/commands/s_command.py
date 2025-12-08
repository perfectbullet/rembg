import json
import os
import webbrowser
from typing import Optional, Tuple, cast

import aiohttp
import click
import gradio as gr
import uvicorn
from asyncer import asyncify
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import Response

from .._version import get_versions
from ..bg import remove
from ..session_factory import new_session
from ..sessions import sessions_names
from ..sessions.base import BaseSession


@click.command(  # type: ignore
    name="s",
    help="for a http server",
)
@click.option(
    "-p",
    "--port",
    default=7000,
    type=int,
    show_default=True,
    help="port",
)
@click.option(
    "-h",
    "--host",
    default="0.0.0.0",
    type=str,
    show_default=True,
    help="host",
)
@click.option(
    "-l",
    "--log_level",
    default="info",
    type=str,
    show_default=True,
    help="log level",
)
@click.option(
    "-t",
    "--threads",
    default=None,
    type=int,
    show_default=True,
    help="number of worker threads",
)
def s_command(port: int, host: str, log_level: str, threads: int) -> None:
    """
    Command-line interface for running the FastAPI web server.

    This function starts the FastAPI web server with the specified port and log level.
    If the number of worker threads is specified, it sets the thread limiter accordingly.
    """
    sessions: dict[str, BaseSession] = {}
    tags_metadata = [
        {
            "name": "Background Removal",
            "description": "Endpoints that perform background removal with different image sources.",
            "externalDocs": {
                "description": "GitHub Source",
                "url": "https://github.com/danielgatis/rembg",
            },
        },
        {
            "name": "Green Screen",
            "description": "Endpoints for replacing image background with green or custom colors (ideal for video production).",
        },
    ]
    app = FastAPI(
        title="Rembg API",
        description="Rembg is a powerful tool to remove image backgrounds and replace them with custom colors. Perfect for video production, photo editing, and more.",
        version=get_versions()["version"],
        contact={
            "name": "Daniel Gatis",
            "url": "https://github.com/danielgatis",
            "email": "danielgatis@gmail.com",
        },
        license_info={
            "name": "MIT License",
            "url": "https://github.com/danielgatis/rembg/blob/main/LICENSE.txt",
        },
        openapi_tags=tags_metadata,
        docs_url="/api",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_credentials=True,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    class CommonQueryParams:
        def __init__(
            self,
            model: str = Query(
                description="Model to use when processing image",
                regex=r"(" + "|".join(sessions_names) + ")",
                default="u2net",
            ),
            a: bool = Query(default=False, description="Enable Alpha Matting"),
            af: int = Query(
                default=240,
                ge=0,
                le=255,
                description="Alpha Matting (Foreground Threshold)",
            ),
            ab: int = Query(
                default=10,
                ge=0,
                le=255,
                description="Alpha Matting (Background Threshold)",
            ),
            ae: int = Query(
                default=10, ge=0, description="Alpha Matting (Erode Structure Size)"
            ),
            om: bool = Query(default=False, description="Only Mask"),
            ppm: bool = Query(default=False, description="Post Process Mask"),
            bgc: Optional[str] = Query(default=None, description="Background Color"),
            extras: Optional[str] = Query(
                default=None, description="Extra parameters as JSON"
            ),
        ):
            self.model = model
            self.a = a
            self.af = af
            self.ab = ab
            self.ae = ae
            self.om = om
            self.ppm = ppm
            self.extras = extras
            self.bgc = (
                cast(Tuple[int, int, int, int], tuple(map(int, bgc.split(","))))
                if bgc
                else None
            )

    class CommonQueryPostParams:
        def __init__(
            self,
            model: str = Form(
                description="Model to use when processing image",
                regex=r"(" + "|".join(sessions_names) + ")",
                default="u2net",
            ),
            a: bool = Form(default=False, description="Enable Alpha Matting"),
            af: int = Form(
                default=240,
                ge=0,
                le=255,
                description="Alpha Matting (Foreground Threshold)",
            ),
            ab: int = Form(
                default=10,
                ge=0,
                le=255,
                description="Alpha Matting (Background Threshold)",
            ),
            ae: int = Form(
                default=10, ge=0, description="Alpha Matting (Erode Structure Size)"
            ),
            om: bool = Form(default=False, description="Only Mask"),
            ppm: bool = Form(default=False, description="Post Process Mask"),
            bgc: Optional[str] = Query(default=None, description="Background Color"),
            extras: Optional[str] = Query(
                default=None, description="Extra parameters as JSON"
            ),
        ):
            self.model = model
            self.a = a
            self.af = af
            self.ab = ab
            self.ae = ae
            self.om = om
            self.ppm = ppm
            self.extras = extras
            self.bgc = (
                cast(Tuple[int, int, int, int], tuple(map(int, bgc.split(","))))
                if bgc
                else None
            )

    def im_without_bg(content: bytes, commons: CommonQueryParams) -> Response:
        kwargs = {}

        if commons.extras:
            try:
                kwargs.update(json.loads(commons.extras))
            except Exception:
                pass

        session = sessions.get(commons.model)
        if session is None:
            session = new_session(commons.model, **kwargs)
            sessions[commons.model] = session

        return Response(
            remove(
                content,
                session=session,
                alpha_matting=commons.a,
                alpha_matting_foreground_threshold=commons.af,
                alpha_matting_background_threshold=commons.ab,
                alpha_matting_erode_size=commons.ae,
                only_mask=commons.om,
                post_process_mask=commons.ppm,
                bgcolor=commons.bgc,
                **kwargs,
            ),
            media_type="image/png",
        )

    def validate_image_file(file: UploadFile) -> None:
        """Validate that uploaded file is a supported image format."""
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        allowed_types = ["image/png", "image/jpeg", "image/jpg"]
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported image format. Supported formats: PNG, JPG, JPEG"
            )

    def get_or_create_session(model: str) -> BaseSession:
        """Get existing session or create a new one for the specified model."""
        session_obj = sessions.get(model)
        if session_obj is None:
            session_obj = new_session(model)
            sessions[model] = session_obj
        return session_obj

    @app.on_event("startup")
    def startup():
        try:
            webbrowser.open(f"http://localhost:{port}")
        except Exception:
            pass

        if threads is not None:
            from anyio import CapacityLimiter
            from anyio.lowlevel import RunVar

            RunVar("_default_thread_limiter").set(CapacityLimiter(threads))

    @app.get(
        path="/api/remove",
        tags=["Background Removal"],
        summary="Remove from URL",
        description="Removes the background from an image obtained by retrieving an URL.",
    )
    async def get_index(
        url: str = Query(
            default=..., description="URL of the image that has to be processed."
        ),
        commons: CommonQueryParams = Depends(),
    ):
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                file = await response.read()
                return await asyncify(im_without_bg)(file, commons)

    @app.post(
        path="/api/remove",
        tags=["Background Removal"],
        summary="Remove from Stream",
        description="Removes the background from an image sent within the request itself.",
    )
    async def post_index(
        file: bytes = File(
            default=...,
            description="Image file (byte stream) that has to be processed.",
        ),
        commons: CommonQueryPostParams = Depends(),
    ):
        return await asyncify(im_without_bg)(file, commons)  # type: ignore

    @app.post(
        path="/api/greenscreen",
        tags=["Green Screen"],
        summary="Replace Background with Green Screen",
        description="Removes the background and replaces it with green color (0, 255, 0). Perfect for video production and chroma keying.",
    )
    async def greenscreen(
        file: UploadFile = File(
            default=...,
            description="Image file (PNG, JPG, JPEG) to process.",
        ),
        model: str = Form(
            description="Model to use when processing image",
            regex=r"(" + "|".join(sessions_names) + ")",
            default="u2net",
        ),
        a: bool = Form(default=False, description="Enable Alpha Matting"),
        af: int = Form(
            default=240,
            ge=0,
            le=255,
            description="Alpha Matting (Foreground Threshold)",
        ),
        ab: int = Form(
            default=10,
            ge=0,
            le=255,
            description="Alpha Matting (Background Threshold)",
        ),
        ae: int = Form(
            default=10, ge=0, description="Alpha Matting (Erode Structure Size)"
        ),
        ppm: bool = Form(default=False, description="Post Process Mask"),
    ):
        """
        Replace image background with green screen color (0, 255, 0, 255).
        Supports PNG, JPG, and JPEG image formats.
        """
        validate_image_file(file)
        content = await file.read()
        
        # Green screen color: (R=0, G=255, B=0, A=255)
        green_bgcolor = (0, 255, 0, 255)
        
        session_obj = get_or_create_session(model)

        result = await asyncify(remove)(
            content,
            session=session_obj,
            alpha_matting=a,
            alpha_matting_foreground_threshold=af,
            alpha_matting_background_threshold=ab,
            alpha_matting_erode_size=ae,
            only_mask=False,
            post_process_mask=ppm,
            bgcolor=green_bgcolor,
        )
        
        return Response(
            result,
            media_type="image/png",
            headers={
                "Content-Disposition": f"attachment; filename=greenscreen_{file.filename}"
            }
        )

    @app.post(
        path="/api/replace-background",
        tags=["Green Screen"],
        summary="Replace Background with Custom Color",
        description="Removes the background and replaces it with a custom RGB color. Specify color as three integers (R, G, B) each in range 0-255.",
    )
    async def replace_background(
        file: UploadFile = File(
            default=...,
            description="Image file (PNG, JPG, JPEG) to process.",
        ),
        red: int = Form(
            default=0,
            ge=0,
            le=255,
            description="Red color component (0-255)",
        ),
        green: int = Form(
            default=255,
            ge=0,
            le=255,
            description="Green color component (0-255)",
        ),
        blue: int = Form(
            default=0,
            ge=0,
            le=255,
            description="Blue color component (0-255)",
        ),
        model: str = Form(
            description="Model to use when processing image",
            regex=r"(" + "|".join(sessions_names) + ")",
            default="u2net",
        ),
        a: bool = Form(default=False, description="Enable Alpha Matting"),
        af: int = Form(
            default=240,
            ge=0,
            le=255,
            description="Alpha Matting (Foreground Threshold)",
        ),
        ab: int = Form(
            default=10,
            ge=0,
            le=255,
            description="Alpha Matting (Background Threshold)",
        ),
        ae: int = Form(
            default=10, ge=0, description="Alpha Matting (Erode Structure Size)"
        ),
        ppm: bool = Form(default=False, description="Post Process Mask"),
    ):
        """
        Replace image background with any custom RGB color.
        Supports PNG, JPG, and JPEG image formats.
        Example: For blue background, use red=0, green=0, blue=255
        """
        validate_image_file(file)
        content = await file.read()
        
        # Custom color with full opacity (alpha=255)
        custom_bgcolor = (red, green, blue, 255)
        
        session_obj = get_or_create_session(model)

        result = await asyncify(remove)(
            content,
            session=session_obj,
            alpha_matting=a,
            alpha_matting_foreground_threshold=af,
            alpha_matting_background_threshold=ab,
            alpha_matting_erode_size=ae,
            only_mask=False,
            post_process_mask=ppm,
            bgcolor=custom_bgcolor,
        )
        
        return Response(
            result,
            media_type="image/png",
            headers={
                "Content-Disposition": f"attachment; filename=custom_bg_{file.filename}"
            }
        )

    def gr_app(app):
        def inference(input_path, model, *args):
            output_path = "output.png"
            a, af, ab, ae, om, ppm, cmd_args = args

            kwargs = {
                "alpha_matting": a,
                "alpha_matting_foreground_threshold": af,
                "alpha_matting_background_threshold": ab,
                "alpha_matting_erode_size": ae,
                "only_mask": om,
                "post_process_mask": ppm,
            }

            if cmd_args:
                kwargs.update(json.loads(cmd_args))
            kwargs["session"] = new_session(model, **kwargs)

            with open(input_path, "rb") as i:
                with open(output_path, "wb") as o:
                    input = i.read()
                    output = remove(input, **kwargs)
                    o.write(output)
            return os.path.join(output_path)

        interface = gr.Interface(
            inference,
            [
                gr.components.Image(type="filepath", label="Input"),
                gr.components.Dropdown(sessions_names, value="u2net", label="Models"),
                gr.components.Checkbox(value=True, label="Alpha matting"),
                gr.components.Slider(
                    value=240, minimum=0, maximum=255, label="Foreground threshold"
                ),
                gr.components.Slider(
                    value=10, minimum=0, maximum=255, label="Background threshold"
                ),
                gr.components.Slider(
                    value=40, minimum=0, maximum=255, label="Erosion size"
                ),
                gr.components.Checkbox(value=False, label="Only mask"),
                gr.components.Checkbox(value=True, label="Post process mask"),
                gr.components.Textbox(label="Arguments"),
            ],
            gr.components.Image(type="filepath", label="Output"),
            concurrency_limit=3,
            analytics_enabled=False,
        )

        app = gr.mount_gradio_app(app, interface, path="/")
        return app

    print(
        f"To access the Swagger UI documentation, go to http://{'localhost' if host == '0.0.0.0' else host}:{port}/api"
    )
    print(
        f"To access the ReDoc documentation, go to http://{'localhost' if host == '0.0.0.0' else host}:{port}/redoc"
    )
    print(
        f"To access the UI, go to http://{'localhost' if host == '0.0.0.0' else host}:{port}"
    )

    uvicorn.run(gr_app(app), host=host, port=port, log_level=log_level)
