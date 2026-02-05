"""MCP Figure Tool for reading figures and converting to base64."""
import base64
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

from loguru import logger
from PIL import Image
from pydantic import BaseModel, Field

from src.config.settings import get_settings
from src.tools.mcp_protocol import MCPTool, MCPToolOutput, ToolStatus


class FigureToolInput(BaseModel):
    """Input schema for Figure Tool."""

    figure_id: str = Field(description="Figure identifier (e.g., 'scatter1')")
    max_size: tuple[int, int] = Field(default=(1024, 1024), description="Max size")


class FigureData(BaseModel):
    """Figure data response."""

    figure_id: str
    base64_image: str
    mime_type: str = "image/png"
    width: int
    height: int


class FigureToolOutput(MCPToolOutput):
    """Output schema for Figure Tool."""

    data: Optional[FigureData] = None


class FigureTool(MCPTool):
    """MCP Tool for reading figure images and converting to base64."""

    name = "get_figure"
    description = "Retrieve a figure image by ID and convert to base64 for VLM analysis."
    input_schema = FigureToolInput

    def __init__(self, figures_dir: Optional[str] = None) -> None:
        """Initialize with figures directory."""
        self._figures_dir = Path(figures_dir or get_settings().figures_dir)
        logger.info(f"FigureTool initialized: {self._figures_dir}")

    def _find_file(self, figure_id: str) -> Optional[Path]:
        """Find figure file by ID."""
        for ext in [".png", ".jpg", ".jpeg", ".webp"]:
            path = self._figures_dir / f"{figure_id}{ext}"
            if path.exists():
                return path
        return None

    def _to_base64(self, image: Image.Image, fmt: str = "PNG") -> tuple[str, str]:
        """Convert PIL Image to base64 string."""
        buf = BytesIO()

        # Convert RGBA to RGB for JPEG
        if fmt.upper() in ["JPEG", "JPG"] and image.mode == "RGBA":
            image = image.convert("RGB")

        image.save(buf, format=fmt.upper() if fmt.upper() != "JPG" else "JPEG")
        buf.seek(0)
        mime = "image/jpeg" if fmt.upper() in ["JPEG", "JPG"] else "image/png"
        return base64.b64encode(buf.getvalue()).decode(), mime

    async def execute(self, **kwargs: Any) -> FigureToolOutput:
        """Execute figure retrieval and base64 conversion."""
        # Validate input
        inp = FigureToolInput(**kwargs)
        logger.info(f"Fetching figure: {inp.figure_id}")

        # Find file
        path = self._find_file(inp.figure_id)
        if not path:
            logger.warning(f"Figure not found: {inp.figure_id}")
            return FigureToolOutput(
                status=ToolStatus.NOT_FOUND,
                message=f"Figure '{inp.figure_id}' not found",
            )

        try:
            # Load and resize image
            image = Image.open(path)
            image.thumbnail(inp.max_size, Image.Resampling.LANCZOS)

            # Convert to base64
            fmt = path.suffix.upper().lstrip(".") or "PNG"
            b64, mime = self._to_base64(image, fmt)

            logger.info(f"Processed figure: {inp.figure_id}, size={image.size}")
            return FigureToolOutput(
                status=ToolStatus.SUCCESS,
                message=f"Retrieved '{inp.figure_id}'",
                data=FigureData(
                    figure_id=inp.figure_id,
                    base64_image=b64,
                    mime_type=mime,
                    width=image.width,
                    height=image.height,
                ),
            )
        except Exception as e:
            logger.error(f"Error processing figure: {e}")
            return FigureToolOutput(status=ToolStatus.ERROR, message=str(e))

    def list_figures(self) -> list[str]:
        """List available figure IDs."""
        if not self._figures_dir.exists():
            return []
        exts = {".png", ".jpg", ".jpeg", ".webp"}
        return sorted(p.stem for p in self._figures_dir.iterdir() if p.suffix.lower() in exts)


# Singleton instance
_figure_tool: Optional[FigureTool] = None


def get_figure_tool() -> FigureTool:
    """Get singleton FigureTool instance."""
    global _figure_tool
    if _figure_tool is None:
        _figure_tool = FigureTool()
    return _figure_tool
