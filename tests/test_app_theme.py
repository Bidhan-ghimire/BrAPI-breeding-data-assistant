"""Exercise the installed Gradio launch theme comparison, without launching a server."""
import json

import gradio as gr
from gradio import utils
from gradio.themes.utils.fonts import FontEncoder

from app_style import build_theme


def test_theme_survives_the_launch_builtin_comparison():
    theme = build_theme()
    # This exact comparison crashed inside Blocks.launch after reporting the URL.
    is_custom = not any(theme.to_dict() == builtin.to_dict()
                        for builtin in utils.BUILT_IN_THEMES.values())
    assert is_custom
    for builtin in utils.BUILT_IN_THEMES.values():
        assert isinstance(theme.to_dict() == builtin.to_dict(), bool)
        assert isinstance(builtin.to_dict() == theme.to_dict(), bool)
    assert json.loads(json.dumps(theme.to_dict(), cls=FontEncoder))


def test_theme_fonts_remain_system_only():
    theme = build_theme()
    for font in [*theme._font, *theme._font_mono]:
        assert type(font) is gr.themes.Font
        assert font.stylesheet() == {"url": None, "css": None}
