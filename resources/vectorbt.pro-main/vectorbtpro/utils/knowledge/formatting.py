# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing content formatting classes.

See `vectorbtpro.utils.knowledge` for the toy dataset.
"""

import inspect
import io
import re
import sys
import time
from pathlib import Path

from vectorbtpro import _typing as tp
from vectorbtpro.utils import checks
from vectorbtpro.utils.config import Configured, flat_merge_dicts
from vectorbtpro.utils.module_ import get_caller_qualname
from vectorbtpro.utils.path_ import check_mkdir
from vectorbtpro.utils.template import CustomTemplate, RepFunc, SafeSub

if tp.TYPE_CHECKING:
    from IPython.display import DisplayHandle as DisplayHandleT
else:
    DisplayHandleT = "IPython.display.DisplayHandle"

__all__ = [
    "ContentFormatter",
    "PlainFormatter",
    "IPythonFormatter",
    "IPythonMarkdownFormatter",
    "IPythonHTMLFormatter",
    "HTMLFileFormatter",
]


class ToMarkdown(Configured):
    """Class for converting text to Markdown.

    Args:
        remove_code_title (Optional[bool]): Whether to remove the `title` attribute from a code block
            and display it above the block.
        even_indentation (Optional[bool]): Whether leading spaces should be adjusted to
            even numbers (e.g., converting 3 spaces to 4).
        newline_before_list (Optional[bool]): Whether a newline should be inserted before list items.
        **kwargs: Keyword arguments for `vectorbtpro.utils.config.Configured`.

    !!! info
        For default settings, see `vectorbtpro._settings.knowledge` and its sub-configuration `formatting`.
    """

    _settings_path: tp.SettingsPath = ["knowledge", "knowledge.formatting"]

    def __init__(
        self,
        remove_code_title: tp.Optional[bool] = None,
        even_indentation: tp.Optional[bool] = None,
        newline_before_list: tp.Optional[bool] = None,
        **kwargs,
    ) -> None:
        Configured.__init__(
            self,
            remove_code_title=remove_code_title,
            even_indentation=even_indentation,
            newline_before_list=newline_before_list,
            **kwargs,
        )

        remove_code_title = self.resolve_setting(remove_code_title, "remove_code_title")
        even_indentation = self.resolve_setting(even_indentation, "even_indentation")
        newline_before_list = self.resolve_setting(newline_before_list, "newline_before_list")

        self._remove_code_title = remove_code_title
        self._even_indentation = even_indentation
        self._newline_before_list = newline_before_list

    @property
    def remove_code_title(self) -> bool:
        """Whether to remove the `title` attribute from a code block and display it above the block.

        Returns:
            bool: True if the `title` attribute should be removed, False otherwise.
        """
        return self._remove_code_title

    @property
    def newline_before_list(self) -> bool:
        """Whether a newline should be inserted before list items.

        Returns:
            bool: True if a newline should be inserted before list items, False otherwise.
        """
        return self._newline_before_list

    @property
    def even_indentation(self) -> bool:
        """Whether leading spaces should be adjusted to even numbers (e.g., converting 3 spaces to 4).

        Returns:
            bool: True if leading spaces should be adjusted to even numbers, False otherwise.
        """
        return self._even_indentation

    def to_markdown(self, text: str) -> str:
        """Return the given text converted to Markdown format.

        Args:
            text (str): Text to convert to Markdown.

        Returns:
            str: Converted Markdown text.
        """
        markdown = text
        if self.remove_code_title:

            def _replace_code_block(match):
                language = match.group(1)
                title = match.group(2)
                code = match.group(3)
                if title:
                    title_md = f"**{title}**\n\n"
                else:
                    title_md = ""
                code_md = f"```{language}\n{code}\n```"
                return title_md + code_md

            code_block_pattern = re.compile(r'```(\w+)\s+title="([^"]*)"\s*\n(.*?)\n```', re.DOTALL)
            markdown = code_block_pattern.sub(_replace_code_block, markdown)

        if self.even_indentation:
            leading_spaces_pattern = re.compile(r"^( +)(?=\S|$|\n)")
            fixed_lines = []
            for line in markdown.splitlines(keepends=True):
                match = leading_spaces_pattern.match(line)
                if match and len(match.group(0)) % 2 != 0:
                    line = " " + line
                fixed_lines.append(line)
            markdown = "".join(fixed_lines)

        if self.newline_before_list:
            markdown = re.sub(r"(?<=[^\n])\n(?=[ \t]*(?:[*+-]\s|\d+\.\s))", "\n\n", markdown)

        return markdown


def to_markdown(text: str, **kwargs) -> str:
    """Return the Markdown conversion of the given text using `ToMarkdown`.

    Args:
        text (str): Text to convert to Markdown.
        **kwargs: Keyword arguments for `ToMarkdown`.

    Returns:
        str: Converted Markdown text.
    """
    return ToMarkdown(**kwargs).to_markdown(text)


class ToHTML(Configured):
    """Class for converting Markdown text to HTML.

    Args:
        resolve_extensions (Optional[bool]): Whether to resolve Markdown extensions,
            favoring `pymdownx` extensions when available.
        make_links (Optional[bool]): Whether to convert raw URLs within HTML `p` and
            `span` tags into hyperlinks.
        frontmatter_to_code (Optional[bool]): Whether to convert frontmatter (YAML) blocks to code blocks.
        **markdown_kwargs: Keyword arguments for Markdown conversion.

    !!! info
        For default settings, see `vectorbtpro._settings.knowledge` and its sub-configuration `formatting`.
    """

    _expected_keys_mode: tp.ExpectedKeysMode = "disable"

    _settings_path: tp.SettingsPath = ["knowledge", "knowledge.formatting"]

    def __init__(
        self,
        resolve_extensions: tp.Optional[bool] = None,
        make_links: tp.Optional[bool] = None,
        frontmatter_to_code: tp.Optional[bool] = None,
        **markdown_kwargs,
    ) -> None:
        Configured.__init__(
            self,
            resolve_extensions=resolve_extensions,
            make_links=make_links,
            frontmatter_to_code=frontmatter_to_code,
            **markdown_kwargs,
        )

        resolve_extensions = self.resolve_setting(resolve_extensions, "resolve_extensions")
        make_links = self.resolve_setting(make_links, "make_links")
        markdown_kwargs = self.resolve_setting(markdown_kwargs, "markdown_kwargs", merge=True)
        frontmatter_to_code = self.resolve_setting(frontmatter_to_code, "frontmatter_to_code")

        self._resolve_extensions = resolve_extensions
        self._make_links = make_links
        self._markdown_kwargs = markdown_kwargs
        self._frontmatter_to_code = frontmatter_to_code

    @property
    def resolve_extensions(self) -> bool:
        """Whether Markdown extensions should be resolved, favoring `pymdownx` extensions when available.

        Returns:
            bool: True if extensions should be resolved, False otherwise.
        """
        return self._resolve_extensions

    @property
    def make_links(self) -> bool:
        """Whether raw URLs in HTML `p` and `span` elements should be converted into clickable links.

        Returns:
            bool: True if raw URLs should be converted into links, False otherwise.
        """
        return self._make_links

    @property
    def markdown_kwargs(self) -> tp.Kwargs:
        """Keyword arguments for `markdown.markdown` for conversion.

        Returns:
            Kwargs: Dictionary of keyword arguments for Markdown conversion.
        """
        return self._markdown_kwargs

    @property
    def frontmatter_to_code(self) -> bool:
        """Whether to convert frontmatter (YAML) blocks to code blocks.

        Returns:
            bool: True if frontmatter (YAML) blocks should be converted to code blocks, False otherwise.
        """
        return self._frontmatter_to_code

    def to_html(self, markdown: str) -> str:
        """Return the HTML conversion of the given Markdown text.

        Args:
            markdown (str): Markdown text to convert to HTML.

        Returns:
            str: Converted HTML text.
        """
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("markdown")
        import markdown as md

        markdown_kwargs = dict(self.markdown_kwargs)
        extensions = markdown_kwargs.pop("extensions", [])
        if self.resolve_extensions:
            from vectorbtpro.utils.module_ import check_installed

            filtered_extensions = [
                ext
                for ext in extensions
                if "." not in ext or check_installed(ext.partition(".")[0])
            ]
            ext_set = set(filtered_extensions)
            remove_fenced_code = "fenced_code" in ext_set and "pymdownx.superfences" in ext_set
            remove_codehilite = "codehilite" in ext_set and "pymdownx.highlight" in ext_set
            if remove_fenced_code or remove_codehilite:
                filtered_extensions = [
                    ext
                    for ext in filtered_extensions
                    if not (
                        (ext == "fenced_code" and remove_fenced_code)
                        or (ext == "codehilite" and remove_codehilite)
                    )
                ]
            extensions = filtered_extensions
        if self.frontmatter_to_code:

            def _looks_like_yaml(lines):
                if not lines or lines[0].strip() == "":
                    return False
                return True

            def _frontmatter_to_code(markdown):
                out = io.StringIO()
                lines = markdown.splitlines(keepends=True)
                i = 0
                n = len(lines)

                while i < n:
                    line = lines[i]
                    if line.strip() == "---":
                        start = i
                        i += 1
                        block = []
                        while i < n and lines[i].strip() != "---":
                            block.append(lines[i])
                            i += 1
                        if i < n and lines[i].strip() == "---":
                            if _looks_like_yaml(block):
                                out.write(lines[start].replace("---", "```yaml", 1))
                                out.write("".join(block))
                                out.write(lines[i].replace("---", "```", 1))
                            else:
                                out.write("".join(lines[start : i + 1]))
                            i += 1
                        else:
                            out.write("".join(lines[start:]))
                            break
                    else:
                        out.write(line)
                        i += 1

                return out.getvalue()

            markdown = _frontmatter_to_code(markdown)

        html = md.markdown(markdown, extensions=extensions, **markdown_kwargs)

        if self.make_links:
            tag_pattern = re.compile(r"<(p|span)(\s[^>]*)?>(.*?)</\1>", re.DOTALL | re.IGNORECASE)
            url_pattern = re.compile(
                r'(https?://[^\s<>"\'`]+?)(?=[.,;:!?)\]]*(?:\s|$))', re.IGNORECASE
            )

            def _replace_urls(match, _url_pattern=url_pattern):
                tag = match.group(1)
                attributes = match.group(2) if match.group(2) else ""
                content = match.group(3)
                parts = re.split(r"(<a\b[^>]*>.*?</a>)", content, flags=re.DOTALL | re.IGNORECASE)
                for i, part in enumerate(parts):
                    if not re.match(r"<a\b[^>]*>.*?</a>", part, re.DOTALL | re.IGNORECASE):
                        part = _url_pattern.sub(r'<a href="\1">\1</a>', part)
                        parts[i] = part
                new_content = "".join(parts)
                return f"<{tag}{attributes}>{new_content}</{tag}>"

            html = tag_pattern.sub(_replace_urls, html)
        return html.strip()


def to_html(text: str, **kwargs) -> str:
    """Return the HTML conversion of the given Markdown text using `ToHTML`.

    Args:
        text (str): Markdown text to convert to HTML.
        **kwargs: Keyword arguments for `ToHTML`.

    Returns:
        str: Converted HTML text.
    """
    return ToHTML(**kwargs).to_html(text)


class FormatHTML(Configured):
    """Class to format HTML.

    This class formats HTML content using a customizable template. It supports code highlighting
    via Pygments if enabled, and allows injection of additional CSS rules, extra HTML elements
    in the `<head>`, and JavaScript or inline scripts in the `<body>`.

    Args:
        html_template (Optional[CustomTemplateLike]): Template for HTML formatting,
            as a string, function, or custom template.
        style_extras (Optional[MaybeList[str]]): Extra CSS rules for the `<style>` element.
        head_extras (Optional[MaybeList[str]]): Extra HTML elements to inject into the `<head>` section.
        body_extras (Optional[MaybeList[str]]): Extra content to insert at the end of the `<body>` section.
        invert_colors (Optional[bool]): Flag to enable color inversion.
        invert_colors_style (Optional[str]): CSS styles applied when colors are inverted.
        auto_scroll (Optional[bool]): Flag to enable automatic scrolling during refreshing.
        auto_scroll_body (Optional[str]): HTML or script to facilitate auto scrolling in the body.
        show_spinner (Optional[bool]): Flag to display a loading spinner during refreshing.
        spinner_style (Optional[str]): CSS style for the spinner.
        spinner_body (Optional[str]): HTML or script for spinner placement.
        use_pygments (Optional[bool]): Flag to enable code highlighting with Pygments.
        pygments_kwargs (KwargsLike): Keyword arguments for `pygments.formatters.HtmlFormatter`.
        template_context (KwargsLike): Additional context for template substitution.
        **kwargs: Keyword arguments for `vectorbtpro.utils.config.Configured`.

    !!! info
        For default settings, see `vectorbtpro._settings.knowledge` and its sub-configuration `formatting`.
    """

    _settings_path: tp.SettingsPath = ["knowledge", "knowledge.formatting"]

    def __init__(
        self,
        html_template: tp.Optional[tp.CustomTemplateLike] = None,
        style_extras: tp.Optional[tp.MaybeList[str]] = None,
        head_extras: tp.Optional[tp.MaybeList[str]] = None,
        body_extras: tp.Optional[tp.MaybeList[str]] = None,
        invert_colors: tp.Optional[bool] = None,
        invert_colors_style: tp.Optional[str] = None,
        auto_scroll: tp.Optional[bool] = None,
        auto_scroll_body: tp.Optional[str] = None,
        show_spinner: tp.Optional[bool] = None,
        spinner_style: tp.Optional[str] = None,
        spinner_body: tp.Optional[str] = None,
        use_pygments: tp.Optional[bool] = None,
        pygments_kwargs: tp.KwargsLike = None,
        template_context: tp.KwargsLike = None,
        **kwargs,
    ) -> None:
        from vectorbtpro.utils.module_ import assert_can_import, check_installed

        Configured.__init__(
            self,
            html_template=html_template,
            style_extras=style_extras,
            head_extras=head_extras,
            body_extras=body_extras,
            invert_colors=invert_colors,
            invert_colors_style=invert_colors_style,
            auto_scroll=auto_scroll,
            auto_scroll_body=auto_scroll_body,
            show_spinner=show_spinner,
            spinner_style=spinner_style,
            spinner_body=spinner_body,
            use_pygments=use_pygments,
            pygments_kwargs=pygments_kwargs,
            template_context=template_context,
            **kwargs,
        )

        html_template = self.resolve_setting(html_template, "html_template")
        invert_colors = self.resolve_setting(invert_colors, "invert_colors")
        invert_colors_style = self.resolve_setting(invert_colors_style, "invert_colors_style")
        auto_scroll = self.resolve_setting(auto_scroll, "auto_scroll")
        auto_scroll_body = self.resolve_setting(auto_scroll_body, "auto_scroll_body")
        show_spinner = self.resolve_setting(show_spinner, "show_spinner")
        spinner_style = self.resolve_setting(spinner_style, "spinner_style")
        spinner_body = self.resolve_setting(spinner_body, "spinner_body")
        use_pygments = self.resolve_setting(use_pygments, "use_pygments")
        pygments_kwargs = self.resolve_setting(pygments_kwargs, "pygments_kwargs", merge=True)
        template_context = self.resolve_setting(template_context, "template_context", merge=True)

        def _prepare_extras(extras):
            if extras is None:
                extras = []
            if isinstance(extras, str):
                extras = [extras]
            if not isinstance(extras, list):
                extras = list(extras)
            return "\n".join(extras)

        if isinstance(html_template, str):
            html_template = SafeSub(html_template)
        elif checks.is_function(html_template):
            html_template = RepFunc(html_template)
        elif not isinstance(html_template, CustomTemplate):
            raise TypeError("HTML template must be a string, function, or template")
        style_extras = _prepare_extras(self.get_setting("style_extras")) + _prepare_extras(
            style_extras
        )
        head_extras = _prepare_extras(self.get_setting("head_extras")) + _prepare_extras(
            head_extras
        )
        body_extras = _prepare_extras(self.get_setting("body_extras")) + _prepare_extras(
            body_extras
        )
        if invert_colors:
            style_extras = "\n".join([style_extras, invert_colors_style])
        if auto_scroll:
            body_extras = "\n".join([body_extras, auto_scroll_body])
        if show_spinner:
            style_extras = "\n".join([style_extras, spinner_style])
            body_extras = "\n".join([body_extras, spinner_body])
        if use_pygments is None:
            use_pygments = check_installed("pygments")
        if use_pygments:
            assert_can_import("pygments")
            from pygments.formatters import HtmlFormatter

            formatter = HtmlFormatter(**pygments_kwargs)
            highlight_css = formatter.get_style_defs(".highlight")
            if style_extras == "":
                style_extras = highlight_css
            else:
                style_extras = highlight_css + "\n" + style_extras

        self._html_template = html_template
        self._style_extras = style_extras
        self._head_extras = head_extras
        self._body_extras = body_extras
        self._template_context = template_context

    @property
    def html_template(self) -> CustomTemplate:
        """Template for HTML formatting, as a string, function, or custom template.

        Returns:
            CustomTemplate: HTML template used for formatting.
        """
        return self._html_template

    @property
    def style_extras(self) -> str:
        """Extra CSS rules for the `<style>` element.

        Returns:
            str: String with additional CSS rules.
        """
        return self._style_extras

    @property
    def head_extras(self) -> str:
        """Extra HTML elements to inject into the `<head>` section.

        Returns:
            str: String with additional head extras.
        """
        return self._head_extras

    @property
    def body_extras(self) -> str:
        """Extra content to insert at the end of the `<body>` section.

        Returns:
            str: String with additional body extras.
        """
        return self._body_extras

    @property
    def template_context(self) -> tp.Kwargs:
        """Additional context for template substitution.

        Returns:
            Kwargs: Dictionary of context variables for template substitution.
        """
        return self._template_context

    def format_html(
        self, title: str = "", html_metadata: str = "", html_content: str = "", **kwargs
    ) -> str:
        """Format HTML content using the configured template.

        Args:
            title (str): Title of the HTML document.
            html_metadata (str): HTML metadata elements, such as meta tags.
            html_content (str): HTML content to format.
            **kwargs: Additional parameters to merge into the template context.

        Returns:
            str: Formatted HTML string.
        """
        return self.html_template.substitute(
            flat_merge_dicts(
                self.template_context,
                dict(
                    title=title,
                    html_metadata=html_metadata,
                    html_content=html_content,
                    style_extras=self.style_extras,
                    head_extras=self.head_extras,
                    body_extras=self.body_extras,
                ),
                kwargs,
            ),
            eval_id="html_template",
        )


def format_html(**kwargs) -> str:
    """Format HTML content using the `FormatHTML` class.

    This function extracts configuration parameters from the provided keyword arguments,
    instantiates a `FormatHTML` object, and returns the resulting formatted HTML.

    Args:
        **kwargs: Keyword arguments for `FormatHTML` and `FormatHTML.format_html`.

    Returns:
        str: Resulting formatted HTML string.
    """
    from vectorbtpro.utils.parsing import get_func_arg_names

    init_kwargs = {}
    for k in get_func_arg_names(FormatHTML.__init__):
        if k in kwargs:
            init_kwargs[k] = kwargs.pop(k)
    return FormatHTML(**init_kwargs).format_html(**kwargs)


class ContentFormatter(Configured):
    """Class for formatting content.

    Args:
        output_to (Optional[Union[str, TextIO]]): Destination for output, which may be a file path or stream.
        flush_output (Optional[bool]): Whether to flush the output immediately after writing.
        buffer_output (Optional[bool]): Whether to buffer output before writing.
        close_output (Optional[bool]): Whether to close the output stream after writing.
        update_interval (Optional[float]): Time interval in seconds for updates.
        minimal_format (Optional[bool]): Boolean indicating if the input is minimally formatted.
        template_context (KwargsLike): Additional context for template substitution.
        **kwargs: Keyword arguments for `vectorbtpro.utils.config.Configured`.

    !!! info
        For default settings, see `vectorbtpro._settings.knowledge` and
        its sub-configurations `formatting` and `formatting.formatter_config`.
    """

    _short_name: tp.ClassVar[tp.Optional[str]] = None
    """Short alias for the class."""

    _settings_path: tp.SettingsPath = [
        "knowledge",
        "knowledge.formatting",
        "knowledge.formatting.formatter_config",
    ]

    def __init__(
        self,
        output_to: tp.Optional[tp.Union[str, tp.TextIO]] = None,
        flush_output: tp.Optional[bool] = None,
        buffer_output: tp.Optional[bool] = None,
        close_output: tp.Optional[bool] = None,
        update_interval: tp.Optional[float] = None,
        minimal_format: tp.Optional[bool] = None,
        template_context: tp.KwargsLike = None,
        **kwargs,
    ) -> None:
        Configured.__init__(
            self,
            output_to=output_to,
            flush_output=flush_output,
            buffer_output=buffer_output,
            close_output=close_output,
            update_interval=update_interval,
            minimal_format=minimal_format,
            template_context=template_context,
            **kwargs,
        )

        output_to = self.resolve_setting(output_to, "output_to")
        flush_output = self.resolve_setting(flush_output, "flush_output")
        buffer_output = self.resolve_setting(buffer_output, "buffer_output")
        close_output = self.resolve_setting(close_output, "close_output")
        update_interval = self.resolve_setting(update_interval, "update_interval")
        minimal_format = self.resolve_setting(minimal_format, "minimal_format")
        template_context = self.resolve_setting(template_context, "template_context", merge=True)

        if isinstance(output_to, (str, Path)):
            output_to = Path(output_to).open("w")
            if close_output is None:
                close_output = True
        else:
            if close_output is None:
                close_output = False

        self._output_to = output_to
        self._flush_output = flush_output
        self._buffer_output = buffer_output
        self._close_output = close_output
        self._update_interval = update_interval
        self._minimal_format = minimal_format
        self._template_context = template_context

        self._last_update = None
        self._lines = []
        self._current_line = []
        self._in_code_block = False
        self._code_block_indent = ""
        self._buffer = []
        self._content = ""

    @property
    def output_to(self) -> tp.Optional[tp.Union[str, tp.TextIO]]:
        """Output destination, which may be a file path or stream.

        Returns:
            Optional[Union[str, TextIO]]: Output destination; None if not set.
        """
        return self._output_to

    @property
    def flush_output(self) -> bool:
        """Boolean indicating if the output should be flushed immediately.

        Returns:
            bool: True if the output should be flushed immediately, False otherwise.
        """
        return self._flush_output

    @property
    def buffer_output(self) -> bool:
        """Boolean indicating if the output is buffered.

        Returns:
            bool: True if the output is buffered, False otherwise.
        """
        return self._buffer_output

    @property
    def close_output(self) -> bool:
        """Boolean indicating if the output stream should be closed after writing.

        Returns:
            bool: True if the output stream should be closed after writing, False otherwise.
        """
        return self._close_output

    @property
    def update_interval(self) -> tp.Optional[float]:
        """Update interval in seconds.

        Returns:
            Optional[float]: Time interval in seconds for updates; None if not set.
        """
        return self._update_interval

    @property
    def minimal_format(self) -> bool:
        """Whether minimal formatting is applied.

        Returns:
            bool: True if minimal formatting is applied, False otherwise.
        """
        return self._minimal_format

    @property
    def template_context(self) -> tp.Kwargs:
        """Additional context for template substitution.

        Returns:
            Kwargs: Dictionary of context variables for template substitution.
        """
        return self._template_context

    @property
    def last_update(self) -> tp.Optional[int]:
        """Timestamp of the last update.

        Returns:
            Optional[int]: UNIX timestamp representing the last update time; None if not set.
        """
        return self._last_update

    @property
    def lines(self) -> tp.List[str]:
        """List of formatted lines.

        Returns:
            List[str]: List containing the formatted lines.
        """
        return self._lines

    @property
    def current_line(self) -> tp.List[str]:
        """List of string segments constituting the current line.

        Returns:
            List[str]: List of string segments for the current line.
        """
        return self._current_line

    @property
    def in_code_block(self) -> bool:
        """Whether the formatter is currently inside a code block.

        Returns:
            bool: True if currently inside a code block, False otherwise.
        """
        return self._in_code_block

    @property
    def code_block_indent(self) -> str:
        """Indentation used for the current code block.

        Returns:
            str: Indentation string for the current code block.
        """
        return self._code_block_indent

    @property
    def buffer(self) -> tp.List[str]:
        """List of buffered strings.

        Returns:
            List[str]: List containing the buffered strings.
        """
        return self._buffer

    @property
    def content(self) -> str:
        """Complete formatted content.

        Returns:
            str: Complete formatted content as a single string.
        """
        return self._content

    def initialize(self) -> None:
        """Initialize the formatter by setting the last update time to the current time.

        Returns:
            None
        """
        self._last_update = time.time()

    def format_line(self, line: str) -> str:
        """Format the provided line to process code block markers.

        Args:
            line (str): Line to format.

        Returns:
            str: Processed line with updated code block state.
        """
        start = 0
        while True:
            idx = line.find("```", start)
            if idx == -1:
                break
            if not self.in_code_block:
                self._in_code_block = True
                if line[:idx].strip() == "":
                    self._code_block_indent = line[:idx]
                else:
                    self._code_block_indent = ""
            else:
                self._in_code_block = False
            start = idx + 3
        return line

    def flush(self, final: bool = False) -> None:
        """Flush the buffered content and process complete and incomplete lines.

        This method processes the current contents of the buffer, formatting complete lines and
        appending them to the overall content.

        Args:
            final (bool): Whether the update finalizes the content.

        Returns:
            None
        """
        new_content = "".join(self.buffer)
        self.buffer.clear()

        lines = new_content.splitlines(keepends=True)
        for line in lines:
            if final or line.endswith("\n") or line.endswith("\r\n"):
                stripped_line = line.rstrip("\r\n")
                self.current_line.append(stripped_line)
                complete_line = "".join(self.current_line)
                formatted_line = self.format_line(complete_line)
                self.lines.append(formatted_line + line[len(stripped_line) :])
                self.current_line.clear()
            else:
                self.current_line.append(line)
        if final and self.current_line:
            complete_line = "".join(self.current_line)
            formatted_line = self.format_line(complete_line)
            self.lines.append(formatted_line)
            self.current_line.clear()

        if final:
            self._content = "".join(self.lines)
        else:
            content = self.lines.copy()
            if self.current_line:
                content.extend(self.current_line)
                if self.in_code_block:
                    content.append("\n" + self.code_block_indent + "```")
            else:
                if self.in_code_block:
                    content.append(self.code_block_indent + "```")
            self._content = "".join(content)

    def buffer_update(self) -> None:
        """If buffering is enabled and an output destination is set, print the buffered content immediately.

        Returns:
            None
        """
        if self.buffer_output and self.output_to is not None:
            print("".join(self.buffer), end="", file=self.output_to, flush=self.flush_output)

    def update(self, final: bool = False) -> None:
        """Update the content by processing the buffer and flushing outputs if necessary.

        Args:
            final (bool): Whether the update finalizes the content.

        Returns:
            None
        """
        self._last_update = time.time()
        if self.buffer:
            self.buffer_update()
        if self.buffer or (final and self.current_line):
            self.flush(final=final)

    def append(self, new_content: str, final: bool = False) -> None:
        """Append new content to the buffer and perform an update if necessary.

        Args:
            new_content (str): String content to append.
            final (bool): Whether the update finalizes the content.

        Returns:
            None
        """
        if not self.buffer_output and self.output_to is not None:
            print(new_content, end="", file=self.output_to, flush=self.flush_output)
        self.buffer.append(new_content)
        if (
            final
            or self.last_update is None
            or self.update_interval is None
            or (time.time() - self.last_update >= self.update_interval)
        ):
            self.update(final=final)

    def append_once(self, content: str) -> None:
        """Append final content to the buffer and finalize the formatting process.

        Args:
            content (str): Final content to append.

        Returns:
            None
        """
        if self.last_update is None:
            self.initialize()
        self.append(content, final=True)
        self.finalize(update=False)

    def finalize(self, update: bool = True) -> None:
        """Perform a final content update and close the output stream if configured.

        Args:
            update (bool): Whether to update the content before finalizing.

        Returns:
            None
        """
        if update:
            self.update(final=True)
        if self.close_output and self.output_to is not None:
            self.output_to.close()

    def __enter__(self) -> tp.Self:
        self.initialize()
        return self

    def __exit__(self, *args) -> None:
        self.finalize()


class PlainFormatter(ContentFormatter):
    """Class for formatting plain content.

    !!! info
        For default settings, see `formatting.formatter_configs.plain` in `vectorbtpro._settings.knowledge`.
    """

    _short_name = "plain"

    _settings_path: tp.SettingsPath = "knowledge.formatting.formatter_configs.plain"

    def buffer_update(self) -> None:
        print("".join(self.buffer), end="")


class IPythonFormatter(ContentFormatter):
    """Class for formatting plain content in IPython.

    Args:
        *args: Positional arguments for `ContentFormatter`.
        **kwargs: Keyword arguments for `ContentFormatter`.

    !!! info
        For default settings, see `formatting.formatter_configs.ipython` in `vectorbtpro._settings.knowledge`.
    """

    _short_name = "ipython"

    _settings_path: tp.SettingsPath = "knowledge.formatting.formatter_configs.ipython"

    def __init__(self, *args, **kwargs) -> None:
        ContentFormatter.__init__(self, *args, **kwargs)

        self._display_handle = None

    @property
    def display_handle(self) -> tp.Optional[DisplayHandleT]:
        """IPython display handle.

        Returns:
            Optional[DisplayHandleT]: IPython display handle; None if not set.
        """
        return self._display_handle

    def initialize(self) -> None:
        ContentFormatter.initialize(self)

        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("IPython")
        from IPython.display import display

        self._display_handle = display("", display_id=True)

    def update_display(self) -> None:
        """Update the IPython display with the current content.

        Returns:
            None
        """
        self.display_handle.update(self.content)

    def update(self, final: bool = False) -> None:
        ContentFormatter.update(self, final=final)
        self.update_display()


class IPythonMarkdownFormatter(IPythonFormatter):
    """Class for formatting Markdown content in IPython.

    Args:
        *args: Positional arguments for `IPythonFormatter`.
        to_markdown_kwargs (KwargsLike): Keyword arguments for `to_markdown`.
        **kwargs: Keyword arguments for `IPythonFormatter`.

    !!! info
        For default settings, see `formatting.formatter_configs.ipython_markdown` in `vectorbtpro._settings.knowledge`.
    """

    _short_name = "ipython_markdown"

    _settings_path: tp.SettingsPath = "knowledge.formatting.formatter_configs.ipython_markdown"

    def __init__(self, *args, to_markdown_kwargs: tp.KwargsLike = None, **kwargs) -> None:
        IPythonFormatter.__init__(
            self,
            *args,
            to_markdown_kwargs=to_markdown_kwargs,
            **kwargs,
        )

        if self.minimal_format:
            to_markdown_kwargs = self.resolve_setting(
                to_markdown_kwargs,
                "to_markdown_kwargs",
                sub_path="minimal_format_config",
                merge=True,
            )
        else:
            to_markdown_kwargs = self.resolve_setting(
                to_markdown_kwargs, "to_markdown_kwargs", merge=True
            )

        self._to_markdown_kwargs = to_markdown_kwargs

    @property
    def to_markdown_kwargs(self) -> tp.Kwargs:
        """Keyword arguments forwarded to `to_markdown`.

        Returns:
            Kwargs: Dictionary of keyword arguments for Markdown conversion.
        """
        return self._to_markdown_kwargs

    def update_display(self) -> None:
        from IPython.display import Markdown

        markdown_content = to_markdown(self.content, **self.to_markdown_kwargs)
        self.display_handle.update(Markdown(markdown_content))


class IPythonHTMLFormatter(IPythonFormatter):
    """Class for formatting HTML content in IPython.

    Args:
        *args: Positional arguments for `IPythonFormatter`.
        to_markdown_kwargs (KwargsLike): Keyword arguments for `to_markdown`.
        to_html_kwargs (KwargsLike): Keyword arguments for `to_html`.
        **kwargs: Keyword arguments for `IPythonFormatter`.

    !!! info
        For default settings, see `formatting.formatter_configs.ipython_html` in `vectorbtpro._settings.knowledge`.
    """

    _short_name = "ipython_html"

    _settings_path: tp.SettingsPath = "knowledge.formatting.formatter_configs.ipython_html"

    def __init__(
        self,
        *args,
        to_markdown_kwargs: tp.KwargsLike = None,
        to_html_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> None:
        IPythonFormatter.__init__(
            self,
            *args,
            to_markdown_kwargs=to_markdown_kwargs,
            to_html_kwargs=to_html_kwargs,
            **kwargs,
        )

        if self.minimal_format:
            to_markdown_kwargs = self.resolve_setting(
                to_markdown_kwargs,
                "to_markdown_kwargs",
                sub_path="minimal_format_config",
                merge=True,
            )
            to_html_kwargs = self.resolve_setting(
                to_html_kwargs, "to_html_kwargs", sub_path="minimal_format_config", merge=True
            )
        else:
            to_markdown_kwargs = self.resolve_setting(
                to_markdown_kwargs, "to_markdown_kwargs", merge=True
            )
            to_html_kwargs = self.resolve_setting(to_html_kwargs, "to_html_kwargs", merge=True)

        self._to_markdown_kwargs = to_markdown_kwargs
        self._to_html_kwargs = to_html_kwargs

    @property
    def to_markdown_kwargs(self) -> tp.Kwargs:
        """Keyword arguments for `to_markdown`.

        Returns:
            Kwargs: Dictionary of keyword arguments for Markdown conversion.
        """
        return self._to_markdown_kwargs

    @property
    def to_html_kwargs(self) -> tp.Kwargs:
        """Keyword arguments for `to_html`.

        Returns:
            Kwargs: Dictionary of keyword arguments for HTML conversion.
        """
        return self._to_html_kwargs

    def update_display(self) -> None:
        from IPython.display import HTML

        markdown_content = to_markdown(self.content, **self.to_markdown_kwargs)
        html_content = to_html(markdown_content, **self.to_html_kwargs)
        self.display_handle.update(HTML(html_content))


class HTMLFileFormatter(ContentFormatter):
    """Class for formatting static HTML files.

    Args:
        *args: Positional arguments for `ContentFormatter`.
        page_title (str): Title of the HTML page.
        refresh_page (Optional[bool]): Determines whether the HTML page should refresh.
        dir_path (Optional[PathLike]): Directory path for saving HTML files.
        mkdir_kwargs (KwargsLike): Keyword arguments for directory creation.

            See `vectorbtpro.utils.path_.check_mkdir`.
        temp_files (Optional[bool]): Indicates if HTML content is saved as temporary files.
        file_prefix_len (Optional[int]): Number of characters for the truncated title prefix.
        file_suffix_len (Optional[int]): Number of characters for the random hash suffix.
        auto_scroll (Optional[bool]): Flag to enable automatic scrolling during refreshing.
        show_spinner (Optional[bool]): Flag to display a loading spinner during refreshing.
        open_browser (Optional[bool]): Flag indicating whether to open the web browser.
        to_markdown_kwargs (KwargsLike): Keyword arguments for `to_markdown`.
        to_html_kwargs (KwargsLike): Keyword arguments for `to_html`.
        format_html_kwargs (KwargsLike): Keyword arguments for `format_html`.
        **kwargs: Keyword arguments for `ContentFormatter`.

    !!! info
        For default settings, see `formatting.formatter_configs.html` in `vectorbtpro._settings.knowledge`.
    """

    _short_name = "html"

    _settings_path: tp.SettingsPath = "knowledge.formatting.formatter_configs.html"

    def __init__(
        self,
        *args,
        page_title: str = "",
        refresh_page: tp.Optional[bool] = None,
        dir_path: tp.Optional[tp.PathLike] = None,
        mkdir_kwargs: tp.KwargsLike = None,
        temp_files: tp.Optional[bool] = None,
        file_prefix_len: tp.Optional[int] = None,
        file_suffix_len: tp.Optional[int] = None,
        auto_scroll: tp.Optional[bool] = None,
        show_spinner: tp.Optional[bool] = None,
        open_browser: tp.Optional[bool] = None,
        to_markdown_kwargs: tp.KwargsLike = None,
        to_html_kwargs: tp.KwargsLike = None,
        format_html_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> None:
        ContentFormatter.__init__(
            self,
            *args,
            page_title=page_title,
            refresh_page=refresh_page,
            dir_path=dir_path,
            mkdir_kwargs=mkdir_kwargs,
            temp_files=temp_files,
            file_prefix_len=file_prefix_len,
            file_suffix_len=file_suffix_len,
            auto_scroll=auto_scroll,
            show_spinner=show_spinner,
            open_browser=open_browser,
            to_markdown_kwargs=to_markdown_kwargs,
            to_html_kwargs=to_html_kwargs,
            format_html_kwargs=format_html_kwargs,
            **kwargs,
        )

        refresh_page = self.resolve_setting(refresh_page, "refresh_page")
        dir_path = self.resolve_setting(dir_path, "dir_path")
        mkdir_kwargs = self.resolve_setting(mkdir_kwargs, "mkdir_kwargs", merge=True)
        temp_files = self.resolve_setting(temp_files, "temp_files")
        file_prefix_len = self.resolve_setting(file_prefix_len, "file_prefix_len")
        file_suffix_len = self.resolve_setting(file_suffix_len, "file_suffix_len")
        auto_scroll = self.resolve_setting(auto_scroll, "auto_scroll")
        show_spinner = self.resolve_setting(show_spinner, "show_spinner")
        open_browser = self.resolve_setting(open_browser, "open_browser")

        if self.minimal_format:
            to_markdown_kwargs = self.resolve_setting(
                to_markdown_kwargs,
                "to_markdown_kwargs",
                sub_path="minimal_format_config",
                merge=True,
            )
            to_html_kwargs = self.resolve_setting(
                to_html_kwargs, "to_html_kwargs", sub_path="minimal_format_config", merge=True
            )
            format_html_kwargs = self.resolve_setting(
                format_html_kwargs,
                "format_html_kwargs",
                sub_path="minimal_format_config",
                merge=True,
            )
        else:
            to_markdown_kwargs = self.resolve_setting(
                to_markdown_kwargs, "to_markdown_kwargs", merge=True
            )
            to_html_kwargs = self.resolve_setting(to_html_kwargs, "to_html_kwargs", merge=True)
            format_html_kwargs = self.resolve_setting(
                format_html_kwargs, "format_html_kwargs", merge=True
            )

        dir_path = self.resolve_setting(dir_path, "dir_path")
        template_context = self.template_context
        if isinstance(dir_path, CustomTemplate):
            cache_dir = self.get_setting("cache_dir", default=None)
            if cache_dir is not None:
                if isinstance(cache_dir, CustomTemplate):
                    cache_dir = cache_dir.substitute(template_context, eval_id="cache_dir")
                template_context = flat_merge_dicts(dict(cache_dir=cache_dir), template_context)
            release_dir = self.get_setting("release_dir", default=None)
            if release_dir is not None:
                if isinstance(release_dir, CustomTemplate):
                    release_dir = release_dir.substitute(template_context, eval_id="release_dir")
                template_context = flat_merge_dicts(dict(release_dir=release_dir), template_context)
            dir_path = dir_path.substitute(template_context, eval_id="dir_path")

        self._page_title = page_title
        self._refresh_page = refresh_page
        self._dir_path = dir_path
        self._mkdir_kwargs = mkdir_kwargs
        self._temp_files = temp_files
        self._file_prefix_len = file_prefix_len
        self._file_suffix_len = file_suffix_len
        self._auto_scroll = auto_scroll
        self._show_spinner = show_spinner
        self._open_browser = open_browser
        self._to_markdown_kwargs = to_markdown_kwargs
        self._to_html_kwargs = to_html_kwargs
        self._format_html_kwargs = format_html_kwargs

        self._file_handle = None

    @property
    def page_title(self) -> str:
        """Title of the HTML page.

        Returns:
            str: Title text for the HTML page.
        """
        return self._page_title

    @property
    def refresh_page(self) -> bool:
        """Determines whether the HTML page should refresh.

        Returns:
            bool: True if the page is set to refresh, otherwise False.
        """
        return self._refresh_page

    @property
    def dir_path(self) -> tp.Optional[tp.Path]:
        """Directory path for saving HTML files.

        Returns:
            Optional[Path]: Directory path where the HTML files are stored, or None if not set.
        """
        return self._dir_path

    @property
    def mkdir_kwargs(self) -> tp.Kwargs:
        """Keyword arguments for directory creation.

        See `vectorbtpro.utils.path_.check_mkdir`.

        Returns:
            Kwargs: Dictionary of keyword arguments used by the directory creation function.
        """
        return self._mkdir_kwargs

    @property
    def temp_files(self) -> bool:
        """Indicates if HTML content is saved as temporary files.

        Returns:
            bool: True if saving as temporary files is enabled, otherwise False.
        """
        return self._temp_files

    @property
    def file_prefix_len(self) -> int:
        """Number of characters used for the HTML file title prefix.

        Returns:
            int: Maximum length allowed for the truncated title prefix.
        """
        return self._file_prefix_len

    @property
    def file_suffix_len(self) -> int:
        """Number of characters used for the random hash suffix.

        Returns:
            int: Length of the random suffix appended to the file name.
        """
        return self._file_suffix_len

    @property
    def auto_scroll(self) -> bool:
        """Specifies whether automatic scrolling is enabled.

        Returns:
            bool: True if auto-scrolling is enabled during page refresh, otherwise False.
        """
        return self._auto_scroll

    @property
    def show_spinner(self) -> bool:
        """Indicates if a loading spinner is displayed during page refresh.

        Returns:
            bool: True if the spinner is enabled, otherwise False.
        """
        return self._show_spinner

    @property
    def open_browser(self) -> bool:
        """Determines whether the default browser should be opened.

        Returns:
            bool: True if the browser should automatically open, otherwise False.
        """
        return self._open_browser

    @property
    def to_markdown_kwargs(self) -> tp.Kwargs:
        """Keyword arguments passed to the `to_markdown` function.

        Returns:
            Kwargs: Dictionary containing settings for the Markdown conversion.
        """
        return self._to_markdown_kwargs

    @property
    def to_html_kwargs(self) -> tp.Kwargs:
        """Keyword arguments passed to the `to_html` function.

        Returns:
            Kwargs: Dictionary containing settings for the HTML conversion.
        """
        return self._to_html_kwargs

    @property
    def format_html_kwargs(self) -> tp.Kwargs:
        """Keyword arguments passed to the `format_html` function.

        Returns:
            Kwargs: Dictionary of keyword arguments used during HTML formatting.
        """
        return self._format_html_kwargs

    @property
    def file_handle(self) -> tp.Optional[tp.TextIO]:
        """File handle associated with the HTML output.

        Returns:
            Optional[TextIO]: Open file object for the HTML output, or None if not initialized.
        """
        return self._file_handle

    def format_html_content(self, html_content: str, final: bool = False) -> str:
        """Return formatted HTML content.

        Args:
            html_content (str): HTML content to format.
            final (bool): Whether the update finalizes the content.

        Returns:
            str: Formatted HTML content.
        """
        _format_html_kwargs = dict(self.format_html_kwargs)
        if not final and self.refresh_page:
            refresh_content = (
                max(1, int(self.update_interval)) if self.update_interval is not None else 1
            )
            head_extras = list(_format_html_kwargs.get("head_extras", []))
            if head_extras is None:
                head_extras = []
            if isinstance(head_extras, str):
                head_extras = [head_extras]
            else:
                head_extras = list(head_extras)
            head_extras.insert(0, f'<meta http-equiv="refresh" content="{refresh_content}">')
            _format_html_kwargs["head_extras"] = head_extras
            html_content = '<div id="overlay" class="overlay"></div>\n' + html_content
            if self.auto_scroll and "auto_scroll" not in _format_html_kwargs:
                _format_html_kwargs["auto_scroll"] = True
            if self.show_spinner and "show_spinner" not in _format_html_kwargs:
                _format_html_kwargs["show_spinner"] = True
        return format_html(
            title=self.page_title,
            html_content=html_content,
            **_format_html_kwargs,
        )

    def initialize(self) -> None:
        ContentFormatter.initialize(self)

        if not self.temp_files:
            import secrets
            import string

            check_mkdir(self.dir_path, **self.mkdir_kwargs)
            page_title = self.page_title.lower().replace(" ", "-")
            if len(page_title) > self.file_prefix_len:
                words = page_title.split("-")
                truncated_page_title = ""
                for word in words:
                    if len(truncated_page_title) + len(word) + 1 <= self.file_prefix_len:
                        truncated_page_title += word + "-"
                    else:
                        break
                truncated_page_title = truncated_page_title.rstrip("-")
            else:
                truncated_page_title = page_title
            suffix_chars = string.ascii_lowercase + string.digits
            random_suffix = "".join(
                secrets.choice(suffix_chars) for _ in range(self.file_suffix_len)
            )
            if truncated_page_title:
                short_filename = f"{truncated_page_title}-{random_suffix}.html"
            else:
                short_filename = f"{random_suffix}.html"
            file_path = self.dir_path / short_filename
            self._file_handle = open(str(file_path.resolve()), "w", encoding="utf-8")
        else:
            import tempfile

            self._file_handle = tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                prefix=get_caller_qualname() + "_",
                suffix=".html",
                delete=False,
            )
        if self.refresh_page:
            html = self.format_html_content("", final=False)
            self.file_handle.write(html)
            self.file_handle.flush()
        if self.open_browser:
            import webbrowser

            webbrowser.open("file://" + str(Path(self.file_handle.name).resolve()))

    def update(self, final: bool = False) -> None:
        ContentFormatter.update(self, final=final)

        markdown_content = to_markdown(self.content, **self.to_markdown_kwargs)
        html_content = to_html(markdown_content, **self.to_html_kwargs)
        html = self.format_html_content(html_content, final=final)
        self.file_handle.seek(0)
        self.file_handle.write(html)
        self.file_handle.truncate()
        self.file_handle.flush()


def resolve_formatter(formatter: tp.ContentFormatterLike) -> tp.MaybeType[ContentFormatter]:
    """Resolve a subclass or instance of `ContentFormatter`.

    Args:
        formatter (ContentFormatterLike): Identifier, subclass, or instance of `ContentFormatter`.

            Supported identifiers:

            * "plain" (`PlainFormatter`): Prints the raw output
            * "ipython" (`IPythonFormatter`): Renders unformatted text in a notebook environment
            * "ipython_markdown" (`IPythonMarkdownFormatter`): Renders Markdown in a notebook environment
            * "ipython_html" (`IPythonHTMLFormatter`): Renders HTML in a notebook environment
            * "ipython_auto": Chooses between "ipython_html" or "plain" based on the environment
            * "html" (`HTMLFileFormatter`): Writes a static HTML page and displays it in a browser

    Returns:
        ContentFormatter: Resolved formatter.

    !!! info
        For default settings, see `formatting` in `vectorbtpro._settings.knowledge`.
    """
    if formatter is None:
        from vectorbtpro._settings import settings

        chat_cfg = settings["knowledge"]["formatting"]
        formatter = chat_cfg["formatter"]
    if isinstance(formatter, str):
        if formatter.lower() == "ipython_auto":
            if checks.in_notebook():
                formatter = "ipython_html"
            else:
                formatter = "plain"
        current_module = sys.modules[__name__]
        found_formatter = None
        for name, cls in inspect.getmembers(current_module, inspect.isclass):
            if name.endswith("Formatter"):
                _short_name = getattr(cls, "_short_name", None)
                if _short_name is not None and _short_name.lower() == formatter.lower():
                    found_formatter = cls
                    break
        if found_formatter is None:
            raise ValueError(f"Invalid formatter: '{formatter}'")
        formatter = found_formatter
    if isinstance(formatter, type):
        checks.assert_subclass_of(formatter, ContentFormatter, arg_name="formatter")
    else:
        checks.assert_instance_of(formatter, ContentFormatter, arg_name="formatter")
    return formatter
