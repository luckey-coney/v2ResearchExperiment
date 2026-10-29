"""Self-contained SVG and strict JSON reports for an actual model-input grid."""
import base64
from html import escape
from io import BytesIO
import json
import math
from pathlib import Path
import unicodedata

from PIL import Image


_FONT = "Microsoft YaHei, Noto Sans CJK SC, sans-serif"
_PALETTE = ("#f9bd42", "#8c72ef", "#29c6c0", "#f47a9b", "#5eacfa", "#c3d957")
_NONFINITE_REASON = "非有限值不可用，已写为 null"


def _json_safe(value, path, nonfinite, active=None):
    """Copy JSON-native values only; never stringify arbitrary model objects."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        nonfinite.append({"path": path or "/", "reason": _NONFINITE_REASON})
        return None
    if not isinstance(value, (dict, list)):
        raise TypeError(f"unsupported JSON value at {path or '/'}: {type(value).__name__}")
    active = set() if active is None else active
    if id(value) in active:
        raise ValueError(f"circular JSON value at {path or '/'}")
    active.add(id(value))
    try:
        if isinstance(value, list):
            return [_json_safe(item, f"{path}/{index}", nonfinite, active)
                    for index, item in enumerate(value)]
        copied = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"JSON object keys must be strings at {path or '/'}")
            pointer = key.replace("~", "~0").replace("/", "~1")
            copied[key] = _json_safe(item, f"{path}/{pointer}", nonfinite, active)
        return copied
    finally:
        active.remove(id(value))


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        numeric = float(value)
    except OverflowError:
        return None
    return numeric if math.isfinite(numeric) else None


def _map_status(values, size, *, regions=False):
    if values is None:
        return {"available": False, "reason": "未提供"}
    if not isinstance(values, list) or len(values) != size:
        return {"available": False, "reason": f"长度必须等于原始网格 Token 数 {size}"}
    if regions:
        valid = all(isinstance(item, int) and not isinstance(item, bool) and item >= 0
                    for item in values)
        reason = "区域标签必须为非负整数（0=背景，1..B=重点区域）"
    else:
        valid = all(_number(item) is not None for item in values)
        reason = "包含非有限值或非数值，问题相关性不可用"
    return {"available": valid, "reason": "可用" if valid else reason}


def _display(value, fallback="未提供"):
    if value is None:
        return fallback
    if isinstance(value, str):
        return value if value else "（空字符串）"
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def _xml(value):
    # XML 1.0 forbids control characters even when HTML-escaped.
    text = "".join(char if (char in "\t\n\r" or " " <= char <= "\ud7ff"
                            or "\ue000" <= char <= "\ufffd" or "\U00010000" <= char <= "\U0010ffff")
                   else "\ufffd" for char in str(value))
    return escape(text, quote=True)


def _wrap(text, width):
    lines = []
    for paragraph in text.splitlines() or [""]:
        current, used = "", 0
        for char in paragraph:
            units = 2 if unicodedata.east_asian_width(char) in "WF" else 1
            if current and used + units > width:
                lines.append(current)
                current, used = "", 0
            current += char
            used += units
        lines.append(current)
    return lines


def _timing_text(value):
    if isinstance(value, dict):
        return "；".join(f"{key}={_timing_text(item)}" for key, item in value.items()) or "未测"
    numeric = _number(value)
    if numeric is not None:
        return f"{numeric:.6g} ms" if numeric >= 0 else "不可用（负时延）"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return "不可用（非有限数值）"
    return "未测"


def _quality_text(record):
    quality = record.get("quality")
    if not isinstance(quality, dict):
        return "评价：未评分"
    protocol = _display(quality.get("protocol"))
    score = _number(quality.get("score"))
    if score is None:
        return f"评价：{protocol}；未评分；" + _display(quality.get("reason"))
    return f"评价：{protocol}，得分={score:.4g}（DEMO 协议，不是官方 VQA 指标）"


def _text(x, y, content, *, size=14, color="#d7e0ef", anchor="start"):
    return (f'<text x="{x:.3f}" y="{y:.3f}" font-size="{size}" '
            f'fill="{color}" text-anchor="{anchor}">{_xml(content)}</text>')


def _cell_rect(x, y, width, height, index, cols, attributes):
    row, col = divmod(index, cols)
    return (f'<rect x="{x + col * width:.4f}" y="{y + row * height:.4f}" '
            f'width="{width:.4f}" height="{height:.4f}" data-index="{index}" '
            f'data-row="{row}" data-col="{col}" {attributes}>')


def _render_svg(record, canvas, q, region_labels, selected, grid_shape, metadata, timing):
    rows, cols = grid_shape
    total = rows * cols
    selection = record.get("selection")
    selection = selection if isinstance(selection, dict) else {}
    details = [
        "问题：" + _display(record.get("question")),
        "回答：" + _display(record.get("answer")),
        _quality_text(record),
    ]
    if "reference_answer" in record:
        details.append("参考答案：" + _display(record["reference_answer"]))
    details.extend([
        "状态：" + _display(selection.get("status")) + "; K/N=" + f"{len(selected)}/{total}",
        "时延：" + timing,
    ])
    if selection.get("reason") is not None:
        details.append("状态原因：" + _display(selection["reason"]))
    if metadata["nonfinite_values"]:
        details.append("记录中存在非有限数值；相关值不可用，JSON 已写为 null。")
    details.extend("数据说明：" + warning for warning in metadata["warnings"])
    lines = [line for detail in details for line in _wrap(detail, 142)]
    top = 78 + len(lines) * 21
    panel_width, gap, margin, image_box = 352, 24, 24, 320
    image_scale = min(image_box / canvas.width, image_box / canvas.height)
    image_width, image_height = canvas.width * image_scale, canvas.height * image_scale
    image_y = top + 54 + (image_box - image_height) / 2
    panel_height = 454
    footer_y = top + panel_height + 30
    height = footer_y + 52
    encoded = BytesIO()
    canvas.save(encoded, format="PNG")
    uri = "data:image/png;base64," + base64.b64encode(encoded.getvalue()).decode("ascii")
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="1152" height="{height}" '
        f'viewBox="0 0 1152 {height}" role="img" aria-labelledby="report-title">',
        '<title id="report-title">视觉 Token 选择报告</title>',
        f'<g font-family="{_FONT}">',
        f'<rect width="1152" height="{height}" fill="#0d1422"/>',
        _text(24, 37, "视觉 Token 选择报告", size=23, color="#f2f5fb"),
    ]
    svg.extend(_text(24, 68 + index * 21, line) for index, line in enumerate(lines))
    # Close the common text group so panel IDs remain direct children of the SVG.
    svg.append("</g>")
    panels = (("question-relevance", "问题相关性热图"),
              ("focus-regions", "重点区域"),
              ("retained-tokens", "实际保留 Token"))
    retained = set(selected)
    for panel_index, (panel_id, title) in enumerate(panels):
        panel_x = margin + panel_index * (panel_width + gap)
        image_x = panel_x + (panel_width - image_width) / 2
        cell_width, cell_height = image_width / cols, image_height / rows
        svg.extend([
            f'<g id="{panel_id}" font-family="{_FONT}">',
            f'<rect x="{panel_x}" y="{top}" width="{panel_width}" height="{panel_height}" '
            'rx="12" fill="#162133" stroke="#2b3a52"/>',
            _text(panel_x + 16, top + 29, title, size=17, color="#f2f5fb"),
            f'<image x="{image_x:.4f}" y="{image_y:.4f}" width="{image_width:.4f}" '
            f'height="{image_height:.4f}" preserveAspectRatio="none" href="{uri}"/>',
        ])
        unavailable = None
        if panel_index == 0 and metadata["q_status"]["available"]:
            numeric_q = [_number(value) for value in q]
            low, high = min(numeric_q), max(numeric_q)
            span = high / 2 - low / 2
            for index, value in enumerate(numeric_q):
                strength = (value / 2 - low / 2) / span if span else 0.5
                red, blue = round(48 + 207 * strength), round(238 - 194 * strength)
                attributes = (f'class="heat-cell" fill="rgb({red},93,{blue})" '
                              'fill-opacity="0.62" stroke="#ffffff" stroke-opacity="0.15" '
                              'stroke-width="0.5"')
                svg.append(_cell_rect(image_x, image_y, cell_width, cell_height,
                                      index, cols, attributes)
                           + f'<title>{index}: q={value:.6g}</title></rect>')
            legend = [f"蓝→红：q 从 {low:.4g} 到 {high:.4g}",
                      "按本例范围线性归一化；相同数值同色。"]
        elif panel_index == 1 and metadata["region_status"]["available"]:
            for index, label in enumerate(region_labels):
                color = _PALETTE[(label - 1) % len(_PALETTE)] if label else "#07101d"
                opacity = "0.48" if label else "0.20"
                attributes = (f'class="region-cell" data-region="{label}" fill="{color}" '
                              f'fill-opacity="{opacity}" stroke="#ffffff" '
                              'stroke-opacity="0.25" stroke-width="0.5"')
                svg.append(_cell_rect(image_x, image_y, cell_width, cell_height,
                                      index, cols, attributes)
                           + f'<title>{index}: 区域 {label}</title></rect>')
                if label and min(cell_width, cell_height) >= 18:
                    row, col = divmod(index, cols)
                    svg.append(_text(image_x + (col + .5) * cell_width,
                                     image_y + (row + .5) * cell_height + 4,
                                     str(label), size=10, color="#ffffff", anchor="middle"))
            region_ids = sorted(set(region_labels) - {0})
            legend = ["0=背景；1..B=重点区域，颜色循环区分。",
                      "区域数：" + str(len(region_ids)) + "；标签见网格提示。"]
        elif panel_index == 2:
            for index in range(total):
                kept = index in retained
                attributes = (f'class="token-cell {"token-kept" if kept else "token-dropped"}" '
                              f'data-selected="{str(kept).lower()}" '
                              f'fill="{"none" if kept else "#07101d"}" '
                              f'fill-opacity="{"0" if kept else "0.84"}" '
                              f'stroke="{"#38eda4" if kept else "#8694a8"}" '
                              f'stroke-opacity="{"1" if kept else "0.30"}" stroke-width="1"')
                svg.append(_cell_rect(image_x, image_y, cell_width, cell_height,
                                      index, cols, attributes)
                           + f'<title>{index}: {"保留" if kept else "移除"}</title></rect>')
            legend = [f"绿色边框=实际保留，K/N={len(selected)}/{total}。",
                      "遮暗=移除；位置始终对应原始网格。"]
        else:
            status = metadata["q_status" if panel_index == 0 else "region_status"]
            unavailable = ("问题相关性不可用" if panel_index == 0 else "重点区域不可用")
            legend = [unavailable] + _wrap(status["reason"], 42)
        if unavailable:
            svg.append(f'<rect x="{image_x:.4f}" y="{image_y:.4f}" width="{image_width:.4f}" '
                       f'height="{image_height:.4f}" fill="#07101d" fill-opacity="0.68"/>')
            svg.append(_text(panel_x + panel_width / 2, top + 220, unavailable,
                             size=17, color="#f2f5fb", anchor="middle"))
        svg.extend(_text(panel_x + 16, top + 397 + index * 20, line, size=12)
                   for index, line in enumerate(legend))
        svg.append("</g>")
    svg.extend([
        f'<g font-family="{_FONT}">',
        _text(24, footer_y, "底图为模型实际输入的预处理后 RGB 画布，不能视为未裁剪原图。", size=13),
        _text(24, footer_y + 23, f"原始网格 {rows}×{cols}；索引从 0 开始，按行排列。", size=13),
        "</g></svg>",
    ])
    return "\n".join(svg) + "\n"


def _render_unmapped_svg(record, canvas, total, metadata, timing):
    """Show the model-input image without inventing a spatial Token mapping."""
    selection = record.get("selection", {})
    details = [
        "问题：" + _display(record.get("question")),
        "回答：" + _display(record.get("answer")),
        _quality_text(record),
    ]
    if "reference_answer" in record:
        details.append("参考答案：" + _display(record["reference_answer"]))
    details.extend([
        f"状态：{_display(selection.get('status'))}; K/N={total}/{total}",
        "状态原因：" + _display(selection.get("reason")),
        "时延：" + timing,
        "空间映射不可用，保留完整视觉序列",
        "问题相关性、重点区域及 Token 位置图均不可用。",
        "底图为模型实际输入的预处理后 RGB 画布，不能视为未裁剪原图。",
    ])
    if metadata["nonfinite_values"]:
        details.append("记录中存在非有限数值；相关值不可用，JSON 已写为 null。")
    lines = [line for detail in details for line in _wrap(detail, 112)]
    image_y = 84 + len(lines) * 21
    scale = min(600 / canvas.width, 480 / canvas.height)
    width, image_height = canvas.width * scale, canvas.height * scale
    height = image_y + image_height + 36
    encoded = BytesIO()
    canvas.save(encoded, format="PNG")
    uri = "data:image/png;base64," + base64.b64encode(encoded.getvalue()).decode("ascii")
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="900" height="{height:.3f}" '
        f'viewBox="0 0 900 {height:.3f}" role="img" aria-labelledby="report-title">',
        '<title id="report-title">视觉 Token 完整保留报告</title>',
        f'<rect width="900" height="{height:.3f}" fill="#0d1422"/>',
        f'<g font-family="{_FONT}">',
        _text(24, 37, "视觉 Token 完整保留报告", size=23, color="#f2f5fb"),
    ]
    svg.extend(_text(24, 68 + index * 21, line) for index, line in enumerate(lines))
    svg.extend([
        f'<image x="24" y="{image_y:.3f}" width="{width:.3f}" height="{image_height:.3f}" '
        f'preserveAspectRatio="none" href="{uri}"/>',
        '</g></svg>',
    ])
    return "\n".join(svg) + "\n"


def write_case(output_dir: Path, record: dict, canvas: Image.Image,
               q: list[float] | None, region_labels: list[int] | None,
               selected_indices: list[int], grid_shape: tuple[int, int] | None) -> dict[str, str]:
    """Write fixed filenames, returning absolute ``json`` and ``svg`` paths.

    ``selected_indices`` is the actual retained set in the original row-major
    grid. ``selection.order`` is never used to infer retained tokens. The input
    record is preserved without mutation; ``report_metadata`` is reserved here.
    Missing/invalid optional maps produce explicit unavailable panels.
    """
    if not isinstance(record, dict):
        raise TypeError("record must be a dictionary")
    if "report_metadata" in record:
        raise ValueError("report_metadata is reserved for report provenance")
    if grid_shape is None:
        total = record.get("visual_tokens_original")
        if isinstance(total, bool) or not isinstance(total, int) or total <= 0:
            raise ValueError("without a grid, visual_tokens_original must be a positive integer")
        if q is not None or region_labels is not None:
            raise ValueError("without a grid, q and region labels cannot be mapped to image positions")
    else:
        if (not isinstance(grid_shape, tuple) or len(grid_shape) != 2
                or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0
                       for item in grid_shape)):
            raise ValueError("grid_shape must be a pair of positive integers or None")
        rows, cols = grid_shape
        total = rows * cols
    if (not isinstance(selected_indices, list)
            or any(isinstance(item, bool) or not isinstance(item, int)
                   or not 0 <= item < total for item in selected_indices)
            or len(set(selected_indices)) != len(selected_indices)):
        raise ValueError("selected_indices must be unique original-grid indices in [0, N)")
    if not isinstance(canvas, Image.Image) or canvas.mode != "RGB":
        raise ValueError("canvas must be the actual preprocessed RGB PIL image")
    if canvas.width <= 0 or canvas.height <= 0:
        raise ValueError("canvas dimensions must be positive")
    if grid_shape is None:
        selection = record.get("selection")
        if (not isinstance(selection, dict) or selection.get("status") != "fallback_full"
                or sorted(selected_indices) != list(range(total))
                or selection.get("indices") != selected_indices
                or record.get("visual_tokens_kept") != total):
            raise ValueError("without a grid, report requires an actual full-keep fallback and matching counts")
    nonfinite = []
    cleaned = _json_safe(record, "", nonfinite)
    clean_q = _json_safe(q, "/report_metadata/q", nonfinite)
    clean_regions = _json_safe(region_labels, "/report_metadata/region_labels", nonfinite)
    warnings = []
    for key, actual in (("visual_tokens_original", total),
                        ("visual_tokens_kept", len(selected_indices))):
        if key in cleaned and cleaned[key] != actual:
            warnings.append(f"{key} 与实际网格/选择不一致；图中 K/N 以实际参数为准。")
    selection = cleaned.get("selection")
    if (isinstance(selection, dict) and "indices" in selection
            and selection["indices"] != selected_indices):
        warnings.append("selection.indices 与绘图参数不一致；实际保留以 selected_indices 为准。")
    metadata = {
        "canvas_source": "模型实际输入的预处理后 RGB 画布；不代表未裁剪原图布局",
        "canvas_size": [canvas.width, canvas.height],
        "grid_shape": None if grid_shape is None else [rows, cols],
        "index_convention": ("原始视觉序列索引、0 基；空间位置不可用"
                             if grid_shape is None else
                             "原始网格、0 基、逐行排列；selection.order 不是实际保留集合"),
        "visual_tokens_original": total,
        "visual_tokens_kept": len(selected_indices),
        "selected_indices": list(selected_indices),
        "q": clean_q,
        "region_labels": clean_regions,
        "q_status": ({"available": False, "reason": "空间映射不可用"} if grid_shape is None
                     else _map_status(q, total)),
        "region_status": ({"available": False, "reason": "空间映射不可用"} if grid_shape is None
                          else _map_status(region_labels, total, regions=True)),
        "nonfinite_values": nonfinite,
        "warnings": warnings,
    }
    timing = _timing_text(record.get("timing_ms"))
    svg = (_render_unmapped_svg(cleaned, canvas, total, metadata, timing)
           if grid_shape is None else
           _render_svg(cleaned, canvas, q, region_labels, selected_indices, grid_shape, metadata, timing))
    cleaned["report_metadata"] = metadata
    json_text = json.dumps(cleaned, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    json_path, svg_path = destination / "result.json", destination / "visual_tokens.svg"
    json_path.write_text(json_text, encoding="utf-8")
    svg_path.write_text(svg, encoding="utf-8")
    return {"json": str(json_path), "svg": str(svg_path)}
