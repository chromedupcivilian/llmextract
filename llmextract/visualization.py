import json
import logging
import textwrap
from typing import Any, Dict, List, Union

from .data_models import AnnotatedDocument, Extraction

logger = logging.getLogger(__name__)

_PALETTE: List[str] = [
    "#D2E3FC",
    "#C8E6C9",
    "#FEF0C3",
    "#F9DEDC",
    "#FFDDBE",
    "#EADDFF",
    "#C4E9E4",
    "#FCE4EC",
    "#E8EAED",
    "#DDE8E8",
]

_HTML_TEMPLATE = textwrap.dedent(
    """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>llmextract Visualization</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; line-height: 1.6; padding: 20px; background-color: #f9f9f9; }
        .container { max-width: 900px; margin: auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #eee; padding-bottom: 10px; margin-bottom: 20px; flex-wrap: wrap; gap: 10px; }
        .header h2 { margin: 0; }
        .selectors { display: flex; gap: 20px; align-items: center; }
        .selector-container label { margin-right: 10px; font-size: 14px; color: #555; }
        .selector { font-size: 14px; padding: 5px; border-radius: 4px; border: 1px solid #ccc; }
        .metadata-panel { font-size: 12px; color: #666; background: #fafafa; padding: 10px; border-radius: 4px; border: 1px solid #eee; margin-bottom: 20px; }
        .metadata-panel code { background: #eee; padding: 2px 4px; border-radius: 3px; }
        .text-display { white-space: pre-wrap; word-wrap: break-word; border: 1px solid #ddd; padding: 15px; border-radius: 4px; margin-bottom: 20px; background-color: #fff; max-height: 50vh; overflow-y: auto; }
        .highlight { position: relative; border-radius: 3px; padding: 2px 4px; cursor: default; display: inline; }
        .tooltip {
            visibility: hidden; opacity: 0; transition: opacity 0.2s;
            background: #333; color: #fff; text-align: left;
            border-radius: 4px; padding: 8px; position: absolute;
            z-index: 10; bottom: 125%; left: 50%;
            transform: translateX(-50%); font-size: 12px;
            width: max-content; max-width: 320px; box-shadow: 0 2px 6px rgba(0,0,0,0.3);
            white-space: normal;
        }
        .highlight:hover .tooltip { visibility: visible; opacity: 1; }
        .legend { margin-bottom: 20px; }
        .legend-item { display: inline-flex; align-items: center; margin-right: 15px; font-size: 14px; }
        .legend-color { width: 15px; height: 15px; border-radius: 3px; margin-right: 5px; }
        .no-result { color: #999; font-style: italic; }
        .small-btn { font-size: 13px; padding: 6px 10px; border-radius: 4px; border: 1px solid #ccc; background: #fff; cursor: pointer; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h2>Extraction Visualization</h2>
            <div class="selectors">
                <div class="selector-container">
                    <label for="doc-selector">Document:</label>
                    <select id="doc-selector" class="selector"></select>
                </div>
                <div class="selector-container">
                    <label for="model-selector">Model:</label>
                    <select id="model-selector" class="selector"></select>
                </div>
                <button id="download-json" class="small-btn">Download JSON</button>
            </div>
        </div>
        <div id="content-area">
            <div class="metadata-panel" id="metadata-panel"></div>
            <div class="legend" id="legend"></div>
            <div class="text-display" id="text-display"></div>
        </div>
        <div id="no-result-area" style="display: none;" class="no-result">
            No result available for this document/model combination.
        </div>
    </div>

    <!-- Embedded data (safe JSON) -->
    <script type="application/json" id="llmextract-data">
__DATA_PLACEHOLDER__
    </script>

    <script>
    (function() {
        function safeParseJsonText(id) {
            const el = document.getElementById(id);
            if (!el) return {};
            try {
                return JSON.parse(el.textContent || "{}");
            } catch (err) {
                console.error("Failed to parse visualization JSON:", err);
                return {};
            }
        }

        const organizedData = safeParseJsonText('llmextract-data');
        const docSelector = document.getElementById('doc-selector');
        const modelSelector = document.getElementById('model-selector');
        const metadataContainer = document.getElementById('metadata-panel');
        const legendContainer = document.getElementById('legend');
        const textContainer = document.getElementById('text-display');
        const contentArea = document.getElementById('content-area');
        const noResultArea = document.getElementById('no-result-area');
        const downloadBtn = document.getElementById('download-json');

        const docIds = Object.keys(organizedData).sort();

        function populateDocSelector() {
            docSelector.innerHTML = '';
            docIds.forEach(docId => {
                const opt = document.createElement('option');
                opt.value = docId;
                opt.textContent = docId;
                docSelector.appendChild(opt);
            });
        }

        function populateModelSelector(selectedDocId) {
            modelSelector.innerHTML = '';
            const modelsObj = organizedData[selectedDocId] || {};
            const models = Object.keys(modelsObj).sort();
            models.forEach(modelId => {
                const opt = document.createElement('option');
                opt.value = modelId;
                opt.textContent = modelId;
                modelSelector.appendChild(opt);
            });
        }

        function escapeHtml(unsafe) {
            if (typeof unsafe !== 'string') unsafe = String(unsafe);
            return unsafe.replace(/&/g, "&amp;").replace(/</g, "&lt;")
                         .replace(/>/g, "&gt;").replace(/"/g, "&quot;")
                         .replace(/'/g, "&#039;");
        }

        function buildLegend(colorMap) {
            legendContainer.innerHTML = '';
            if (!colorMap) return;
            Object.entries(colorMap).forEach(([cls, color]) => {
                const item = document.createElement('div');
                item.className = 'legend-item';
                item.innerHTML = `<div class="legend-color" style="background-color:${color};"></div>${escapeHtml(cls)}`;
                legendContainer.appendChild(item);
            });
        }

        function buildHighlightedHtml(text, extractions, colorMap) {
            if (!text) return '';
            if (!Array.isArray(extractions) || extractions.length === 0) {
                return escapeHtml(text);
            }

            const cleansed = [];
            const bounds = new Set([0, text.length]);

            for (const ext of extractions) {
                if (!ext || !ext.char_interval) continue;
                const s = Number(ext.char_interval.start);
                const e = Number(ext.char_interval.end);
                if (!Number.isFinite(s) || !Number.isFinite(e)) {
                    console.warn("Skipping extraction with non-numeric interval:", ext);
                    continue;
                }
                const start = Math.max(0, Math.min(text.length, Math.floor(s)));
                const end = Math.max(0, Math.min(text.length, Math.floor(e)));
                if (start >= end) {
                    console.warn("Skipping extraction with invalid interval:", ext);
                    continue;
                }
                bounds.add(start);
                bounds.add(end);
                cleansed.push(Object.assign({}, ext, {char_interval: {start, end}}));
            }

            const sortedBounds = Array.from(bounds).sort((a,b) => a - b);
            let html = '';

            for (let i = 0; i < sortedBounds.length - 1; i++) {
                const segStart = sortedBounds[i];
                const segEnd = sortedBounds[i + 1];
                const segmentText = text.slice(segStart, segEnd);

                const covering = cleansed.filter(ext => (
                    ext.char_interval.start <= segStart && ext.char_interval.end >= segEnd
                ));

                if (covering.length === 0) {
                    html += escapeHtml(segmentText);
                    continue;
                }

                function buildTooltipFor(ext) {
                    let t = `<strong>${escapeHtml(ext.extraction_class)}</strong>`;
                    const attrs = ext.attributes || {};
                    const attrKeys = Object.keys(attrs || {});
                    if (attrKeys.length > 0) {
                        t += '<hr style="margin:4px 0; border-color:#555;">';
                        for (const k of attrKeys) {
                            const val = attrs[k];
                            const valStr = (typeof val === 'object') ? JSON.stringify(val) : String(val);
                            t += `<div><em>${escapeHtml(k)}</em>: ${escapeHtml(valStr)}</div>`;
                        }
                    }
                    return t;
                }

                if (covering.length === 1) {
                    const ext = covering[0];
                    const color = (colorMap && colorMap[ext.extraction_class]) || '#ccc';
                    const tooltip = buildTooltipFor(ext);
                    html += `<span class="highlight" style="background-color:${color};">` +
                            `${escapeHtml(segmentText)}` +
                            `<span class="tooltip">${tooltip}</span></span>`;
                } else {
                    const uniqueClasses = Array.from(new Set(covering.map(e => e.extraction_class)));
                    const colors = uniqueClasses.map(c => (colorMap && colorMap[c]) || '#ccc');
                    const cappedColors = colors.slice(0, 6);
                    const stop = 100 / cappedColors.length;
                    const parts = [];
                    for (let j = 0; j < cappedColors.length; j++) {
                        const startPct = (j * stop).toFixed(2);
                        const endPct = ((j + 1) * stop).toFixed(2);
                        parts.push(`${cappedColors[j]} ${startPct}%, ${cappedColors[j]} ${endPct}%`);
                    }
                    const gradient = `linear-gradient(90deg, ${parts.join(', ')})`;

                    let combinedTooltip = '';
                    for (const ext of covering) {
                        combinedTooltip += buildTooltipFor(ext);
                        combinedTooltip += '<div style="height:6px;"></div>';
                    }
                    html += `<span class="highlight" style="background:${gradient};">` +
                            `${escapeHtml(segmentText)}` +
                            `<span class="tooltip">${combinedTooltip}</span></span>`;
                }
            }

            return html;
        }

        function render() {
            const selectedDocId = docSelector.value;
            const selectedModelId = modelSelector.value;
            const docData = (organizedData[selectedDocId] || {})[selectedModelId];

            if (!docData) {
                contentArea.style.display = 'none';
                noResultArea.style.display = 'block';
                return;
            }

            contentArea.style.display = 'block';
            noResultArea.style.display = 'none';

            let metadataHtml = '';
            if (docData.metadata) {
                for (const [k,v] of Object.entries(docData.metadata)) {
                    const display = (typeof v === 'object') ? JSON.stringify(v) : String(v);
                    metadataHtml += `<strong>${escapeHtml(k)}:</strong> <code>${escapeHtml(display)}</code><br>`;
                }
            }
            metadataContainer.innerHTML = metadataHtml;

            buildLegend(docData.colorMap || {});

            try {
                textContainer.innerHTML = buildHighlightedHtml(docData.text || '', docData.extractions || [], docData.colorMap || {});
            } catch (err) {
                console.error("Rendering error:", err);
                textContainer.innerHTML = escapeHtml(docData.text || '');
            }
        }

        docSelector.addEventListener('change', function() {
            populateModelSelector(docSelector.value);
            render();
        });
        modelSelector.addEventListener('change', render);

        downloadBtn.addEventListener('click', function() {
            const blob = new Blob([JSON.stringify(organizedData, null, 2)], {type: 'application/json'});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'llmextract_visualization.json';
            document.body.appendChild(a);
            a.click();
            a.remove();
            URL.revokeObjectURL(url);
        });

        if (docIds.length > 0) {
            populateDocSelector();
            populateModelSelector(docIds[0]);
            docSelector.value = docIds[0];
            if (modelSelector.options.length > 0) {
                modelSelector.selectedIndex = 0;
            }
            render();
        } else {
            noResultArea.textContent = "No documents were provided for visualization.";
            noResultArea.style.display = 'block';
        }
    })();
    </script>
</body>
</html>
"""
)


def _assign_colors(extractions: List[Extraction]) -> Dict[str, str]:
    unique = sorted({(ext.extraction_class or "unknown") for ext in extractions})
    return {cls: _PALETTE[i % len(_PALETTE)] for i, cls in enumerate(unique)}


def _serialize_extraction(ext: Extraction) -> Dict[str, Any]:
    data: Dict[str, Any] = {
        "extraction_class": ext.extraction_class,
        "extraction_text": ext.extraction_text,
        "attributes": ext.attributes or {},
    }
    if ext.char_interval:
        try:
            start = int(ext.char_interval.start)
            end = int(ext.char_interval.end)
            data["char_interval"] = {"start": start, "end": end}
        except Exception as e:
            logger.warning(
                "Failed to serialize char_interval for extraction %r: %s", ext, e
            )
    if getattr(ext, "id", None) is not None:
        data["id"] = ext.id
    return data


def visualize(docs_or_doc: Union[AnnotatedDocument, List[AnnotatedDocument]]) -> str:
    docs = [docs_or_doc] if isinstance(docs_or_doc, AnnotatedDocument) else docs_or_doc
    if not docs:
        return "<p>No documents to visualize.</p>"

    organized: Dict[str, Dict[str, Any]] = {}

    for i, doc in enumerate(docs):
        doc_id = doc.metadata.get("doc_id") if isinstance(doc.metadata, dict) else None
        doc_id = doc_id or f"Document_{i + 1}"
        model_name = (
            doc.metadata.get("model_name") if isinstance(doc.metadata, dict) else None
        )
        model_name = (
            model_name or doc.metadata.get("model")
            if isinstance(doc.metadata, dict)
            else None
        )
        model_name = model_name or "Unknown_Model"

        organized.setdefault(doc_id, {})

        valid_extractions = [
            ext for ext in doc.extractions if ext.char_interval is not None
        ]
        color_map = _assign_colors(valid_extractions)

        serial_extractions = []
        for ext in valid_extractions:
            ser = _serialize_extraction(ext)
            if "char_interval" in ser:
                s = ser["char_interval"]["start"]
                e = ser["char_interval"]["end"]
                if s < 0 or e < 0 or s >= e or s > len(doc.text) or e < 0:
                    logger.warning(
                        "Skipping extraction with out-of-bounds interval (doc=%s, model=%s): %s",
                        doc_id,
                        model_name,
                        ser,
                    )
                    continue
                ser["char_interval"]["start"] = max(0, min(len(doc.text), s))
                ser["char_interval"]["end"] = max(0, min(len(doc.text), e))
            serial_extractions.append(ser)

        organized[doc_id][model_name] = {
            "text": doc.text,
            "extractions": serial_extractions,
            "colorMap": color_map,
            "metadata": doc.metadata or {},
        }

    try:
        organized_json = json.dumps(
            organized, ensure_ascii=False, default=str, indent=2
        )
    except Exception as e:
        logger.exception("Failed to serialize organized data for visualization: %s", e)
        organized_json = "{}"

    # Prevent accidental script-close or HTML comment injection
    organized_json = organized_json.replace("</script", "<\\/script")
    organized_json = organized_json.replace("<!--", "<\\!--")

    return _HTML_TEMPLATE.replace("__DATA_PLACEHOLDER__", organized_json)
