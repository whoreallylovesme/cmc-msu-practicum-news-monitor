"""Read-only monitoring dashboard with server-side requests to Web Master."""

import logging
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import urlencode, urlsplit

import httpx
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, ctx, dcc, html
from flask import Response, g, has_request_context, request
from plotly.subplots import make_subplots


ROOT = Path(__file__).resolve().parent
ACCENT = "#9f3919"
LOGGER = logging.getLogger(__name__)
UNAVAILABLE = "Данные временно недоступны. Попробуйте обновить страницу позже."


def create_dashboard(master_url, timeout, settings=None):
    settings = settings or {}

    def master(path, params=None):
        try:
            result = httpx.get(
                master_url + (path if path.startswith("/") else "/api/" + path),
                headers={"Authorization": request.headers.get("Authorization", "")},
                params=params, timeout=timeout,
            )
            result.raise_for_status()
            return result.json()
        except (httpx.HTTPError, ValueError) as error:
            LOGGER.exception("Dashboard data request failed")
            raise RuntimeError(UNAVAILABLE) from error

    dashboard = Dash(
        __name__, title="Мониторинг обсуждений", assets_folder=str(ROOT / "assets"),
        update_title="Загрузка…", serve_locally=True,
    )

    @dashboard.server.before_request
    def authenticate():
        try:
            result = httpx.get(
                master_url + "/api/config",
                headers={"Authorization": request.headers.get("Authorization", "")},
                timeout=timeout,
            )
            if result.status_code == 401:
                return Response("Для просмотра войдите в систему.", status=401,
                                headers={"WWW-Authenticate": result.headers.get("www-authenticate", 'Basic realm="News Monitor"')})
            result.raise_for_status()
            g.monitor_config = result.json()
        except (httpx.HTTPError, ValueError):
            LOGGER.exception("Dashboard authentication request failed")
            return Response(UNAVAILABLE, status=502)

    def layout():
        config = getattr(g, "monitor_config", {}) if has_request_context() else {}
        entity = config.get("monitoring", {}).get("query", "Обсуждения")
        return html.Main([
            dcc.Store(id="report-data"), dcc.Store(id="selection"),
            dcc.Interval(id="refresh", interval=int(settings.get("refresh_seconds", 30)) * 1000),
            html.Header([html.H1(entity), html.P("Упоминания и настроение обсуждений", className="subtle")], className="masthead"),
            html.Div([
                html.Label([html.Span("Период"), dcc.Dropdown(
                    id="bucket", options=[{"label": "Неделя", "value": "week"}, {"label": "День", "value": "day"}],
                    value=settings.get("default_bucket", "week"), clearable=False,
                )], className="control"),
                html.A("Скачать отчёт", id="report", download="hn-monitor-report.html", className="button secondary", style={"display": "none"}),
            ], className="filters"),
            html.P("", id="notice", role="status", className="notice empty"),
            html.Section([
                html.Div([html.H2("Динамика"), html.Div(id="summary", className="summary")], className="chart-heading"),
                html.P(id="scale-description", className="subtle"),
                dcc.Loading(dcc.Graph(
                    id="chart", figure=empty_figure(), responsive=True,
                    style={"width": "100%", "minWidth": 0, "height": 430},
                    config={"displaylogo": False, "responsive": True,
                            "toImageButtonOptions": {"filename": "news-monitor"},
                            "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
                ), type="dot", color=ACCENT),
            ], className="panel chart-panel"),
            html.Section([html.H2("По источникам"), html.Div(id="source-breakdown")], className="panel"),
            html.Section([
                html.Div([html.H2("Комментарии"), html.Button("Все периоды", id="clear-selection", className="secondary")], className="section-heading"),
                html.P(id="selected-period", className="subtle"),
                html.Div([
                    html.Label([html.Span("Поиск"), dcc.Input(id="search", placeholder="Слово или фраза", type="search", debounce=True)], className="control"),
                    html.Button("Найти", id="find", className="secondary"),
                ], className="actions"),
                html.P(id="evidence-caption", className="subtle"),
                dcc.Loading(html.Div(id="evidence"), type="dot", color=ACCENT),
            ], className="panel"),
        ], className="layout")

    dashboard.layout = layout

    @dashboard.callback(
        Output("chart", "figure"), Output("summary", "children"), Output("scale-description", "children"),
        Output("report", "href"), Output("report", "style"), Output("report-data", "data"),
        Output("source-breakdown", "children"), Output("notice", "children"), Output("notice", "className"),
        Input("bucket", "value"), Input("refresh", "n_intervals"),
    )
    def refresh(bucket, _ticks):
        try:
            config = g.monitor_config
            entity = config["monitoring"]["query"]
            pipeline = config.get("pipeline", {})
            models = master("/services/ml/models", {"entity": entity})["items"]
            model_id = pipeline.get("model_id")
            kind = pipeline.get("model_kind", "hard_label")
            if model_id:
                model = next((row for row in models if row["id"] == model_id), None)
                if model is None or model["kind"] != kind or model["entity"].casefold() != entity.casefold() or model.get("version") is None:
                    raise RuntimeError(UNAVAILABLE)
            else:
                model = max((row for row in models if row["kind"] == kind and row.get("version") is not None),
                            key=lambda row: row.get("updated_at", ""), default=None)
            params = {"entity": entity, "bucket": bucket}
            if model:
                params["model_id"] = model["id"]
            data = master("analytics", params)
            scale = config["score_scale"]
            hint = f'Шкала {scale["min"]:g}–{scale["max"]:g}. Нажмите на график, чтобы открыть комментарии за период.'
            return (timeline_figure(data["periods"], scale), summary(data["summary"]), hint,
                    "/services/web_master/api/report?" + urlencode(params) if model else "",
                    {} if model else {"display": "none"},
                    {"entity": entity, "model_id": model["id"] if model else None, "bucket": bucket},
                    source_breakdown(data["sources"]), "", "notice empty")
        except RuntimeError:
            return empty_figure(), [], "", "", {"display": "none"}, None, [], UNAVAILABLE, "notice error"

    @dashboard.callback(
        Output("evidence", "children"), Output("evidence-caption", "children"),
        Output("selected-period", "children"), Output("selection", "data"),
        Input("chart", "clickData"), Input("find", "n_clicks"), Input("clear-selection", "n_clicks"),
        Input("report-data", "data"), State("search", "value"), State("selection", "data"),
    )
    def evidence(clicked, _find, _clear, data, query, previous):
        if not data:
            return [], "", "", None
        try:
            period = previous.get("period") if previous and previous.get("bucket") == data["bucket"] else None
            if ctx.triggered_id == "clear-selection":
                period = None
            if ctx.triggered_id == "chart" and clicked and clicked.get("points"):
                period = clicked["points"][0].get("customdata")
            params = {"entity": data["entity"], "limit": 30}
            if data.get("model_id"):
                params["model_id"] = data["model_id"]
            if period:
                first = date.fromisoformat(period)
                params.update(from_date=period, to_date=(first + timedelta(days=7 if data["bucket"] == "week" else 1)).isoformat())
            if query and query.strip():
                params["query"] = query.strip()
            result = master("comments", params)
            label = "Период с " + period if period else "Все периоды"
            return comments(result["items"], scored=bool(data.get("model_id"))), count_caption(result), label, {"period": period, "bucket": data["bucket"]}
        except (RuntimeError, ValueError):
            return [html.P(UNAVAILABLE, className="notice error")], "", "", previous

    return dashboard


def empty_figure(message="Пока нет данных для отображения."):
    figure = go.Figure()
    figure.add_annotation(text=message, x=.5, y=.5, xref="paper", yref="paper", showarrow=False)
    figure.update_layout(template="plotly_white", height=370, autosize=True, margin=dict(l=35, r=25, t=30, b=35),
                          paper_bgcolor="#fffdfa", plot_bgcolor="#fffdfa", font=dict(color="#6a6057", size=14))
    figure.update_xaxes(visible=False)
    figure.update_yaxes(visible=False)
    return figure


def timeline_figure(points, scale):
    if not points:
        return empty_figure("Пока нет данных для отображения.")
    dates = [row["date"] for row in points]
    figure = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[.72, .28], vertical_spacing=.12)
    figure.add_trace(go.Scatter(
        x=dates, y=[row["score"] for row in points], name="Средний балл", mode="lines+markers",
        customdata=dates, line=dict(color=ACCENT, width=3), marker=dict(size=8), connectgaps=False,
        hovertemplate="%{x|%d.%m.%Y}<br>Балл %{y:.2f}<extra></extra>",
    ), row=1, col=1)
    figure.add_hline(y=(scale["min"] + scale["max"]) / 2, line_dash="dot", line_color="#b78465", row=1, col=1)
    figure.add_trace(go.Bar(
        x=dates, y=[row["comments"] for row in points], name="Комментарии в тредах", customdata=dates,
        marker_color="#cdb39d", hovertemplate="%{x|%d.%m.%Y}<br>Комментарии в тредах %{y}<extra></extra>",
    ), row=2, col=1)
    figure.add_trace(go.Bar(
        x=dates, y=[row["mentions"] for row in points], name="Упоминания", customdata=dates,
        marker_color=ACCENT, hovertemplate="%{x|%d.%m.%Y}<br>Упоминания %{y}<extra></extra>",
    ), row=2, col=1)
    figure.update_layout(
        template="plotly_white", height=430, margin=dict(l=50, r=20, t=15, b=35),
        paper_bgcolor="#fffdfa", plot_bgcolor="#fffdfa", font=dict(family="system-ui", color="#40362e"),
        autosize=True, barmode="group", showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.05, x=0),
        hovermode="x unified", clickmode="event+select", dragmode="zoom",
    )
    figure.update_yaxes(range=[scale["min"], scale["max"]], title_text="Балл", row=1, col=1)
    figure.update_yaxes(title_text="Количество", rangemode="tozero", row=2, col=1)
    figure.update_xaxes(type="date", gridcolor="#eee5db")
    figure.update_yaxes(gridcolor="#eee5db", zeroline=False)
    return figure


def summary(values):
    average = f'{values["score"]:.2f}' if values.get("score") is not None else "—"
    return [html.Span([html.Strong(average), " средний балл"]),
            html.Span([html.Strong(str(values["mentions"])), " упоминаний"]),
            html.Span([html.Strong(str(values["comments"])), " комментариев в тредах"]),
            html.Span([html.Strong(str(values["relevant"])), " оценено по теме"])]


def source_breakdown(sources):
    if not sources:
        return html.P("Пока нет данных об источниках.", className="subtle")
    rows = []
    for source in sources:
        name = "Hacker News" if source["source"] == "hacker_news" else source["source"]
        score = f'{source["score"]:.2f}' if source.get("score") is not None else "—"
        rows.append(html.Tr([html.Td(name), html.Td(source["comments"]), html.Td(source["mentions"]), html.Td(score)]))
    return html.Table([
        html.Thead(html.Tr([html.Th("Источник"), html.Th("Комментарии"), html.Th("Упоминания"), html.Th("Средний балл")])),
        html.Tbody(rows),
    ], className="source-table")


def count_caption(result):
    return f'Показано {len(result["items"])} из {result["total"]}'


def comments(items, scored=True):
    if not items:
        return [html.P("Комментарии не найдены. Попробуйте другой период или запрос.", className="subtle")]
    result = []
    for item in items:
        parsed = urlsplit(item.get("url") or "")
        title = html.Strong(item["story_title"])
        if parsed.scheme in {"https", "http"} and parsed.hostname != "example.invalid":
            title = html.A(item["story_title"], href=item["url"], target="_blank", rel="noopener noreferrer")
        details = [item["published_at"], item["source"]]
        if scored:
            details.append(f'Балл {item["score"]:.2f}' if item["score"] is not None else "Мимо темы")
        children = [title, html.P(" · ".join(details), className="metadata"), html.P(item["text"])]
        if item.get("parent_text"):
            children.append(html.Details([html.Summary("Контекст ответа"), html.P(item["parent_text"])]))
        result.append(html.Article(children, className="comment"))
    return result
