"""Plotly Dash interface with server-side callbacks to Web Master."""

import json
import logging
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import urlencode, urlsplit

import httpx
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, ctx, dcc, html, no_update
from flask import Response, g, has_request_context, request
from plotly.subplots import make_subplots


ROOT = Path(__file__).resolve().parent
ACCENT = "#9f3919"
SOURCES = {
    "hn_snapshot": "Hacker News · снимок",
    "hacker_news": "Hacker News · живой",
}
TEACHERS = {"hn_snapshot": "dataset", "hacker_news": "reviewed"}
TEACHER_LABELS = {
    "dataset": "Готовая разметка датасета",
    "reviewed": "Проверенная разметка",
    "service": "Отдельный сервис учителя",
    "live": "Модель через llama.cpp",
}


LOGGER = logging.getLogger(__name__)


def create_dashboard(master_url, timeout):
    def master(path, *, method="GET", body=None, params=None):
        headers = {"Authorization": request.headers.get("Authorization", "")}
        try:
            result = httpx.request(
                method, master_url + (path if path.startswith("/") else "/api/" + path), headers=headers,
                json=body, params=params, timeout=timeout,
            )
        except httpx.RequestError as error:
            LOGGER.exception("Web Master request failed")
            raise RuntimeError("Web Master недоступен. Проверь состояние сервисов.") from error
        try:
            data = result.json()
        except ValueError as error:
            LOGGER.exception("Web Master response is not JSON")
            raise RuntimeError("Web Master вернул некорректный ответ.") from error
        if result.is_error:
            detail = data.get("detail", "Не удалось выполнить запрос.")
            raise RuntimeError(detail if isinstance(detail, str) else json.dumps(detail, ensure_ascii=False))
        return data

    dashboard = Dash(
        __name__, title="HN Monitor", assets_folder=str(ROOT / "assets"),
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
        except httpx.RequestError:
            return Response("Web Master недоступен. Проверь состояние сервисов.", status=502)
        if result.is_error:
            return Response(
                result.content, status=result.status_code,
                headers={"WWW-Authenticate": result.headers.get("www-authenticate", 'Basic realm="News Monitor"')},
                content_type=result.headers.get("content-type", "application/json"),
            )
        g.monitor_config = result.json()
        g.monitor_role = g.monitor_config["role"]

    def control(title, component):
        return html.Label([html.Span(title), component], className="control")

    def layout():
        config = {"entities": [], "sources": [], "monitoring": {}}
        role = "viewer"
        if has_request_context():
            role = getattr(g, "monitor_role", "viewer")
            config = getattr(g, "monitor_config", config)
        source = config["monitoring"].get("source")
        entity = config["monitoring"].get("query")
        return html.Main([
            dcc.Store(id="role", data=role),
            dcc.Store(id="cursor"), dcc.Store(id="refresh", data=0),
            dcc.Store(id="created-model"), dcc.Store(id="job"),
            dcc.Store(id="model-state"), dcc.Store(id="timeline", data={}),
            dcc.Store(id="score-scale", data=config.get("score_scale", {"min": 0, "max": 5})),
            dcc.Store(id="selection"),
            dcc.Interval(id="job-poll", interval=1500, disabled=True),
            dcc.Interval(id="status-poll", interval=30000),
            html.Header([
                html.Div([
                    html.H1("Настроение обсуждений"),
                    html.P("Отношение к выбранной сущности в комментариях", className="subtle"),
                ]),
                html.Div([
                    html.A("Swagger API", href="/gateway-docs", target="_blank"),
                    html.Span("Офлайн-снимок", className="badge", hidden=not config.get("offline_demo", False)),
                    html.Span("Администратор" if role == "admin" else "Читатель", className="badge"),
                ], className="header-links"),
            ], className="masthead"),
            html.Div([
                control("Сущность", dcc.Dropdown(
                    id="entity", options=[{"label": row["name"], "value": row["name"]} for row in config["entities"]],
                    value=entity, clearable=False,
                )),
                control("Модель", dcc.Dropdown(id="model", options=[], placeholder="Создай модель", clearable=False)),
                control("Период", dcc.Dropdown(
                    id="bucket", options=[{"label": "Неделя", "value": "week"}, {"label": "День", "value": "day"}],
                    value="week", clearable=False,
                )),
                html.A("Скачать отчёт", id="report", className="button secondary", style={"display": "none"}),
            ], className="filters"),
            html.P("", id="notice", role="status", className="notice empty"),
            html.Section([
                html.H2("Управление"),
                html.Div([
                    control("Источник", dcc.Dropdown(
                        id="source", options=[{"label": SOURCES.get(row["id"], row["id"]), "value": row["id"]} for row in config["sources"]],
                        value=source, clearable=False,
                    )),
                    html.Button("Загрузить батч", id="collect"),
                    html.Button("Следующий батч", id="next", className="secondary", disabled=True),
                ], className="actions"),
                html.Div([
                    control("Классификатор", dcc.Dropdown(id="kind", options=[
                        {"label": "NLI с калибровкой", "value": "semantic"},
                        {"label": "Мягкие метки", "value": "softmax"},
                        {"label": "Один класс", "value": "hard_label"},
                        {"label": "Малая нейросеть", "value": "tiny_mlp"},
                    ], value="hard_label", clearable=False)),
                    html.Button("Создать модель", id="create"),
                    control("Разметка", dcc.Dropdown(id="teacher", options=[
                        {"label": TEACHER_LABELS["dataset"], "value": "dataset"},
                        {"label": TEACHER_LABELS["reviewed"], "value": "reviewed"},
                        {"label": TEACHER_LABELS["service"], "value": "service"},
                    ] + ([] if config.get("offline_demo") else [{"label": "Модель через llama.cpp", "value": "live"}]),
                        value=TEACHERS.get(source, "dataset"), clearable=False)),
                    control("Режим", dcc.Dropdown(id="mode", options=[
                        {"label": "С нуля", "value": "from_scratch"},
                        {"label": "Дообучение", "value": "incremental"},
                    ], value="from_scratch", clearable=False)),
                    html.Button("Обучить", id="train"),
                    html.Button("Оценить комментарии", id="score", className="secondary"),
                ], className="actions"),
                html.P(id="training", className="subtle", role="status"),
            ], id="admin", className="panel", style={} if role == "admin" else {"display": "none"}),
            html.Section([
                html.Div([html.H2("Динамика"), html.Div(id="summary", className="summary")], className="chart-heading"),
                html.P(id="scale-description", className="subtle"),
                dcc.Loading(dcc.Graph(
                    id="chart", figure=empty_figure(), responsive=True,
                    style={"width": "100%", "minWidth": 0, "height": 430},
                    config={"displaylogo": False, "responsive": True,
                            "toImageButtonOptions": {"filename": "hn-monitor"},
                            "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
                ), type="dot", color=ACCENT),
            ], className="panel chart-panel"),
            html.Section([
                html.H2("По источникам"),
                html.P("Упоминания в собранных комментариях: имя, алиас, небольшая опечатка или простая отсылка к явно названной сущности. Один комментарий считается один раз.", className="subtle"),
                html.Div(id="source-breakdown"),
            ], className="panel"),
            html.Section([
                html.Div([
                    html.H2("Комментарии"),
                    html.Button("Все периоды", id="clear-selection", className="secondary"),
                ], className="section-heading"),
                html.P(id="selected-period", className="subtle"),
                html.Div([
                    control("Поиск по архиву", dcc.Input(id="search", placeholder="Слово или фраза", type="search", debounce=True)),
                    html.Button("Найти", id="find", className="secondary"),
                ], className="actions"),
                html.P(id="evidence-caption", className="subtle"),
                dcc.Loading(html.Div(id="evidence"), type="dot", color=ACCENT),
            ], className="panel"),
            html.Div([
                html.Section([html.H2("Оценка модели"), html.Div(id="model-metrics")], className="panel"),
                html.Section([
                    html.H2("Сервисы"), html.Div(id="status", className="statuses"),
                    html.H3("Журнал загрузок"), html.Div(id="loads"),
                ], className="panel"),
            ], className="details-grid"),
        ], className="layout")

    dashboard.layout = layout

    @dashboard.callback(Output("teacher", "value"), Input("source", "value"))
    def teacher_for_source(source):
        return TEACHERS.get(source, "dataset")

    @dashboard.callback(
        Output("model", "options"), Output("model", "value"),
        Input("entity", "value"), Input("refresh", "data"),
        State("created-model", "data"), State("model", "value"),
    )
    def models(entity, _refresh, created, current):
        if not entity:
            return [], None
        try:
            rows = master("/services/ml/models", params={"entity": entity})["items"]
        except RuntimeError:
            return [], None
        options = [{"label": f'{row["kind"]} · {row["id"][:8]}' + (f' · v{row["version"]}' if row.get("version") else ""),
                    "value": row["id"]} for row in rows]
        available = {row["id"] for row in rows}
        selected = created if created in available else current
        if selected not in available:
            preferred = next((row for row in rows if row.get("recommended")), None)
            selected = preferred["id"] if preferred else rows[0]["id"] if rows else None
        return options, selected

    @dashboard.callback(
        Output("notice", "children"), Output("notice", "className"),
        Output("cursor", "data"), Output("refresh", "data"), Output("created-model", "data"),
        Output("job", "data"), Output("job-poll", "disabled"), Output("training", "children"),
        Input("collect", "n_clicks"), Input("next", "n_clicks"), Input("create", "n_clicks"),
        Input("train", "n_clicks"), Input("score", "n_clicks"), Input("job-poll", "n_intervals"),
        Input("entity", "value"), Input("source", "value"),
        State("model", "value"), State("kind", "value"), State("teacher", "value"), State("mode", "value"),
        State("cursor", "data"), State("refresh", "data"), State("job", "data"),
        prevent_initial_call=True,
        running=[(Output("collect", "disabled"), True, False), (Output("create", "disabled"), True, False)],
    )
    def operate(_collect, _next, _create, _train, _score, _ticks, entity, source,
                model_id, kind, teacher, mode, cursor, refresh, job):
        trigger = ctx.triggered_id
        unchanged = [no_update] * 8
        if trigger in {"entity", "source"}:
            unchanged[2] = None
            return unchanged
        if g.monitor_role != "admin":
            return ["Для изменения данных войди как администратор.", "notice error", *unchanged[2:]]
        try:
            if trigger in {"collect", "next"}:
                result = master("import", method="POST", body={
                    "entity": entity, "source": source, "cursor": cursor if trigger == "next" else None, "limit": 100,
                })
                return [f'Получено {result["received"]}, сохранено {result["stored"]}.', "notice",
                        result.get("next_cursor"), (refresh or 0) + 1, no_update, no_update, no_update, no_update]
            if trigger == "create":
                model = master("/services/ml/models", method="POST", body={"entity": entity, "kind": kind})
                return ["Модель создана. Можно запустить обучение.", "notice", no_update,
                        (refresh or 0) + 1, model["id"], no_update, no_update, no_update]
            if trigger == "job-poll":
                if not job:
                    return unchanged
                result = master("/services/ml/jobs/" + job["job_id"])
                progress = f' · {result.get("labeled", 0)}/{result["total"]}' if result.get("total") else ""
                label = f'Обучение · {result.get("stage", result["status"])}{progress}'
                if result["status"] in {"queued", "running"}:
                    unchanged[7] = label
                    return unchanged
                if result["status"] == "failed":
                    return [str(result.get("error", "Ошибка обучения.")), "notice error", no_update,
                            no_update, no_update, None, True, label]
                message = result.get("reason", "Обучение завершено. Можно оценить комментарии.")
                if result["status"] == "skipped" and result.get("teacher") == "reviewed":
                    message = "Новой проверенной разметки нет. Для оценки комментариев используется текущая версия."
                return [message, "notice", no_update, (refresh or 0) + 1,
                        no_update, None, True, label]
            if not model_id:
                raise RuntimeError("Сначала создай модель.")
            if trigger == "train":
                started = master(f"/services/ml/models/{model_id}/train", method="POST", body={
                    "teacher": teacher, "mode": mode, "limit": 10000 if teacher == "dataset" else 3000,
                })
                return ["Обучение запущено.", "notice", no_update, no_update,
                        no_update, started, False, "Подготовка данных…"]
            if trigger == "score":
                result = master("score", method="POST", body={"model_id": model_id, "entity": entity, "limit": 10000})
                return [f'Версия {result["version"]} оценила {result["processed"]} комментариев.', "notice",
                        no_update, (refresh or 0) + 1, no_update, no_update, no_update, no_update]
        except RuntimeError as error:
            return [str(error), "notice error", *unchanged[2:]]
        return unchanged

    @dashboard.callback(Output("next", "disabled"), Input("cursor", "data"))
    def continuation(cursor):
        return not bool(cursor)

    @dashboard.callback(
        Output("chart", "figure"), Output("summary", "children"), Output("scale-description", "children"),
        Output("model-metrics", "children"), Output("report", "href"), Output("report", "style"),
        Output("model-state", "data"), Output("timeline", "data"),
        Output("source-breakdown", "children"),
        Input("model", "value"), Input("bucket", "value"), Input("entity", "value"), Input("refresh", "data"),
        State("score-scale", "data"),
    )
    def timeline(model_id, bucket, entity, _refresh, scale):
        if not entity:
            return empty_figure(), [], "Выбери сущность для мониторинга.", [], "", {"display": "none"}, None, {}, []
        try:
            model = master(f"/services/ml/models/{model_id}/summary") if model_id else None
            active = model is not None and model.get("version") is not None
            params = {"entity": entity, "bucket": bucket}
            if active:
                params["model_id"] = model_id
            analytics = master("analytics", params=params)
            points = analytics["periods"]
            hint = f'В собранных комментариях. Шкала {scale["min"]:g}–{scale["max"]:g}, нейтральная точка {(scale["min"] + scale["max"]) / 2:g}.'
            hint += " Нажми на график, чтобы открыть комментарии за период." if active else " Обучи модель и оцени комментарии, чтобы увидеть тональность."
            return (timeline_figure(points, scale), summary(analytics["summary"]), hint,
                    model_metrics(model) if active else [],
                    "/services/web_master/api/report?" + urlencode(params) if active else "", {} if active else {"display": "none"}, model,
                    {"entity": entity, "model_id": model_id if active else None, "bucket": bucket, "points": points},
                    source_breakdown(analytics["sources"]))
        except RuntimeError as error:
            return empty_figure(str(error)), [], str(error), [], "", {"display": "none"}, None, {}, []

    @dashboard.callback(
        Output("evidence", "children"), Output("evidence-caption", "children"),
        Output("selected-period", "children"), Output("selection", "data"),
        Input("chart", "clickData"), Input("find", "n_clicks"), Input("clear-selection", "n_clicks"),
        Input("timeline", "data"), State("search", "value"), State("selection", "data"),
    )
    def evidence(clicked, _find, _clear, data, query, previous):
        if not data:
            return [], "", "", None
        try:
            if ctx.triggered_id == "find" and query and query.strip():
                result = master("comments", params={"entity": data["entity"], "query": query.strip(), "limit": 30})
                return comments(result["items"], scored=False), count_caption(result), "Результаты поиска по архиву", None
            if not data.get("model_id"):
                result = master("comments", params={"entity": data["entity"], "limit": 30})
                return comments(result["items"], scored=False), count_caption(result), "Последние собранные комментарии", None
            period = None
            if ctx.triggered_id == "chart" and clicked and clicked.get("points"):
                period = clicked["points"][0].get("customdata")
            params = {"entity": data["entity"], "model_id": data["model_id"], "limit": 30}
            if period:
                first = date.fromisoformat(period)
                end = first + timedelta(days=7 if data["bucket"] == "week" else 1)
                params.update(from_date=period, to_date=end.isoformat())
            result = master("comments", params=params)
            label = "Период с " + period if period else "Последние оценки"
            return comments(result["items"]), count_caption(result), label, period
        except (RuntimeError, ValueError) as error:
            return [html.P(str(error), className="notice error")], "", "", previous

    @dashboard.callback(
        Output("status", "children"), Output("loads", "children"),
        Input("refresh", "data"), Input("status-poll", "n_intervals"),
    )
    def service_status(_refresh, _ticks):
        try:
            state = master("status")
            services = state["services"]
            history = state["loads"][:5]
            status = [html.Span(
                name + " · " + ("OK" if row["status"] in {"ok", "healthy", "ready"} else "недоступен"),
                className="badge " + ("ok" if row["status"] in {"ok", "healthy", "ready"} else "bad"),
            ) for name, row in services.items()]
            loads = [html.P(" · ".join(str(row.get(key, "")) for key in ("timestamp", "entity", "source", "status")),
                            className="metadata") for row in history]
            return status, loads or [html.P("Загрузок пока нет.", className="subtle")]
        except RuntimeError as error:
            return [html.P(str(error), className="notice error")], []

    return dashboard


def empty_figure(message="Загрузи данные, обучи модель и нажми «Оценить комментарии». "):
    figure = go.Figure()
    figure.add_annotation(text=message, x=.5, y=.5, xref="paper", yref="paper", showarrow=False)
    figure.update_layout(template="plotly_white", height=370, autosize=True, margin=dict(l=35, r=25, t=30, b=35),
                          paper_bgcolor="#fffdfa", plot_bgcolor="#fffdfa", font=dict(color="#6a6057", size=14))
    figure.update_xaxes(visible=False)
    figure.update_yaxes(visible=False)
    return figure


def timeline_figure(points, scale):
    if not points:
        return empty_figure("У этой версии пока нет оценок. Нажми «Оценить комментарии». ")
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
        return html.P("Загрузи первый батч, чтобы увидеть статистику по источникам.", className="subtle")
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
        return [html.P("В выбранном периоде комментариев нет. Выбери другой период.", className="subtle")]
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


def model_metrics(model):
    metrics = model.get("metrics") or {}
    holdout = metrics.get("holdout") or {}
    fidelity = holdout.get("teacher_fidelity_candidates") or holdout.get("teacher_fidelity") or {}
    values = [html.Dt("Версия"), html.Dd(str(model["version"])),
              html.Dt("Классификатор"), html.Dd(model["kind"]),
              html.Dt("Разметка"), html.Dd(TEACHER_LABELS.get(model.get("teacher"), model.get("teacher") or "—")),
              html.Dt("Проверочных комментариев"), html.Dd(str(holdout.get("items", "—")))]
    if "class_agreement" in fidelity:
        label = "Совпадение с разметкой" if model.get("teacher") in {"dataset", "reviewed"} else "Совпадение с учителем"
        values.extend([html.Dt(label), html.Dd(f'{fidelity["class_agreement"]:.1%}')])
    reference = metrics.get("assistant_reference") or {}
    quality = reference.get("student") or holdout.get("assistant_reviewed") or {}
    if quality:
        values.extend([
            html.Dt("F1 полярности"), html.Dd(f'{quality["sentiment_polarity_macro_f1"]:.1%}' if quality.get("sentiment_polarity_macro_f1") is not None else "—"),
            html.Dt("Точность полярности"), html.Dd(f'{quality["sentiment_polarity_accuracy"]:.1%}' if quality.get("sentiment_polarity_accuracy") is not None else "—"),
            html.Dt("Ошибка балла (MAE)"), html.Dd(f'{quality["score_mae"]:.2f}' if quality.get("score_mae") is not None else "—"),
        ])
    result = [html.Dl(values)]
    if quality:
        label = "Слепая разметка ассистента" if reference.get("student") else "Отложенная обучающая разметка ассистента"
        result.append(html.P(f'{label}: {quality["items"]} комментариев. Требует подтверждения человеком.', className="subtle"))
        if quality.get("sentiment_polarity_items") != quality["items"]:
            result.append(html.P(
                f'Точность полярности считается по {quality["sentiment_polarity_items"]} '
                'комментариям, размеченным как относящиеся к сущности.', className="subtle",
            ))
        if quality.get("unscored_relevant_items"):
            result.append(html.P(
                f'Балл получен для {quality["scored_items"]} комментариев по теме; '
                f'{quality["unscored_relevant_items"]} отклонены моделью. MAE считается по полученным баллам.',
                className="subtle",
            ))
    elif model.get("teacher") == "dataset":
        result.append(html.P("Согласие с готовыми метками датасета; это не независимая оценка точности человеком.", className="subtle"))
    elif model.get("teacher") == "reviewed":
        result.append(html.P("Обучение на проверенной ассистентом разметке. Новые комментарии участвуют в обучении после разметки.", className="subtle"))
    if holdout.get("quality_status") == "unavailable":
        result.append(html.P("Для этой версии нет размеченных проверочных комментариев. Качество после дообучения ещё не измерено.", className="subtle"))
    result.append(html.Details([html.Summary("Все метрики"), html.Pre(json.dumps(metrics, ensure_ascii=False, indent=2))]))
    history = [html.P(f'v{row["version"]} · {row.get("mode", "from_scratch")} · {row["trained_at"]}', className="metadata")
               for row in model.get("versions", [])]
    if history:
        result.append(html.Details([html.Summary("История версий"), *history]))
    return result
