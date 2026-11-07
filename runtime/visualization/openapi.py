"""Combine service contracts for the diagnostic Swagger page."""

SERVICES = (
    ("collector", "Collector"), ("storage", "Storage"), ("ml", "ML"),
    ("web_master", "Web Master"), ("visualization", "Visualization"), ("teacher", "Teacher"),
)
METHODS = {"get", "post", "put", "patch", "delete", "options", "head", "trace"}
DESCRIPTIONS = {
    "collector": "Источники, получение батча и журнал загрузок.",
    "storage": "CRUD комментариев, сохранение оценок и разметки, агрегация данных.",
    "ml": "Все операции моделей: создание, обучение, проверка, предсказания и состояние задач. Читателю доступны GET-запросы.",
    "web_master": "Сценарии конвейера: импорт → пакетная оценка → аналитика и отчёт. Методов управления моделями здесь нет.",
    "visualization": "Интерактивный дашборд доступен по адресу /. REST-маршрут проверяет его доступность.",
    "teacher": "Метаданные учителя и разметка примеров. Опциональный сервис для обучения.",
}


def _rewrite(value, service):
    if isinstance(value, dict):
        return {key: (service + "__" + item if key == "operationId" else _rewrite(item, service))
                for key, item in value.items()}
    if isinstance(value, list):
        return [_rewrite(item, service) for item in value]
    if isinstance(value, str):
        if value.startswith("#/components/"):
            parts = value.split("/")
            if parts[2] == "securitySchemes":
                return "#/components/securitySchemes/HTTPBasic"
            parts[3] = service + "__" + parts[3]
            return "/".join(parts)
        if value.startswith("#/paths/"):
            return "#/paths/~1services~1" + service + value[len("#/paths/"):]
    return value


def merge_schemas(schemas, unavailable=()):
    result = {
        "openapi": "3.1.0",
        "info": {"title": "HN Monitor — all services", "version": "1.0.0",
                 "description": "Контракты сгруппированы по сервисам. Войдите через HTTP Basic. "
                 "Управление моделями находится только в ML; читатель может читать модели и результаты, "
                 "операции записи доступны администратору. Баллы через gateway используют настроенную шкалу."},
        "tags": [{"name": title, "description": DESCRIPTIONS[service]} for service, title in SERVICES],
        "paths": {},
        "components": {"securitySchemes": {"HTTPBasic": {"type": "http", "scheme": "basic"}}},
        "security": [{"HTTPBasic": []}],
    }
    if unavailable:
        result["info"]["description"] += " Unavailable optional services: " + ", ".join(unavailable) + "."
    for service, title in SERVICES:
        if service not in schemas:
            continue
        schema = _rewrite(schemas[service], service)
        for category, definitions in schema.get("components", {}).items():
            if category != "securitySchemes":
                result["components"].setdefault(category, {}).update(
                    {service + "__" + name: value for name, value in definitions.items()})
        for path, item in schema.get("paths", {}).items():
            item.pop("servers", None)
            for method, operation in item.items():
                if method in METHODS:
                    operation["tags"] = [title]
                    operation["security"] = [{"HTTPBasic": []}]
                    operation.setdefault("operationId", service + "__" + method + "__" + path)
                    operation.pop("servers", None)
            result["paths"]["/services/" + service + path] = item
    return result
