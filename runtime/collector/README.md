# Collector

`GET /sources` перечисляет зарегистрированные источники. `POST /batches` принимает `entity`, `aliases`, `source`, `limit` и необязательный `cursor`. Ответ содержит `items` и `next_cursor`. `limit` считается в комментариях. Для продолжения передайте тот же источник, сущность и псевдонимы. Один тред может продолжаться в нескольких батчах.

Для демонстрации на реальных данных используйте `source: hn_snapshot` и сущность `ChatGPT`. В `demo/hn_10k.json` находятся 10 000 публичных комментариев из 60 обсуждений, а `demo/hn_1k.json` содержит 1000 строк — точное подмножество большой версии. 48 тредов 2024–2025 годов используются для обучения, 12 других тредов 2026 года — для проверки. В нём есть даты, заголовки, контекст родительских ответов и ссылки на оригиналы. В демо-снимок добавлены готовые A–F метки с источником и отпечатком контекста; они позволяют обучать малый классификатор без учителя. Окно выборки, правила отбора и контрольная сумма записаны в `demo/hn_10k.metadata.json` и `demo/hn_1k.metadata.json`. `python runtime/collector/fetch_snapshot.py` из корня проекта повторяет сбор через публичный Algolia API в `.local-data/hn_snapshot.raw.json`, не заменяя размеченное демо; `--small-demo` оставляет только 144 поздних комментария. Результат может измениться после редактирования или удаления записей в индексе.

`hn_snapshot` возвращает батчи выбранной реальной выгрузки; `hacker_news` запрашивает обсуждения онлайн.

## Как добавить источник

1. Унаследуйте класс от `NewsProvider` из `lib/news.py`. Для источника с сетевыми запросами задайте `requires_network = True`. Тогда офлайн-режим скроет его и отклонит прямой запрос к нему.
2. В `batch(terms, cursor, limit)` приведите данные API к `News` с вложенными `NewsComment`. Используйте даты с часовым поясом, стабильные идентификаторы и URL оригиналов. Для числовых идентификаторов других платформ добавьте префикс источника, например `example:42`, чтобы исключить совпадения с HN.
3. Импортируйте новый класс в `runtime/collector/main.py` и зарегистрируйте его вызовом `register_provider(YourProvider(...))`. Он появится в `GET /sources`. Collector преобразует `News` и `NewsComment` в общий ответ для остальных сервисов.

```python
from datetime import datetime, timezone

from lib.news import News, NewsComment, NewsProvider


class ExampleProvider(NewsProvider):
    def __init__(self):
        super().__init__("example", "Example media")

    def batch(self, terms, cursor, limit):
        date = datetime(2026, 10, 1, tzinfo=timezone.utc)
        comment = NewsComment(
            id="example:comment:1",
            date=date,
            text="The new feature worked well for me.",
            url="https://example.invalid/comments/1",
        )
        news = News(
            id="example:story:1",
            source=self.name,
            date=date,
            title="Example product release",
            description="Summary from the source API",
            url="https://example.invalid/stories/1",
            comments=(comment,),
        )
        if not any(term.casefold() in news.title.casefold() for term in terms):
            return [], None
        return [news], None
```
