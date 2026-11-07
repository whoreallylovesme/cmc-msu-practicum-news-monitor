"""Validated client for the independent model-labeling service."""

from lib.teacher_contracts import TeacherRequest, TeacherResponse, input_hash


class TeacherService:
    def __init__(self, config, session):
        self.session = session
        self.url = str(config.get("teacher_service_url", "")).rstrip("/")
        if not self.url.startswith(("http://", "https://")):
            raise ValueError("Set teacher_service_url before requesting service labels")
        self.timeout = float(config.get("teacher_service_timeout", 180))
        self.metadata = self._request("GET", "/metadata")
        revision = self.metadata.get("revision") if isinstance(self.metadata, dict) else None
        if not isinstance(revision, str) or not revision:
            raise ValueError("Teacher service returned no model revision")
        self.revision = revision
        self.namespace = "service-" + revision

    def _request(self, method, path, body=None):
        response = self.session.request(method, self.url + path, json=body, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def label(self, entity, items, aliases=(), matching=None):
        request = TeacherRequest.model_validate({
            "entity": entity, "aliases": list(aliases), "matching": matching or {}, "items": items,
        })
        response = TeacherResponse.model_validate(self._request("POST", "/label", request.model_dump()))
        if response.revision != self.revision:
            raise ValueError("Teacher revision changed during training; retry with one pinned revision")
        expected = {item.id: input_hash(entity, item, aliases, matching) for item in request.items}
        if len(response.items) != len(expected) or len({item.comment_id for item in response.items}) != len(expected):
            raise ValueError("Teacher response must contain each requested comment exactly once")
        if any(expected.get(item.comment_id) != item.input_hash for item in response.items):
            raise ValueError("Teacher labels do not match the requested full input context")
        return response.items
