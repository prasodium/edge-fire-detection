"""Outbound alarm notifications: MQTT (primary - works fully offline on a
local broker, appropriate for a building with no guaranteed internet),
Firebase Cloud Messaging (optional, for remote/mobile push), and a generic
webhook (optional, for integration with existing building management
systems).

Each channel fails independently and logs rather than raising, so a
notification outage never blocks the local GPIO alarm (which is the safety-
critical path and must never depend on network availability).
"""
from __future__ import annotations

import json
from typing import Any

from utils.logger import get_logger

logger = get_logger("alarm.notifier")

try:
    import paho.mqtt.client as mqtt

    _HAS_MQTT = True
except ImportError:
    _HAS_MQTT = False


class MqttNotifier:
    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg
        self._client = None
        if _HAS_MQTT and cfg.get("enabled", True):
            self._client = mqtt.Client()
            try:
                self._client.connect(cfg["broker_host"], cfg.get("broker_port", 1883), keepalive=30)
                self._client.loop_start()
                logger.info("Connected to MQTT broker %s:%s", cfg["broker_host"], cfg.get("broker_port"))
            except Exception:
                logger.exception("MQTT connect failed - notifications on this channel will be skipped")
                self._client = None
        elif cfg.get("enabled", True):
            logger.warning("paho-mqtt not installed - MQTT notifications disabled")

    def publish(self, payload: dict[str, Any]) -> None:
        if self._client is None:
            return
        try:
            self._client.publish(
                self._cfg["topic"], json.dumps(payload), qos=self._cfg.get("qos", 1),
                retain=self._cfg.get("retain", False),
            )
        except Exception:
            logger.exception("MQTT publish failed")


class FirebaseNotifier:
    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg
        self._app = None
        if cfg.get("enabled", False):
            try:
                import firebase_admin
                from firebase_admin import credentials

                creds = credentials.Certificate(cfg["credentials_path"])
                self._app = firebase_admin.initialize_app(creds)
                logger.info("Firebase Admin SDK initialized")
            except Exception:
                logger.exception("Firebase init failed - push notifications disabled")
                self._app = None

    def publish(self, payload: dict[str, Any]) -> None:
        if self._app is None:
            return
        try:
            from firebase_admin import messaging

            message = messaging.Message(
                notification=messaging.Notification(
                    title="FIRE ALARM",
                    body=f"{payload.get('context_label')} (confidence {payload.get('confidence'):.0%})",
                ),
                data={k: str(v) for k, v in payload.items()},
                topic=self._cfg.get("fcm_topic", "fire_alerts"),
            )
            messaging.send(message)
        except Exception:
            logger.exception("Firebase publish failed")


class WebhookNotifier:
    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg

    def publish(self, payload: dict[str, Any]) -> None:
        if not self._cfg.get("enabled", False) or not self._cfg.get("url"):
            return
        try:
            import requests

            requests.post(self._cfg["url"], json=payload, timeout=3)
        except Exception:
            logger.exception("Webhook publish failed")


class NotificationDispatcher:
    """Fans an alarm event out to every configured channel."""

    def __init__(self, cfg: dict) -> None:
        self._mqtt = MqttNotifier(cfg.get("mqtt", {})) if cfg.get("mqtt", {}).get("enabled") else None
        self._firebase = FirebaseNotifier(cfg.get("firebase", {}))
        self._webhook = WebhookNotifier(cfg.get("webhook", {}))

    def notify(self, payload: dict[str, Any]) -> None:
        if self._mqtt is not None:
            self._mqtt.publish(payload)
        self._firebase.publish(payload)
        self._webhook.publish(payload)
