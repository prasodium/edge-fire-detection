"""GPIO control for buzzer, relay (siren/FACP interlock) and status LED.

Uses gpiozero on a real Pi; falls back to a logging mock everywhere else so
the alarm/dashboard code can be developed and unit-tested off-Pi.
"""
from __future__ import annotations

import threading
import time

from utils.logger import get_logger

logger = get_logger("alarm.gpio")

try:
    from gpiozero import Buzzer, LED, OutputDevice  # type: ignore

    _HAS_GPIOZERO = True
except (ImportError, Exception):  # gpiozero raises on import off-Pi without mock pin factory
    _HAS_GPIOZERO = False


class _MockOutputDevice:
    """Drop-in replacement for gpiozero output devices on non-Pi hosts."""

    def __init__(self, pin: int, name: str) -> None:
        self._pin = pin
        self._name = name
        self.is_active = False

    def on(self) -> None:
        self.is_active = True
        logger.debug("[MOCK GPIO] %s (pin %s) -> ON", self._name, self._pin)

    def off(self) -> None:
        self.is_active = False
        logger.debug("[MOCK GPIO] %s (pin %s) -> OFF", self._name, self._pin)

    def close(self) -> None:
        pass


class GpioController:
    """Owns the buzzer, relay and status LED. Thread-safe, idempotent on/off."""

    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg
        self._lock = threading.Lock()
        self._blink_thread: threading.Thread | None = None
        self._blink_stop = threading.Event()

        if _HAS_GPIOZERO and cfg.get("backend", "gpiozero") == "gpiozero":
            self._buzzer = Buzzer(cfg["buzzer_pin"])
            self._relay = OutputDevice(cfg["relay_pin"], active_high=cfg.get("active_high", True))
            self._led = LED(cfg["led_pin"])
            logger.info("GpioController using real gpiozero backend")
        else:
            self._buzzer = _MockOutputDevice(cfg.get("buzzer_pin", -1), "buzzer")
            self._relay = _MockOutputDevice(cfg.get("relay_pin", -1), "relay")
            self._led = _MockOutputDevice(cfg.get("led_pin", -1), "led")
            logger.info("GpioController using mock backend (not running on a Pi / gpiozero unavailable)")

    def activate_alarm(self, blink_hz: float = 4.0) -> None:
        with self._lock:
            self._buzzer.on()
            self._relay.on()
            self._start_led_blink(blink_hz)
        logger.warning("GPIO alarm ACTIVATED (buzzer + relay + LED)")

    def silence(self) -> None:
        with self._lock:
            self._buzzer.off()
            self._relay.off()
            self._stop_led_blink()
            self._led.off()
        logger.info("GPIO alarm SILENCED")

    def _start_led_blink(self, hz: float) -> None:
        self._stop_led_blink()
        self._blink_stop.clear()
        interval = 1.0 / (2 * max(hz, 0.1))

        def _blink() -> None:
            while not self._blink_stop.is_set():
                self._led.on()
                time.sleep(interval)
                self._led.off()
                time.sleep(interval)

        self._blink_thread = threading.Thread(target=_blink, name="led-blink", daemon=True)
        self._blink_thread.start()

    def _stop_led_blink(self) -> None:
        self._blink_stop.set()
        if self._blink_thread is not None:
            self._blink_thread.join(timeout=1.0)
            self._blink_thread = None

    def close(self) -> None:
        self.silence()
        for device in (self._buzzer, self._relay, self._led):
            try:
                device.close()
            except Exception:
                logger.exception("Error closing GPIO device")
