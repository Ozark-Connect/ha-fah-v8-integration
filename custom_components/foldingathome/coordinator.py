"""DataUpdateCoordinator for Folding@home."""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import timedelta
from typing import Any

import aiohttp
from homeassistant.core import HomeAssistant
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed

from .const import DOMAIN, WEBSOCKET_PATH, UPDATE_INTERVAL, WEBSOCKET_TIMEOUT

_LOGGER = logging.getLogger(__name__)


class FAHDataUpdateCoordinator(DataUpdateCoordinator[dict[str, Any]]):
    """Coordinator for FAH client data."""

    def __init__(self, hass: HomeAssistant, host: str, port: int) -> None:
        """Initialize coordinator."""
        super().__init__(
            hass,
            _LOGGER,
            name=f"{DOMAIN}_{host}",
            update_interval=timedelta(seconds=UPDATE_INTERVAL),
        )
        self.host = host
        self.port = port
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._session: aiohttp.ClientSession | None = None
        self._listen_task: asyncio.Task | None = None
        # At most one reconnect loop runs at a time. Parallel loops close each
        # other's sockets and keep the connection in a disconnect/reconnect cycle.
        self._reconnect_task: asyncio.Task | None = None
        self._connect_lock = asyncio.Lock()
        self._shutdown: bool = False
        self._connected: bool = False
        self._reconnect_delay: int = 10  # Start with 10 seconds

    @property
    def ws_url(self) -> str:
        """Get WebSocket URL."""
        return f"ws://{self.host}:{self.port}{WEBSOCKET_PATH}"

    @property
    def machine_id(self) -> str | None:
        """Get machine ID from data."""
        if self.data:
            info = self.data.get("info") or {}
            return info.get("id")
        return None

    @property
    def machine_name(self) -> str:
        """Get machine name from data."""
        if self.data:
            info = self.data.get("info") or {}
            return info.get("mach_name", "FAH Client")
        return "FAH Client"

    async def async_initialize(self) -> None:
        """Initialize the coordinator - try to connect but don't fail if offline."""
        _LOGGER.info("Initializing FAH coordinator for %s", self.host)
        try:
            await self._connect()
        except Exception as err:
            _LOGGER.warning(
                "FAH client %s not available at startup: %s. Will keep trying.",
                self.host, err
            )
            self._mark_disconnected()
            # Start reconnection loop
            self._schedule_reconnect()

    def _mark_disconnected(self) -> None:
        """Mark the client unreachable so its entities become unavailable.

        Without this the entities keep the last state they received. A rig
        that went to sleep while folding then still reads as folding, and
        automations that wake a rig only when its switch is off never wake it.
        """
        self._connected = False
        if self.last_update_success:
            self.async_set_update_error(UpdateFailed(f"FAH client {self.host} is not connected"))

    def _apply_incremental_update(self, update: list) -> None:
        """Apply an incremental update to the stored state."""
        if not self.data or len(update) < 2:
            return

        new_data = dict(self.data)
        path = update[:-1]
        value = update[-1]

        current = new_data
        for i, key in enumerate(path[:-1]):
            if isinstance(current, dict):
                if key not in current or current[key] is None:
                    current[key] = {}
                if isinstance(current[key], dict):
                    current[key] = dict(current[key])
                current = current[key]
            elif isinstance(current, list):
                idx = int(key) if isinstance(key, (int, str)) and str(key).isdigit() else key
                if isinstance(idx, int) and 0 <= idx < len(current):
                    if current[idx] is None:
                        current[idx] = {}
                    if isinstance(current[idx], dict):
                        current[idx] = dict(current[idx])
                    current = current[idx]
                else:
                    return
            else:
                return

        final_key = path[-1]
        if isinstance(current, dict):
            current[final_key] = value
            if path == ["groups", "", "config", "paused"]:
                _LOGGER.info("FAH %s paused changed to: %s", self.host, value)
            elif path == ["groups", "", "config", "finish"]:
                _LOGGER.info("FAH %s finish changed to: %s", self.host, value)
            self.async_set_updated_data(new_data)
        elif isinstance(current, list) and isinstance(final_key, int):
            if 0 <= final_key < len(current):
                current[final_key] = value
                self.async_set_updated_data(new_data)

    async def _async_update_data(self) -> dict[str, Any]:
        """Called by HA coordinator on interval - just return current data."""
        # Don't try to connect here - let the listener/reconnect handle it
        # This prevents two things trying to connect at once
        if not self._connected:
            raise UpdateFailed(f"FAH client {self.host} is not connected")
        return self.data or {}

    async def _connect(self) -> None:
        """Establish WebSocket connection.

        The lock serializes callers (reconnect loop, user commands), so one
        caller never tears down a socket that another caller just opened.
        """
        async with self._connect_lock:
            if self._shutdown:
                return
            await self._connect_locked()

    async def _connect_locked(self) -> None:
        """Open a new connection. The caller holds _connect_lock."""
        # Stop the old listener and close the old socket first
        await self._stop_listener()

        _LOGGER.info("Connecting to FAH client at %s", self.ws_url)

        self._session = aiohttp.ClientSession()
        try:
            ws = await self._session.ws_connect(
                self.ws_url,
                timeout=aiohttp.ClientTimeout(total=WEBSOCKET_TIMEOUT),
            )
            self._ws = ws

            # Wait for initial state
            msg = await ws.receive(timeout=WEBSOCKET_TIMEOUT)
            if msg.type == aiohttp.WSMsgType.TEXT:
                data = msg.data
                if data != "ping":
                    parsed = json.loads(data)
                    if parsed is not None and isinstance(parsed, dict):
                        self.async_set_updated_data(parsed)
                        _LOGGER.info("Connected to FAH %s, received initial state", self.host)
            elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                raise aiohttp.ClientError(f"WebSocket closed immediately: {msg.type}")

            # Connection successful
            self._connected = True
            self._reconnect_delay = 10  # Reset backoff on success
            if not self.last_update_success:
                # Entities become available again (the first message may not
                # have been a full state dict)
                self.async_set_updated_data(self.data or {})

            # Start a listener bound to this socket
            self._listen_task = self.hass.async_create_background_task(
                self._listen(ws), f"fah_listener_{self.host}"
            )

        except BaseException:
            # BaseException: a cancelled connect must also release the session
            self._mark_disconnected()
            await self._cleanup()
            raise

    async def _stop_listener(self) -> None:
        """Close the current socket and wait for its listener to exit."""
        task = self._listen_task
        self._listen_task = None
        # Cleanup first: self._ws no longer matches the listener's socket,
        # so the listener exits without scheduling a reconnect.
        await self._cleanup()
        if task is not None and not task.done() and task is not asyncio.current_task():
            task.cancel()
            await asyncio.wait([task])

    async def _listen(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        """Listen for WebSocket messages on one socket."""
        _LOGGER.debug("FAH WebSocket listener started for %s", self.host)
        # FAH sends updates every ~1 second when active, use 60s timeout to detect dead connections
        receive_timeout = 60

        try:
            while not self._shutdown:
                try:
                    msg = await asyncio.wait_for(
                        ws.receive(),
                        timeout=receive_timeout
                    )
                except asyncio.TimeoutError:
                    _LOGGER.warning("FAH %s: no data received for %ds, assuming dead connection", self.host, receive_timeout)
                    break

                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = msg.data
                    if data == "ping":
                        continue
                    try:
                        parsed = json.loads(data)
                        if parsed is None:
                            continue
                        if isinstance(parsed, dict):
                            self.async_set_updated_data(parsed)
                        elif isinstance(parsed, list) and len(parsed) >= 2:
                            self._apply_incremental_update(parsed)
                    except json.JSONDecodeError:
                        _LOGGER.warning("Invalid JSON from FAH %s", self.host)
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    _LOGGER.error("WebSocket error from %s: %s", self.host, ws.exception())
                    break
                elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSING, aiohttp.WSMsgType.CLOSED):
                    _LOGGER.info("WebSocket to %s closed", self.host)
                    break

        except asyncio.CancelledError:
            _LOGGER.debug("WebSocket listener for %s cancelled", self.host)
            raise
        except Exception as err:
            _LOGGER.error("WebSocket listener error for %s: %s", self.host, err)
        finally:
            # A newer connection replaced this socket: leave its state alone
            if self._ws is ws:
                self._mark_disconnected()
                await self._cleanup()
                if not self._shutdown:
                    _LOGGER.info("FAH %s disconnected, will reconnect", self.host)
                    self._schedule_reconnect()

    def _schedule_reconnect(self) -> None:
        """Start the reconnect loop if it is not already running."""
        if self._shutdown:
            return
        if self._reconnect_task is not None and not self._reconnect_task.done():
            return

        self._reconnect_task = self.hass.async_create_background_task(
            self._reconnect(), f"fah_reconnect_{self.host}"
        )

    def _cancel_reconnect(self) -> None:
        """Cancel a pending reconnect loop."""
        task = self._reconnect_task
        self._reconnect_task = None
        if task is not None and not task.done() and task is not asyncio.current_task():
            task.cancel()

    async def _reconnect(self) -> None:
        """Retry the connection with backoff until it succeeds or shutdown."""
        while not self._shutdown:
            _LOGGER.info("FAH %s: reconnecting in %d seconds", self.host, self._reconnect_delay)
            await asyncio.sleep(self._reconnect_delay)

            if self._shutdown:
                return

            # Increase delay for next time (exponential backoff, max 30 sec)
            self._reconnect_delay = min(self._reconnect_delay * 2, 30)

            try:
                await self._connect()
                return
            except Exception as err:
                _LOGGER.warning("FAH %s reconnection failed: %s", self.host, err)

    async def _cleanup(self) -> None:
        """Clean up WebSocket and session."""
        if self._ws is not None:
            try:
                if not self._ws.closed:
                    await self._ws.close()
            except Exception:
                pass
            self._ws = None

        if self._session is not None:
            try:
                if not self._session.closed:
                    await self._session.close()
            except Exception:
                pass
            self._session = None

    async def async_shutdown(self) -> None:
        """Shutdown coordinator."""
        _LOGGER.info("Shutting down FAH coordinator for %s", self.host)
        self._shutdown = True
        self._cancel_reconnect()
        await self._stop_listener()

    async def async_send_command(self, command: dict[str, Any]) -> None:
        """Send command to FAH client."""
        _LOGGER.info("Sending command to FAH %s: %s (connected: %s)", self.host, command, self._connected)

        # Reset backoff - user wants action now
        self._reconnect_delay = 10

        # If not connected, try to connect first
        if not self._connected or self._ws is None or self._ws.closed:
            _LOGGER.info("FAH %s: not connected, attempting connection for command", self.host)
            # Connect now instead of waiting for the backoff delay
            self._cancel_reconnect()
            try:
                await self._connect()
            except Exception as err:
                _LOGGER.error("FAH %s: cannot send command, connection failed: %s", self.host, err)
                self._schedule_reconnect()
                return

        # Try to send
        if self._ws is not None and not self._ws.closed:
            try:
                await self._ws.send_str(json.dumps(command))
                _LOGGER.info("Sent command to FAH %s: %s", self.host, command)
            except Exception as err:
                _LOGGER.error("FAH %s: failed to send command: %s", self.host, err)
                self._mark_disconnected()
                self._schedule_reconnect()
